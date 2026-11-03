########################################################################################################################
# Retrain the best transformer configuration with an EMA shadow model for stable epoch selection.
# Supports multi-seed retraining and shadow-only ensembling (strategy b: average sigmoid probabilities
# across seeds pre-calibration, fit one calibrator per method on the ensemble validation probability,
# apply to the ensemble test probability).
# Reference: Morales-Brotons et al. (2024). arXiv:2411.18704 (EMA decay = 0.999).
# Saves each seed's raw and shadow models as .pth files, plus a companion context .pth for SHAP reload.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import copy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from time import time
from torch.utils.data import DataLoader
from typing import Optional
from .Baseline_Retraining import _calibrate_and_build_probs, _evaluate_test
from .Logger import Logger
from .Transformer_Modeling import ArrayDataset, TransformerClassifier

########################################################################################################################
# Compute weighted BCE loss over a DataLoader (no grad)
########################################################################################################################


def _weighted_loss(M: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   cost_mul: int,
                   device: torch.device) -> float:
    """
    :param M: nn.Module. Model in eval mode.
    :param loader: DataLoader. Data to evaluate loss on.
    :param criterion: nn.Module. Per-sample loss (reduction='none').
    :param cost_mul: int. Multiplier for negative class.
    :param device: torch.device.
    :return: float. Weight-normalized mean loss.
    """
    running_loss: float = 0.0
    running_weight: float = 0.0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            logits = M(xb).view(-1)
            loss_raw = criterion(logits, yb)
            w = torch.where(yb == 0, torch.full_like(yb, cost_mul), torch.ones_like(yb))
            running_loss += (loss_raw * w).sum().item()
            running_weight += w.sum().item()
    return running_loss / (running_weight + 1e-12)

########################################################################################################################
# RNG-protected wrapper — save/restore RNG state around a callable that must not advance it
########################################################################################################################


def _rng_protected(fn):
    """
    Decorator that saves and restores the CPU (and CUDA, if available) RNG state around fn.
    Prevents shadow/test passes from perturbing the raw model's training trajectory.
    """
    def _wrapped(*args, **kwargs):
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        result = fn(*args, **kwargs)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda)
        return result
    return _wrapped


_weighted_loss_protected = _rng_protected(_weighted_loss)

########################################################################################################################
# Collect logits from a DataLoader for a given model
########################################################################################################################


def _collect_logits(M: nn.Module,
                    loader: DataLoader,
                    device: torch.device) -> tuple[np.ndarray, list]:
    """
    :param M: nn.Module. Model in eval mode.
    :param loader: DataLoader. Data to score.
    :param device: torch.device.
    :return: (logits np.ndarray, list of record IDs)
    """
    logits_all: list = []
    rec_ids: list = []
    with torch.no_grad():
        for xb, _, rid in loader:
            xb = xb.to(device)
            logits = M(xb).view(-1)
            logits_all.append(logits.cpu().numpy())
            rec_ids.extend(list(rid))
    logits: np.ndarray = np.concatenate(logits_all)
    logits = np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    return logits, rec_ids

########################################################################################################################
# Train a single (raw + shadow) model pair with a given seed
########################################################################################################################


def _train_single_seed(best_config: dict,
                       train_ds: ArrayDataset,
                       val_ds: ArrayDataset,
                       test_ds: ArrayDataset,
                       seed: int,
                       scheduler_patience: int,
                       ema_decay: float,
                       print_func) -> dict:
    """
    :param best_config: dict. Hyperparameter configuration.
    :param train_ds: ArrayDataset. Training partition.
    :param val_ds: ArrayDataset. Validation partition.
    :param test_ds: ArrayDataset. Test partition.
    :param seed: int. Random seed for this training run.
    :param scheduler_patience: int. Patience for ReduceLROnPlateau.
    :param ema_decay: float. EMA decay rate for shadow model updates.
    :param print_func: callable. logger.info or print, for per-epoch progress.
    :return: dict with keys raw_state_dict, shadow_state_dict, shadow_val_logits, shadow_test_logits,
             raw_val_logits, raw_test_logits, test_rec_ids, best_epoch, best_shadow_epoch, history, elapsed,
             n_params.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ####################################################################################################################
    # Extract configuration
    ####################################################################################################################
    d_model: int = int(best_config['d_model'])
    n_heads: int = int(best_config['n_heads'])
    n_layers: int = int(best_config['n_layers'])
    n_units: int = int(best_config['n_units'])
    lr: float = float(best_config['lr'])
    cost_mul: int = int(best_config['cost_mul'])
    handle_nan: str = str(best_config['handle_nan'])
    factor: float = float(best_config['factor'])
    max_epochs: int = int(best_config['max_epochs'])
    batch_size: int = int(best_config['batch_size'])
    dropout: float = float(best_config['dropout'])

    ####################################################################################################################
    # Device and seeding (seed must be set before loader shuffle order or model init)
    ####################################################################################################################
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    ####################################################################################################################
    # Data loaders
    ####################################################################################################################
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    n_feat: int = train_ds.X.shape[2]

    ####################################################################################################################
    # Raw model, optimizer, scheduler, criterion
    ####################################################################################################################
    M = TransformerClassifier(feature_dim=n_feat,
                              d_model=d_model,
                              n_head=n_heads,
                              n_units=n_units,
                              n_layers=n_layers,
                              task='binary',
                              handle_nan=handle_nan,
                              dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(M.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=factor,
                                                           patience=scheduler_patience)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    ####################################################################################################################
    # Shadow model — EMA copy of raw, never updated by optimizer
    ####################################################################################################################
    shadow = TransformerClassifier(feature_dim=n_feat,
                                   d_model=d_model,
                                   n_head=n_heads,
                                   n_units=n_units,
                                   n_layers=n_layers,
                                   task='binary',
                                   handle_nan=handle_nan,
                                   dropout=dropout).to(device)
    shadow.load_state_dict(copy.deepcopy(M.state_dict()))
    shadow.eval()
    for p in shadow.parameters():
        p.requires_grad_(False)

    ####################################################################################################################
    # Book-keeping
    ####################################################################################################################
    best_val_loss: float = float('inf')
    best_state_dict = None
    best_epoch: int = 0
    best_shadow_val_loss: float = float('inf')
    best_shadow_state_dict = None
    best_shadow_epoch: int = 0
    history: list[dict] = []

    ####################################################################################################################
    # Training loop
    ####################################################################################################################
    t0: float = time()
    for epoch in range(max_epochs):
        M.train()
        train_running_loss: float = 0.0
        train_running_weight: float = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = M(xb).view(-1)
            loss_raw = criterion(logits, yb)
            w = torch.where(yb == 0, torch.full_like(yb, cost_mul), torch.ones_like(yb))
            loss = (loss_raw * w).sum() / (w.sum() + 1e-12)
            train_running_loss += (loss_raw * w).sum().item()
            train_running_weight += w.sum().item()
            loss.backward()
            optimizer.step()

            # EMA update (RNG-protected)
            rng_cpu = torch.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            with torch.no_grad():
                for p_shadow, p_raw in zip(shadow.parameters(), M.parameters()):
                    p_shadow.data.mul_(ema_decay).add_(p_raw.data, alpha=1.0 - ema_decay)
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state(rng_cuda)

        train_loss: float = train_running_loss / (train_running_weight + 1e-12)

        M.eval()
        val_loss: float = _weighted_loss(M, val_loader, criterion, cost_mul, device)
        shadow_val_loss: float = _weighted_loss_protected(shadow, val_loader, criterion, cost_mul, device)
        test_loss: float = _weighted_loss_protected(M, test_loader, criterion, cost_mul, device)
        shadow_test_loss: float = _weighted_loss_protected(shadow, test_loader, criterion, cost_mul, device)

        current_lr: float = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(M.state_dict())
        if shadow_val_loss < best_shadow_val_loss:
            best_shadow_val_loss = shadow_val_loss
            best_shadow_epoch = epoch + 1
            best_shadow_state_dict = copy.deepcopy(shadow.state_dict())

        history.append({'seed': seed,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss_raw': val_loss,
                        'val_loss_shadow': shadow_val_loss,
                        'test_loss_raw': test_loss,
                        'test_loss_shadow': shadow_test_loss,
                        'lr': current_lr})

        if (epoch + 1) % 10 == 0:
            print_func(f'[seed={seed}] Epoch {epoch + 1}/{max_epochs} — '
                       f'train={train_loss:.4f}, val_raw={val_loss:.4f}, val_shadow={shadow_val_loss:.4f}, '
                       f'test_raw={test_loss:.4f}, test_shadow={shadow_test_loss:.4f} '
                       f'(best_raw={best_val_loss:.4f} @ {best_epoch}, '
                       f'best_shadow={best_shadow_val_loss:.4f} @ {best_shadow_epoch}) — '
                       f'LR={current_lr:.2e}')

    elapsed: float = round(time() - t0, 3)

    ####################################################################################################################
    # Restore best states and collect logits from both models
    ####################################################################################################################
    if best_state_dict is not None:
        M.load_state_dict(best_state_dict)
    if best_shadow_state_dict is not None:
        shadow.load_state_dict(best_shadow_state_dict)

    M.eval()
    shadow.eval()
    raw_val_logits, _ = _collect_logits(M, val_loader, device)
    raw_test_logits, _ = _collect_logits(M, test_loader, device)
    shadow_val_logits, _ = _collect_logits(shadow, val_loader, device)
    shadow_test_logits, test_rec_ids = _collect_logits(shadow, test_loader, device)

    return {'raw_state_dict': copy.deepcopy(M.state_dict()),
            'shadow_state_dict': copy.deepcopy(shadow.state_dict()),
            'raw_val_logits': raw_val_logits,
            'raw_test_logits': raw_test_logits,
            'shadow_val_logits': shadow_val_logits,
            'shadow_test_logits': shadow_test_logits,
            'test_rec_ids': test_rec_ids,
            'best_epoch': best_epoch,
            'best_shadow_epoch': best_shadow_epoch,
            'history': history,
            'elapsed': elapsed,
            'n_params': sum(p.numel() for p in M.parameters())}

########################################################################################################################
# Multi-seed transformer retraining with shadow-only ensemble (strategy b)
########################################################################################################################


def tf_retrain(best_config: dict,
               train_ds: ArrayDataset,
               val_ds: ArrayDataset,
               test_ds: ArrayDataset,
               out_dir: str,
               file_suffix: str = '',
               seeds: list[int] = [0, 1, 2, 42],
               scheduler_patience: int = 20,
               ema_decay: float = 0.999,
               logger: Optional[Logger] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param best_config: dict. Best hyperparameter configuration from cross-validation.
    :param train_ds: ArrayDataset. Retraining training partition.
    :param val_ds: ArrayDataset. Retraining validation partition (for calibration).
    :param test_ds: ArrayDataset. Held-out test partition.
    :param out_dir: str. Directory to save .pth model files and context files.
    :param file_suffix: str. Suffix appended to model filenames (e.g., '_unimodal').
    :param seeds: list[int]. Seeds to train with. Each produces a saved raw and shadow model.
    :param scheduler_patience: int. Patience for ReduceLROnPlateau.
    :param ema_decay: float. EMA decay rate for shadow model updates.
    :param logger: Optional Logger. If provided, prints per-epoch progress via logger.info; else uses print.
    :return: A tuple of (df_probs, df_test, df_history) with rows for single-seed and ensemble reports.
    """
    print_func = print if logger is None else logger.info
    os.makedirs(out_dir, exist_ok=True)

    ####################################################################################################################
    # Train each seed and save its .pth files + context companion
    ####################################################################################################################
    per_seed_results: dict[int, dict] = {}
    for seed in seeds:
        print_func(f'=== Training seed {seed} ===')
        result: dict = _train_single_seed(best_config=best_config,
                                          train_ds=train_ds,
                                          val_ds=val_ds,
                                          test_ds=test_ds,
                                          seed=seed,
                                          scheduler_patience=scheduler_patience,
                                          ema_decay=ema_decay,
                                          print_func=print_func)
        per_seed_results[seed] = result

        raw_path: str = os.path.join(out_dir, f'Best_Model_seed{seed}{file_suffix}.pth')
        shadow_path: str = os.path.join(out_dir, f'Best_Model_Shadow_seed{seed}{file_suffix}.pth')
        torch.save(result['raw_state_dict'], raw_path)
        torch.save(result['shadow_state_dict'], shadow_path)

        context: dict = {'best_config': best_config,
                         'n_feat': train_ds.X.shape[2],
                         'seed': seed,
                         'best_epoch': result['best_epoch'],
                         'best_shadow_epoch': result['best_shadow_epoch'],
                         'elapsed': result['elapsed']}
        context_path: str = os.path.join(out_dir, f'Model_Context_seed{seed}{file_suffix}.pth')
        torch.save(context, context_path)
        print_func(f'Saved seed {seed}: raw -> {raw_path}, shadow -> {shadow_path}, context -> {context_path}')

    ####################################################################################################################
    # Assemble history across seeds
    ####################################################################################################################
    all_history: list[dict] = []
    for seed in seeds:
        all_history.extend(per_seed_results[seed]['history'])
    df_history: pd.DataFrame = pd.DataFrame(all_history)

    ####################################################################################################################
    # Common per-partition labels and record IDs
    ####################################################################################################################
    y_val: np.ndarray = val_ds.y.numpy()
    y_test: np.ndarray = test_ds.y.numpy()
    test_rec_ids: list = per_seed_results[seeds[0]]['test_rec_ids']
    n_params: int = per_seed_results[seeds[0]]['n_params']

    ####################################################################################################################
    # Per-seed shadow reporting (one model row per seed)
    ####################################################################################################################
    all_df_probs: list[pd.DataFrame] = []
    all_df_test: list[pd.DataFrame] = []

    for seed in seeds:
        r: dict = per_seed_results[seed]

        for model_label, logit_key_val, logit_key_test, epoch_key in [
            ('Raw', 'raw_val_logits', 'raw_test_logits', 'best_epoch'),
            ('Shadow', 'shadow_val_logits', 'shadow_test_logits', 'best_shadow_epoch'),
        ]:
            val_logits: np.ndarray = r[logit_key_val]
            test_logits: np.ndarray = r[logit_key_test]
            val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits))
            test_prob: np.ndarray = 1 / (1 + np.exp(-test_logits))

            df_p: pd.DataFrame = _calibrate_and_build_probs(y_test, test_prob,
                                                            val_logits, test_logits,
                                                            val_prob, y_val)
            df_p.insert(0, 'rec_id', test_rec_ids)
            df_p.insert(1, 'seed', seed)
            df_p.insert(2, 'Model', model_label)
            all_df_probs.append(df_p)

            df_t: pd.DataFrame = _evaluate_test(df_p, y_test, n_params, r['elapsed'], best_config, 'TF')
            df_t = df_t.drop(columns='seed')
            df_t.insert(1, 'seed', seed)
            df_t.insert(2, 'Model', model_label)
            df_t.insert(3, 'best_epoch', r[epoch_key])
            all_df_test.append(df_t)

    ####################################################################################################################
    # Concatenate and return
    ####################################################################################################################
    df_probs: pd.DataFrame = pd.concat(all_df_probs, ignore_index=True)
    df_test: pd.DataFrame = pd.concat(all_df_test, ignore_index=True)

    return df_probs, df_test, df_history
