########################################################################################################################
# Acronyms used in this script
# B = Batch size (number of samples in a batch, or simply the total sample size when not performing any batching)
# L = Length of the sequence (number of timestamps / encounters for a given a sample)
# D = Dimension of the time embedding vector (also identical to the d_model in a transformer model)
# F = Number of features
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import contextlib
import copy
import io
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from time import time
from torch.utils.data import DataLoader
from typing import Optional, Union
from .binary_metrics import binary_metrics

########################################################################################################################
# Base class
########################################################################################################################


class DL_Class(nn.Module):
    """
    A superclass for various model classes that will be used in this module.
    """
    def __init__(self):
        super().__init__()
        # An identifier of the physical location of the model
        self.dummy_param: nn.Parameter = nn.Parameter(torch.empty(0))

    def set_device(self, device_str: Union[str, torch.device]):
        """
        :param device_str: A string or torch.device object. The physical location of the model to be set.
        """
        assert type(device_str) in [str, torch.device], \
            f'device_str must be a torch.device object or a string. Now its type is {type(device_str)}.'
        self.to(device_str)

    def get_device(self):
        """
        :return: A string. The current physical location of the model.
        """
        return self.dummy_param.device

    def get_n_params(self):
        """
        :return: An integer. The number of parameters of the model.
        """
        return sum(p.numel() for p in self.parameters())

########################################################################################################################
# Time2Vec — continuous time encoding
########################################################################################################################


class Time2Vec(DL_Class):
    """
    This class implements the time positional encoding module. It is necessary for irregular time-series data. This
    time embedding vector will be aligned with the feature embedding vector, both with shape (B, L, D).
    Reference:
    1. Kazemi et al. (2019) "Time2Vec: Learning a vector representation of time". DOI: 10.48550/arXiv.1907.05321.
    """
    def __init__(self, out_dim: int):
        """
        :param out_dim: A positive integer. Size of the time embedding vector.
        """
        super().__init__()
        assert int(out_dim) > 1, f'out_dim must be > 1. Got {out_dim}.'
        # 1-to-1 linear layer capturing the linear trend of the time series
        self.linear: nn.Linear = nn.Linear(in_features=1, out_features=1)
        # Learnable periodic sinusoid component of time
        self.freqs: nn.Parameter = nn.Parameter(torch.randn(int(out_dim) - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        :param t: The number of timestamps for each encounter in the batch.
        :return: the concatenated components of trend and periodicity.
        """
        trend: torch.Tensor = self.linear(t)                                     # (B, L, 1)
        periodic: torch.Tensor = torch.sin(t * self.freqs)                       # (B, L, D-1)
        return torch.cat([trend, periodic], dim=-1)                       # (B, L, D)

########################################################################################################################
# Masked feature encoder
########################################################################################################################


class MaskedFeatureEncoder(DL_Class):
    """
    Projects feature vectors (with NaNs) into dense embeddings using only observed feature values. The masking mechanism
    being used is inspired by BRITS and GRU-D, but our approach is implemented as a simplified projection layer for
    transformer compatibility.
    References:
    1. Cao et al. (2018) "BRITS: Bidirectional Recurrent Imputation for Time Series".
    DOI: 10.48550/arXiv.1806.00770.
    2. Che et al. (2018) "Recurrent Neural Networks for Missing Value Imputation in Time Series".
    DOI: 10.48550/arXiv.1606.01865.
    """
    def __init__(self, feature_dim: int, d_model: int):
        """
        :param feature_dim: A positive integer. F, number of input features.
        :param d_model: A positive integer. D, embedding dimension.
        """
        super().__init__()
        try:
            feature_dim = int(feature_dim)
        except Exception as e:
            raise TypeError(f"feature_dim must be a positive integer. Got {feature_dim} instead.") from e
        assert feature_dim > 1, f"feature_dim must be a positive integer. Got {feature_dim} instead."
        try:
            d_model = int(d_model)
        except Exception as e:
            raise TypeError(f"d_model must be a positive integer. Got {d_model} instead.") from e
        assert d_model > 1, f"d_model must be a positive integer. Got {d_model} instead."
        self.weights = nn.Parameter(torch.randn(feature_dim, d_model))              # (F, D)
        self.bias = nn.Parameter(torch.zeros(d_model))                              # (D, )
        nn.init.xavier_uniform_(self.weights)                                       # Xavier initialization of weights

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: Feature dataset of dimension (B, L, F) with NaNs preserved.
        :param mask: Optional tensor of the same dimension of x indicating the missingness mask of x.
        :return: The encoded feature dataset of dimension (B, L, D).
        """
        mask: torch.Tensor = (~torch.isnan(x)).float() if mask is None else mask.float()
        x0: torch.Tensor = torch.where(mask.bool(), x, torch.zeros_like(x))         # (B, L, F)
        return torch.einsum('blf,fd->bld', x0, self.weights) + self.bias            # (B, L, D)

########################################################################################################################
# Loading an array dataset
########################################################################################################################


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, rec_ids: np.ndarray):
        self.X: torch.Tensor = torch.tensor(X, dtype=torch.float32)      # (N, 96, F)
        self.y: torch.Tensor = torch.tensor(y, dtype=torch.float32)      # (N,)
        self.rec_ids: np.ndarray = rec_ids                               # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], str(self.rec_ids[idx])

########################################################################################################################
# Cost-sensitive weight tensor
########################################################################################################################


def make_weight_tensor(y: torch.Tensor, cost_mul: float) -> torch.Tensor:
    """
    :param y: torch.Tensor. The actual outcome label.
    :param cost_mul: float. The multiplier of the negative class.
    :return:
    A torch.Tensor indicating the per-sample cost multiplier.
    """
    return torch.where(y == 0,
                       torch.as_tensor(cost_mul, dtype=y.dtype, device=y.device),
                       torch.as_tensor(1, dtype=y.dtype, device=y.device))

########################################################################################################################
# Transformer classifier
########################################################################################################################


class TransformerClassifier(DL_Class):
    """
    Transformer-based classifier for (potentially irregular) time-series data.
    - Handles per-feature missingness via masked feature encoding.
    - Handles irregular sampling via continuous time encoding (Time2Vec).
    - Handles timestamp-level masking via Transformer attention mask.
    """
    def __init__(self,
                 feature_dim: int,
                 d_model: int,
                 n_head: int = 4,
                 n_units: int = 128,
                 n_layers: int = 2,
                 task: str = 'binary',
                 dropout: float = 0.1,
                 handle_nan: str = 'implicit'):
        """
        :param feature_dim: A positive integer. F, number of input features.
        :param d_model: A positive integer. Embedding dimension.
        :param n_head: A positive integer. Number of attention heads.
        :param n_units: A positive integer. Feedforward layer units.
        :param n_layers: A positive integer. Number of transformer encoder layers.
        :param task: A string. 'binary' or 'regress'.
        :param dropout: A float. Dropout rate.
        :param handle_nan: A string. 'explicit', 'implicit', or 'ffill'.
        """
        assert task in ['binary', 'regress']
        assert handle_nan in ['explicit', 'implicit', 'ffill']
        super().__init__()
        self.task = task
        self.handle_nan = handle_nan
        self.feature_encoder = MaskedFeatureEncoder(feature_dim, d_model)
        self.time_encoder = Time2Vec(out_dim=d_model)
        self.miss_encoder = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=n_units,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.embed_ln = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, 1))
        self.__init_weights()

    def __init_weights(self):
        """
        Xavier uniform weight initialization (with bias initialized as 0)
        """
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                param.data.fill_(0.0)

    def _forward_fill(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward filling feature values from previous timestamp.
        :param x: torch.Tensor. Feature dataset.
        :return:
        x_ffill: torch.Tensor. Forward-filled feature dataset.
        """
        x_ffill = x.clone()
        B, L, F = x_ffill.shape
        for t in range(1, L):
            nan_mask = torch.isnan(x_ffill[:, t, :])
            prev_vals = x_ffill[:, t - 1, :]
            fill_mask = nan_mask & (~torch.isnan(prev_vals))
            x_ffill[:, t, :] = torch.where(fill_mask, prev_vals, x_ffill[:, t, :])
        return x_ffill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core model forwarding function.
        :param x: torch.Tensor. Feature dataset.
        :return:
        A 1-d torch.Tensor of logits for subsequent sigmoid calculation.
        """
        B, L, F = x.shape
        timestamp: torch.Tensor = (torch.arange(L, dtype=torch.float32, device=x.device)
                                   .view(1, L, 1)
                                   .expand(B, L, 1))
        if self.handle_nan == 'ffill':
            x = self._forward_fill(x)

        M = (~torch.isnan(x)).float()                                   # (B, L, F)
        pad_mask = (M.sum(dim=2) == 0)                                  # (B, L)

        if self.handle_nan == 'explicit':
            x0 = torch.where(M.bool(), x, torch.zeros_like(x))
            x_feat = self.feature_encoder(x0, mask=M)                   # (B, L, D)
            x_miss = self.miss_encoder(M)                               # (B, L, D)
            x_time = self.time_encoder(timestamp)                       # (B, L, D)
            x_embed = x_feat + x_time + x_miss
        else:
            x_feat = self.feature_encoder(x, mask=None)                 # (B, L, D)
            x_time = self.time_encoder(timestamp)                       # (B, L, D)
            x_embed = x_feat + x_time

        x_embed = self.embed_ln(x_embed)
        x_trans = self.transformer(x_embed, src_key_padding_mask=pad_mask)  # (B, L, D)
        attn_scores = self.attn_pool(x_trans).squeeze(-1)                   # (B, L)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)                    # (B, L)
        pooled = torch.sum(x_trans * attn_weights.unsqueeze(-1), dim=1)     # (B, D)
        return self.fc(pooled).squeeze(-1)                                  # (B,)

    def forward_for_shap(self,
                         x0: torch.Tensor,
                         timestamp: torch.Tensor,
                         mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward function for SHAP value calculation.
        :param x0: torch.Tensor. Feature dataset.
        :param timestamp: torch.Tensor. Timestamp dataset.
        :param mask: torch.Tensor. Missingness mask for x0.
        :return:
        A 1-d torch.Tensor of logits for subsequent sigmoid calculation.
        """

        M = mask.float().to(x0.device)
        pad_mask = (M.sum(dim=2) == 0)

        if self.handle_nan == 'ffill':
            x_ffill = x0.clone()
            B, L, F = x_ffill.shape
            for t in range(1, L):
                miss = (M[:, t, :] == 0)
                prev_obs = (M[:, t - 1, :] == 1)
                fill_mask = miss & prev_obs
                x_ffill[:, t, :] = torch.where(fill_mask, x_ffill[:, t - 1, :], x_ffill[:, t, :])
            x_used = x_ffill
        else:
            x_used = x0

        if self.handle_nan == 'explicit':
            x_gated = torch.where(M.bool(), x_used, torch.zeros_like(x_used))
            x_feat = self.feature_encoder(x_gated, mask=M)
            x_miss = self.miss_encoder(M)
            x_time = self.time_encoder(timestamp)
            x_embed = x_feat + x_time + x_miss
        else:
            x_feat = self.feature_encoder(x_used, mask=M)
            x_time = self.time_encoder(timestamp)
            x_embed = x_feat + x_time

        x_embed = self.embed_ln(x_embed)
        x_trans = self.transformer(x_embed, src_key_padding_mask=pad_mask)
        attn_scores = self.attn_pool(x_trans).squeeze(-1)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(x_trans * attn_weights.unsqueeze(-1), dim=1)
        return self.fc(pooled).squeeze(-1)

########################################################################################################################
# Training function
########################################################################################################################


def train_tf(M: TransformerClassifier,
             train_loader: DataLoader,
             val_loader: DataLoader,
             device: torch.device,
             cost_mul: Optional[int] = None,
             lr: float = 1e-4,
             max_epochs: int = 200,
             scheduler_factor: float = 0.5,
             scheduler_patience: int = 10) -> tuple[TransformerClassifier, np.ndarray, np.ndarray, int]:
    """
    Helper function for model training.
    :param M: TransformerClassifier object.
    :param train_loader: DataLoader object for the training partition.
    :param val_loader: DataLoader object for the validation partition.
    :param device: torch.device. Specifying the device to run the process.
    :param cost_mul: float. The multiplier of the negative class.
    :param lr: float. Learning rate of the AdamW optimizer.
    :param max_epochs: int. Maximum training epochs.
    :param scheduler_factor: float. Factor of the ReduceLROnPlateau scheduler for the learning rate (lr).
    :param scheduler_patience: int. Patience of the ReduceLROnPlateau scheduler for the learning rate (lr).
    :return:
    M: TransformerClassifier object. Trained model.
    val_logits: np.ndarray. Logits of the model estimates evaluated on the validation data.
    val_labels: np.ndarray. True outcome label of the validation data.
    best_epoch: int. The training epoch with the lowest binary cross-entropy loss.
    """
    M = M.to(device)
    criterion = nn.BCEWithLogitsLoss() if cost_mul is None else nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.AdamW(M.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=scheduler_factor,
                                                           patience=scheduler_patience)
    best_val_loss: float = float('inf')
    best_state_dict = None
    best_epoch: int = 0

    for epoch in range(max_epochs):

        # Training pass
        M.train()

        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = M(xb).view(-1)
            loss_raw = criterion(logits, yb)

            if cost_mul is None:
                loss = loss_raw
            else:
                w = make_weight_tensor(yb, cost_mul)
                loss = (loss_raw * w).sum() / (w.sum() + 1e-12)

            loss.backward()
            optimizer.step()

        # Validation pass
        M.eval()
        val_running_loss: float = 0.0
        val_running_weight: float = 0.0

        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).view(-1)
                logits = M(xb).view(-1)
                val_loss_raw = criterion(logits, yb)

                if cost_mul is None:
                    val_running_loss += val_loss_raw.item() * xb.size(0)
                else:
                    w = make_weight_tensor(yb, cost_mul)
                    val_running_loss += (val_loss_raw * w).sum().item()
                    val_running_weight += w.sum().item()

        if cost_mul is None:
            val_loss: float = val_running_loss / len(val_loader.dataset)
        else:
            val_loss: float = val_running_loss / (val_running_weight + 1e-12)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(M.state_dict())

        if (epoch + 1) % 10 == 0:
            current_lr: float = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{max_epochs} — val_loss={val_loss:.4f} '
                  f'(best={best_val_loss:.4f} @ epoch {best_epoch}) — LR={current_lr:.2e}',
                  flush=True)

    # Restore best weights
    if best_state_dict is not None:
        M.load_state_dict(best_state_dict)

    # Collect validation logits from best model
    M.eval()
    val_logits_all, val_labels_all = [], []
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            logits = M(xb).view(-1)
            val_logits_all.append(logits.cpu())
            val_labels_all.append(yb.cpu())
    val_logits = torch.cat(val_logits_all).numpy()
    val_labels = torch.cat(val_labels_all).numpy()

    return M, val_logits, val_labels, best_epoch

########################################################################################################################
# Collect logits from a DataLoader
########################################################################################################################


def collect_logits(M: TransformerClassifier,
                   loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Collect the estimated logits from a trained Transformer model.
    :param M: TransformerClassifier object.
    :param loader: DataLoader. Feature+Target dataset in a given partition.
    :return:
    y_true: np.ndarray. The true outcome labels.
    y_logit: np.ndarray. The estimated logits.
    rec_ids: list. The identifiers of the records.
    """
    M.eval()
    device = next(M.parameters()).device
    y_true_all, y_logit_all, rec_ids = [], [], []
    with torch.no_grad():
        for xb, yb, rid in loader:
            xb = xb.to(device)
            logits = M(xb).view(-1)
            y_true_all.append(yb.view(-1).cpu())
            y_logit_all.append(logits.cpu())
            rec_ids.extend(list(rid))
    y_true = torch.cat(y_true_all).numpy()
    y_logit = torch.cat(y_logit_all).numpy()
    y_logit = np.nan_to_num(y_logit, nan=0.0, posinf=30.0, neginf=-30.0)
    return y_true, y_logit, rec_ids

########################################################################################################################
# HPO grid runner — single device
########################################################################################################################


def run_hpo_grid(configs: list[tuple],
                 train_ds: ArrayDataset,
                 val_ds: ArrayDataset,
                 device: torch.device,
                 batch_size: int,
                 dropout: float,
                 max_epochs: int,
                 scheduler_patience: int,
                 random_state: int,
                 rank: int = 0) -> pd.DataFrame:
    """
    Run the hyperparameter optimization (HPO) with a given grid specified by configs.
    :param configs: A list of (d_model, n_heads, n_layers, n_units, lr, cost_mul, handle_nan, factor) tuples.
    :param train_ds: Training ArrayDataset.
    :param val_ds: Validation ArrayDataset.
    :param device: A torch.device. Device to run on.
    :param batch_size: An integer. Batch size.
    :param dropout: A float. Drop out ratio.
    :param max_epochs: An integer. Maximum training epochs.
    :param scheduler_patience: An integer. ReduceLROnPlateau patience.
    :param random_state: An integer. Random seed.
    :param rank: An integer. GPU rank for logging prefix.
    :return: A pd.DataFrame of validation performance per config.
    """
    n_feat: int = train_ds.X.shape[2]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    records: list[dict] = []

    for d_model, n_heads, n_layers, n_units, lr, cost_mul, handle_nan, factor in configs:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        M = TransformerClassifier(feature_dim=n_feat,
                                  d_model=d_model,
                                  n_head=n_heads,
                                  n_units=n_units,
                                  n_layers=n_layers,
                                  task='binary',
                                  handle_nan=handle_nan,
                                  dropout=dropout).to(device)

        t0: float = time()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            M, val_logits, val_labels, best_epoch = train_tf(M, train_loader, val_loader,
                                                             device=device,
                                                             cost_mul=cost_mul,
                                                             lr=lr,
                                                             max_epochs=max_epochs,
                                                             scheduler_factor=factor,
                                                             scheduler_patience=scheduler_patience)
        n_params: int = sum(p.numel() for p in M.parameters() if p.requires_grad)
        elapsed: float = time() - t0
        print(f'[GPU{rank}] d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, '
              f'n_units={n_units}, lr={lr:.0e}, cost_mul={cost_mul}, '
              f'handle_nan={handle_nan}, factor={factor}\n' + buffer.getvalue(), flush=True)

        val_logits = np.nan_to_num(val_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        y_val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits))
        metrics: dict = binary_metrics(y_true=val_labels, y_prob=y_val_prob)
        metrics = {'Algorithm': 'TF',
                   'd_model': d_model,
                   'n_heads': n_heads,
                   'n_layers': n_layers,
                   'n_units': n_units,
                   'lr': lr,
                   'cost_mul': cost_mul,
                   'handle_nan': handle_nan,
                   'factor': factor,
                   'max_epochs': max_epochs,
                   'batch_size': batch_size,
                   'dropout': dropout,
                   'n_feat': n_feat,
                   'n_Params': n_params,
                   'scheduler_patience': scheduler_patience,
                   'seed': random_state,
                   'best_epoch': best_epoch,
                   'Elapsed_training_time': elapsed} | {k: v for k, v in metrics.items() if not k.endswith('LIST')}
        records.append(metrics)
    return pd.DataFrame(records)

########################################################################################################################
# Parallel HPO worker
########################################################################################################################


def parallel_hpo_worker(rank: int,
                        configs: list[tuple],
                        train_ds: ArrayDataset,
                        val_ds: ArrayDataset,
                        out_dir: str,
                        fold_id: str,
                        batch_size: int,
                        dropout: float,
                        max_epochs: int,
                        scheduler_patience: int,
                        random_state: int):
    """
    :param rank: An integer. GPU index for this worker.
    :param configs: A list of tuples. Config chunk assigned to this worker.
    :param train_ds: Training ArrayDataset.
    :param val_ds: Validation ArrayDataset.
    :param out_dir: A string. Directory to write worker results CSV.
    :param fold_id: A string. Identifier of the fold (e.g., 'V1', 'V2', 'V3', 'V4').
    :param batch_size: An integer. Batch size.
    :param dropout: A float. Drop out ratio.
    :param max_epochs: An integer. Maximum training epochs.
    :param scheduler_patience: An integer. ReduceLROnPlateau patience.
    :param random_state: An integer. Random seed.
    """
    device = torch.device(f'cuda:{rank}')

    df_val: pd.DataFrame = run_hpo_grid(configs=configs,
                                        train_ds=train_ds,
                                        val_ds=val_ds,
                                        device=device,
                                        batch_size=batch_size,
                                        dropout=dropout,
                                        max_epochs=max_epochs,
                                        scheduler_patience=scheduler_patience,
                                        random_state=random_state,
                                        rank=rank)

    out_path: str = os.path.join(out_dir, f'Fold_{fold_id}_Worker_{rank}_Results.csv')
    df_val.to_csv(out_path, index=False)
    print(f'[GPU{rank}] Fold {fold_id} results saved to {out_path}', flush=True)
