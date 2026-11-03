########################################################################################################################
# Acronyms used in this script
# B = Batch size
# L = Sequence length (number of bins per record; here L=96)
# D = Embedding dimension (d_model in the transformer)
# F = Number of features
# F_temp = Number of temporal (varying-across-bin) features
# F_stat = Number of static (constant-across-bin) features
########################################################################################################################

########################################################################################################################
# Two-stream fusion transformer for time-series with a mix of temporal (sensor) and static (survey/daily) features.
# - Temporal stream: MaskedFeatureEncoder -> Time2Vec -> TransformerEncoder -> attention pooling -> (B, D).
# - Static stream: mask-embedding MLP (tunable width d_static and depth n_static_layers) on constant-across-bin
#   features taken from bin 0 -> (B, d_static).
# - Fusion: concatenate the two representations, then a linear head (intermediate / representation-level fusion).
# The static features bypass the temporal attention bottleneck so they reach the classifier undiluted.
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
from itertools import product
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
        self.dummy_param: nn.Parameter = nn.Parameter(torch.empty(0))

    def set_device(self, device_str: Union[str, torch.device]):
        assert type(device_str) in [str, torch.device], \
            f'device_str must be a torch.device object or a string. Now its type is {type(device_str)}.'
        self.to(device_str)

    def get_device(self):
        return self.dummy_param.device

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

########################################################################################################################
# Time2Vec — continuous time encoding
########################################################################################################################


class Time2Vec(DL_Class):
    """
    Time positional encoding. Reference: Kazemi et al. (2019) arXiv:1907.05321.
    """
    def __init__(self, out_dim: int):
        super().__init__()
        assert int(out_dim) > 1, f'out_dim must be > 1. Got {out_dim}.'
        self.linear: nn.Linear = nn.Linear(in_features=1, out_features=1)
        self.freqs: nn.Parameter = nn.Parameter(torch.randn(int(out_dim) - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        trend: torch.Tensor = self.linear(t)                             # (B, L, 1)
        periodic: torch.Tensor = torch.sin(t * self.freqs)              # (B, L, D-1)
        return torch.cat([trend, periodic], dim=-1)                      # (B, L, D)

########################################################################################################################
# Masked feature encoder (temporal stream)
########################################################################################################################


class MaskedFeatureEncoder(DL_Class):
    """
    Projects (B, L, F_temp) feature vectors with NaNs into (B, L, D) embeddings using only observed values.
    Inspired by BRITS / GRU-D. References: Cao et al. (2018) arXiv:1806.00770; Che et al. (2018) arXiv:1606.01865.
    """
    def __init__(self, feature_dim: int, d_model: int):
        super().__init__()
        feature_dim = int(feature_dim)
        d_model = int(d_model)
        assert feature_dim >= 1, f'feature_dim must be >= 1. Got {feature_dim}.'
        assert d_model > 1, f'd_model must be > 1. Got {d_model}.'
        self.weights = nn.Parameter(torch.randn(feature_dim, d_model))   # (F_temp, D)
        self.bias = nn.Parameter(torch.zeros(d_model))                   # (D,)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask_t: torch.Tensor = (~torch.isnan(x)).float() if mask is None else mask.float()
        x0: torch.Tensor = torch.where(mask_t.bool(), x, torch.zeros_like(x))
        return torch.einsum('blf,fd->bld', x0, self.weights) + self.bias

########################################################################################################################
# Static feature encoder — mask-embedding MLP (static stream), tunable width and depth
########################################################################################################################


class StaticEncoder(DL_Class):
    """
    Encodes constant-across-bin static features (B, F_stat) with NaNs into (B, d_static).
    Layer 1 is an additive mask embedding (symmetric with the temporal 'explicit' path):
        h = val_proj(x_gated) + miss_proj(mask)
    followed by ReLU + Dropout, then (n_static_layers - 1) additional Linear->ReLU->Dropout blocks.
    NaN values are gated to zero; the missingness pattern is fed through its own linear projection.
    n_static_layers counts weight layers: 1 = projection only; 2 = projection + one hidden linear; etc.
    """
    def __init__(self, static_dim: int, d_static: int = 128,
                 n_static_layers: int = 2, dropout: float = 0.1):
        """
        :param static_dim: int. F_stat, number of static features.
        :param d_static: int. Hidden/output width of the static stream (tunable).
        :param n_static_layers: int. Number of weight layers (>= 1) (tunable).
        :param dropout: float. Dropout rate (matched to the temporal stream).
        """
        super().__init__()
        static_dim = int(static_dim)
        d_static = int(d_static)
        n_static_layers = int(n_static_layers)
        assert static_dim >= 1, f'static_dim must be >= 1. Got {static_dim}.'
        assert d_static > 1, f'd_static must be > 1. Got {d_static}.'
        assert n_static_layers >= 1, f'n_static_layers must be >= 1. Got {n_static_layers}.'

        # Layer 1: mask-embedding projection (value + missingness), each static_dim -> d_static
        self.val_proj: nn.Linear = nn.Linear(static_dim, d_static)
        self.miss_proj: nn.Linear = nn.Linear(static_dim, d_static)

        # Post-projection stack: ReLU + Dropout, then (n_static_layers - 1) hidden blocks
        hidden: list = [nn.ReLU(), nn.Dropout(p=dropout)]
        for _ in range(n_static_layers - 1):
            hidden.append(nn.Linear(d_static, d_static))
            hidden.append(nn.ReLU())
            hidden.append(nn.Dropout(p=dropout))
        self.mlp: nn.Sequential = nn.Sequential(*hidden)

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        :param x_static: (B, F_stat) static features with NaNs.
        :return: (B, d_static) static representation.
        """
        mask: torch.Tensor = (~torch.isnan(x_static)).float()
        x_gated: torch.Tensor = torch.where(mask.bool(), x_static, torch.zeros_like(x_static))
        h: torch.Tensor = self.val_proj(x_gated) + self.miss_proj(mask)
        return self.mlp(h)

########################################################################################################################
# Cost-sensitive weight tensor
########################################################################################################################


def make_weight_tensor(y: torch.Tensor, cost_mul: float) -> torch.Tensor:
    return torch.where(y == 0,
                       torch.as_tensor(cost_mul, dtype=y.dtype, device=y.device),
                       torch.as_tensor(1, dtype=y.dtype, device=y.device))

########################################################################################################################
# Array dataset
########################################################################################################################


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, rec_ids: np.ndarray):
        self.X: torch.Tensor = torch.tensor(X, dtype=torch.float32)      # (N, L, F)
        self.y: torch.Tensor = torch.tensor(y, dtype=torch.float32)      # (N,)
        self.rec_ids: np.ndarray = rec_ids                               # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], str(self.rec_ids[idx])

########################################################################################################################
# Two-stream fusion transformer classifier
########################################################################################################################


class FusionTransformerClassifier(DL_Class):
    """
    Two-stream classifier: temporal transformer encoder + static mask-embedding MLP, fused by concatenation.
    The model receives the full (B, L, F) tensor and internally slices temporal vs static columns using the
    index lists supplied at construction. Static features are read from bin 0 (they are constant across bins).
    """
    def __init__(self,
                 temporal_idxs: list[int],
                 static_idxs: list[int],
                 d_model: int,
                 n_head: int = 4,
                 n_units: int = 128,
                 n_layers: int = 2,
                 d_static: int = 128,
                 n_static_layers: int = 2,
                 task: str = 'binary',
                 dropout: float = 0.1,
                 handle_nan: str = 'explicit'):
        """
        :param temporal_idxs: list[int]. Column indices of temporal (varying-across-bin) features.
        :param static_idxs: list[int]. Column indices of static (constant-across-bin) features.
        :param d_model: int. Temporal embedding dimension.
        :param n_head: int. Number of attention heads.
        :param n_units: int. Feedforward layer units in the transformer.
        :param n_layers: int. Number of transformer encoder layers.
        :param d_static: int. Static stream hidden/output width.
        :param n_static_layers: int. Static stream weight-layer count.
        :param task: str. 'binary' or 'regress'.
        :param dropout: float. Dropout rate (shared by both streams).
        :param handle_nan: str. 'explicit', 'implicit', or 'ffill' (temporal stream only).
        """
        assert task in ['binary', 'regress']
        assert handle_nan in ['explicit', 'implicit', 'ffill']
        assert len(temporal_idxs) >= 1, 'need at least one temporal feature'
        assert len(static_idxs) >= 1, 'need at least one static feature'
        super().__init__()
        self.task = task
        self.handle_nan = handle_nan
        self.register_buffer('temporal_idxs', torch.tensor(temporal_idxs, dtype=torch.long))
        self.register_buffer('static_idxs', torch.tensor(static_idxs, dtype=torch.long))

        n_temp: int = len(temporal_idxs)
        n_stat: int = len(static_idxs)

        # Temporal stream
        self.feature_encoder = MaskedFeatureEncoder(n_temp, d_model)
        self.time_encoder = Time2Vec(out_dim=d_model)
        self.miss_encoder = nn.Linear(n_temp, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=n_units,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.embed_ln = nn.LayerNorm(d_model)

        # Static stream
        self.static_encoder = StaticEncoder(n_stat, d_static=d_static,
                                            n_static_layers=n_static_layers, dropout=dropout)

        # Fusion head
        self.fc = nn.Linear(d_model + d_static, 1)
        self.__init_weights()

    def __init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                param.data.fill_(0.0)

    def _forward_fill(self, x: torch.Tensor) -> torch.Tensor:
        x_ffill = x.clone()
        B, L, F = x_ffill.shape
        for t in range(1, L):
            nan_mask = torch.isnan(x_ffill[:, t, :])
            prev_vals = x_ffill[:, t - 1, :]
            fill_mask = nan_mask & (~torch.isnan(prev_vals))
            x_ffill[:, t, :] = torch.where(fill_mask, prev_vals, x_ffill[:, t, :])
        return x_ffill

    def _encode_temporal(self, x_temp: torch.Tensor) -> torch.Tensor:
        """
        :param x_temp: (B, L, F_temp) temporal features with NaNs.
        :return: (B, D) pooled temporal representation.
        """
        B, L, F = x_temp.shape
        timestamp: torch.Tensor = (torch.arange(L, dtype=torch.float32, device=x_temp.device)
                                   .view(1, L, 1).expand(B, L, 1))
        if self.handle_nan == 'ffill':
            x_temp = self._forward_fill(x_temp)

        M = (~torch.isnan(x_temp)).float()                              # (B, L, F_temp)
        pad_mask = (M.sum(dim=2) == 0)                                  # (B, L)

        if self.handle_nan == 'explicit':
            x0 = torch.where(M.bool(), x_temp, torch.zeros_like(x_temp))
            x_feat = self.feature_encoder(x0, mask=M)
            x_miss = self.miss_encoder(M)
            x_time = self.time_encoder(timestamp)
            x_embed = x_feat + x_time + x_miss
        else:
            x_feat = self.feature_encoder(x_temp, mask=None)
            x_time = self.time_encoder(timestamp)
            x_embed = x_feat + x_time

        x_embed = self.embed_ln(x_embed)
        x_trans = self.transformer(x_embed, src_key_padding_mask=pad_mask)
        attn_scores = self.attn_pool(x_trans).squeeze(-1)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(x_trans * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, L, F) full feature tensor. Temporal and static columns are sliced internally.
        :return: (B,) logits.
        """
        x_temp: torch.Tensor = x.index_select(dim=2, index=self.temporal_idxs)          # (B, L, F_temp)
        x_stat: torch.Tensor = x.index_select(dim=2, index=self.static_idxs)[:, 0, :]   # (B, F_stat) from bin 0

        temporal_repr: torch.Tensor = self._encode_temporal(x_temp)     # (B, D)
        static_repr: torch.Tensor = self.static_encoder(x_stat)         # (B, d_static)

        fused: torch.Tensor = torch.cat([temporal_repr, static_repr], dim=-1)
        return self.fc(fused).squeeze(-1)                               # (B,)

    def forward_for_shap(self,
                         x0: torch.Tensor,
                         mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        SHAP-friendly forward. Expects the full (B, L, F) gated tensor x0 and its mask.
        Slices temporal/static internally; the static path reads bin 0.
        """
        device = x0.device
        M_full = mask.float().to(device) if mask is not None else (~torch.isnan(x0)).float()

        x_temp = x0.index_select(dim=2, index=self.temporal_idxs)
        M_temp = M_full.index_select(dim=2, index=self.temporal_idxs)
        x_stat = x0.index_select(dim=2, index=self.static_idxs)[:, 0, :]
        M_stat = M_full.index_select(dim=2, index=self.static_idxs)[:, 0, :]

        B, L, F = x_temp.shape
        timestamp = (torch.arange(L, dtype=torch.float32, device=device).view(1, L, 1).expand(B, L, 1))
        pad_mask = (M_temp.sum(dim=2) == 0)

        if self.handle_nan == 'ffill':
            x_ffill = x_temp.clone()
            for t in range(1, L):
                miss = (M_temp[:, t, :] == 0)
                prev_obs = (M_temp[:, t - 1, :] == 1)
                fill_mask = miss & prev_obs
                x_ffill[:, t, :] = torch.where(fill_mask, x_ffill[:, t - 1, :], x_ffill[:, t, :])
            x_temp_used = x_ffill
        else:
            x_temp_used = x_temp

        if self.handle_nan == 'explicit':
            x_gated = torch.where(M_temp.bool(), x_temp_used, torch.zeros_like(x_temp_used))
            x_feat = self.feature_encoder(x_gated, mask=M_temp)
            x_miss = self.miss_encoder(M_temp)
            x_time = self.time_encoder(timestamp)
            x_embed = x_feat + x_time + x_miss
        else:
            x_feat = self.feature_encoder(x_temp_used, mask=M_temp)
            x_time = self.time_encoder(timestamp)
            x_embed = x_feat + x_time

        x_embed = self.embed_ln(x_embed)
        x_trans = self.transformer(x_embed, src_key_padding_mask=pad_mask)
        attn_scores = self.attn_pool(x_trans).squeeze(-1)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        temporal_repr = torch.sum(x_trans * attn_weights.unsqueeze(-1), dim=1)

        # Static stream (mask embedding), using gated bin-0 values and their mask
        x_stat_gated = torch.where(M_stat.bool(), x_stat, torch.zeros_like(x_stat))
        h_stat = self.static_encoder.val_proj(x_stat_gated) + self.static_encoder.miss_proj(M_stat)
        static_repr = self.static_encoder.mlp(h_stat)

        fused = torch.cat([temporal_repr, static_repr], dim=-1)
        return self.fc(fused).squeeze(-1)

########################################################################################################################
# Training function
########################################################################################################################


def train_tf(M: FusionTransformerClassifier,
             train_loader: DataLoader,
             val_loader: DataLoader,
             device: torch.device,
             cost_mul: Optional[int] = None,
             lr: float = 1e-4,
             max_epochs: int = 200,
             scheduler_factor: float = 0.5,
             scheduler_patience: int = 10) -> tuple[FusionTransformerClassifier, np.ndarray, np.ndarray, int]:
    """
    Train a FusionTransformerClassifier with best-val-loss checkpointing.
    :return: (best model, val_logits, val_labels, best_epoch)
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
                  f'(best={best_val_loss:.4f} @ epoch {best_epoch}) — LR={current_lr:.2e}', flush=True)

    if best_state_dict is not None:
        M.load_state_dict(best_state_dict)

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


def collect_logits(M: FusionTransformerClassifier,
                   loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list]:
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
                 temporal_idxs: list[int],
                 static_idxs: list[int],
                 device: torch.device,
                 batch_size: int,
                 dropout: float,
                 max_epochs: int,
                 scheduler_patience: int,
                 random_state: int,
                 rank: int = 0) -> pd.DataFrame:
    """
    Run HPO over configs of
    (d_model, n_heads, n_layers, n_units, lr, cost_mul, handle_nan, factor, d_static, n_static_layers).
    """
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    n_temp: int = len(temporal_idxs)
    n_stat: int = len(static_idxs)
    records: list[dict] = []

    for (d_model, n_heads, n_layers, n_units, lr, cost_mul, handle_nan, factor,
         d_static, n_static_layers) in configs:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        M = FusionTransformerClassifier(temporal_idxs=temporal_idxs,
                                        static_idxs=static_idxs,
                                        d_model=d_model,
                                        n_head=n_heads,
                                        n_units=n_units,
                                        n_layers=n_layers,
                                        d_static=d_static,
                                        n_static_layers=n_static_layers,
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
              f'n_units={n_units}, lr={lr:.0e}, cost_mul={cost_mul}, handle_nan={handle_nan}, '
              f'factor={factor}, d_static={d_static}, n_static_layers={n_static_layers}\n'
              + buffer.getvalue(), flush=True)

        val_logits = np.nan_to_num(val_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        y_val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits))
        metrics: dict = binary_metrics(y_true=val_labels, y_prob=y_val_prob)
        metrics = {'Algorithm': 'TF_Fusion',
                   'd_model': d_model,
                   'n_heads': n_heads,
                   'n_layers': n_layers,
                   'n_units': n_units,
                   'lr': lr,
                   'cost_mul': cost_mul,
                   'handle_nan': handle_nan,
                   'factor': factor,
                   'd_static': d_static,
                   'n_static_layers': n_static_layers,
                   'max_epochs': max_epochs,
                   'batch_size': batch_size,
                   'dropout': dropout,
                   'n_temp': n_temp,
                   'n_stat': n_stat,
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
                        temporal_idxs: list[int],
                        static_idxs: list[int],
                        out_dir: str,
                        fold_id: str,
                        batch_size: int,
                        dropout: float,
                        max_epochs: int,
                        scheduler_patience: int,
                        random_state: int):
    device = torch.device(f'cuda:{rank}')
    df_val: pd.DataFrame = run_hpo_grid(configs=configs,
                                        train_ds=train_ds,
                                        val_ds=val_ds,
                                        temporal_idxs=temporal_idxs,
                                        static_idxs=static_idxs,
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
