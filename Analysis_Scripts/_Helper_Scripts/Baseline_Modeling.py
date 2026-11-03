########################################################################################################################
# Import packages
########################################################################################################################
import copy
import numpy as np
import pandas as pd
import torch
import warnings
from itertools import product
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from time import time
from xgboost import XGBClassifier
from .binary_metrics import binary_metrics
from .Logger import Logger, create_log
from typing import Optional

########################################################################################################################
# Elastic net modeling
########################################################################################################################


def en_modeling(X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                C_grid: tuple[float] = (0.001, 0.01, 0.1, 1, 10, 100),
                l1_ratio_grid: tuple[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
                cost_mul_grid: tuple[int] = (1, 2, 5, 10),
                max_iter: int = 1000,
                random_state: int = 42,
                logger: Optional[Logger] = None):

    # Logging
    print_func = print if logger is None else logger.info
    debug_func = print if logger is None else logger.debug

    # Create an empty list to store model performance
    records: list[dict] = []

    # Loop over the grid and create the associated model
    full_grid: list[tuple] = list(product(C_grid,
                                          l1_ratio_grid,
                                          cost_mul_grid))
    for ele_idx, (C, l1_ratio, cost_mul) in enumerate(full_grid, 1):
        class_weight: dict[int, int] = {0: cost_mul, 1: 1}
        M = LogisticRegression(penalty='elasticnet',
                               C=C,
                               l1_ratio=l1_ratio,
                               class_weight=class_weight,
                               solver='saga',
                               max_iter=max_iter,
                               random_state=random_state,
                               n_jobs=-1)

        # Fit the model with the training dataset
        t0: float = time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            M.fit(X_train, y_train)
        elapsed: float = round(time() - t0, 3)

        # Evaluate the model with the validation dataset
        y_val_prob: np.ndarray = M.predict_proba(X_val)[:, 1]

        metrics: dict = binary_metrics(y_true=y_val, y_prob=y_val_prob)
        prefix: dict = {'Algorithm': 'EN',
                        'Element_index': ele_idx,
                        'C': C,
                        'l1_ratio': l1_ratio,
                        'cost_mul': cost_mul,
                        'max_iter': max_iter,
                        'seed': random_state,
                        'Elapsed_training_time': elapsed}
        metrics = prefix | {k: v for k, v in metrics.items() if not k.endswith('LIST')}
        records.append(metrics)
        debug_func(prefix)

    # Output: Organize validation performance statistics into a pandas.DataFrame
    df_val: pd.DataFrame = pd.DataFrame(records)
    print_func(f'Training and validation of the grid (size = {len(full_grid)}) completed.')
    return df_val

########################################################################################################################
# SVM modeling
########################################################################################################################


def svm_modeling(X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 C_grid: tuple[float] = (0.001, 0.01, 0.1, 1, 10, 100),
                 penalty_grid: tuple[str] = ('l1', 'l2'),
                 cost_mul_grid: tuple[int] = (1, 2, 5, 10),
                 max_iter: int = 1000,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):

    # Logging
    print_func = print if logger is None else logger.info
    debug_func = print if logger is None else logger.debug

    # Create an empty list to store model performance
    records: list[dict] = []

    # Loop over the grid and create the associated model
    full_grid: list[tuple] = list(product(C_grid,
                                          penalty_grid,
                                          cost_mul_grid))
    for ele_idx, (C, penalty, cost_mul) in enumerate(full_grid, 1):
        class_weight: dict[int, int] = {0: cost_mul, 1: 1}
        M = LinearSVC(C=C,
                      penalty=penalty,
                      dual=penalty == 'l2',
                      class_weight=class_weight,
                      max_iter=max_iter,
                      random_state=random_state)

        # Fit the model with the training dataset
        t0: float = time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            M.fit(X_train, y_train)
        elapsed: float = round(time() - t0, 3)

        # Evaluate the model with the validation dataset
        val_logits: np.ndarray = M.decision_function(X_val)
        y_val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits))
        metrics: dict = binary_metrics(y_true=y_val, y_prob=y_val_prob)
        prefix: dict = {'Algorithm': 'SVM',
                        'Element_index': ele_idx,
                        'C': C,
                        'penalty': penalty,
                        'dual': penalty == 'l2',
                        'cost_mul': cost_mul,
                        'max_iter': max_iter,
                        'seed': random_state,
                        'Elapsed_training_time': elapsed}
        metrics = prefix | {k: v for k, v in metrics.items() if not k.endswith('LIST')}
        records.append(metrics)
        debug_func(prefix)

    # Output: Organize validation performance statistics into a pandas.DataFrame
    df_val: pd.DataFrame = pd.DataFrame(records)
    print_func(f'Training and validation of the grid (size = {len(full_grid)}) completed.')
    return df_val

########################################################################################################################
# XGBoost modeling
########################################################################################################################


def xgb_modeling(X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 max_depth_grid: tuple[int] = (4, 6, 8),
                 n_estimators_grid: tuple[int] = (100, 300, 500),
                 learning_rate_grid: tuple[float] = (0.1, 0.3, 0.5),
                 reg_lambda_grid: tuple[float] = (0.1, 1, 10),
                 cost_mul_grid: tuple[int] = (1, 2, 5, 10),
                 random_state: int = 42,
                 logger: Optional[Logger] = None):

    # Logging
    print_func = print if logger is None else logger.info
    debug_func = print if logger is None else logger.debug

    # Create an empty list to store model performance
    records: list[dict] = []

    # Loop over the grid and create the associated model
    full_grid: list[tuple] = list(product(max_depth_grid,
                                          n_estimators_grid,
                                          learning_rate_grid,
                                          reg_lambda_grid,
                                          cost_mul_grid))
    for ele_idx, (max_depth, n_estimators, learning_rate, reg_lambda, cost_mul) in enumerate(full_grid, 1):
        sample_weight: np.ndarray = np.where(y_train == 0, cost_mul, 1.0)
        M = XGBClassifier(max_depth=max_depth,
                          n_estimators=n_estimators,
                          learning_rate=learning_rate,
                          reg_lambda=reg_lambda,
                          random_state=random_state,
                          eval_metric='logloss',
                          device='cuda' if torch.cuda.is_available() else 'cpu')

        # Fit the model with the training dataset and sample weights
        t0: float = time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Falling back to prediction.*")
            M.fit(X_train, y_train, sample_weight=sample_weight)
            elapsed: float = round(time() - t0, 3)

            # Evaluate the model with the validation dataset
            y_val_prob: np.ndarray = M.predict_proba(X_val)[:, 1]

        metrics: dict = binary_metrics(y_true=y_val, y_prob=y_val_prob)
        prefix: dict = {'Algorithm': 'XGB',
                        'Element_index': ele_idx,
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'reg_lambda': reg_lambda,
                        'cost_mul': cost_mul,
                        'seed': random_state,
                        'Elapsed_training_time': elapsed}
        metrics = prefix | {k: v for k, v in metrics.items() if not k.endswith('LIST')}
        records.append(metrics)
        debug_func(prefix)

    # Output: Organize validation performance statistics into a pandas.DataFrame
    df_val: pd.DataFrame = pd.DataFrame(records)
    print_func(f'Training and validation of the grid (size = {len(full_grid)}) completed.')
    return df_val

########################################################################################################################
# Perceptron (ANN) modeling
########################################################################################################################


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        layers: list = []
        in_size: int = input_size
        for _ in range(n_layers):
            layers.append(torch.nn.Linear(in_size, hidden_size))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            in_size = hidden_size
        layers.append(torch.nn.Linear(in_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def ann_modeling(X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 hidden_size_grid: tuple[int] = (128, 256, 512),
                 n_layers_grid: tuple[int] = (2, 3),
                 lr_grid: tuple[float] = (1e-3, 1e-4, 1e-5),
                 cost_mul_grid: tuple[int] = (1, 2, 5, 10),
                 factor_grid: tuple[float] = (0.1,),
                 batch_size_grid: tuple[int] = (32,),
                 dropout_grid: tuple[float] = (0.1,),
                 n_epochs: int = 200,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):

    # Logging
    print_func = print if logger is None else logger.info
    debug_func = print if logger is None else logger.debug

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert to tensors
    X_train_t: torch.Tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t: torch.Tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t: torch.Tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t: torch.Tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Data loader for training
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

    # Input size
    input_size: int = X_train.shape[1]

    # Create an empty list to store model performance
    records: list[dict] = []

    # Loop over the grid
    full_grid: list[tuple] = list(product(hidden_size_grid,
                                          n_layers_grid,
                                          lr_grid,
                                          cost_mul_grid,
                                          factor_grid,
                                          batch_size_grid,
                                          dropout_grid))
    for ele_idx, (hidden_size, n_layers, lr, cost_mul, factor, batch_size, dropout) in enumerate(full_grid, 1):

        # Reproducibility
        torch.manual_seed(random_state)

        # Instantiate model
        M = MLP(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(M.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=factor,
                                                               patience=10)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training
        best_val_loss: float = float('inf')
        best_epoch: int = 0
        best_state: dict = {}
        t0: float = time()
        for epoch in range(n_epochs):

            # Training pass
            M.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits: torch.Tensor = M(xb)
                loss_raw: torch.Tensor = criterion(logits, yb)
                w: torch.Tensor = torch.where(yb == 0,
                                              torch.full_like(yb, cost_mul),
                                              torch.ones_like(yb))
                loss: torch.Tensor = (loss_raw * w).sum() / (w.sum() + 1e-12)
                loss.backward()
                optimizer.step()

            # Validation pass
            M.eval()
            with torch.no_grad():
                val_logits: torch.Tensor = M(X_val_t)
                val_loss_raw: torch.Tensor = criterion(val_logits, y_val_t)
                w_val: torch.Tensor = torch.where(y_val_t == 0,
                                                  torch.full_like(y_val_t, cost_mul),
                                                  torch.ones_like(y_val_t))
                val_loss: float = ((val_loss_raw * w_val).sum() / (w_val.sum() + 1e-12)).item()

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in M.state_dict().items()}

            scheduler.step(val_loss)
            if (epoch + 1) % 10 == 0:
                current_lr: float = optimizer.param_groups[0]['lr']
                debug_func(f'[hidden={hidden_size}, layers={n_layers}, lr={lr:.0e}, cost_mul={cost_mul}, '
                           f'factor={factor}] Epoch {epoch + 1}/{n_epochs} — val_loss={val_loss:.4f} '
                           f'(best={best_val_loss:.4f} @ epoch {best_epoch}) — LR={current_lr:.2e}')

        elapsed: float = round(time() - t0, 3)

        # Restore best weights
        M.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Evaluate on validation set
        M.eval()
        with torch.no_grad():
            val_logits_np: np.ndarray = M(X_val_t).cpu().numpy()
        y_val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits_np))

        metrics: dict = binary_metrics(y_true=y_val, y_prob=y_val_prob)
        prefix: dict = {'Algorithm': 'ANN',
                        'hidden_size': hidden_size,
                        'n_layers': n_layers,
                        'lr': lr,
                        'cost_mul': cost_mul,
                        'factor': factor,
                        'batch_size': batch_size,
                        'dropout': dropout,
                        'n_epochs': n_epochs,
                        'seed': random_state,
                        'best_epoch': best_epoch,
                        'Elapsed_training_time': elapsed}
        metrics = prefix | {k: v for k, v in metrics.items() if not k.endswith('LIST')}
        records.append(metrics)
        debug_func(prefix)

    # Output: Organize validation performance statistics into a pandas.DataFrame
    df_val: pd.DataFrame = pd.DataFrame(records)
    print_func(f'Training and validation of the grid (size = {len(full_grid)}) completed.')
    return df_val
