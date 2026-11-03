########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import torch
import warnings
import xgboost as xgb
from .binary_metrics import binary_metrics
from .Baseline_Modeling import MLP
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from time import time
from xgboost import XGBClassifier

########################################################################################################################
# Define a function for Platt scaling for logits in the validation set
########################################################################################################################


def fit_platt(val_logits: np.ndarray, val_labels: np.ndarray):
    """
    Fit a 2-parameter Platt scaler: p = sigmoid(a * logit + b).
    Returns a fitted sklearn LogisticRegression with a single feature (the logit).
    """
    platt = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    platt.fit(val_logits.reshape(-1, 1), val_labels)
    return platt

########################################################################################################################
# Define a helper function for temperature scaling
########################################################################################################################


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    def nll(T):
        probs = np.clip(1 / (1 + np.exp(-val_logits / T)), 1e-15, 1 - 1e-15)
        return -np.mean(val_labels * np.log(probs) + (1 - val_labels) * np.log(1 - probs))
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded').x)

########################################################################################################################
# Held-out evaluation
########################################################################################################################


def _evaluate_test(df_probs: pd.DataFrame,
                   y_test: np.ndarray,
                   n_params: int,
                   elapsed: float,
                   best_config: dict,
                   algorithm: str) -> pd.DataFrame:
    """Shared test evaluation logic for all baseline retrain functions."""
    prob_label_dict: dict[str, str] = {'Raw': 'y_prob',
                                       'Isotonic': 'y_prob_isotonic',
                                       'Platt': 'y_prob_platt',
                                       'Temperature': 'y_prob_temperature'}
    test_records: list[dict] = []
    for prob_label, col in prob_label_dict.items():
        y_prob: np.ndarray = df_probs[col].to_numpy()
        metrics: dict = binary_metrics(y_true=y_test, y_prob=y_prob, threshold=0.5)
        metrics = {'Algorithm': algorithm,
                   'Scaling': prob_label,
                   'n_Params': n_params,
                   'Training_Time': elapsed} | best_config | {k: v for k, v in metrics.items() if not k.endswith('LIST')}
        test_records.append(metrics)
    return pd.DataFrame(test_records)

########################################################################################################################
# Calibration
########################################################################################################################


def _calibrate_and_build_probs(y_test: np.ndarray,
                               y_test_prob: np.ndarray,
                               val_logits: np.ndarray,
                               test_logits: np.ndarray,
                               y_val_prob: np.ndarray,
                               y_val: np.ndarray) -> pd.DataFrame:
    """Shared calibration logic for all baseline retrain functions."""
    platt = fit_platt(val_logits, y_val)
    y_test_prob_platt: np.ndarray = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-3, y_max=1-1e-3)
    iso.fit(y_val_prob, y_val)
    y_test_prob_iso: np.ndarray = iso.predict(y_test_prob)
    T: float = fit_temperature(val_logits, y_val)
    y_test_prob_temp: np.ndarray = np.clip(1 / (1 + np.exp(-test_logits / T)), 1e-7, 1 - 1e-7)
    return pd.DataFrame({'y_test': y_test,
                         'y_prob': y_test_prob,
                         'y_prob_isotonic': y_test_prob_iso,
                         'y_prob_platt': y_test_prob_platt,
                         'y_prob_temperature': y_test_prob_temp})

########################################################################################################################
# Retrain EN model with the best hyperparameter configuration and evaluate
########################################################################################################################


def en_retrain(best_config: dict,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               X_test: np.ndarray,
               y_test: np.ndarray,
               return_model: bool = False):
    """
    Retrain the best EN config and evaluate on test set.
    :param best_config: dict. Best config.
    :param X_train: np.ndarray. Training features.
    :param y_train: np.ndarray. Training labels.
    :param X_val: np.ndarray. Validation features.
    :param y_val: np.ndarray. Validation labels.
    :param X_test: np.ndarray. Test features.
    :param y_test: np.ndarray. Test labels.
    :return: A tuple of (df_probs, df_test).
    """
    cw = {0: int(best_config['cost_mul']), 1: 1}
    M = LogisticRegression(penalty='elasticnet',
                           C=float(best_config['C']),
                           l1_ratio=float(best_config['l1_ratio']),
                           class_weight=cw,
                           solver='saga',
                           max_iter=int(best_config['max_iter']),
                           random_state=int(best_config['seed']),
                           n_jobs=-1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        t0: float = time()
        M.fit(X_train, y_train)
        elapsed: float = round(time() - t0, 3)

    y_val_prob: np.ndarray = M.predict_proba(X_val)[:, 1]
    y_test_prob: np.ndarray = M.predict_proba(X_test)[:, 1]
    val_logits: np.ndarray = M.decision_function(X_val)
    test_logits: np.ndarray = M.decision_function(X_test)
    n_params: int = int(M.coef_.size + M.intercept_.size)

    df_probs = _calibrate_and_build_probs(y_test, y_test_prob, val_logits,
                                          test_logits, y_val_prob, y_val)
    df_test = _evaluate_test(df_probs, y_test, n_params, elapsed, best_config, 'EN')
    if return_model:
        return df_probs, df_test, M
    else:
        return df_probs, df_test

########################################################################################################################
# Retrain SVM model with the best hyperparameter configuration and evaluate
########################################################################################################################


def svm_retrain(best_config: dict,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                return_model: bool = False):
    """
    Retrain the best SVM config and evaluate on test set.
    :param best_config: dict. Best config.
    :param X_train: np.ndarray. Training features.
    :param y_train: np.ndarray. Training labels.
    :param X_val: np.ndarray. Validation features.
    :param y_val: np.ndarray. Validation labels.
    :param X_test: np.ndarray. Test features.
    :param y_test: np.ndarray. Test labels.
    :return: A tuple of (df_probs, df_test).
    """
    penalty: str = str(best_config['penalty'])
    cw = {0: int(best_config['cost_mul']), 1: 1}
    M = LinearSVC(C=float(best_config['C']),
                  penalty=penalty,
                  dual=penalty == 'l2',
                  class_weight=cw,
                  max_iter=int(best_config['max_iter']),
                  random_state=int(best_config['seed']))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        t0: float = time()
        M.fit(X_train, y_train)
        elapsed: float = round(time() - t0, 3)

    val_logits: np.ndarray = M.decision_function(X_val)
    test_logits: np.ndarray = M.decision_function(X_test)
    y_val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits))
    y_test_prob: np.ndarray = 1 / (1 + np.exp(-test_logits))
    n_params: int = int(M.coef_.size + M.intercept_.size)

    df_probs = _calibrate_and_build_probs(y_test, y_test_prob, val_logits,
                                          test_logits, y_val_prob, y_val)
    df_test = _evaluate_test(df_probs, y_test, n_params, elapsed, best_config, 'SVM')
    if return_model:
        return df_probs, df_test, M
    else:
        return df_probs, df_test

########################################################################################################################
# Retrain XGB model with the best hyperparameter configuration and evaluate
########################################################################################################################


def xgb_retrain(best_config: dict,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                return_model: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrain the best XGB config and evaluate on test set.
    :param best_config: dict. Best config.
    :param X_train: np.ndarray. Training features.
    :param y_train: np.ndarray. Training labels.
    :param X_val: np.ndarray. Validation features.
    :param y_val: np.ndarray. Validation labels.
    :param X_test: np.ndarray. Test features.
    :param y_test: np.ndarray. Test labels.
    :return: A tuple of (df_probs, df_test).
    """
    cost_mul: int = int(best_config['cost_mul'])
    sample_weight: np.ndarray = np.where(y_train == 0, cost_mul, 1.0)
    M = XGBClassifier(max_depth=int(best_config['max_depth']),
                      n_estimators=int(best_config['n_estimators']),
                      learning_rate=float(best_config['learning_rate']),
                      reg_lambda=float(best_config['reg_lambda']),
                      random_state=int(best_config['seed']),
                      eval_metric='logloss',
                      device='cuda' if torch.cuda.is_available() else 'cpu')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        t0: float = time()
        M.fit(X_train, y_train, sample_weight=sample_weight)
        elapsed: float = round(time() - t0, 3)

    dval: xgb.DMatrix = xgb.DMatrix(X_val)
    dtest: xgb.DMatrix = xgb.DMatrix(X_test)
    y_val_prob: np.ndarray = M.get_booster().predict(dval)
    y_test_prob: np.ndarray = M.get_booster().predict(dtest)
    y_val_prob_c: np.ndarray = np.clip(y_val_prob, 1e-7, 1 - 1e-7)
    y_test_prob_c: np.ndarray = np.clip(y_test_prob, 1e-7, 1 - 1e-7)
    val_logits: np.ndarray = np.log(y_val_prob_c / (1 - y_val_prob_c))
    test_logits: np.ndarray = np.log(y_test_prob_c / (1 - y_test_prob_c))
    df_trees: pd.DataFrame = M.get_booster().trees_to_dataframe()
    n_internal: int = int((df_trees['Feature'] != 'Leaf').sum())
    n_leaves: int = int((df_trees['Feature'] == 'Leaf').sum())
    n_params: int = 2 * n_internal + n_leaves

    df_probs = _calibrate_and_build_probs(y_test, y_test_prob, val_logits,
                                          test_logits, y_val_prob, y_val)
    df_test = _evaluate_test(df_probs, y_test, n_params, elapsed, best_config, 'XGB')
    if return_model:
        return df_probs, df_test, M
    else:
        return df_probs, df_test


########################################################################################################################
# Retrain ANN model with the best hyperparameter configuration and evaluate
########################################################################################################################

def ann_retrain(best_config: dict,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                return_model: bool = False):
    """
    Retrain the best ANN config and evaluate on test set.
    :param best_config: dict. Best config.
    :param X_train: np.ndarray. Training features.
    :param y_train: np.ndarray. Training labels.
    :param X_val: np.ndarray. Validation features.
    :param y_val: np.ndarray. Validation labels.
    :param X_test: np.ndarray. Test features.
    :param y_test: np.ndarray. Test labels.
    :return: A tuple of (df_probs, df_test).
    """
    random_state: int = int(best_config['seed'])
    hidden_size: int = int(best_config['hidden_size'])
    n_layers: int = int(best_config['n_layers'])
    lr: float = float(best_config['lr'])
    cost_mul: int = int(best_config['cost_mul'])
    factor: float = float(best_config['factor'])
    n_epochs: int = int(best_config['n_epochs'])
    batch_size: int = int(best_config['batch_size'])

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(random_state)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_size: int = X_train.shape[1]
    M = MLP(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(M.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=factor, patience=10)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss: float = float('inf')
    best_state: dict = {}
    t0: float = time()
    for epoch in range(n_epochs):
        M.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = M(xb)
            loss_raw = criterion(logits, yb)
            w = torch.where(yb == 0, torch.full_like(yb, cost_mul), torch.ones_like(yb))
            loss = (loss_raw * w).sum() / (w.sum() + 1e-12)
            loss.backward()
            optimizer.step()

        M.eval()
        with torch.no_grad():
            val_logits_t = M(X_val_t)
            val_loss_raw = criterion(val_logits_t, y_val_t)
            w_val = torch.where(y_val_t == 0, torch.full_like(y_val_t, cost_mul), torch.ones_like(y_val_t))
            val_loss = ((val_loss_raw * w_val).sum() / (w_val.sum() + 1e-12)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in M.state_dict().items()}
        scheduler.step(val_loss)
    elapsed: float = round(time() - t0, 3)

    M.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    M.eval()
    with torch.no_grad():
        val_logits: np.ndarray = M(X_val_t).cpu().numpy()
        test_logits: np.ndarray = M(X_test_t).cpu().numpy()

    y_val_prob: np.ndarray = 1 / (1 + np.exp(-val_logits))
    y_test_prob: np.ndarray = 1 / (1 + np.exp(-test_logits))
    n_params: int = sum(p.numel() for p in M.parameters())

    df_probs = _calibrate_and_build_probs(y_test, y_test_prob, val_logits,
                                          test_logits, y_val_prob, y_val)
    df_test = _evaluate_test(df_probs, y_test, n_params, elapsed, best_config, 'ANN')
    if return_model:
        return df_probs, df_test, M
    else:
        return df_probs, df_test
