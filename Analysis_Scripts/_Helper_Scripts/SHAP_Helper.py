########################################################################################################################
# Compute GradientSHAP attributions for the two-stream FusionTransformerClassifier.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap
from .Transformer_Fusion_Modeling import FusionTransformerClassifier

########################################################################################################################
# Helper function to handle missing values
########################################################################################################################


def _to_float32_with_nan(a: np.ndarray) -> np.ndarray:
    """
    :param a: np.ndarray. Input array, possibly object dtype with NaN-like entries.
    :return: np.ndarray. float32 array with NaN preserved.
    """
    a = np.asarray(a)
    if a.dtype == object:
        a = np.where(pd.isna(a), np.nan, a)
    return a.astype(np.float32, copy=False)

########################################################################################################################
# Core function to compute SHAP values for one fusion model
########################################################################################################################


def get_shap(model: FusionTransformerClassifier,
             X_train: np.ndarray,
             X_test: np.ndarray,
             feature_names: list[str],
             n_samples: int | None = None,
             batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    :param model: FusionTransformerClassifier. Trained model; will be set to eval mode here.
    :param X_train: np.ndarray. Training features (N_train, L, F), NaN preserved. Supplies the GradientShap
                    baseline distribution.
    :param X_test: np.ndarray. Test features (N_test, L, F), NaN preserved. Attributions computed for these.
    :param feature_names: list[str]. Column names, length F.
    :param n_samples: int or None. Number of (baseline, alpha) interpolation draws per input. None defaults to
                      N_train (full baseline pool); smaller values (e.g. 100-200) trade fidelity for speed.
    :param batch_size: int. Number of test records processed per forward/backward pass.
    :return:
    shap_3d: np.ndarray (N_test, L, F). Attributions, zeroed at unobserved (NaN) raw-input positions.
    M_test: np.ndarray (N_test, L, F). Observedness mask for X_test (1 = observed, 0 = NaN in the raw input).
    """
    X_train = _to_float32_with_nan(X_train)
    X_test = _to_float32_with_nan(X_test)
    device = next(model.parameters()).device

    assert X_train.ndim == 3, f'X_train must be (N,L,F), got {X_train.shape}'
    assert X_test.ndim == 3, f'X_test must be (N,L,F), got {X_test.shape}'
    n_train, L, F = X_train.shape
    n_test = X_test.shape[0]
    assert X_test.shape[1:] == (L, F), f'Train (L,F)=({L},{F}) but test {X_test.shape[1:]}'
    assert len(feature_names) == F, f'len(feature_names)={len(feature_names)}, expected F={F}'

    if n_samples is None:
        n_samples = n_train

    # Fill NaNs in the training set with the per-(bin, feature) median across training records.
    # Baselines passed to GradientShap must be valid model inputs (no NaN).
    mall: np.ndarray = np.nanmedian(X_train, axis=0)                           # (L, F)
    mall = np.where(np.isnan(mall), 0.5, mall).astype(np.float32)
    X_train_fill: np.ndarray = np.where(np.isnan(X_train), mall[None], X_train).astype(np.float32)
    X_baseline: torch.Tensor = torch.tensor(X_train_fill, dtype=torch.float32, device=device)  # (N_train, L, F)

    # Fill NaNs in the test set the same way; missingness is communicated to the model via M_test.
    M_test: np.ndarray = (~np.isnan(X_test)).astype(np.float32)                # (N_test, L, F)
    X_test_fill: np.ndarray = np.where(np.isnan(X_test), mall[None], X_test).astype(np.float32)

    model.eval()
    explainer = GradientShap(forward_func=model.forward_for_shap)

    attrs: list[torch.Tensor] = []
    for i in range(0, n_test, batch_size):
        x: torch.Tensor = torch.tensor(X_test_fill[i:i + batch_size], dtype=torch.float32, device=device)
        m: torch.Tensor = torch.tensor(M_test[i:i + batch_size], dtype=torch.float32, device=device)
        a: torch.Tensor = explainer.attribute(inputs=x,
                                              baselines=X_baseline,
                                              additional_forward_args=(m,),
                                              n_samples=n_samples)
        attrs.append(a.detach().cpu())

    shap_3d: np.ndarray = torch.cat(attrs, dim=0).numpy()                      # (N_test, L, F)

    # Post-hoc zeroing: eliminate residual gradient noise at unobserved (NaN) positions. Note this is orthogonal
    # to the static-feature bin-0-only property below: static features that ARE observed have M_test=1 at every
    # bin (the raw value is broadcast, so it is non-NaN everywhere), so this multiply does not additionally zero
    # their unused bins — that zeroing already happens via the model's own computational graph (no gradient path
    # from bins 1..L-1 to the output for static columns).
    shap_3d = shap_3d * M_test

    return shap_3d, M_test

########################################################################################################################
# Summarize per-feature importance, correctly weighting static vs. temporal features
########################################################################################################################


def summarize_shap(shap_3d: np.ndarray,
                   M_test: np.ndarray,
                   feature_names: list[str],
                   static_idxs: list[int],
                   temporal_idxs: list[int]) -> pd.DataFrame:
    """
    Computes mean |SHAP| per feature using the correct denominator for each feature type. Static (constant-
    across-bin) features carry essentially all their attribution mass at bin index 0 — the only position
    forward_for_shap reads for them — so their denominator is the observed-record count at bin 0. Temporal
    features can carry attribution at every observed bin, so their denominator is the observed-cell count
    across (N_test, L), matching prior convention. Using the (N_test, L)-wide denominator for static features
    would understate their importance by a factor of roughly L, since only 1 of L bins actually contributes.
    :param shap_3d: np.ndarray (N_test, L, F). Attributions, already zeroed at unobserved positions.
    :param M_test: np.ndarray (N_test, L, F). Observedness mask for the same test set.
    :param feature_names: list[str]. Column names, length F.
    :param static_idxs: list[int]. Column indices of static (constant-across-bin) features.
    :param temporal_idxs: list[int]. Column indices of temporal (varying-across-bin) features.
    :return: pd.DataFrame with columns ['Feature', 'FeatureType', 'MeanAbsSHAP'], sorted descending within type.
    """
    n_feat: int = shap_3d.shape[2]
    mean_abs_shap: np.ndarray = np.full(n_feat, np.nan, dtype=np.float64)

    if len(temporal_idxs) > 0:
        observed_count: np.ndarray = M_test[:, :, temporal_idxs].sum(axis=(0, 1)).clip(min=1)
        mean_abs_shap[temporal_idxs] = (np.abs(shap_3d[:, :, temporal_idxs]).sum(axis=(0, 1))
                                        / observed_count)

    if len(static_idxs) > 0:
        # Only bin 0 carries real attribution for static features; use it directly rather than summing over L.
        observed_count_static: np.ndarray = M_test[:, 0, static_idxs].sum(axis=0).clip(min=1)
        mean_abs_shap[static_idxs] = (np.abs(shap_3d[:, 0, static_idxs]).sum(axis=0)
                                      / observed_count_static)

    static_set: set[int] = set(static_idxs)
    feature_type: list[str] = ['Static' if i in static_set else 'Temporal' for i in range(n_feat)]
    df_summary: pd.DataFrame = pd.DataFrame({'Feature': feature_names,
                                             'FeatureType': feature_type,
                                             'MeanAbsSHAP': mean_abs_shap})
    return df_summary.sort_values(['FeatureType', 'MeanAbsSHAP'], ascending=[True, False]).reset_index(drop=True)
