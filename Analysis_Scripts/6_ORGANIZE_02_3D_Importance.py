########################################################################################################################
# Compute GradientSHAP feature importance for all 4 transformer models (standard + fusion × unimodal + multimodal)
# and map the results onto the 564-feature data dictionary as new columns.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
import torch
import warnings
from captum.attr import GradientShap
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing_Fusion import preprocess_data
from _Helper_Scripts.Transformer_Modeling import TransformerClassifier
from _Helper_Scripts.Transformer_Fusion_Modeling import FusionTransformerClassifier
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Runtime parameters
########################################################################################################################
OUTCOME: str = 'BSI_GSI_Case'
BKGD_COLS: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                         'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                         'bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']
SEEDS: list[int] = [0, 1, 2, 42]
N_SAMPLES: int = 200
SHAP_BATCH_SIZE: int = 16
DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################################################################
# Model specifications
########################################################################################################################
MODEL_SPECS: dict = {
    'TF_unimodal': {
        'model_class': 'standard',
        'modality': 'unimodal',
        'result_dir': '_Results_Retraining_v2/TF_unimodal_SEEDED',
    },
    'TF_multimodal': {
        'model_class': 'standard',
        'modality': 'multimodal',
        'result_dir': '_Results_Retraining_v2/TF_multimodal_SEEDED',
    },
    'TF_Fusion_unimodal': {
        'model_class': 'fusion',
        'modality': 'unimodal',
        'result_dir': '_Results_Retraining_Fusion_v2/TF_Fusion_unimodal',
    },
    'TF_Fusion_multimodal': {
        'model_class': 'fusion',
        'modality': 'multimodal',
        'result_dir': '_Results_Retraining_Fusion_v2/TF_Fusion_multimodal',
    },
}

########################################################################################################################
# Paths
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'
X_PATH: str = os.path.join(IN_DIR, 'Features_v4.parquet')
Y_PATH: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
DICT_IN_PATH: str = '_Final_Results/Supplementary_Table_1_-_Data_Dictionary_v2.csv'
DICT_OUT_PATH: str = '_Final_Results/Supplementary_Table_1_-_Data_Dictionary_v3.csv'
os.makedirs(os.path.dirname(DICT_OUT_PATH), exist_ok=True)

########################################################################################################################
# Load features and outcome
########################################################################################################################
logger.info('Loading features and outcome...')
df_X: pd.DataFrame = pd.read_parquet(X_PATH)
df_y: pd.DataFrame = pd.read_csv(Y_PATH)
logger.info(f'df_X: {df_X.shape}, df_y: {df_y.shape}')
logger.print('-' * 100)

########################################################################################################################
# Build TRAIN / TEST partitions from Partition_Retraining
########################################################################################################################
df_y_train: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TRAIN'].copy().reset_index(drop=True)
y_train_ids: list[int] = df_y_train['Record_ID'].unique().tolist()
df_X_train: pd.DataFrame = df_X.set_index('Record_ID').loc[y_train_ids].reset_index()

df_y_test: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TEST'].copy().reset_index(drop=True)
y_test_ids: list[int] = df_y_test['Record_ID'].unique().tolist()
df_X_test: pd.DataFrame = df_X.set_index('Record_ID').loc[y_test_ids].reset_index()

logger.info(f'TRAIN: {len(y_train_ids)} records; TEST: {len(y_test_ids)} records')
logger.print('-' * 100)

########################################################################################################################
# Load the data dictionary from the 2D script
########################################################################################################################
logger.info(f'Loading data dictionary from {DICT_IN_PATH}...')
df_dict: pd.DataFrame = pd.read_csv(DICT_IN_PATH)
logger.info(f'Dictionary shape: {df_dict.shape}')
logger.print('=' * 100)


########################################################################################################################
# Helper: float32 conversion preserving NaN
########################################################################################################################
def _to_float32_with_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == object:
        a = np.where(pd.isna(a), np.nan, a)
    return a.astype(np.float32, copy=False)


########################################################################################################################
# Helper: GradientSHAP for standard TransformerClassifier (3-arg forward_for_shap: x0, timestamp, mask)
########################################################################################################################
def compute_shap_standard(model: TransformerClassifier,
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          n_samples: int,
                          batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    X_train = _to_float32_with_nan(X_train)
    X_test = _to_float32_with_nan(X_test)
    device = next(model.parameters()).device
    n_train, L, F = X_train.shape
    n_test = X_test.shape[0]

    # Fill NaN baselines with per-(bin, feature) median
    mall: np.ndarray = np.nanmedian(X_train, axis=0)                            # (L, F)
    mall = np.where(np.isnan(mall), 0.5, mall).astype(np.float32)
    X_train_fill: np.ndarray = np.where(np.isnan(X_train), mall[None], X_train).astype(np.float32)
    X_baseline: torch.Tensor = torch.tensor(X_train_fill, dtype=torch.float32, device=device)

    M_test: np.ndarray = (~np.isnan(X_test)).astype(np.float32)
    X_test_fill: np.ndarray = np.where(np.isnan(X_test), mall[None], X_test).astype(np.float32)

    # Positional timestamp: arange(L), same as TransformerClassifier.forward generates internally
    T: torch.Tensor = torch.arange(L, dtype=torch.float32, device=device)       # (L,)

    model.eval()
    explainer = GradientShap(forward_func=lambda x, t, m: model.forward_for_shap(x, t, m))

    attrs: list[torch.Tensor] = []
    for i in range(0, n_test, batch_size):
        x: torch.Tensor = torch.tensor(X_test_fill[i:i + batch_size], dtype=torch.float32, device=device)
        b: int = x.shape[0]
        t: torch.Tensor = T.unsqueeze(0).expand(b, -1).unsqueeze(-1)
        m: torch.Tensor = torch.tensor(M_test[i:i + batch_size], dtype=torch.float32, device=device)
        a: torch.Tensor = explainer.attribute(inputs=x,
                                              baselines=X_baseline,
                                              additional_forward_args=(t, m),
                                              n_samples=n_samples)
        attrs.append(a.detach().cpu())

    shap_3d: np.ndarray = torch.cat(attrs, dim=0).numpy()
    shap_3d = shap_3d * M_test # post-hoc zeroing
    return shap_3d, M_test


########################################################################################################################
# Helper: GradientSHAP for FusionTransformerClassifier (2-arg forward_for_shap: x0, mask)
########################################################################################################################
def compute_shap_fusion(model: FusionTransformerClassifier,
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        n_samples: int,
                        batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    X_train = _to_float32_with_nan(X_train)
    X_test = _to_float32_with_nan(X_test)
    device = next(model.parameters()).device
    n_train, L, F = X_train.shape
    n_test = X_test.shape[0]

    mall: np.ndarray = np.nanmedian(X_train, axis=0)
    mall = np.where(np.isnan(mall), 0.5, mall).astype(np.float32)
    X_train_fill: np.ndarray = np.where(np.isnan(X_train), mall[None], X_train).astype(np.float32)
    X_baseline: torch.Tensor = torch.tensor(X_train_fill, dtype=torch.float32, device=device)

    M_test: np.ndarray = (~np.isnan(X_test)).astype(np.float32)
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

    shap_3d: np.ndarray = torch.cat(attrs, dim=0).numpy()
    shap_3d = shap_3d * M_test
    return shap_3d, M_test


########################################################################################################################
# Helper: summarize SHAP with correct static vs. temporal denominator
# Reuses the logic from SHAP_Helper.summarize_shap but allows all-temporal mode for standard transformers.
########################################################################################################################
def summarize_shap(shap_3d: np.ndarray,
                   M_test: np.ndarray,
                   feature_names: list[str],
                   static_idxs: list[int],
                   temporal_idxs: list[int]) -> pd.DataFrame:
    n_feat: int = shap_3d.shape[2]
    mean_abs_shap: np.ndarray = np.full(n_feat, np.nan, dtype=np.float64)

    if len(temporal_idxs) > 0:
        observed_count: np.ndarray = M_test[:, :, temporal_idxs].sum(axis=(0, 1)).clip(min=1)
        mean_abs_shap[temporal_idxs] = (np.abs(shap_3d[:, :, temporal_idxs]).sum(axis=(0, 1))
                                        / observed_count)

    if len(static_idxs) > 0:
        observed_count_static: np.ndarray = M_test[:, 0, static_idxs].sum(axis=0).clip(min=1)
        mean_abs_shap[static_idxs] = (np.abs(shap_3d[:, 0, static_idxs]).sum(axis=0)
                                      / observed_count_static)

    return pd.DataFrame({'Feature': feature_names, 'MeanAbsSHAP': mean_abs_shap})


########################################################################################################################
# Helper: load a standard TransformerClassifier from saved context + checkpoint
########################################################################################################################
def load_standard_model(context: dict, raw_path: str, device: torch.device) -> TransformerClassifier:
    cfg: dict = context['best_config']
    model = TransformerClassifier(
        feature_dim=int(context['n_feat']),
        d_model=int(cfg['d_model']),
        n_head=int(cfg['n_heads']),
        n_units=int(cfg['n_units']),
        n_layers=int(cfg['n_layers']),
        task='binary',
        dropout=float(cfg['dropout']),
        handle_nan=str(cfg['handle_nan']),
    ).to(device)
    model.load_state_dict(torch.load(raw_path, map_location=device, weights_only=True))
    model.eval()
    return model


########################################################################################################################
# Helper: load a FusionTransformerClassifier from saved context + checkpoint
########################################################################################################################
def load_fusion_model(context: dict, raw_path: str, device: torch.device) -> FusionTransformerClassifier:
    cfg: dict = context['best_config']
    model = FusionTransformerClassifier(
        temporal_idxs=list(context['temporal_idxs']),
        static_idxs=list(context['static_idxs']),
        d_model=int(cfg['d_model']),
        n_head=int(cfg['n_heads']),
        n_units=int(cfg['n_units']),
        n_layers=int(cfg['n_layers']),
        d_static=int(cfg['d_static']),
        n_static_layers=int(cfg['n_static_layers']),
        task='binary',
        dropout=float(cfg['dropout']),
        handle_nan=str(cfg['handle_nan']),
    ).to(device)
    model.load_state_dict(torch.load(raw_path, map_location=device, weights_only=True))
    model.eval()
    return model


########################################################################################################################
# Main loop: iterate over all 4 transformer models
########################################################################################################################
preprocess_cache: dict[str, tuple] = {}                                          # cache per modality

for model_name, spec in MODEL_SPECS.items():
    logger.print('=' * 100)
    logger.info(f'Processing {model_name}...')
    modality: str = spec['modality']
    model_class: str = spec['model_class']
    result_dir: str = spec['result_dir']

    # ------------------------------------------------------------------------------------------------------------------
    # Preprocess once per modality (reuse for standard + fusion of same modality)
    # ------------------------------------------------------------------------------------------------------------------
    if modality not in preprocess_cache:
        logger.info(f'Preprocessing for modality={modality}...')
        pp_result = preprocess_data(
            df_X_train=df_X_train,
            df_X_val=df_X_test,
            bkgd_cols=BKGD_COLS,
            groupby_col='Record_ID',
            sd_away=3,
            dim='3D',
            modality=modality,
            logger=logger,
        )
        X_train_3D, X_test_3D, ids_train, ids_test, feats, pp_static_idxs, pp_temporal_idxs = pp_result
        assert list(ids_train) == y_train_ids, f'{modality}: ids_train order mismatch'
        assert list(ids_test) == y_test_ids, f'{modality}: ids_test order mismatch'
        logger.info(f'{modality}: X_train {X_train_3D.shape}, X_test {X_test_3D.shape}')
        logger.info(f'{modality}: {len(pp_static_idxs)} static, {len(pp_temporal_idxs)} temporal features')
        preprocess_cache[modality] = (X_train_3D, X_test_3D, feats, pp_static_idxs, pp_temporal_idxs)
    else:
        X_train_3D, X_test_3D, feats, pp_static_idxs, pp_temporal_idxs = preprocess_cache[modality]
        logger.info(f'Reusing cached preprocessing for modality={modality}')

    logger.print('-' * 100)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute SHAP per seed
    # ------------------------------------------------------------------------------------------------------------------
    shap_per_seed: list[np.ndarray] = []
    M_test_ref: np.ndarray | None = None

    for seed in SEEDS:
        raw_path: str = os.path.join(result_dir, f'Best_Model_seed{seed}_{modality}.pth')
        ctx_path: str = os.path.join(result_dir, f'Model_Context_seed{seed}_{modality}.pth')
        context: dict = torch.load(ctx_path, map_location='cpu', weights_only=False)

        # Sanity: n_feat must match preprocessed data
        assert int(context['n_feat']) == X_train_3D.shape[2], \
            f'{model_name} seed={seed}: n_feat={context["n_feat"]} vs preprocessed F={X_train_3D.shape[2]}'

        # Load model
        if model_class == 'fusion':
            # Extra sanity: verify static/temporal indices match context
            assert list(context['temporal_idxs']) == pp_temporal_idxs, \
                f'{model_name} seed={seed}: temporal_idxs mismatch'
            assert list(context['static_idxs']) == pp_static_idxs, \
                f'{model_name} seed={seed}: static_idxs mismatch'
            model = load_fusion_model(context, raw_path, DEVICE)
        else:
            model = load_standard_model(context, raw_path, DEVICE)
        logger.info(f'seed={seed}: loaded from {raw_path}')

        # Compute SHAP
        if model_class == 'fusion':
            shap_3d, M_test = compute_shap_fusion(model, X_train_3D, X_test_3D, N_SAMPLES, SHAP_BATCH_SIZE)
        else:
            shap_3d, M_test = compute_shap_standard(model, X_train_3D, X_test_3D, N_SAMPLES, SHAP_BATCH_SIZE)

        shap_per_seed.append(shap_3d)
        if M_test_ref is None:
            M_test_ref = M_test
        else:
            assert np.array_equal(M_test_ref, M_test), f'{model_name} seed={seed}: M_test mismatch'

        logger.info(f'seed={seed}: SHAP computed, shape={shap_3d.shape}')

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    logger.print('-' * 100)

    # ------------------------------------------------------------------------------------------------------------------
    # Average SHAP across seeds, then summarize
    # ------------------------------------------------------------------------------------------------------------------
    shap_3d_mean: np.ndarray = np.mean(np.stack(shap_per_seed, axis=0), axis=0)  # (N_test, L, F)
    logger.info(f'Averaged SHAP across {len(SEEDS)} seeds. Shape: {shap_3d_mean.shape}')

    # For standard transformers: treat ALL features as temporal (no static stream in the architecture).
    # For fusion transformers: use the static/temporal split from preprocessing.
    if model_class == 'standard':
        s_static: list[int] = []
        s_temporal: list[int] = list(range(X_train_3D.shape[2]))
    else:
        s_static = pp_static_idxs
        s_temporal = pp_temporal_idxs

    df_summary: pd.DataFrame = summarize_shap(shap_3d_mean, M_test_ref, feats, s_static, s_temporal)

    # ------------------------------------------------------------------------------------------------------------------
    # Map onto dictionary
    # ------------------------------------------------------------------------------------------------------------------
    col_name: str = f'{model_name}_MeanAbsSHAP'
    shap_map: dict[str, float] = dict(zip(df_summary['Feature'], df_summary['MeanAbsSHAP']))
    df_dict[col_name] = df_dict['Feature_Name'].map(shap_map)
    n_mapped: int = df_dict[col_name].notna().sum()
    logger.info(f'{model_name}: {n_mapped}/{len(df_dict)} features mapped to dictionary')
    logger.print('-' * 100)

########################################################################################################################
# Save updated dictionary
########################################################################################################################
logger.print('=' * 100)
shap_cols: list[str] = [f'{mn}_MeanAbsSHAP' for mn in MODEL_SPECS]
df_dict[shap_cols] = df_dict[shap_cols].fillna('N/A')
df_dict.to_csv(DICT_OUT_PATH, index=False)
logger.info(f'Saved full data dictionary to {DICT_OUT_PATH}')
logger.info(f'Final shape: {df_dict.shape}')

# Summary of mapped features per model
for model_name in MODEL_SPECS:
    col: str = f'{model_name}_MeanAbsSHAP'
    n: int = df_dict[col].notna().sum()
    logger.info(f'  {col}: {n}/{len(df_dict)} mapped')

logger.print('=' * 100)
