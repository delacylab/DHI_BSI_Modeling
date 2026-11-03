########################################################################################################################
# Organize the SHAP values for the best transformer (fusion) model across 4 random seeds
# Computes GradientSHAP attributions for each seed's raw model on the fixed TEST partition, then averages the
# resulting attribution tensors elementwise (valid by SHAP's linearity: SHAP of a mean prediction equals the
# mean of the per-model SHAP values). Also runs a per-seed vs. averaged top-10 overlap check as a sanity diagnostic.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
import torch
import warnings
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing_Fusion import preprocess_data
from _Helper_Scripts.SHAP_Helper import get_shap, summarize_shap
from _Helper_Scripts.Transformer_Fusion_Modeling import FusionTransformerClassifier
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Runtime parameters
########################################################################################################################
MODALITY: str = 'multimodal'
OUTCOME: str = 'BSI_GSI_Case'
BKGD_COLS: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                        'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                        'bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']
SEEDS: list[int] = [0, 1, 2, 42]

# GradientShap interpolation samples per input. None reproduces the reference script's default (= N_train, the
# full baseline pool), which is slow for this model size; 200 is a common practical value that trades some
# fidelity for a large speedup. Tune based on wall-clock and GPU memory.
N_SAMPLES: int | None = 200
SHAP_BATCH_SIZE: int = 16

########################################################################################################################
# Paths
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'
X_PATH: str = os.path.join(IN_DIR, 'Features_v4.parquet')
Y_PATH: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
MODEL_DIR: str = f'_Results_Retraining_Fusion_v2/TF_Fusion_{MODALITY}/'
OUT_DIR: str = f'_Final_Results/_Results_SHAP/TF_Fusion_{MODALITY}/'
os.makedirs(OUT_DIR, exist_ok=True)

model_paths: dict[int, tuple[str, str]] = \
    {seed: (os.path.join(MODEL_DIR, f'Best_Model_seed{seed}_{MODALITY}.pth'),
            os.path.join(MODEL_DIR, f'Model_Context_seed{seed}_{MODALITY}.pth')) for seed in SEEDS}

########################################################################################################################
# Load features and outcome
########################################################################################################################
df_X: pd.DataFrame = pd.read_parquet(X_PATH)
df_y: pd.DataFrame = pd.read_csv(Y_PATH)

########################################################################################################################
# Build TRAIN / TEST partitions from Partition_Retraining, aligned by Record_ID.
# VAL is not needed for SHAP; TRAIN supplies the GradientShap baseline distribution and TEST is what we explain.
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
# Preprocess. preprocess_data requires a 'val' partition; we pass df_X_test in that slot since we only need a
# two-way TRAIN/TEST split here (no separate val is used for SHAP), and val/test receive identical treatment
# (winsorize + scale using TRAIN-fit parameters; no fitting happens on either). Variables are named for what they
# represent (TEST), not for the parameter slot they were passed through.
########################################################################################################################
logger.info('Preprocessing...')
X_train_3D, X_test_3D, ids_train, ids_test, feats, static_idxs, temporal_idxs = preprocess_data(
    df_X_train=df_X_train,
    df_X_val=df_X_test,
    bkgd_cols=BKGD_COLS,
    groupby_col='Record_ID',
    sd_away=3,
    dim='3D',
    modality=MODALITY,
    logger=logger,
)
assert list(ids_train) == y_train_ids, 'ids_train order mismatch'
assert list(ids_test) == y_test_ids, 'ids_test order mismatch'
assert len(feats) == X_train_3D.shape[2], 'feats count mismatch'
assert len(static_idxs) + len(temporal_idxs) == X_train_3D.shape[2], 'index count mismatch'
logger.info(f'X_train {X_train_3D.shape}, X_test {X_test_3D.shape}')
logger.info(f'static features: {len(static_idxs)}, temporal features: {len(temporal_idxs)}')
logger.print('-' * 100)

########################################################################################################################
# Compute SHAP per seed
########################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shap_per_seed: list[np.ndarray] = []
summary_per_seed: list[pd.DataFrame] = []
M_test_ref: np.ndarray | None = None

for seed in SEEDS:
    raw_path, context_path = model_paths[seed]
    context: dict = torch.load(context_path, map_location='cpu', weights_only=False)
    best_config: dict = context['best_config']
    temporal_idxs_ctx: list[int] = list(context['temporal_idxs'])
    static_idxs_ctx: list[int] = list(context['static_idxs'])

    # Sanity: the indices recomputed by the current preprocessing run must match what the saved model used.
    assert temporal_idxs_ctx == temporal_idxs, f'seed={seed}: temporal_idxs mismatch vs. saved context'
    assert static_idxs_ctx == static_idxs, f'seed={seed}: static_idxs mismatch vs. saved context'
    assert int(context['n_feat']) == X_train_3D.shape[2], f'seed={seed}: n_feat mismatch vs. saved context'

    M = FusionTransformerClassifier(temporal_idxs=temporal_idxs_ctx,
                                    static_idxs=static_idxs_ctx,
                                    d_model=int(best_config['d_model']),
                                    n_head=int(best_config['n_heads']),
                                    n_units=int(best_config['n_units']),
                                    n_layers=int(best_config['n_layers']),
                                    d_static=int(best_config['d_static']),
                                    n_static_layers=int(best_config['n_static_layers']),
                                    task='binary',
                                    handle_nan=str(best_config['handle_nan']),
                                    dropout=float(best_config['dropout'])).to(device)
    M.load_state_dict(torch.load(raw_path, map_location=device, weights_only=True))
    M.eval()
    logger.info(f'seed={seed}: model loaded from {raw_path}')

    shap_3d, M_test = get_shap(model=M,
                               X_train=X_train_3D,
                               X_test=X_test_3D,
                               feature_names=feats,
                               n_samples=N_SAMPLES,
                               batch_size=SHAP_BATCH_SIZE)
    shap_per_seed.append(shap_3d)
    if M_test_ref is None:
        M_test_ref = M_test
    else:
        assert np.array_equal(M_test_ref, M_test), f'seed={seed}: M_test mismatch — test set should be fixed'

    df_summary_seed: pd.DataFrame = summarize_shap(shap_3d, M_test, feats, static_idxs, temporal_idxs)
    df_summary_seed.insert(0, 'seed', seed)
    summary_per_seed.append(df_summary_seed)
    logger.info(f'seed={seed}: SHAP computed, shape={shap_3d.shape}')
    logger.print('-' * 100)

########################################################################################################################
# Average across seeds (valid by SHAP linearity: SHAP of the mean prediction = mean of the per-seed SHAP values).
# Averaging happens on the raw tensors first; the summary is then computed once from the averaged tensor, not by
# averaging 4 separately-normalized per-seed summaries.
########################################################################################################################
shap_3d_mean: np.ndarray = np.mean(np.stack(shap_per_seed, axis=0), axis=0)    # (N_test, L, F)
df_summary_mean: pd.DataFrame = summarize_shap(shap_3d_mean, M_test_ref, feats, static_idxs, temporal_idxs)
logger.info(f'Averaged SHAP across {len(SEEDS)} seeds. shap_3d_mean shape: {shap_3d_mean.shape}')
logger.print('-' * 100)

########################################################################################################################
# Per-seed agreement check: top-10 features by seed vs. by the averaged tensor
########################################################################################################################
for seed_df in summary_per_seed:
    seed: int = int(seed_df['seed'].iloc[0])
    top10_seed: list[str] = seed_df.sort_values('MeanAbsSHAP', ascending=False)['Feature'].head(10).tolist()
    top10_mean: list[str] = df_summary_mean.sort_values('MeanAbsSHAP', ascending=False)['Feature'].head(10).tolist()
    overlap: int = len(set(top10_seed) & set(top10_mean))
    logger.info(f'seed={seed}: {overlap}/10 features overlap with averaged top-10')
logger.print('-' * 100)

########################################################################################################################
# Save outputs
########################################################################################################################
np.savez_compressed(os.path.join(OUT_DIR, f'SHAP_Averaged_{MODALITY}.npz'),
                    shap_3d_mean=shap_3d_mean,
                    X_test_3D=X_test_3D,
                    M_test=M_test_ref,
                    feats=np.array(feats),
                    static_idxs=np.array(static_idxs),
                    temporal_idxs=np.array(temporal_idxs),
                    ids_test=np.array(ids_test))

df_summary_mean.to_csv(os.path.join(OUT_DIR, f'SHAP_Summary_Averaged_{MODALITY}.csv'), index=False)
pd.concat(summary_per_seed, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, f'SHAP_Summary_PerSeed_{MODALITY}.csv'), index=False)

logger.info(f'Saved averaged SHAP tensor, summary, and per-seed diagnostics to {OUT_DIR}')
