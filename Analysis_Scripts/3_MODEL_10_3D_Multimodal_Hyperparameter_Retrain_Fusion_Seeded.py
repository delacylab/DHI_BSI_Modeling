########################################################################################################################
# Retrain the top hyperparameter configuration for the two-stream Fusion transformer (multimodal)
# with the retraining seeds [0, 1, 2, 42] on TRAIN + VAL + TEST from Partition_Retraining.
# Reads the HPO validation summary, picks the best config by mean AUROC (ECE tiebreaker) across the 4 CV folds,
# preprocesses via Preprocessing_Fusion (dim='3D', modality='multimodal'), and calls tf_retrain.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
import warnings
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing_Fusion import preprocess_data
from _Helper_Scripts.Transformer_Fusion_Modeling import ArrayDataset
from _Helper_Scripts.Transformer_Fusion_Retraining_Seeded import tf_retrain
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################################################################################################
# Create the logging file
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
SCHEDULER_PATIENCE: int = 20
EMA_DECAY: float = 0.999
SEEDS: list[int] = [0, 1, 2, 42]

########################################################################################################################
# Paths
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'
X_PATH: str = os.path.join(IN_DIR, 'Features_v4.parquet')
Y_PATH: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
HPO_DIR: str = f'_Results_Validation_Fusion/TF_Fusion_{MODALITY}/'
HPO_CSV: str = os.path.join(HPO_DIR, f'TF_Fusion_{MODALITY}_Validation.csv')
OUT_DIR: str = f'_Results_Retraining_Fusion_v2/TF_Fusion_{MODALITY}/'
os.makedirs(OUT_DIR, exist_ok=True)

########################################################################################################################
# Load HPO results and identify the best configuration by mean AUROC (ECE tiebreaker) across the 4 CV folds
########################################################################################################################
logger.info(f'Loading HPO results from {HPO_CSV}')
df_hpo: pd.DataFrame = pd.read_csv(HPO_CSV)
logger.info(f'HPO results shape: {df_hpo.shape}')

params: list[str] = ['d_model', 'n_heads', 'n_layers', 'n_units', 'lr',
                     'cost_mul', 'handle_nan', 'factor', 'd_static', 'n_static_layers',
                     'max_epochs', 'batch_size', 'dropout', 'seed']

if 'Element_index' not in df_hpo.columns:
    df_hpo = df_hpo.sort_values(by=['k_fold'] + params).reset_index(drop=True)
    df_hpo.insert(2, 'Element_index', df_hpo.groupby('k_fold').cumcount() + 1)

df_hpo_gb: pd.DataFrame = df_hpo.groupby('Element_index')[['AUROC', 'ECE']].mean().reset_index()
df_hpo_gb = df_hpo_gb.sort_values(by=['AUROC', 'ECE'], ascending=[False, True]).reset_index(drop=True)

best_element_idx: int = int(df_hpo_gb.iloc[0]['Element_index'])
df_hpo_best: pd.DataFrame = df_hpo.loc[df_hpo['Element_index'] == best_element_idx]

best_config_dict: dict = {}
for c in params:
    assert df_hpo_best[c].nunique() == 1, f'Non-unique {c} within Element_index {best_element_idx}'
    best_config_dict[c] = df_hpo_best[c].unique().tolist()[0]
    logger.info(f'[Model={MODALITY}] {c} = {best_config_dict[c]}')

logger.info(f'CV mean AUROC = {df_hpo_gb.iloc[0]["AUROC"]:.4f}, '
            f'CV mean ECE = {df_hpo_gb.iloc[0]["ECE"]:.4f}')
logger.print('-' * 100)

########################################################################################################################
# Load features and outcome
########################################################################################################################
df_X: pd.DataFrame = pd.read_parquet(X_PATH)
df_y: pd.DataFrame = pd.read_csv(Y_PATH)
logger.info(f'Feature parquet: {df_X.shape}; Outcome CSV: {df_y.shape}')

########################################################################################################################
# Build TRAIN / VAL / TEST partitions from Partition_Retraining, aligned by Record_ID
########################################################################################################################
df_y_train: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TRAIN'].copy().reset_index(drop=True)
y_train_ids: list[int] = df_y_train['Record_ID'].unique().tolist()
df_X_train: pd.DataFrame = df_X.set_index('Record_ID').loc[y_train_ids].reset_index()

df_y_val: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'VAL'].copy().reset_index(drop=True)
y_val_ids: list[int] = df_y_val['Record_ID'].unique().tolist()
df_X_val: pd.DataFrame = df_X.set_index('Record_ID').loc[y_val_ids].reset_index()

df_y_test: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TEST'].copy().reset_index(drop=True)
y_test_ids: list[int] = df_y_test['Record_ID'].unique().tolist()
df_X_test: pd.DataFrame = df_X.set_index('Record_ID').loc[y_test_ids].reset_index()

logger.info(f'TRAIN: {len(y_train_ids)} records; VAL: {len(y_val_ids)} records; TEST: {len(y_test_ids)} records')
logger.print('-' * 100)

########################################################################################################################
# Preprocess (modality filter and static/temporal detection happen inside)
########################################################################################################################
logger.info('Preprocessing...')
X_train_3D, X_val_3D, X_test_3D, ids_train, ids_val, ids_test, feats, static_idxs, temporal_idxs = preprocess_data(
    df_X_train=df_X_train,
    df_X_val=df_X_val,
    df_X_test=df_X_test,
    bkgd_cols=BKGD_COLS,
    groupby_col='Record_ID',
    sd_away=3,
    dim='3D',
    modality=MODALITY,
    logger=logger,
)
assert list(ids_train) == y_train_ids, 'ids_train order mismatch'
assert list(ids_val) == y_val_ids, 'ids_val order mismatch'
assert list(ids_test) == y_test_ids, 'ids_test order mismatch'
assert len(feats) == X_train_3D.shape[2], 'feats count mismatch'
assert len(static_idxs) + len(temporal_idxs) == X_train_3D.shape[2], 'index count mismatch'
logger.info(f'X_train {X_train_3D.shape}, X_val {X_val_3D.shape}, X_test {X_test_3D.shape}')
logger.info(f'static features: {len(static_idxs)}, temporal features: {len(temporal_idxs)}')
logger.print('-' * 100)

########################################################################################################################
# Build ArrayDatasets
########################################################################################################################
y_train: np.ndarray = df_y_train[OUTCOME].to_numpy(dtype=np.float32)
y_val: np.ndarray = df_y_val[OUTCOME].to_numpy(dtype=np.float32)
y_test: np.ndarray = df_y_test[OUTCOME].to_numpy(dtype=np.float32)
train_ds: ArrayDataset = ArrayDataset(X_train_3D, y_train, np.array(ids_train))
val_ds: ArrayDataset = ArrayDataset(X_val_3D, y_val, np.array(ids_val))
test_ds: ArrayDataset = ArrayDataset(X_test_3D, y_test, np.array(ids_test))

########################################################################################################################
# Retrain across seeds
########################################################################################################################
logger.info(f'Retraining with seeds {SEEDS}...')
df_probs, df_test, df_history = tf_retrain(best_config=best_config_dict,
                                           train_ds=train_ds,
                                           val_ds=val_ds,
                                           test_ds=test_ds,
                                           temporal_idxs=temporal_idxs,
                                           static_idxs=static_idxs,
                                           out_dir=OUT_DIR,
                                           file_suffix=f'_{MODALITY}',
                                           seeds=SEEDS,
                                           scheduler_patience=SCHEDULER_PATIENCE,
                                           ema_decay=EMA_DECAY,
                                           logger=logger)

df_test['Algorithm'] = f'TF_Fusion_{MODALITY}_' + df_test['Model']

########################################################################################################################
# Save outputs
########################################################################################################################
probs_path: str = os.path.join(OUT_DIR, f'TF_Fusion_{MODALITY}_Probability_SEEDED.csv')
perf_path: str = os.path.join(OUT_DIR, f'TF_Fusion_{MODALITY}_Performance_SEEDED.csv')
hist_path: str = os.path.join(OUT_DIR, f'TF_Fusion_{MODALITY}_History_SEEDED.csv')
df_probs.to_csv(probs_path, index=False)
df_test.to_csv(perf_path, index=False)
df_history.to_csv(hist_path, index=False)
logger.info(f'Saved probabilities to {probs_path}')
logger.info(f'Saved performance to {perf_path}')
logger.info(f'Saved history to {hist_path}')
logger.print('-' * 100)
