########################################################################################################################
# Retrain each baseline model with the best hyperparameter configuration
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import os
import numpy as np
import pandas as pd
import warnings
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing import preprocess_data
from _Helper_Scripts.Transformer_Modeling import ArrayDataset
from _Helper_Scripts.Transformer_Retraining_Seeded import tf_retrain
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
# Create the logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Modality, outcome, background columns
########################################################################################################################
MODALITY: str = 'unimodal'
OUTCOME: str = 'BSI_GSI_Case'
BKGD_COLS: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                        'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                        'bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']

########################################################################################################################
# Book-keeping for best configuration
########################################################################################################################
params: list[str] = ['d_model', 'n_heads', 'n_layers', 'n_units', 'lr',
                     'cost_mul', 'handle_nan', 'factor', 'max_epochs', 'batch_size',
                     'dropout', 'seed']
best_config_dict: dict = {k: None for k in params}

########################################################################################################################
# Identify the best hyperparameter configurations in each model
########################################################################################################################
HPO_DIR: str = '_Results_Validation/'
model_str: str = MODALITY
log_head: str = f'[Model={model_str}] '
hpo_path: str = os.path.join(HPO_DIR, f'TF_{model_str}/TF_{model_str}_Validation.csv')
df_hpo: pd.DataFrame = pd.read_csv(hpo_path)


df_hpo = df_hpo.sort_values(by=['k_fold'] + list(best_config_dict)).reset_index(drop=True)
df_hpo.insert(2, 'Element_index', df_hpo.groupby('k_fold').cumcount() + 1)
df_hpo_gb: pd.DataFrame = df_hpo.groupby('Element_index')[['AUROC', 'ECE']].mean().reset_index()
df_hpo_gb = df_hpo_gb.sort_values(by=['AUROC', 'ECE'], ascending=[False, True]).reset_index(drop=True)
best_config_idx: int = df_hpo_gb.loc[0, 'Element_index']

# Record the actual best configuration
df_hpo_best: pd.DataFrame = df_hpo.loc[df_hpo['Element_index'] == best_config_idx]
for c in best_config_dict.keys():
    assert df_hpo_best[c].nunique() == 1
    v = df_hpo_best[c].unique().tolist()[0]
    best_config_dict[c] = v
    logger.info(f'{log_head}{c} = {v}')
logger.print('-' * 100)

########################################################################################################################
# Load data
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'
X_PATH: str = os.path.join(IN_DIR, 'Features_v4.parquet')
Y_PATH: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
df_X: pd.DataFrame = pd.read_parquet(X_PATH)
df_y: pd.DataFrame = pd.read_csv(Y_PATH)

########################################################################################################################
# Specify the outcome label column in df_y
########################################################################################################################
OUTCOME: str = 'BSI_GSI_Case'

########################################################################################################################
# Obtain the partitions
########################################################################################################################
df_y_train: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TRAIN'].copy()
df_y_val: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'VAL'].copy()
df_y_test: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TEST'].copy()
y_train_ids: list[int] = df_y_train['Record_ID'].tolist()
y_val_ids: list[int] = df_y_val['Record_ID'].tolist()
y_test_ids: list[int] = df_y_test['Record_ID'].tolist()
df_X_train: pd.DataFrame = df_X.set_index('Record_ID').loc[y_train_ids].reset_index()
df_X_val: pd.DataFrame = df_X.set_index('Record_ID').loc[y_val_ids].reset_index()
df_X_test: pd.DataFrame = df_X.set_index('Record_ID').loc[y_test_ids].reset_index()

########################################################################################################################
# Preprocessing
########################################################################################################################
logger.info(f'Data preprocessing...')
X_train_3D, X_val_3D, X_test_3D, ids_train, ids_val, ids_test, feats = preprocess_data(df_X_train=df_X_train,
                                                                                       df_X_val=df_X_val,
                                                                                       df_X_test=df_X_test,
                                                                                       bkgd_cols=BKGD_COLS,
                                                                                       groupby_col='Record_ID',
                                                                                       sd_away=3,
                                                                                       dim='3D',
                                                                                       logger=logger)
assert list(ids_train) == y_train_ids, 'ids_train order mismatch'
assert list(ids_val) == y_val_ids, 'ids_val order mismatch'
assert list(ids_test) == y_test_ids, 'ids_test order mismatch'
assert len(feats) == X_train_3D.shape[2], 'feats count mismatch'

# Filter to unimodal features
if MODALITY == 'unimodal':
    unimodal_feats: list[str] = [f for f in feats if not f.endswith('@survey')]
    feat_idxs: list[int] = [feats.index(f) for f in unimodal_feats]
    X_train_3D = X_train_3D[:, :, feat_idxs]
    X_val_3D = X_val_3D[:, :, feat_idxs]
    X_test_3D = X_test_3D[:, :, feat_idxs]
    feats = unimodal_feats
    assert len(feats) == X_train_3D.shape[2]
logger.info(f'{log_head}Preprocessed: X_train {X_train_3D.shape}, X_val {X_val_3D.shape}, X_test {X_test_3D.shape}')

# Build ArrayDatasets
y_train: np.ndarray = df_y_train[OUTCOME].to_numpy(dtype=np.float32)
y_val: np.ndarray = df_y_val[OUTCOME].to_numpy(dtype=np.float32)
y_test: np.ndarray = df_y_test[OUTCOME].to_numpy(dtype=np.float32)
train_ds: ArrayDataset = ArrayDataset(X_train_3D, y_train, np.array(ids_train))
val_ds: ArrayDataset = ArrayDataset(X_val_3D, y_val, np.array(ids_val))
test_ds: ArrayDataset = ArrayDataset(X_test_3D, y_test, np.array(ids_test))

########################################################################################################################
# Transformer retraining
########################################################################################################################
# OUT_DIR: str = '_Results_Retraining/'
OUT_DIR: str = '_Results_Retraining_v2/'
os.makedirs(OUT_DIR, exist_ok=True)
df_probs, df_test, df_history = tf_retrain(best_config=best_config_dict,
                                           train_ds=train_ds,
                                           val_ds=val_ds,
                                           test_ds=test_ds,
                                           out_dir=f'{OUT_DIR}/TF_{MODALITY}_SEEDED/',
                                           file_suffix=f'_{MODALITY}',
                                           seeds=[0, 1, 2, 42],
                                           scheduler_patience=20,
                                           ema_decay=0.999,
                                           logger=logger)
df_test['Algorithm'] = f'TF_{MODALITY}_' + df_test['Model']

# Save outputs
df_probs.to_csv(os.path.join(OUT_DIR, f'TF_{MODALITY}_Probability_SEEDED.csv'), index=False)
df_test.to_csv(os.path.join(OUT_DIR, f'TF_{MODALITY}_Performance_SEEDED.csv'), index=False)
df_history.to_csv(os.path.join(OUT_DIR, f'TF_{MODALITY}_History_SEEDED.csv'), index=False)
