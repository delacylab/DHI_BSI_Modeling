########################################################################################################################
# Run baseline modeling (Elastic Net, SVM, XGBoost) on the 2D datasets (mean-aggregated)
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import os
import pandas as pd
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing import preprocess_data
from _Helper_Scripts.Baseline_Modeling import en_modeling, svm_modeling, xgb_modeling, ann_modeling

########################################################################################################################
# Create the logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Load data
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'
X_PATH: str = os.path.join(IN_DIR, 'Features_v4.parquet')
Y_PATH: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
df_X: pd.DataFrame = pd.read_parquet(X_PATH)
df_y: pd.DataFrame = pd.read_csv(Y_PATH)

########################################################################################################################
# Specify the background columns in df_X
########################################################################################################################
BKGD_COLS: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                        'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                        'bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']

########################################################################################################################
# Specify the outcome label column in df_y
########################################################################################################################
OUTCOME: str = 'BSI_GSI_Case'

########################################################################################################################
# Obtain the full training partition in df_y and df_X
########################################################################################################################
K_FOLD_STRS: list[str] = sorted([k for k in df_y['Partition_Validation'].unique().tolist() if k != 'TEST'])
df_y = df_y[df_y['Partition_Validation'] != 'TEST'].copy()
y_ids: list[int] = df_y['Record_ID'].unique().tolist()
df_y = df_y.copy().set_index('Record_ID').loc[y_ids].reset_index()
df_X = df_X[df_X['Record_ID'].isin(y_ids)].copy().set_index('Record_ID').loc[y_ids].reset_index()
logger.info(f'Full training feature dataset loaded with dimension = {df_X.shape}')

########################################################################################################################
# Book-keeping records
########################################################################################################################
en_results: list[pd.DataFrame] = []
svm_results: list[pd.DataFrame] = []
xgb_results: list[pd.DataFrame] = []
ann_results: list[pd.DataFrame] = []

########################################################################################################################
# Create the internal training and validation pairs
########################################################################################################################
for k_fold_str in K_FOLD_STRS:

    # Logging
    log_head: str = f'[{k_fold_str}] '

    # Subset to training samples relative to the fold
    df_y_train = df_y[df_y['Partition_Validation'] != k_fold_str].copy()
    y_train_ids: list[int] = df_y_train['Record_ID'].unique().tolist()
    df_X_train = df_X.set_index('Record_ID').loc[y_train_ids].reset_index()

    # Subset to validation samples relative to the fold
    df_y_val = df_y[df_y['Partition_Validation'] == k_fold_str].copy()
    y_val_ids: list[int] = df_y_val['Record_ID'].unique().tolist()
    df_X_val = df_X.set_index('Record_ID').loc[y_val_ids].reset_index()

    # Preprocessing
    logger.info(f'{log_head}Data preprocessing...')
    feat_train_2D, feat_val_2D = preprocess_data(df_X_train=df_X_train,
                                                 df_X_val=df_X_val,
                                                 bkgd_cols=BKGD_COLS,
                                                 groupby_col='Record_ID',
                                                 sd_away=3,
                                                 impute='Median',
                                                 dim='2D',
                                                 logger=logger)
    logger.info(f'{log_head}Data preprocessed.')
    assert feat_train_2D['Record_ID'].tolist() == list(y_train_ids)
    assert feat_val_2D['Record_ID'].tolist() == list(y_val_ids)
    logger.print('-'*100)

    # Elastic net modeling
    logger.info(f'{log_head}Elastic net models - Running...')
    df_en_result: pd.DataFrame = en_modeling(X_train=feat_train_2D.drop(columns='Record_ID').to_numpy(),
                                             y_train=df_y_train[OUTCOME].to_numpy(),
                                             X_val=feat_val_2D.drop(columns='Record_ID').to_numpy(),
                                             y_val=df_y_val[OUTCOME].to_numpy(),
                                             logger=logger)
    df_en_result.insert(1, 'k_fold', k_fold_str)
    en_results.append(df_en_result)
    logger.info(f'{log_head}Elastic net models - Completed.')
    logger.print('-'*100)

    # SVM modeling
    logger.info(f'{log_head}SVM models - Running...')
    df_svm_result: pd.DataFrame = svm_modeling(X_train=feat_train_2D.drop(columns='Record_ID').to_numpy(),
                                               y_train=df_y_train[OUTCOME].to_numpy(),
                                               X_val=feat_val_2D.drop(columns='Record_ID').to_numpy(),
                                               y_val=df_y_val[OUTCOME].to_numpy(),
                                               logger=logger)
    df_svm_result.insert(1, 'k_fold', k_fold_str)
    svm_results.append(df_svm_result)
    logger.info(f'{log_head}SVM models - Completed.')
    logger.print('-'*100)

    # XGBoost modeling
    logger.info(f'{log_head}XGBoost models - Running...')
    df_xgb_result: pd.DataFrame = xgb_modeling(X_train=feat_train_2D.drop(columns='Record_ID').to_numpy(),
                                               y_train=df_y_train[OUTCOME].to_numpy(),
                                               X_val=feat_val_2D.drop(columns='Record_ID').to_numpy(),
                                               y_val=df_y_val[OUTCOME].to_numpy(),
                                               logger=logger)
    df_xgb_result.insert(1, 'k_fold', k_fold_str)
    xgb_results.append(df_xgb_result)
    logger.info(f'{log_head}XGBoost models - Completed.')
    logger.print('-'*100)

    # ANN modeling
    logger.info(f'{log_head}ANN models - Running...')
    df_ann_result: pd.DataFrame = ann_modeling(X_train=feat_train_2D.drop(columns='Record_ID').to_numpy(),
                                               y_train=df_y_train[OUTCOME].to_numpy(),
                                               X_val=feat_val_2D.drop(columns='Record_ID').to_numpy(),
                                               y_val=df_y_val[OUTCOME].to_numpy(),
                                               logger=logger)
    df_ann_result.insert(1, 'k_fold', k_fold_str)
    ann_results.append(df_ann_result)
    logger.info(f'{log_head}ANN models - Completed.')
    logger.print('-'*100)

########################################################################################################################
# Organize the performance statistics for each algorithm
########################################################################################################################
df_en_all: pd.DataFrame = pd.concat(en_results, axis=0)
df_svm_all: pd.DataFrame = pd.concat(svm_results, axis=0)
df_xgb_all: pd.DataFrame = pd.concat(xgb_results, axis=0)
df_ann_all: pd.DataFrame = pd.concat(ann_results, axis=0)

########################################################################################################################
# Export results
########################################################################################################################
OUT_DIR: str = '_Results_Validation/'
os.makedirs(OUT_DIR, exist_ok=True)

EN_OUT_PATH: str = os.path.join(OUT_DIR, 'EN_Validation.csv')
df_en_all.to_csv(EN_OUT_PATH, index=False)

SVM_OUT_PATH: str = os.path.join(OUT_DIR, 'SVM_Validation.csv')
df_svm_all.to_csv(SVM_OUT_PATH, index=False)

XGB_OUT_PATH: str = os.path.join(OUT_DIR, 'XGB_Validation.csv')
df_xgb_all.to_csv(XGB_OUT_PATH, index=False)

ANN_OUT_PATH: str = os.path.join(OUT_DIR, 'ANN_Validation.csv')
df_ann_all.to_csv(ANN_OUT_PATH, index=False)
