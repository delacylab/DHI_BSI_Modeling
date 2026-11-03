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
from copy import deepcopy
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing import preprocess_data
from _Helper_Scripts.Baseline_Retraining import en_retrain, svm_retrain, xgb_retrain, ann_retrain
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
# Create the logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Specify the seeds for retraining
########################################################################################################################
SEEDS: list[int] = [0, 1, 2, 42]

########################################################################################################################
# Book-keeping for best configuration
########################################################################################################################
best_config_dict: dict = {'ANN': {k: None for k in ['hidden_size',
                                                    'n_layers',
                                                    'lr',
                                                    'cost_mul',
                                                    'factor',
                                                    'batch_size',
                                                    'dropout',
                                                    'n_epochs',
                                                    'seed']},
                          'EN': {k: None for k in ['C',
                                                   'l1_ratio',
                                                   'cost_mul',
                                                   'max_iter',
                                                   'seed']},
                          'SVM': {k: None for k in ['C',
                                                    'penalty',
                                                    'dual',
                                                    'cost_mul',
                                                    'max_iter',
                                                    'seed']},
                          'XGB': {k: None for k in ['max_depth',
                                                    'n_estimators',
                                                    'learning_rate',
                                                    'reg_lambda',
                                                    'cost_mul',
                                                    'seed']}}

########################################################################################################################
# Identify the best hyperparameter configurations in each model
########################################################################################################################
HPO_DIR: str = '_Results_Validation/'
for model_str in best_config_dict.keys():
    hpo_path: str = os.path.join(HPO_DIR, f'{model_str}_Validation.csv')
    df_hpo: pd.DataFrame = pd.read_csv(hpo_path)

    # Identify the index of the best configuration
    if 'Element_index' not in df_hpo.columns:
        df_hpo.insert(2, 'Element_index', df_hpo.groupby('k_fold').cumcount() + 1)
    df_hpo_gb: pd.DataFrame = df_hpo.groupby('Element_index')[['AUROC', 'ECE']].mean().reset_index()
    df_hpo_gb = df_hpo_gb.sort_values(by=['AUROC', 'ECE'], ascending=[False, True]).reset_index(drop=True)
    best_config_idx: int = df_hpo_gb.loc[0, 'Element_index']

    # Record the actual best configuration
    df_hpo_best: pd.DataFrame = df_hpo.loc[df_hpo['Element_index'] == best_config_idx]
    for c in best_config_dict[model_str].keys():
        assert df_hpo_best[c].nunique() == 1
        v = df_hpo_best[c].unique().tolist()[0]
        best_config_dict[model_str][c] = v
        logger.info(f'[Model={model_str}] {c} = {v}')
    logger.print('-'*100)

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
feat_train_2D, feat_val_2D, feat_test_2D = preprocess_data(df_X_train=df_X_train,
                                                           df_X_val=df_X_val,
                                                           df_X_test=df_X_test,
                                                           bkgd_cols=BKGD_COLS,
                                                           groupby_col='Record_ID',
                                                           sd_away=3,
                                                           impute='Median',
                                                           dim='2D',
                                                           logger=logger)
logger.info(f'Data preprocessed.')
assert feat_train_2D['Record_ID'].tolist() == list(y_train_ids)
assert feat_val_2D['Record_ID'].tolist() == list(y_val_ids)
assert feat_test_2D['Record_ID'].tolist() == list(y_test_ids)

X_train_np: np.ndarray = feat_train_2D.drop(columns='Record_ID').to_numpy()
y_train_np: np.ndarray = df_y_train[OUTCOME].to_numpy()
X_val_np: np.ndarray = feat_val_2D.drop(columns='Record_ID').to_numpy()
y_val_np: np.ndarray = df_y_val[OUTCOME].to_numpy()
X_test_np: np.ndarray = feat_test_2D.drop(columns='Record_ID').to_numpy()
y_test_np: np.ndarray = df_y_test[OUTCOME].to_numpy()
logger.print('-' * 100)

########################################################################################################################
# Retraining the 4 algorithms with different seeds
########################################################################################################################
prob_result_dict: dict = {model: [] for model in best_config_dict.keys()}
test_result_dict: dict = {model: [] for model in best_config_dict.keys()}

for model, config_dict in best_config_dict.items():
    retrain_func = {'EN': en_retrain,
                    'SVM': svm_retrain,
                    'XGB': xgb_retrain,
                    'ANN': ann_retrain}[model]
    for seed in SEEDS:
        logger.info(f'Retraining {model} at seed={seed}...')
        config_dict_cur = deepcopy(config_dict)
        config_dict_cur['seed'] = seed
        probs_cur, test_cur = retrain_func(best_config=config_dict_cur,
                                           X_train=X_train_np,
                                           y_train=y_train_np,
                                           X_val=X_val_np,
                                           y_val=y_val_np,
                                           X_test=X_test_np,
                                           y_test=y_test_np)
        probs_cur.insert(0, 'seed', seed)
        test_cur['seed'] = seed
        prob_result_dict[model].append(probs_cur)
        test_result_dict[model].append(test_cur)
        logger.info(f'Retraining {model} at seed={seed} completed.')
        logger.print('-'*100)

########################################################################################################################
# Export the probability estimates
########################################################################################################################
OUT_DIR: str = '_Results_Retraining_v2'
os.makedirs(OUT_DIR, exist_ok=True)
for model, results in prob_result_dict.items():
    prob_path: str = os.path.join(OUT_DIR, f'{model}_Probability_SEEDED.csv')
    pd.concat(results, axis=0).to_csv(prob_path, index=False)
logger.info(f'Estimated probabilities for the 4 models exported to {OUT_DIR}.')

########################################################################################################################
# Export the performance statistics
########################################################################################################################
for model, results in test_result_dict.items():
    test_path: str = os.path.join(OUT_DIR, f'{model}_Performance_SEEDED.csv')
    pd.concat(results, axis=0).to_csv(test_path, index=False)
logger.info(f'Held-out performance statistics for the 4 models exported to {OUT_DIR}.')
