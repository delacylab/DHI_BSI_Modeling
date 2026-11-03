########################################################################################################################
# Record the feature importance score for each feature and each 2D model
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import os
import torch
import xgboost as xgb
import warnings
from captum.attr import GradientShap
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
# Load the data dictionary file
########################################################################################################################
dict_path: str = '_Final_Results/Supplementary_Table_1_-_Data_Dictionary_v1.csv'
df_dict: pd.DataFrame = pd.read_csv(dict_path)
logger.info(f'Number of variables in the dictionary: {len(df_dict)}')
logger.print('-'*100)

# Identify the feature names
feats: list[str] = df_dict['Feature_Name'].tolist()

########################################################################################################################
# Specify the seeds for retraining
########################################################################################################################
SEEDS: list[int] = [0, 1, 2, 42]

########################################################################################################################
# Retraining 2D models (see 3_MODEL_02_2D_Hyperparameter_Retrain_Seeded.py
########################################################################################################################

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
feat_names_2D: list[str] = feat_train_2D.drop(columns='Record_ID').columns.tolist()
logger.info(f'{len(feat_names_2D)} features found for 2D data.')
logger.print('-' * 100)

########################################################################################################################
# Retraining the 4 algorithms with different seeds
########################################################################################################################
model_dict: dict = {model: [] for model in best_config_dict.keys()}

for model, config_dict in best_config_dict.items():
    retrain_func = {'EN': en_retrain,
                    'SVM': svm_retrain,
                    'XGB': xgb_retrain,
                    'ANN': ann_retrain}[model]
    for seed in SEEDS:
        logger.info(f'Retraining {model} at seed={seed}...')
        config_dict_cur = deepcopy(config_dict)
        config_dict_cur['seed'] = seed
        _, _, M_cur = retrain_func(best_config=config_dict_cur,
                                   X_train=X_train_np,
                                   y_train=y_train_np,
                                   X_val=X_val_np,
                                   y_val=y_val_np,
                                   X_test=X_test_np,
                                   y_test=y_test_np,
                                   return_model=True)
        model_dict[model].append(M_cur)
        logger.info(f'Retraining {model} at seed={seed} completed.')
        logger.print('-'*100)

########################################################################################################################
# Extract feature importance for ANN (mean |SHAP| averaged across 4 seeds)
########################################################################################################################
N_SAMPLES: int = 200
ann_shap_arrays: list[np.ndarray] = []
for i, fitted_model in enumerate(model_dict['ANN']):
    fitted_model.eval()
    device = next(fitted_model.parameters()).device
    X_baseline_t = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)

    explainer = GradientShap(fitted_model)
    attrs = explainer.attribute(inputs=X_test_t,
                                baselines=X_baseline_t,
                                n_samples=N_SAMPLES)
    shap_values: np.ndarray = attrs.detach().cpu().numpy()  # (N_test, 1013)
    mean_abs_shap: np.ndarray = np.abs(shap_values).mean(axis=0)
    ann_shap_arrays.append(mean_abs_shap)
    logger.info(f'ANN seed {SEEDS[i]}: GradientSHAP computed')

# Average mean|SHAP| across 4 seeds
avg_ann_shap: np.ndarray = np.mean(ann_shap_arrays, axis=0)
assert len(avg_ann_shap) == len(feat_names_2D)

# Map to dictionary (exclude !!NA columns)
shap_all: pd.Series = pd.Series(avg_ann_shap, index=feat_names_2D)
shap_no_na: pd.Series = shap_all[~shap_all.index.str.endswith('!!NA')]
df_dict['ANN_MeanAbsSHAP'] = df_dict['Feature_Name'].map(shap_no_na)
n_mapped: int = df_dict['ANN_MeanAbsSHAP'].notna().sum()
logger.info(f'ANN: {n_mapped}/{len(df_dict)} dictionary features mapped (mean |SHAP|)')
logger.print('-' * 100)

########################################################################################################################
# Extract feature importance for EN and SVM (signed linear coefficients, averaged across 4 seeds)
########################################################################################################################
for model in ['EN', 'SVM']:
    coef_arrays: list[np.ndarray] = []
    for fitted_model in model_dict[model]:
        coef_arrays.append(fitted_model.coef_.ravel())
    avg_coef: np.ndarray = np.mean(coef_arrays, axis=0)     # Average coefficients across 4 seeds
    assert len(avg_coef) == len(feat_names_2D), \
        f'{model}: coef length {len(avg_coef)} != feat_names_2D length {len(feat_names_2D)}'
    coef_all: pd.Series = pd.Series(avg_coef, index=feat_names_2D)      # Build series over all 2D columns
    coef_no_na: pd.Series = coef_all[~coef_all.index.str.endswith('!!NA')]  # Map only non-!!NA features
    df_dict[f'{model}_Coefficient'] = df_dict['Feature_Name'].map(coef_no_na)
    n_mapped: int = df_dict[f'{model}_Coefficient'].notna().sum()
    logger.info(f'{model}: {n_mapped}/{len(df_dict)} dictionary features mapped (signed coefficients)')
logger.print('-' * 100)

########################################################################################################################
# Extract feature importance for XGB (mean |SHAP| averaged across 4 seeds)
########################################################################################################################
xgb_shap_arrays: list[np.ndarray] = []
for i, fitted_model in enumerate(model_dict['XGB']):
    contribs = fitted_model.get_booster().predict(
        xgb.DMatrix(X_test_np, feature_names=feat_names_2D),
        pred_contribs=True)  # (N_test, 1014) — last column is the bias term
    shap_values = contribs[:, :-1]  # (N_test, 1013)
    mean_abs_shap: np.ndarray = np.abs(shap_values).mean(axis=0)
    xgb_shap_arrays.append(mean_abs_shap)
    logger.info(f'XGB seed {SEEDS[i]}: SHAP computed')

# Average mean|SHAP| across 4 seeds
avg_xgb_shap: np.ndarray = np.mean(xgb_shap_arrays, axis=0)
assert len(avg_xgb_shap) == len(feat_names_2D)

# Map to dictionary (exclude !!NA columns)
shap_all: pd.Series = pd.Series(avg_xgb_shap, index=feat_names_2D)
shap_no_na: pd.Series = shap_all[~shap_all.index.str.endswith('!!NA')]
df_dict['XGB_MeanAbsSHAP'] = df_dict['Feature_Name'].map(shap_no_na)
n_mapped: int = df_dict['XGB_MeanAbsSHAP'].notna().sum()
logger.info(f'XGB: {n_mapped}/{len(df_dict)} dictionary features mapped (mean |SHAP|)')
logger.print('-' * 100)

########################################################################################################################
# Organize updated df_dict and export
########################################################################################################################
for col in ['ANN_MeanAbsSHAP', 'EN_Coefficient', 'SVM_Coefficient', 'XGB_MeanAbsSHAP']:
    df_dict[col] = df_dict[col].fillna('N/A')
os.makedirs('_Final_Results/', exist_ok=True)
df_dict.to_csv('_Final_Results/Supplementary_Table_1_-_Data_Dictionary_v2.csv', index=False)
