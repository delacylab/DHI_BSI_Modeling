########################################################################################################################
# Plot the ROC curves across 8 models using seed-pooled (ensembled) probability estimates
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from _Helper_Scripts.Logger import create_log
from sklearn.metrics import roc_curve, auc
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Century Gothic"]

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Runtime parameters
########################################################################################################################
SEEDS: list[int] = [0, 1, 2, 42]
PROB_COL: str = 'y_prob_platt'

########################################################################################################################
# Model paths and ordering
########################################################################################################################
IN_DIR_2D: str = '_Results_Retraining_v2/'
IN_DIR_FUSION: str = '_Results_Retraining_Fusion_v2/'
ML_MODELS: list[str] = ['ANN', 'EN', 'SVM', 'XGB']
TF_MODELS_2D: list[str] = ['TF_unimodal', 'TF_multimodal']
TF_MODELS_FUSION: list[str] = ['TF_Fusion_unimodal', 'TF_Fusion_multimodal']
MODEL_ORDER: list[str] = ['ANN', 'EN', 'SVM', 'XGB',
                          'TF_unimodal', 'TF_Fusion_unimodal',
                          'TF_multimodal', 'TF_Fusion_multimodal']

########################################################################################################################
# Load test partition Record_IDs from Outcome_v2
########################################################################################################################
Y_PATH: str = '_Helper_Files/15min_Data/Outcome_v2.csv'
OUTCOME: str = 'BSI_GSI_Case'
df_y: pd.DataFrame = pd.read_csv(Y_PATH)
df_y_test: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TEST'].copy().reset_index(drop=True)
test_record_ids: list[int] = df_y_test['Record_ID'].tolist()

# Get BeiweId mapping from features parquet
df_feat: pd.DataFrame = pd.read_parquet('_Helper_Files/15min_Data/Features_v1.parquet',
                                        columns=['Record_ID', 'BeiweId'])
record_to_beiwe: dict[int, str] = (df_feat.drop_duplicates('Record_ID')
                                   .set_index('Record_ID')['BeiweId'].to_dict())
test_beiwe_ids: list[str] = [record_to_beiwe[rid] for rid in test_record_ids]
id_df: pd.DataFrame = pd.DataFrame({'Record_ID': test_record_ids, 'BeiweId': test_beiwe_ids})

########################################################################################################################
# Load and pool predicted probabilities across 4 seeds for each model
########################################################################################################################
pooled_probs: dict[str, pd.DataFrame] = {}

for model in ML_MODELS:
    df_prob: pd.DataFrame = pd.read_csv(os.path.join(IN_DIR_2D, f'{model}_Probability_SEEDED.csv'))
    df_prob['pos'] = df_prob.groupby('seed').cumcount()
    pooled: pd.DataFrame = df_prob.groupby('pos')[['y_test', PROB_COL]].mean().reset_index(drop=True)
    pooled['Record_ID'] = test_record_ids
    pooled_probs[model] = pooled
    logger.info(f'{model}: pooled {df_prob["seed"].nunique()} seeds → {len(pooled)} records')

for model in TF_MODELS_2D:
    df_prob: pd.DataFrame = pd.read_csv(os.path.join(IN_DIR_2D, f'{model}_Probability_SEEDED.csv'))
    if 'Model' in df_prob.columns:
        df_prob = df_prob[df_prob['Model'] == 'Raw']
    pooled: pd.DataFrame = df_prob.groupby('rec_id')[['y_test', PROB_COL]].mean().reset_index()
    pooled = pooled.rename(columns={'rec_id': 'Record_ID'})
    pooled_probs[model] = pooled
    logger.info(f'{model}: pooled → {len(pooled)} records')

for model in TF_MODELS_FUSION:
    df_prob: pd.DataFrame = pd.read_csv(os.path.join(IN_DIR_FUSION, model, f'{model}_Probability_SEEDED.csv'))
    if 'Model' in df_prob.columns:
        df_prob = df_prob[df_prob['Model'] == 'Raw']
    pooled: pd.DataFrame = df_prob.groupby('rec_id')[['y_test', PROB_COL]].mean().reset_index()
    pooled = pooled.rename(columns={'rec_id': 'Record_ID'})
    pooled_probs[model] = pooled
    logger.info(f'{model}: pooled → {len(pooled)} records')

# Normalize row order across all models (sort by Record_ID)
for model in MODEL_ORDER:
    pooled_probs[model] = pooled_probs[model].sort_values('Record_ID').reset_index(drop=True)

# Verify y_test consistency across all models
ref_y: np.ndarray = pooled_probs[MODEL_ORDER[0]]['y_test'].values
for model in MODEL_ORDER[1:]:
    assert np.array_equal(ref_y, pooled_probs[model]['y_test'].values), f'{model} y_test mismatch after alignment'
logger.info('All models aligned by Record_ID; y_test consistency verified')
logger.print('-'*100)

########################################################################################################################
# Compute ROC curves from pooled probabilities
########################################################################################################################
fpr_dict: dict[str, np.ndarray] = {}
tpr_dict: dict[str, np.ndarray] = {}
auc_dict: dict[str, float] = {}

for m in MODEL_ORDER:
    y_true: np.ndarray = ref_y
    y_prob: np.ndarray = pooled_probs[m][PROB_COL].values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr_dict[m] = fpr
    tpr_dict[m] = tpr
    auc_dict[m] = auc(fpr, tpr)
    logger.info(f'{m}: AUROC = {auc_dict[m]:.3f}')

logger.print('-'*100)

########################################################################################################################
# Plot — one panel per model (row 0: baselines, row 1: transformer variants)
########################################################################################################################
N_ROWS, N_COLS = 2, 4
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, 8), sharex=True, sharey=True)

for idx, m in enumerate(MODEL_ORDER):
    row_idx: int = idx // N_COLS
    col_idx: int = idx % N_COLS
    ax = axes[row_idx, col_idx]

    # ROC curve
    auroc: float = auc_dict[m]
    ax.plot(fpr_dict[m], tpr_dict[m], color='tab:orange', linewidth=3, label=f'ROC curve\n(area = {auroc:.2f})')

    # Chance line
    ax.plot([0, 1], [0, 1], linestyle='--', color='tab:blue', linewidth=1.8, label='Chance level\n(area = 0.50)')

    if m in ['ANN', 'EN', 'SVM', 'XGB']:
        model_str = m
    else:
        model_str = {'TF_unimodal': 'Transformer\n(unimodal, standard)',
                     'TF_Fusion_unimodal': 'Transformer\n(unimodal, fusion)',
                     'TF_multimodal': 'Transformer\n(multimodal, standard)',
                     'TF_Fusion_multimodal': 'Transformer\n(multimodal, fusion)'}[m]

    ax.set_title(model_str, fontweight='bold', fontsize=20)
    ax.legend(loc='lower right', fontsize=14)

    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.5)

    if row_idx == N_ROWS - 1:
        ax.set_xlabel('False positive rate', fontsize=15)
    if col_idx == 0:
        ax.set_ylabel('True positive rate', fontsize=15)

fig.tight_layout(rect=[0, 0, 1, 0.96])

OUT_DIR: str = '_Final_Results/'
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, 'ROC_Curves_Pooled.png'), dpi=300)
# plt.close(fig)