########################################################################################################################
# Plot the DCA curves across 8 models using seed-pooled (ensembled) probability estimates
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
# Define the helper functions to compute standardized net benefit (sNB)
########################################################################################################################


def nb(y_true_, y_prob_, pt, standard=False):
    y_pred_ = (y_prob_ >= pt).astype(int)
    TP = np.sum((y_pred_ == 1) & (y_true_ == 1))
    FP = np.sum((y_pred_ == 1) & (y_true_ == 0))
    n = len(y_true_)
    nb = (TP / n) - (FP / n) * (pt / (1 - pt))
    snb = nb / np.mean(y_true_)
    return snb if standard else nb


def nb_from_counts(tp, fp, n, pt, prevalence, standard=True):
    nb = (tp / n) - (fp / n) * (pt / (1 - pt))
    return nb / prevalence if standard else nb


def compute_treat_all_nb(y_true, pts, standard=True):
    n = len(y_true)
    prev = np.mean(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    return [nb_from_counts(tp, fp, n, pt, prev, standard=standard) for pt in pts]


########################################################################################################################
# Define the policy probability thresholds for the computation of sNB
########################################################################################################################
n_probs: int = 1000
POLICY_PROBS = np.array([i / n_probs for i in range(1, n_probs)], dtype=float)
assert np.all((POLICY_PROBS > 0) & (POLICY_PROBS < 1)), 'Each element in POLICY_PROBS must be strictly within (0, 1)'

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
# Compute sNB from pooled probabilities
########################################################################################################################
snb_dict: dict[str, np.ndarray] = {}
for m in MODEL_ORDER:
    y_true: np.ndarray = ref_y
    y_prob: np.ndarray = pooled_probs[m][PROB_COL].values
    snb_vals: np.ndarray = np.array([nb(y_true, y_prob, pt, standard=True) for pt in POLICY_PROBS])
    snb_dict[m] = snb_vals
    logger.info(f'{m}: sNB computed across {len(POLICY_PROBS)} thresholds')

logger.print('-'*100)

########################################################################################################################
# Create the plot
########################################################################################################################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))

# Plot the decision curves one by one
colors = {
    'ANN': "#CC3311",                  # Red
    'EN': "#009988",                   # Teal
    'SVM': "#EE3377",                  # Magenta
    'XGB': "#33BBEE",                  # Cyan
    'TF_unimodal': "#EEAABB",          # Yellow
    'TF_multimodal': "#AA4499",        # Purple
    'TF_Fusion_unimodal': "#EE7733",   # Orange
    'TF_Fusion_multimodal': "#0077BB"  # Blue
}
all_snb_vals = []
for model_idx, (m, snb_vals) in enumerate(snb_dict.items()):
    color: str = colors[m]
    if m in ['ANN', 'EN', 'SVM', 'XGB']:
        model_str: str = m
    else:
        model_str: str = {'TF_unimodal': 'Transformer (unimodal, standard)',
                          'TF_Fusion_unimodal': 'Transformer (unimodal, fusion)',
                          'TF_multimodal': 'Transformer (multimodal, standard)',
                          'TF_Fusion_multimodal': 'Transformer (multimodal, fusion)'}[m]
    ax.plot(POLICY_PROBS, snb_vals,
            label=model_str, alpha=0.8,
            linewidth=2,
            linestyle='-',
            color=color,
            zorder=2)
    all_snb_vals.extend(snb_vals)
    if model_idx == 3:
        treat_all_vals = np.asarray(compute_treat_all_nb(ref_y, POLICY_PROBS, standard=True), dtype=float)
        ax.plot(POLICY_PROBS, treat_all_vals,
                linestyle=':', linewidth=3, label='Treat all', zorder=1, alpha=0.9, color='grey')

ax.axhline(0.0,
           linestyle='-', linewidth=4, label='Treat none', zorder=0, alpha=0.9, color='black')

ax.set_xlabel('Threshold probability (p' + r'$_t$' + ')', fontsize=15)
ax.set_ylabel('Standardized net benefit (sNB)', fontsize=15, labelpad=10)
ax.set_xlim(0, 1)
ax.set_ylim(-1.02, 1.02)
ax.grid(alpha=0.5)
ax.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.legend(ncols=2)

OUT_DIR: str = '_Final_Results/'
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, 'DCA_Curves_Pooled.png'), dpi=300)
# plt.show()