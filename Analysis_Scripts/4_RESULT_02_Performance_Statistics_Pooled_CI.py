########################################################################################################################
# Compute cluster-bootstrap 95% confidence intervals for performance metrics across all models.
# Resamples by participant (not by record) to respect within-participant correlation.
# Predicted probabilities are first averaged elementwise across 4 seeds, then bootstrapped.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
from _Helper_Scripts.binary_metrics import binary_metrics
from _Helper_Scripts.Logger import create_log

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Runtime parameters
########################################################################################################################
N_BOOT: int = 2000
RANDOM_STATE: int = 42
PROB_COLS: list[str] = ['y_prob', 'y_prob_isotonic', 'y_prob_platt', 'y_prob_temperature']
SCALE_NAMES: list[str] = ['Raw', 'Isotonic', 'Platt', 'Temperature']
SCALE_MAP: dict[str, str] = dict(zip(PROB_COLS, SCALE_NAMES))

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
# Metrics to include in output (matching 4_RESULT_01c structure, excluding non-bootstrappable metrics)
########################################################################################################################
SINGLE_THRESH: list[str] = ['AUPRC', 'AUROC', 'AveragePrecision',
                            'BrierScore', 'BrierSkillScore',
                            'CalibrationIntercept', 'CalibrationSlope',
                            'CoxSnellPseudoR2', 'DiscriminationSlope',
                            'ECE', 'LogLoss', 'MAPE',
                            'McFaddenPseudoR2', 'NagelkerkePseudoR2', 'O:ERatio']
MULTI_THRESH: list[str] = ['#FN', '#TP', '#TN', '#FP',
                           'Accuracy', 'BalancedAccuracy', 'CohenKappa', 'DOR', 'F1',
                           'MCC', 'NetBenefit', 'NNE', 'NPV', 'Precision', 'Recall', 'Specificity',
                           'StandardizedNetBenefit', 'YoudenIndex']

# Row labels for the output spreadsheet (MULTI_THRESH appears twice: once at 0.5, once at prevalence)
ROW_LABELS: list[str] = SINGLE_THRESH + MULTI_THRESH + MULTI_THRESH
ROW_KEYS: list[str] = (SINGLE_THRESH +
                       [f'{m}@half' for m in MULTI_THRESH] +
                       [f'{m}@prev' for m in MULTI_THRESH])

logger.info(f'Bootstrap parameters: N_BOOT={N_BOOT}, RANDOM_STATE={RANDOM_STATE}')
logger.info(f'Output rows: {len(SINGLE_THRESH)} single-thresh + '
            f'{len(MULTI_THRESH)} multi-thresh @0.5 + {len(MULTI_THRESH)} multi-thresh @prev = {len(ROW_LABELS)}')
logger.print('-'*100)

########################################################################################################################
# Load test partition Record_IDs from Outcome_v2
########################################################################################################################
Y_PATH: str = '_Helper_Files/15min_Data/Outcome_v2.csv'
OUTCOME: str = 'BSI_GSI_Case'
df_y: pd.DataFrame = pd.read_csv(Y_PATH)
df_y_test: pd.DataFrame = df_y[df_y['Partition_Retraining'] == 'TEST'].copy().reset_index(drop=True)
test_record_ids: list[int] = df_y_test['Record_ID'].tolist()
n_test: int = len(test_record_ids)

# Get BeiweId mapping from features parquet
df_feat: pd.DataFrame = pd.read_parquet('_Helper_Files/15min_Data/Features_v1.parquet',
                                        columns=['Record_ID', 'BeiweId'])
record_to_beiwe: dict[int, str] = (df_feat.drop_duplicates('Record_ID')
                                   .set_index('Record_ID')['BeiweId'].to_dict())
test_beiwe_ids: list[str] = [record_to_beiwe[rid] for rid in test_record_ids]
id_df: pd.DataFrame = pd.DataFrame({'Record_ID': test_record_ids, 'BeiweId': test_beiwe_ids})

# Fixed prevalence threshold from the original test set
test_prevalence: float = float(df_y_test[OUTCOME].mean())
logger.info(f'Test partition: {n_test} records from {len(set(test_beiwe_ids))} participants')
logger.info(f'Fixed test-set prevalence threshold: {test_prevalence:.4f}')
logger.print('-'*100)

########################################################################################################################
# Load and pool predicted probabilities across 4 seeds for each model
########################################################################################################################
pooled_probs: dict[str, pd.DataFrame] = {}

for model in ML_MODELS:
    df_prob: pd.DataFrame = pd.read_csv(os.path.join(IN_DIR_2D, f'{model}_Probability_SEEDED.csv'))
    df_prob['pos'] = df_prob.groupby('seed').cumcount()
    pooled: pd.DataFrame = df_prob.groupby('pos')[['y_test'] + PROB_COLS].mean().reset_index(drop=True)
    pooled['Record_ID'] = test_record_ids
    pooled['BeiweId'] = test_beiwe_ids
    pooled_probs[model] = pooled
    logger.info(f'{model}: pooled {df_prob["seed"].nunique()} seeds → {len(pooled)} records')

for model in TF_MODELS_2D:
    df_prob: pd.DataFrame = pd.read_csv(os.path.join(IN_DIR_2D, f'{model}_Probability_SEEDED.csv'))
    if 'Model' in df_prob.columns:
        df_prob = df_prob[df_prob['Model'] == 'Raw']
    pooled: pd.DataFrame = df_prob.groupby('rec_id')[['y_test'] + PROB_COLS].mean().reset_index()
    pooled = pooled.rename(columns={'rec_id': 'Record_ID'})
    pooled = pooled.merge(id_df, on='Record_ID')
    pooled_probs[model] = pooled
    logger.info(f'{model}: pooled → {len(pooled)} records')

for model in TF_MODELS_FUSION:
    df_prob: pd.DataFrame = pd.read_csv(os.path.join(IN_DIR_FUSION, model, f'{model}_Probability_SEEDED.csv'))
    if 'Model' in df_prob.columns:
        df_prob = df_prob[df_prob['Model'] == 'Raw']
    pooled: pd.DataFrame = df_prob.groupby('rec_id')[['y_test'] + PROB_COLS].mean().reset_index()
    pooled = pooled.rename(columns={'rec_id': 'Record_ID'})
    pooled = pooled.merge(id_df, on='Record_ID')
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
# Pre-build participant → row index mapping for cluster resampling
########################################################################################################################
beiwe_arr: np.ndarray = pooled_probs[MODEL_ORDER[0]]['BeiweId'].values
unique_beiwe: np.ndarray = np.unique(beiwe_arr)
n_participants: int = len(unique_beiwe)
beiwe_to_idx: dict[str, np.ndarray] = {bid: np.where(beiwe_arr == bid)[0] for bid in unique_beiwe}
logger.info(f'{n_participants} unique participants indexed for cluster resampling')
logger.print('-'*100)

########################################################################################################################
# Cluster bootstrap
########################################################################################################################
rng: np.random.Generator = np.random.default_rng(RANDOM_STATE)
logger.info(f'Starting cluster bootstrap: {N_BOOT} iterations, {n_participants} clusters')

# Pre-extract numpy arrays for speed
model_prob_arrays: dict[str, dict[str, np.ndarray]] = {}
for model in MODEL_ORDER:
    model_prob_arrays[model] = {col: pooled_probs[model][col].values for col in PROB_COLS}

# Storage: boot_arrays[model][scale][key] = list of metric values across bootstrap iterations
boot_arrays: dict[str, dict[str, dict[str, list[float]]]] = {
    m: {s: {k: [] for k in ROW_KEYS} for s in SCALE_NAMES} for m in MODEL_ORDER}

n_skipped: int = 0
for b in range(N_BOOT):
    if (b + 1) % 500 == 0:
        logger.info(f'  Bootstrap iteration {b + 1}/{N_BOOT}')

    # Resample participants with replacement
    sampled_beiwe: np.ndarray = rng.choice(unique_beiwe, size=n_participants, replace=True)
    boot_idx: np.ndarray = np.concatenate([beiwe_to_idx[bid] for bid in sampled_beiwe])
    y_true_boot: np.ndarray = ref_y[boot_idx]

    # Skip degenerate resamples (all 0 or all 1)
    if len(np.unique(y_true_boot)) < 2:
        n_skipped += 1
        continue

    for model in MODEL_ORDER:
        for prob_col, scale in SCALE_MAP.items():
            y_prob_boot: np.ndarray = model_prob_arrays[model][prob_col][boot_idx]
            try:
                res_half: dict = binary_metrics(y_true=y_true_boot, y_prob=y_prob_boot, threshold=0.5)
                res_prev: dict = binary_metrics(y_true=y_true_boot, y_prob=y_prob_boot, threshold=test_prevalence)
            except Exception:
                continue

            for m_name in SINGLE_THRESH:
                boot_arrays[model][scale][m_name].append(res_half.get(m_name, np.nan))
            for m_name in MULTI_THRESH:
                boot_arrays[model][scale][f'{m_name}@half'].append(res_half.get(m_name, np.nan))
                boot_arrays[model][scale][f'{m_name}@prev'].append(res_prev.get(m_name, np.nan))

logger.info(f'Bootstrap complete. Skipped {n_skipped} degenerate resamples out of {N_BOOT}.')
logger.print('-'*100)

########################################################################################################################
# Compute percentile CIs and format as "(lower, upper)" rounded to 3 decimal places
########################################################################################################################
OUT_DIR: str = '_Final_Results/'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH: str = os.path.join(OUT_DIR, 'All_Performance_Bootstrap_CI.xlsx')

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    for scale in SCALE_NAMES:
        rows: list[dict] = []
        for label, key in zip(ROW_LABELS, ROW_KEYS):
            row: dict = {'Metrics': label}
            for model in MODEL_ORDER:
                vals: np.ndarray = np.array(boot_arrays[model][scale][key])
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    lo: float = np.percentile(vals, 2.5)
                    hi: float = np.percentile(vals, 97.5)
                    row[model] = f'({lo:.3f}, {hi:.3f})'
                else:
                    row[model] = 'N/A'
            rows.append(row)
        df_out: pd.DataFrame = pd.DataFrame(rows)
        df_out.to_excel(writer, sheet_name=scale, index=False)
        logger.info(f'Wrote {scale} tab: {df_out.shape}')

logger.info(f'Bootstrap CIs exported to {OUT_PATH}')
logger.print('-'*100)