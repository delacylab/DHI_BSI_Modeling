########################################################################################################################
# Partitioning the participant-day record by stratifying over participants and the binary outcome label
# To ensure that the training, validation, and test sets all share similar outcome label distributions, we need to
# do a seed-hacking technique to identify the random seed that best preserve the distributions across the partitions.
#
# Retraining partition (Partition_Retraining) uses a 7:1:2 TRAIN:VAL:TEST split. This is achieved by keeping the
# 5-fold outer StratifiedGroupKFold split (used to define TEST and V1-V4) and halving the fold that would otherwise
# be VAL — half becomes new VAL (~10%), half joins TRAIN. Partition_Validation is unaffected.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
from _Helper_Scripts.Logger import create_log
from sklearn.model_selection import StratifiedGroupKFold

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Specify the runtime parameters
########################################################################################################################
TEST_RATIO: float = 0.2
K_FOLD: int = 4
OUTCOME_LABEL: str = 'BSI_GSI_Case'

########################################################################################################################
# Specify the input directory
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'

########################################################################################################################
# Load the feature file
########################################################################################################################
data_path: str = os.path.join(IN_DIR, 'Features_v4.parquet')
df: pd.DataFrame = pd.read_parquet(data_path)
logger.info(f'Loaded: {data_path}  Shape: {df.shape}')
logger.print('=' * 100)

########################################################################################################################
# Create df_y using the non-feature column subset
########################################################################################################################
needed_cols: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                          'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case']
df_y: pd.DataFrame = df[needed_cols].copy()
df_y = df_y.sort_values(by=['Record_ID', 'BeiweId']).drop_duplicates(subset=['Record_ID', 'BeiweId'], keep='first')
df_y.reset_index(drop=True, inplace=True)
target_prev: float = df_y[OUTCOME_LABEL].mean()
logger.info(f'Shape of df_y = {df_y.shape}; '
            f'# participant-day records = {df_y.Record_ID.nunique()}; # BeiweId = {df_y.BeiweId.nunique()}')
logger.info(f'Prevalence of the outcome label = {target_prev*100:.2f}%')

########################################################################################################################
# Identify the best random seed to align the outcome distributions between the training and the test partitions
########################################################################################################################
best_seed: int = -1
best_drift: float = 1.0
n_splits: int = int(1 / TEST_RATIO)
for seed in range(50):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_folds: list[tuple] = list(sgkf.split(X=df_y, y=df_y[OUTCOME_LABEL], groups=df_y['BeiweId']))
    te_idx: np.ndarray = all_folds[0][1]
    va_idx: np.ndarray = all_folds[1][1]
    tr_idx: np.ndarray = np.concatenate([all_folds[i][1] for i in range(2, n_splits)])
    tr_prev: float = df_y.iloc[tr_idx][OUTCOME_LABEL].mean()
    va_prev: float = df_y.iloc[va_idx][OUTCOME_LABEL].mean()
    te_prev: float = df_y.iloc[te_idx][OUTCOME_LABEL].mean()
    max_drift: float = max(abs(tr_prev - target_prev),
                           abs(va_prev - target_prev),
                           abs(te_prev - target_prev))
    if max_drift < best_drift:
        best_drift, best_seed = max_drift, seed
        logger.info(f'seed={seed:3d}: train={tr_prev*100:.2f}%, val={va_prev*100:.2f}%, '
                    f'test={te_prev*100:.2f}%, max_drift={max_drift*100:.2f}%')
logger.info(f'Best seed: {best_seed} with max drift {best_drift*100:.2f}%')
logger.print('-' * 100)

########################################################################################################################
# Use the identified seed to obtain all 5 outer folds:
#   Fold 1 -> TEST (Partition_Validation + Partition_Retraining)
#   Fold 2 -> halved: one half -> VAL (Partition_Retraining), other half -> TRAIN (Partition_Retraining)
#   Folds 3-5 -> TRAIN (Partition_Retraining)
# All partitions are participant-disjoint by construction.
########################################################################################################################
sgkf_outer = StratifiedGroupKFold(n_splits=int(1/TEST_RATIO), shuffle=True, random_state=best_seed)
all_outer_folds: list[tuple] = list(sgkf_outer.split(X=df_y, y=df_y[OUTCOME_LABEL], groups=df_y['BeiweId']))

te_idx_final: np.ndarray = all_outer_folds[0][1]      # fold 1 -> TEST (unchanged from 6:2:2)
old_va_idx: np.ndarray = all_outer_folds[1][1]        # fold 2 -> to be halved

########################################################################################################################
# Sweep seeds to find the sub-split of old fold 2 that minimizes prevalence drift between the two halves
# (one half -> new VAL, other half -> extra TRAIN). Participants remain disjoint via StratifiedGroupKFold.
########################################################################################################################
old_va_records: pd.DataFrame = df_y.iloc[old_va_idx].reset_index(drop=True)
target_va_prev: float = df_y[OUTCOME_LABEL].mean()

best_va_seed: int = -1
best_va_drift: float = 1.0
for seed in range(50):
    sgkf_va = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    extra_tr_local, new_va_local = next(sgkf_va.split(X=old_va_records,
                                                      y=old_va_records[OUTCOME_LABEL],
                                                      groups=old_va_records['BeiweId']))
    new_va_prev: float = old_va_records.iloc[new_va_local][OUTCOME_LABEL].mean()
    extra_tr_prev: float = old_va_records.iloc[extra_tr_local][OUTCOME_LABEL].mean()
    max_drift_va: float = max(abs(new_va_prev - target_va_prev),
                              abs(extra_tr_prev - target_va_prev))
    if max_drift_va < best_va_drift:
        best_va_drift, best_va_seed = max_drift_va, seed
        logger.info(f'fold2 sub-split seed={seed:3d}: new_val={new_va_prev*100:.2f}%, '
                    f'extra_train={extra_tr_prev*100:.2f}%, max_drift={max_drift_va*100:.2f}%')
logger.info(f'Best fold2 sub-split seed: {best_va_seed} with max drift {best_va_drift*100:.2f}%')

sgkf_va_final = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=best_va_seed)
extra_tr_local, new_va_local = next(sgkf_va_final.split(X=old_va_records,
                                                        y=old_va_records[OUTCOME_LABEL],
                                                        groups=old_va_records['BeiweId']))
extra_tr_idx: np.ndarray = old_va_idx[extra_tr_local]
va_idx_final: np.ndarray = old_va_idx[new_va_local]
tr_idx_final: np.ndarray = np.concatenate(
    [np.concatenate([all_outer_folds[i][1] for i in range(2, int(1/TEST_RATIO))]),
     extra_tr_idx]
)

tr_prev_final: float = df_y.iloc[tr_idx_final][OUTCOME_LABEL].mean()
va_prev_final: float = df_y.iloc[va_idx_final][OUTCOME_LABEL].mean()
te_prev_final: float = df_y.iloc[te_idx_final][OUTCOME_LABEL].mean()
logger.info(f'Outer split (7:1:2, seed={best_seed}, fold2 sub-seed={best_va_seed}): '
            f'train_prev={tr_prev_final*100:.2f}%, val_prev={va_prev_final*100:.2f}%, '
            f'test_prev={te_prev_final*100:.2f}%')
logger.info(f'Outer sizes: TRAIN={len(tr_idx_final)}, VAL={len(va_idx_final)}, TEST={len(te_idx_final)}')

df_y['Partition_Retraining'] = 'TRAIN'
df_y.loc[va_idx_final, 'Partition_Retraining'] = 'VAL'
df_y.loc[te_idx_final, 'Partition_Retraining'] = 'TEST'

df_y['Partition_Validation'] = 'UNASSIGNED'
df_y.loc[te_idx_final, 'Partition_Validation'] = 'TEST'
logger.info(f'Partition_Retraining: {len(te_idx_final)} TEST, {len(va_idx_final)} VAL, {len(tr_idx_final)} TRAIN')
logger.print('-' * 100)

########################################################################################################################
# Identify the best random seed to align the outcome distributions across internal validation sets
# Inner CV is performed on the full training set from the outer TEST split (folds 2-5 combined),
# i.e., tr_idx + va from Partition_Retraining perspective, which is Partition_Validation's non-TEST records
########################################################################################################################
df_y_train: pd.DataFrame = df_y[df_y['Partition_Validation'] == 'UNASSIGNED'].reset_index(drop=True)
target_prev = df_y_train[OUTCOME_LABEL].mean()

best_inner_seed: int = -1
best_max_drift: float = 1.0
for seed in range(50):
    sgkf = StratifiedGroupKFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    fold_prevs: list[float] = []
    for _, val_idx in sgkf.split(df_y_train, df_y_train[OUTCOME_LABEL], df_y_train['BeiweId']):
        fold_prevs.append(df_y_train.iloc[val_idx][OUTCOME_LABEL].mean())
    max_drift: float = max(abs(p - target_prev) for p in fold_prevs)
    if max_drift < best_max_drift:
        best_max_drift = max_drift
        best_inner_seed = seed
logger.info(f'Best inner seed: {best_inner_seed} with max fold-drift {best_max_drift*100:.2f}%')
logger.print('-' * 100)

########################################################################################################################
# Use the identified seed to perform the stratification and obtain the validation sets
########################################################################################################################
sgkf_inner = StratifiedGroupKFold(n_splits=K_FOLD, shuffle=True, random_state=best_inner_seed)
for fold_num, (_, val_idx) in enumerate(sgkf_inner.split(df_y_train, df_y_train[OUTCOME_LABEL],
                                                         df_y_train['BeiweId']), start=1):
    df_y_train.loc[val_idx, 'Partition_Validation'] = f'V{fold_num}'

# Merge Partition_Validation labels back into df_y
df_y = df_y.drop(columns='Partition_Validation').merge(
    df_y_train[['Record_ID', 'Partition_Validation']], on='Record_ID', how='left')
df_y.loc[df_y['Partition_Validation'].isna(), 'Partition_Validation'] = 'TEST'

# Sanity checks
assert df_y['Partition_Validation'].isna().sum() == 0
assert df_y['Partition_Retraining'].isna().sum() == 0
assert set(df_y['Partition_Validation'].unique()) == {'TEST', 'V1', 'V2', 'V3', 'V4'}
assert set(df_y['Partition_Retraining'].unique()) == {'TRAIN', 'VAL', 'TEST'}

# Participant-disjoint checks across all partitions
pids_by_val: dict[str, set] = {lbl: set(df_y.loc[df_y['Partition_Validation'] == lbl, 'BeiweId'])
                               for lbl in ['TEST', 'V1', 'V2', 'V3', 'V4']}
for l1 in pids_by_val:
    for l2 in pids_by_val:
        if l1 < l2:
            assert pids_by_val[l1] & pids_by_val[l2] == set(), f'participant leakage: {l1} vs {l2}'

pids_by_ret: dict[str, set] = {lbl: set(df_y.loc[df_y['Partition_Retraining'] == lbl, 'BeiweId'])
                               for lbl in ['TRAIN', 'VAL', 'TEST']}
for l1 in pids_by_ret:
    for l2 in pids_by_ret:
        if l1 < l2:
            assert pids_by_ret[l1] & pids_by_ret[l2] == set(), f'participant leakage: {l1} vs {l2}'
logger.print('-' * 100)

########################################################################################################################
# Confirm the distributions
########################################################################################################################
logger.info('Partition_Validation distributions:')
for part in ['TEST', 'V1', 'V2', 'V3', 'V4']:
    sub: pd.DataFrame = df_y[df_y['Partition_Validation'] == part]
    logger.info(f'  [{part}] n={len(sub)}, participants={sub["BeiweId"].nunique()}, '
                f'prevalence={sub[OUTCOME_LABEL].mean()*100:.2f}%')
logger.print('-' * 100)
logger.info('Partition_Retraining distributions:')
for part in ['TRAIN', 'VAL', 'TEST']:
    sub = df_y[df_y['Partition_Retraining'] == part]
    logger.info(f'  [{part}] n={len(sub)}, participants={sub["BeiweId"].nunique()}, '
                f'prevalence={sub[OUTCOME_LABEL].mean()*100:.2f}%')
logger.print('-' * 100)

########################################################################################################################
# Save
########################################################################################################################
out_path: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
df_y.to_csv(out_path, index=False)
logger.info(f'Saved: {out_path}  Shape: {df_y.shape}')
