########################################################################################################################
# Compute ICC for each sensor feature using a linear random-intercept model.
# Steps: (1) load 15-min epoch features, (2) aggregate to record level via nanmean,
#         (3) fit feature ~ 1 + (1 | BeiweId) per feature, (4) extract ICC = tau^2 / (tau^2 + sigma^2).
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
import statsmodels.formula.api as smf
import warnings

########################################################################################################################
# Paths
########################################################################################################################
FEATURE_PATH: str = '_Helper_Files/15min_Data/Features_v4.parquet'
OUTCOME_PATH: str = '_Helper_Files/15min_Data/Outcome_v2.csv'
OUT_DIR: str = '_Final_Results/'
os.makedirs(OUT_DIR, exist_ok=True)

########################################################################################################################
# Load data
########################################################################################################################
df_feat: pd.DataFrame = pd.read_parquet(FEATURE_PATH)
df_y: pd.DataFrame = pd.read_csv(OUTCOME_PATH, usecols=['Record_ID', 'BeiweId'])

# Exclude ID, metadata, outcome, and index columns
EXCLUDE_COLS: list[str] = ['Record_ID', 'BeiweId', 'AppleId',
                           'Sex', 'Survey_Day', 'Survey_Time', 'Bin_Start_Time', 'Bin_End_Time',
                           'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                           'with_home_address', 'bin_index', 'is_pad']
# Exclude @survey, @roc, and @clock suffixes (indexing artifacts), but keep @daily (derived sensor features)
EXCLUDE_SUFFIXES: list[str] = ['@survey', '@clock', '@roc']
sensor_cols: list[str] = [c for c in df_feat.columns
                          if c not in EXCLUDE_COLS
                          and not any(c.endswith(s) for s in EXCLUDE_SUFFIXES)
                          and pd.api.types.is_numeric_dtype(df_feat[c])]
print(f'Loaded {len(df_feat)} epoch rows, {len(sensor_cols)} sensor features (after excluding index/metadata/outcome)')

########################################################################################################################
# Aggregate to record level via nanmean
########################################################################################################################
df_record: pd.DataFrame = df_feat.groupby('Record_ID')[sensor_cols].mean()  # pandas .mean() skips NaN by default
df_record = df_record.reset_index()
df_record = df_record.merge(df_y[['Record_ID', 'BeiweId']], on='Record_ID', how='inner')

n_records: int = len(df_record)
n_participants: int = df_record['BeiweId'].nunique()
print(f'Record-level: {n_records} records, {n_participants} participants')

# Report remaining missingness after aggregation
n_missing_per_feature: pd.Series = df_record[sensor_cols].isna().sum()
n_features_with_missing: int = int((n_missing_per_feature > 0).sum())
print(f'Features with any remaining NaN after aggregation: {n_features_with_missing}/{len(sensor_cols)}')
if n_features_with_missing > 0:
    print(f'  Max missing records per feature: {n_missing_per_feature.max()}')
    print(f'  Median missing records (among those with any): '
          f'{n_missing_per_feature[n_missing_per_feature > 0].median():.0f}')

########################################################################################################################
# Fit linear mixed model per feature and extract ICC
########################################################################################################################
results: list[dict] = []
n_failed: int = 0

for i, col in enumerate(sensor_cols):
    if (i + 1) % 25 == 0 or (i + 1) == len(sensor_cols):
        print(f'  Fitting feature {i + 1}/{len(sensor_cols)}: {col}')

    df_fit: pd.DataFrame = df_record[['BeiweId', col]].dropna()
    n_obs: int = len(df_fit)
    n_pids: int = df_fit['BeiweId'].nunique()

    # Skip if too few observations or participants
    if n_obs < 10 or n_pids < 3:
        results.append({'Feature': col, 'ICC': np.nan, 'tau2': np.nan, 'sigma2': np.nan,
                        'n_obs': n_obs, 'n_participants': n_pids, 'status': 'skipped_insufficient_data'})
        n_failed += 1
        continue

    # Check for zero variance
    if df_fit[col].std() == 0:
        results.append({'Feature': col, 'ICC': np.nan, 'tau2': np.nan, 'sigma2': np.nan,
                        'n_obs': n_obs, 'n_participants': n_pids, 'status': 'skipped_zero_variance'})
        n_failed += 1
        continue

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lmm = smf.mixedlm(f'Q("{col}") ~ 1', data=df_fit, groups=df_fit['BeiweId'])
            lmm_fit = lmm.fit(reml=True)
        tau2: float = float(lmm_fit.cov_re.iloc[0, 0])
        sigma2: float = float(lmm_fit.scale)
        icc: float = tau2 / (tau2 + sigma2) if (tau2 + sigma2) > 0 else np.nan
        results.append({'Feature': col, 'ICC': round(icc, 4), 'tau2': tau2, 'sigma2': sigma2,
                        'n_obs': n_obs, 'n_participants': n_pids, 'status': 'converged'})
    except Exception as e:
        results.append({'Feature': col, 'ICC': np.nan, 'tau2': np.nan, 'sigma2': np.nan,
                        'n_obs': n_obs, 'n_participants': n_pids, 'status': f'failed: {str(e)[:80]}'})
        n_failed += 1

########################################################################################################################
# Compile results
########################################################################################################################
df_icc: pd.DataFrame = pd.DataFrame(results)
df_converged: pd.DataFrame = df_icc[df_icc['status'] == 'converged']

print(f'\n{"="*80}')
print(f'RESULTS: {len(df_converged)} converged / {len(sensor_cols)} total features ({n_failed} failed/skipped)')
print(f'{"="*80}')

if len(df_converged) > 0:
    icc_vals: pd.Series = df_converged['ICC']
    print(f'ICC across sensor features:')
    print(f'  Min:    {icc_vals.min():.4f}')
    print(f'  Q1:     {icc_vals.quantile(0.25):.4f}')
    print(f'  Median: {icc_vals.median():.4f}')
    print(f'  Q3:     {icc_vals.quantile(0.75):.4f}')
    print(f'  Max:    {icc_vals.max():.4f}')
    print(f'  Mean:   {icc_vals.mean():.4f}')
    print(f'  SD:     {icc_vals.std():.4f}')

    # Top 10 highest ICC
    print(f'\nTop 10 features by ICC:')
    for _, row in df_converged.nlargest(10, 'ICC').iterrows():
        print(f'  {row["Feature"]}: {row["ICC"]:.4f}')

    # Bottom 10 lowest ICC
    print(f'\nBottom 10 features by ICC:')
    for _, row in df_converged.nsmallest(10, 'ICC').iterrows():
        print(f'  {row["Feature"]}: {row["ICC"]:.4f}')

########################################################################################################################
# Export full table
########################################################################################################################
OUT_PATH: str = os.path.join(OUT_DIR, 'Sensor_Feature_ICC.csv')
df_icc.to_csv(OUT_PATH, index=False)
print(f'\nFull ICC table exported to {OUT_PATH}')