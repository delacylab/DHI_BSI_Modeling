########################################################################################################################
# Overview: This script identifies the potential outcome variable
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
# Set up the flag of whether to subset to a single participant-day survey
########################################################################################################################
SINGLE_SURVEY_PER_DAY: bool = True
logger.info(f'SINGLE_SURVEY_PER_DAY = {SINGLE_SURVEY_PER_DAY}')
logger.print('-'*100)

########################################################################################################################
# Load the iOS configuration file if needed
########################################################################################################################
IOS_PATH: str = '_Helper_Files/iOS_Identification.csv'
df_ios: pd.DataFrame = pd.read_csv(IOS_PATH)
df_ios = df_ios[(df_ios['iOS'] == 1) & (df_ios['BeiweId'].notna())]

########################################################################################################################
# Load the data
########################################################################################################################
df_path: str = '../00_Data/01_Cleaned_Data/D5_End_of_Day_Self_Reports_Data/EOD_v1.csv'
df: pd.DataFrame = pd.read_csv(df_path)
df['RecordedDate'] = pd.to_datetime(df['RecordedDate'])
df.rename(columns={'RecordedDate': 'Survey_Time'}, inplace=True)

########################################################################################################################
# Subset to the iOS users
########################################################################################################################
df = df[df['BeiweId'].isin(df_ios['BeiweId'].unique())]
logger.info(f'# involved Beiwe IDs: {df["BeiweId"].nunique()}')

########################################################################################################################
# Filter df by columns
########################################################################################################################
bkgd_cols: list[str] = ['BeiweId', 'AppleId', 'Survey_Time']
bsi_cols: list[str] = [c for c in df.columns if c.startswith('BSI')]
bsi_cols: list[str] = [c for c in bsi_cols
                       if not (c.endswith('Ave') or c.endswith('.rnd') or c.endswith('manual'))]
df = df[df['Survey_Time'].notna()]
df = df[bkgd_cols + bsi_cols].sort_values(by=['BeiweId', 'Survey_Time']).reset_index(drop=True)
logger.info(f'# involved participant-day records: {len(df)}')

########################################################################################################################
# Subset to a single survey per participant per day if needed
########################################################################################################################
df['Survey_Date'] = pd.to_datetime(df['Survey_Time']).dt.normalize()
df.insert(3, 'Survey_Day',
          (df['Survey_Date'] - df.groupby('BeiweId')['Survey_Date'].transform('min')).dt.days)
if SINGLE_SURVEY_PER_DAY:
    df = (df.sort_values(['BeiweId', 'Survey_Time'])
          .drop_duplicates(subset=['BeiweId', 'Survey_Date'], keep='last').reset_index(drop=True))
    logger.info(f'# involved participant_day records (after requiring a single survey per day): {len(df)}')
df.drop(columns='Survey_Date', inplace=True)
logger.print('-'*100)

########################################################################################################################
# Compute and visualize the summary statistics of Survey_Time
########################################################################################################################
df.insert(4, 'Survey_Time_Hour', df['Survey_Time'].dt.round('h').dt.hour)
logger.info(f'Survey time (mean):                {df["Survey_Time_Hour"].mean():.1f}')
logger.info(f'Survey time (sd):                  {df["Survey_Time_Hour"].std():.1f}')
logger.info(f'Survey time (min):                 {df["Survey_Time_Hour"].min():.1f}')
logger.info(f'Survey time (25th percentile):     {df["Survey_Time_Hour"].quantile(.25):.1f}')
logger.info(f'Survey time (median):              {df["Survey_Time_Hour"].median():.1f}')
logger.info(f'Survey time (75th percentile):     {df["Survey_Time_Hour"].quantile(.75):.1f}')
logger.info(f'Survey time (max):                 {df["Survey_Time_Hour"].max():.1f}')

plt.hist(df['Survey_Time_Hour'], bins=[x - 0.5 for x in range(25)], edgecolor='black')
mean_hour = df['Survey_Time_Hour'].mean()
median_hour = df['Survey_Time_Hour'].median()
plt.axvline(mean_hour, linestyle=':', color='red', linewidth=3, label=f'Mean = {mean_hour:.1f}')
plt.axvline(median_hour, linestyle='--', color='gold', linewidth=3, label=f'Median = {median_hour:.1f}')
plt.xlabel('Survey time hour')
plt.ylabel('Count')
plt.title('Distribution of Survey Time by Hour')
plt.xticks(range(24))
plt.xlim(-0.5, 23.5)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.legend()
out_fig_dir: str = '_Helper_Files/Figures/'
os.makedirs(out_fig_dir, exist_ok=True)
plt.savefig(os.path.join(out_fig_dir, 'Survey_time_distribution.png'), dpi=300)
plt.close()
logger.print('-'*100)

########################################################################################################################
# Verify the prorated subscale scores from the data against recomputation from raw items
########################################################################################################################
SOM_ITEMS: list[int] = [1, 4, 7, 10, 13, 16]
DEP_ITEMS: list[int] = [2, 5, 8, 11, 14, 17]
ANX_ITEMS: list[int] = [3, 6, 9, 12, 15, 18]
SUBSCALE_MAP: dict[str, list[int]] = {'SOM': SOM_ITEMS, 'DEP': DEP_ITEMS, 'ANX': ANX_ITEMS}

for subscale, items in SUBSCALE_MAP.items():
    item_cols: list[str] = [f'BSI_{i}' for i in items]
    n_answered: pd.Series = df[item_cols].notna().sum(axis=1)
    raw_sum: pd.Series = df[item_cols].sum(axis=1)
    recomputed: pd.Series = (raw_sum * 6 / n_answered).where(n_answered > 0)
    existing: pd.Series = df[f'BSI_{subscale}_prorate']
    match: pd.Series = recomputed.eq(existing) | (recomputed.isna() & existing.isna())
    n_mismatch: int = (~match).sum()
    if n_mismatch > 0:
        max_diff: float = (recomputed - existing).abs().max()
        logger.info(f'BSI_{subscale}_prorate: {n_mismatch} mismatches (max diff = {max_diff:.6f})')
    else:
        logger.info(f'BSI_{subscale}_prorate: verified (all match)')
logger.print('-'*100)

########################################################################################################################
# Discard the original BSI_GSI columns (direct proration of all 18 items, not literature-supported).
# The BSI-18 GSI is defined as SOM + DEP + ANX (Derogatis, 2001; Franke et al., 2017).
# Proration must be performed at the subscale level (Franke et al., 2017; Michel et al., 2024; Franke, 2000),
# then GSI is computed as the sum of the prorated subscale scores.
########################################################################################################################
df.drop(columns=['BSI_GSI_prorate', 'BSI_GSI_sum', 'BSI_GSI_miss'], inplace=True)

########################################################################################################################
# Apply subscale validity rule and compute GSI from subscale prorates
# Rule (Franke et al., 2017; Michel et al., 2024; Franke, 2000): ≤1 missing per 6-item subscale.
# Proration method: person-mean imputation within each subscale.
# GSI = SOM + DEP + ANX; valid if and only if all three subscales are valid.
########################################################################################################################
SUBSCALE_MISS_MAX: int = 1

for subscale in ['SOM', 'DEP', 'ANX']:
    miss_col: str = f'BSI_{subscale}_miss'
    prorate_col: str = f'BSI_{subscale}_prorate'
    n_invalid: int = (df[miss_col] > SUBSCALE_MISS_MAX).sum()
    df.loc[df[miss_col] > SUBSCALE_MISS_MAX, prorate_col] = np.nan
    logger.info(f'{prorate_col}: {n_invalid} records invalidated (>{SUBSCALE_MISS_MAX} missing items)')

any_subscale_invalid: pd.Series = (
    df['BSI_SOM_prorate'].isna() | df['BSI_DEP_prorate'].isna() | df['BSI_ANX_prorate'].isna()
)
df['BSI_GSI_prorate'] = np.where(
    any_subscale_invalid, np.nan,
    df['BSI_SOM_prorate'] + df['BSI_DEP_prorate'] + df['BSI_ANX_prorate']
)

n_valid: int = df['BSI_GSI_prorate'].notna().sum()
n_total: int = len(df)
n_invalid_gsi: int = any_subscale_invalid.sum()
logger.info(f'BSI_GSI_prorate: {n_invalid_gsi} records invalidated (any subscale invalid)')
logger.info(f'Valid GSI records: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)')
logger.print('-'*100)

########################################################################################################################
# Remove invalid GSI records and visualize the distribution of BSI_GSI_prorate
########################################################################################################################
n_before: int = len(df)
df = df[df['BSI_GSI_prorate'].notna()].reset_index(drop=True)
n_after: int = len(df)
logger.info(f'Removed {n_before - n_after} records with invalid GSI')
logger.info(f'Remaining records: {n_after}')
logger.info(f'Remaining participants: {df["BeiweId"].nunique()}')

gsi: pd.Series = df['BSI_GSI_prorate']
logger.info(f'BSI_GSI_prorate (mean):   {gsi.mean():.1f}')
logger.info(f'BSI_GSI_prorate (sd):     {gsi.std():.1f}')
logger.info(f'BSI_GSI_prorate (median): {gsi.median():.1f}')
logger.info(f'BSI_GSI_prorate (min):    {gsi.min():.1f}')
logger.info(f'BSI_GSI_prorate (max):    {gsi.max():.1f}')
logger.info(f'BSI_GSI_prorate (% = 0):  {100*(gsi == 0).mean():.1f}%')

fig, ax = plt.subplots(figsize=(10, 5))
bins: np.ndarray = np.arange(-0.5, 73.5, 1)
ax.hist(gsi, bins=bins, edgecolor='black', linewidth=0.3)
ax.axvline(gsi.mean(), linestyle=':', color='red', linewidth=2, label=f'Mean = {gsi.mean():.1f}')
ax.axvline(gsi.median(), linestyle='--', color='gold', linewidth=2, label=f'Median = {gsi.median():.1f}')
ax.set_xlabel('BSI-18 GSI (prorated sum score)')
ax.set_ylabel('Count')
ax.set_title('Distribution of BSI-18 Global Severity Index')
ax.set_xlim(-0.5, 72.5)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_fig_dir, 'BSI_GSI_prorate_distribution.png'), dpi=300)
plt.close(fig)
logger.print('-'*100)

########################################################################################################################
# Identify the sex of the participant for each record in df
########################################################################################################################
baseline_path: str = '../00_Data/01_Cleaned_Data/D6_Apple_Baseline_Outcomes/Apple_Baseline_v4.csv'
df_baseline: pd.DataFrame = pd.read_csv(baseline_path, usecols=['AppleId', 'BeiweId', 'Sex'])
apple_beiwe_map: dict[str, str] = {p[0]: p[1] for p in zip(df['AppleId'], df['BeiweId'])}
df_baseline = df_baseline[df_baseline['AppleId'].isin(apple_beiwe_map.keys())]
df_baseline['BeiweId'] = df_baseline['AppleId'].map(apple_beiwe_map)
assert df_baseline['BeiweId'].isna().sum() == 0
assert df_baseline['Sex'].isna().sum() == 0
df = pd.merge(left=df, right=df_baseline[['BeiweId', 'Sex']], on='BeiweId', how='left')
df['Sex'] = df['Sex'].map({1: 'F', 2: 'M'})
logger.info(f'Sex distribution: {df["Sex"].value_counts().to_dict()}')

########################################################################################################################
# Convert prorated GSI to T-score using published general-population norms
# Reference: Michel et al. (2024), "The Brief Symptom Inventory in the Swiss general population", PLOS ONE.
#   DOI: https://doi.org/10.1371/journal.pone.0305192
#   T-score conversion: S2 Appendix, Table S8 (GSI-18)
#   N = 1,238 representative general-population adults, ages 18-75
#   Normalized (area) T-scores: mean=50, SD=10
########################################################################################################################
# GSI-18 raw-to-T lookup: {raw_sum: (T_all, T_male, T_female)}
GSI18_T_TABLE: dict[int, tuple[int, int, int]] = {
    0: (36, 36, 35), 1: (42, 42, 42), 2: (46, 47, 45), 3: (48, 50, 47),
    4: (50, 52, 49), 5: (52, 54, 51), 6: (54, 56, 53), 7: (55, 57, 54),
    8: (57, 59, 56), 9: (59, 60, 57), 10: (60, 61, 59), 11: (61, 62, 60),
    12: (62, 63, 61), 13: (63, 64, 62), 14: (63, 65, 62), 15: (64, 66, 63),
    16: (65, 66, 64), 17: (65, 67, 64), 18: (66, 67, 65), 19: (66, 68, 65),
    20: (67, 69, 65), 21: (67, 70, 66), 22: (68, 70, 66), 23: (68, 70, 67),
    24: (69, 70, 68), 25: (69, 71, 69), 26: (70, 71, 69), 27: (70, 71, 70),
    28: (71, 71, 71), 29: (72, 71, 72), 30: (72, 72, 72), 31: (73, 72, 74),
    32: (73, 72, 75), 33: (74, 72, 76), 34: (74, 73, 76), 35: (75, 73, 76),
    36: (75, 74, 76), 37: (76, 75, 77), 38: (76, 75, 77), 39: (77, 76, 78),
    40: (79, 78, 79), 41: (79, 78, 79), 42: (80, 79, 80), 43: (80, 79, 80),
    44: (80, 79, 80), 45: (80, 79, 80), 46: (80, 79, 80),
}
T_MAX: tuple[int, int, int] = (80, 80, 80)
SEX_COL_IDX: dict[str, int] = {'M': 1, 'F': 2}


def gsi_raw_to_t(raw: float, sex: str) -> int:
    """:param raw: float. Prorated GSI sum score.
    :param sex: str. 'M' or 'F'.
    :return: int. Normalized T-score.
    """
    col_idx: int = SEX_COL_IDX[sex]
    raw_int: int = int(round(raw))
    raw_int = max(0, raw_int)
    if raw_int > 46:
        return T_MAX[col_idx]
    return GSI18_T_TABLE[raw_int][col_idx]


assert df['Sex'].isin(['M', 'F']).all()
df['BSI_GSI_T'] = df.apply(lambda r: gsi_raw_to_t(r['BSI_GSI_prorate'], r['Sex']), axis=1)

logger.info(f'BSI_GSI_T (mean):   {df["BSI_GSI_T"].mean():.1f}')
logger.info(f'BSI_GSI_T (sd):     {df["BSI_GSI_T"].std():.1f}')
logger.info(f'BSI_GSI_T (median): {df["BSI_GSI_T"].median():.1f}')
logger.info(f'BSI_GSI_T (min):    {df["BSI_GSI_T"].min()}')
logger.info(f'BSI_GSI_T (max):    {df["BSI_GSI_T"].max()}')

########################################################################################################################
# Binarize GSI T-score into caseness (Derogatis, 2001: T >= 63)
########################################################################################################################
CASENESS_T: int = 63
for sex, idx in SEX_COL_IDX.items():
    raw_threshold: int = min(k for k, v in GSI18_T_TABLE.items() if v[idx] >= CASENESS_T)
    logger.info(f'T >= {CASENESS_T} corresponds to raw GSI >= {raw_threshold} ({sex})')
logger.print('-'*100)

df['BSI_GSI_Case'] = (df['BSI_GSI_T'] >= CASENESS_T).astype(int)

n_cases: int = df['BSI_GSI_Case'].sum()
n_total: int = len(df)
prevalence: float = 100 * n_cases / n_total
logger.info(f'Caseness (T >= {CASENESS_T}): {n_cases}/{n_total} ({prevalence:.1f}%)')

for sex in ['M', 'F']:
    mask: pd.Series = df['Sex'] == sex
    n_sex: int = mask.sum()
    n_cases_sex: int = df.loc[mask, 'BSI_GSI_Case'].sum()
    logger.info(f'  {sex}: {n_cases_sex}/{n_sex} ({100*n_cases_sex/n_sex:.1f}%)')

n_participants_total: int = df['BeiweId'].nunique()
n_participants_ever_case: int = df.loc[df['BSI_GSI_Case'] == 1, 'BeiweId'].nunique()
logger.info(f'Participants ever case: {n_participants_ever_case}/{n_participants_total} '
            f'({100*n_participants_ever_case/n_participants_total:.1f}%)')
logger.print('-'*100)

########################################################################################################################
# Verify prevalence against published general-population benchmarks
# Michel et al. (2024): GSI-only caseness in Swiss general population (T>=63 on GSI only)
#   Overall: 10.2% (CI: 8.6-12.1%)
#   Male:    10.6% (CI: 8.1-13.7%)
#   Female:  10.5% (CI: 8.3-13.1%)
# Note: our survey asks "today"; Michel et al. normative data based on "past 7 days" recall.
########################################################################################################################
logger.info('--- Prevalence benchmarking (Michel et al., 2024) ---')
logger.info(f'Observed overall prevalence:  {prevalence:.1f}%  (expected: 10.2%, CI: 8.6-12.1%)')
for sex, expected, ci in [('M', 10.6, '8.1-13.7'), ('F', 10.5, '8.3-13.1')]:
    mask: pd.Series = df['Sex'] == sex
    obs: float = 100 * df.loc[mask, 'BSI_GSI_Case'].sum() / mask.sum()
    logger.info(f'Observed {sex} prevalence:        {obs:.1f}%  (expected: {expected}%, CI: {ci}%)')
logger.print('-'*100)

########################################################################################################################
# Visualize the T-score distribution with caseness threshold
########################################################################################################################
fig, ax = plt.subplots(figsize=(10, 5))
bins_t: np.ndarray = np.arange(34.5, 81.5, 1)
ax.hist(df.loc[df['BSI_GSI_Case'] == 0, 'BSI_GSI_T'], bins=bins_t,
        edgecolor='black', linewidth=0.3, color='steelblue', label='Non-case')
ax.hist(df.loc[df['BSI_GSI_Case'] == 1, 'BSI_GSI_T'], bins=bins_t,
        edgecolor='black', linewidth=0.3, color='tomato', label='Case')
ax.axvline(CASENESS_T, linestyle='-', color='red', linewidth=2,
           label=f'T = {CASENESS_T} (caseness threshold)')
ax.axvline(50, linestyle='--', color='gray', linewidth=1.5, label='T = 50 (normative mean)')
ax.set_xlabel('BSI-18 GSI T-score (sex-specific)')
ax.set_ylabel('Count')
# ax.set_title(f'Distribution of BSI-18 GSI T-scores (caseness: {n_cases}/{n_total}, {prevalence:.1f}%)')
ax.set_xlim(34.5, 80.5)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_fig_dir, 'BSI_GSI_T_distribution.png'), dpi=300)
plt.close(fig)

########################################################################################################################
# Export
########################################################################################################################
out_path: str = '_Helper_Files/Outcome_v0.csv'
export_cols: list[str] = ['BeiweId', 'AppleId', 'Survey_Time', 'Survey_Day', 'Sex',
                          'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case']
df[export_cols].to_csv(out_path, index=False)
logger.info(f'Exported {len(df)} records to {out_path}')
logger.info(f'Columns: {export_cols}')
logger.print('-'*100)
