########################################################################################################################
# Preparing survey features
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
from _Helper_Scripts.Logger import create_log

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Specify the input directory
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'

########################################################################################################################
# Load the feature file
########################################################################################################################
data_path: str = os.path.join(IN_DIR, 'Features_v3.parquet')
df: pd.DataFrame = pd.read_parquet(data_path)
logger.print(f'Loaded: {data_path}  Shape: {df.shape}')
logger.print('=' * 100)

########################################################################################################################
# Create a mapping from Apple ID to Beiwe ID
########################################################################################################################
id_map: dict = {i[0]: i[1] for i in df[['AppleId', 'BeiweId']].values}

########################################################################################################################
# Load the survey data
########################################################################################################################
SURVEY_PATH: str = '../00_Data/01_Cleaned_Data/D6_Apple_Baseline_Outcomes/Apple_Baseline_v4.csv'
df_survey: pd.DataFrame = pd.read_csv(SURVEY_PATH)
df_survey = df_survey[df_survey['AppleId'].notna()]
df_survey['BeiweId'] = df_survey['AppleId'].map(id_map)
df_survey = df_survey[df_survey['BeiweId'].notna()].reset_index(drop=True)
df_survey['RecordedDate'] = pd.to_datetime(df_survey['RecordedDate'])
assert sorted(df['AppleId'].unique()) == sorted(df['AppleId'].unique())

########################################################################################################################
# Load the survey data dictionary to clean the data columns
########################################################################################################################
dfs_dict: pd.DataFrame = pd.read_excel('../00_Data/99_Dictionaries/D6_Dictionary_Apple_Baseline.xlsx',
                                       sheet_name=None)
unwanted_cols: list[str] = dfs_dict['Deleted']['Variable_Name'].unique()
df_survey.drop(columns=unwanted_cols, errors='ignore', inplace=True)
dict_cols: list[str] = set(dfs_dict['Cleaned']['Variable_Name'])
df_survey.drop(columns=set(df_survey.columns) - dict_cols, errors='ignore', inplace=True)

########################################################################################################################
# Further filter columns
########################################################################################################################
df_survey.rename(columns={c: c.replace('.0', '') for c in df_survey.columns if c.endswith('.0')}, inplace=True)
other_cols_to_drop: list[str] = ['BirthDate_r', 'ConsentDate_fixed',
                                 'EndDate7Day', 'StartDate7Day', 'EndDate', 'StartDate', 'BirthYear_fixed',
                                 'RiskGroup', 'Duration..in.seconds.', 'Healthcare', 'Continue.screen']
other_cols_to_drop += [c for c in df_survey.columns
                       if c.startswith('Compensation^') or
                       c.startswith('HealthcareProv') or
                       c.endswith('skip') or
                       c.endswith('SKIP')]
other_cols_to_drop += [c for c in df_survey.columns if df_survey[c].notna().sum() == 0]
other_cols_to_drop += [
    # Structural: near-constant column
    'TAPS.2_Q2_Alcohol',
    # Ambiguous definition + vanishing signal
    'Withdrawal_reason^active_inv',
    'Withdrawal_reason^active_ppt',
    'Withdrawal_reason^passive',
    # Redundant with TAPS.2_*Risk aggregates: main substance items
    'TAPS.2_Q1_Tobacco',
    'TAPS.2_Q3_Cannabis',
    'TAPS.2_Q4_Stimulants',
    'TAPS.2_Q5_Heroin',
    'TAPS.2_Q6_Opioids',
    'TAPS.2_Q7_AnxMeds',
    'TAPS.2_Q8_ADHDMeds',
    'TAPS.2_Q9a_OtherDrug',
    # Redundant with TAPS.2_*Risk aggregates: sub-items
    'TAPS.2_Q1sub_Tobacco_1',
    'TAPS.2_Q1sub_Tobacco_2',
    'TAPS.2_Q2sub_Alcohol_1',
    'TAPS.2_Q2sub_Alcohol_2',
    'TAPS.2_Q2sub_Alcohol_3',
    'TAPS.2_Q3sub_Cannab_1',
    'TAPS.2_Q3sub_Cannab_2',
    'TAPS.2_Q4sub_Stimul_1',
    'TAPS.2_Q4sub_Stimul_2',
    'TAPS.2_Q7sub_AnxMeds_1',
    'TAPS.2_Q7sub_AnxMeds_2',
    # Redundant with TAPS.2_OtherDrugRisk
    'TAPS.2_Q9b_MDMA',
    'TAPS.2_Q9b_Psilocybin',
    'TAPS.2_Q9b_Whipits',
    'TAPS.2_Q9b_eCig',
    # Vanishing signal (97.6% NaN, N=20) - near-constant after imputation
    'SUDtx',
]

df_survey.drop(columns=other_cols_to_drop, inplace=True)

########################################################################################################################
# Date-time validation: Survey time is earlier than the outcome time (for at least a day)
########################################################################################################################
df_participant: pd.DataFrame = (df.sort_values(by=['Record_ID', 'bin_index'])
                                .drop_duplicates(subset=['Record_ID'], keep='first'))
diff_time: list = []
for _, row in df_participant.iterrows():
    row_dict = row.to_dict()
    record_id: int = row_dict['Record_ID']
    beiwe_id: str = row_dict['BeiweId']
    event_time = row_dict['Survey_Time']
    survey_dict = df_survey.loc[df_survey['BeiweId'] == beiwe_id].iloc[0].to_dict()
    survey_time = survey_dict['RecordedDate']
    diff_time.append(event_time - survey_time)
    assert (event_time - survey_time) >= pd.Timedelta(days=1)
logger.info(f'Difference between survey and ED time:')
logger.info(f'Mean: {np.mean(diff_time)}')
logger.info(f'Median: {np.median(diff_time)}')
logger.info(f'Max: {np.max(diff_time)}')
logger.info(f'Min: {np.min(diff_time)}')
logger.print('-'*100)

########################################################################################################################
# Update df_survey and check missingness and constant
########################################################################################################################
df_survey.drop(columns=['AppleId', 'RecordedDate'], inplace=True)
df_survey.rename(columns={k: f'{k}@survey' for k in df_survey.columns if k != 'BeiweId'}, inplace=True)
to_drop: list[str] = []
logger.info(f'{df_survey.shape[1]-1} survey features extracted.')
for col_idx, col in enumerate(df_survey.columns):
    if col == 'BeiweId':
        continue
    logger.debug(f'[{col_idx}. {col}] Missingness pct = {df_survey[col].isna().mean()*100:.2f}%')
    if df_survey[col].dropna().nunique() == 1:
        logger.debug(f'[{col_idx}. {col}] Constant detected.')
        to_drop.append(col)
logger.print('-'*100)
logger.info(f'{len(to_drop)} survey features were constant and will be removed.')
df_survey.drop(columns=to_drop, inplace=True)

########################################################################################################################
# Merge df_survey with df, then export
########################################################################################################################
df_out: pd.DataFrame = pd.merge(left=df, right=df_survey, on=['BeiweId'], how='left')
assert len(df_out) == len(df)
assert df_out.shape[1] == df.shape[1] + df_survey.shape[1] - 1
out_path: str = os.path.join(IN_DIR, 'Features_v4.parquet')
df_out.to_parquet(out_path, index=False)
logger.print(f'Saved: {out_path}  Shape: {df_out.shape}  Records: {df_out["Record_ID"].nunique()}')
