########################################################################################################################
# Extract day-based sensor aggregates from accelerometer data
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
import warnings
from _Helper_Scripts.Logger import create_log
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Specify the input directory
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'

########################################################################################################################
# Runtime parameters
########################################################################################################################
MAX_SAMPLES_PER_DAY: int = 720
MIN_COVERAGE_FRAC: float = 0.30
MIN_COVERAGE_SAMPLES: int = int(MIN_COVERAGE_FRAC * MAX_SAMPLES_PER_DAY)
INACTIVE_CADENCE_THRESHOLD: int = 20
STD_VM_THRESHOLD: float = 0.02
SLEEP_MIN_DURATION_MIN: int = 60
MAX_BRIDGE_GAP_MIN: int = 1
MAX_WAKE_GAP_MIN: int = 20
PEAK_CADENCE_N: int = 30
ACTIVE_BOUT_MIN_DURATION: int = 2

logger.info(f'MAX_SAMPLES_PER_DAY = {MAX_SAMPLES_PER_DAY} (~50% duty cycle ceiling)')
logger.info(f'MIN_COVERAGE_FRAC   = {MIN_COVERAGE_FRAC} (>= {MIN_COVERAGE_SAMPLES} observed 1-min bins required)')
logger.print('-'*100)

########################################################################################################################
# Load the latest feature dataset
########################################################################################################################
data_path: str = os.path.join(IN_DIR, 'Features_v1.parquet')
df: pd.DataFrame = pd.read_parquet(data_path)
logger.info(f'Feature dataset loaded with dimension: {df.shape}')
logger.info(f'Number of Beiwe IDs: {df.BeiweId.nunique()}')
logger.info(f'Number of records: {df.Record_ID.nunique()}')
logger.print('-'*100)

########################################################################################################################
# Load accelerometer aggregate file
########################################################################################################################
logger.info('Loading accelerometer aggregate file...')
df_acc: pd.DataFrame = pd.read_csv('_Helper_Files/Sensor_Aggregates_1min/Accelerometer_Aggregate_Overall.csv')
df_acc['MT_time'] = pd.to_datetime(df_acc['MT_time'])
df_acc = df_acc.sort_values(['BeiweId', 'MT_time']).reset_index(drop=True)
logger.info(f'Total rows loaded: {df_acc.shape[0]:,}')
logger.info(f'Rows with acc_availability == 1: {int((df_acc["acc_availability"] == 1).sum()):,}')
logger.info(f'Rows with acc_availability == 0: {int((df_acc["acc_availability"] == 0).sum()):,}')
logger.print('-'*100)

########################################################################################################################
# Identify the background columns of df (101952 rows)
########################################################################################################################
df_bkgd: pd.DataFrame = df[['Record_ID', 'BeiweId', 'AppleId', 'Survey_Day', 'Survey_Time',
                            'BSI_GSI_Case', 'bin_index', 'Bin_Start_Time', 'Bin_End_Time']]

########################################################################################################################
# For each record ('Record_ID'), identify the earliest Bin_Start_Time and the latest Bin_End_Time to define the time
# for daily aggregate data extraction (1062 rows). The window is always a fixed 24h per the new pipeline design.
########################################################################################################################
df_record: pd.DataFrame = (df_bkgd.groupby('Record_ID')
                           .agg(BeiweId=('BeiweId', 'first'),
                                daily_start=('Bin_Start_Time', 'min'),
                                daily_end=('Bin_End_Time', 'max'))
                           .reset_index())
assert len(df_record) == df['Record_ID'].nunique(), 'record count mismatch'
assert ((df_record['daily_end'] - df_record['daily_start']) == pd.Timedelta('24h')).all(), 'expected fixed 24h window'
logger.info(f'Built per-record window index: {df_record.shape}')
logger.print('-'*100)

########################################################################################################################
# Pre-index observed accelerometer rows by BeiweId for fast per-record slicing
########################################################################################################################
df_acc_obs: pd.DataFrame = df_acc[df_acc['acc_availability'] == 1].reset_index(drop=True)
acc_by_beiwe: dict[str, pd.DataFrame] = {bid: g.reset_index(drop=True)
                                         for bid, g in df_acc_obs.groupby('BeiweId')}
logger.info(f'Indexed observed accelerometer data by {len(acc_by_beiwe)} participants')
logger.print('-'*100)

########################################################################################################################
# Helper: detect sleep periods on the 1-min grid
########################################################################################################################


def detect_sleep(df_win_obs: pd.DataFrame,
                 daily_start: pd.Timestamp,
                 daily_end: pd.Timestamp) -> dict:
    """
    :param df_win_obs: pd.DataFrame. Observed accelerometer rows in the daily window.
    :param daily_start: pd.Timestamp. Window start (inclusive).
    :param daily_end: pd.Timestamp. Window end (exclusive).
    :return: dict. Sleep detection features.
    """
    full_grid: pd.DatetimeIndex = pd.date_range(start=daily_start, end=daily_end, freq='1min', inclusive='left')
    df_grid: pd.DataFrame = pd.DataFrame({'MT_time': full_grid})
    df_slim: pd.DataFrame = df_win_obs[['MT_time', 'cadence_spm', 'std_vm']]
    df_merged: pd.DataFrame = df_grid.merge(df_slim, on='MT_time', how='left')

    observed: pd.Series = df_merged['cadence_spm'].notna() & df_merged['std_vm'].notna()
    inactive: pd.Series = (observed &
                           (df_merged['cadence_spm'] < INACTIVE_CADENCE_THRESHOLD) &
                           (df_merged['std_vm'] < STD_VM_THRESHOLD))
    df_merged['status'] = 'unobserved'
    df_merged.loc[observed & ~inactive, 'status'] = 'active'
    df_merged.loc[inactive, 'status'] = 'inactive'

    status_arr: np.ndarray = df_merged['status'].values.copy()
    i: int = 0
    while i < len(status_arr):
        if status_arr[i] == 'unobserved':
            j: int = i
            while j < len(status_arr) and status_arr[j] == 'unobserved':
                j += 1
            gap_len: int = j - i
            before_ok: bool = i > 0 and status_arr[i - 1] == 'inactive'
            after_ok: bool = j < len(status_arr) and status_arr[j] == 'inactive'
            if gap_len <= MAX_BRIDGE_GAP_MIN and before_ok and after_ok:
                status_arr[i:j] = 'inactive'
            i = j
        else:
            i += 1
    status_arr[status_arr == 'unobserved'] = 'active'
    df_merged['status_bridged'] = status_arr

    df_merged['is_inactive'] = (df_merged['status_bridged'] == 'inactive').astype(int)
    df_merged['run_id'] = (df_merged['is_inactive'] != df_merged['is_inactive'].shift()).cumsum()
    inactive_df: pd.DataFrame = df_merged[df_merged['is_inactive'] == 1]

    if inactive_df.empty:
        return {'sleep_detected': False, 'n_sleep_periods': 0,
                'longest_sleep_min': np.nan, 'most_recent_sleep_min': np.nan}

    run_table: pd.DataFrame = (inactive_df.groupby('run_id')
                               .agg(start_idx=('MT_time', 'first'),
                                    end_idx=('MT_time', 'last'),
                                    duration=('is_inactive', 'sum'))
                               .reset_index())
    sleep_candidates: pd.DataFrame = (run_table[run_table['duration'] >= SLEEP_MIN_DURATION_MIN]
                                      .sort_values('start_idx')
                                      .reset_index(drop=True))

    if sleep_candidates.empty:
        return {'sleep_detected': False, 'n_sleep_periods': 0,
                'longest_sleep_min': np.nan, 'most_recent_sleep_min': np.nan}

    merged_sleep: list[dict] = []
    current: dict = sleep_candidates.iloc[0].to_dict()
    for _, nxt in sleep_candidates.iloc[1:].iterrows():
        wake_gap_min: float = (nxt['start_idx'] - current['end_idx']).total_seconds() / 60
        if wake_gap_min <= MAX_WAKE_GAP_MIN:
            current['end_idx'] = nxt['end_idx']
            current['duration'] = (current['end_idx'] - current['start_idx']).total_seconds() / 60 + 1
        else:
            merged_sleep.append(current)
            current = nxt.to_dict()
    merged_sleep.append(current)
    df_merged_sleep: pd.DataFrame = pd.DataFrame(merged_sleep)

    return {'sleep_detected': True,
            'n_sleep_periods': len(df_merged_sleep),
            'longest_sleep_min': float(df_merged_sleep['duration'].max()),
            'most_recent_sleep_min': float(df_merged_sleep.iloc[-1]['duration'])}


########################################################################################################################
# Helper: extract physical activity and behavioral regularity features
########################################################################################################################


def extract_activity_features(df_win_obs: pd.DataFrame) -> dict:
    """
    :param df_win_obs: pd.DataFrame. Observed accelerometer rows in the daily window.
    :return: dict. Activity features (active minutes per obs hr, steps per obs hr, peak cadence, bout stats).
    """
    if df_win_obs.empty:
        return {'active_min_per_obs_hr': np.nan,
                'steps_per_obs_hr': np.nan,
                'peak_cadence_avg': np.nan,
                'active_bout_count': np.nan,
                'longest_active_bout_min': np.nan}

    n_observed: int = len(df_win_obs)
    obs_hours: float = n_observed / 60.0

    cadence: pd.Series = df_win_obs['cadence_spm'].fillna(0)
    is_active: pd.Series = cadence >= INACTIVE_CADENCE_THRESHOLD
    active_min_per_obs_hr: float = int(is_active.sum()) / obs_hours

    steps_total: float = df_win_obs['steps'].fillna(0).sum()
    steps_per_obs_hr: float = steps_total / obs_hours

    cadence_nonzero: pd.Series = cadence[cadence > 0]
    if len(cadence_nonzero) >= PEAK_CADENCE_N:
        peak_cadence_avg: float = float(cadence_nonzero.nlargest(PEAK_CADENCE_N).mean())
    elif len(cadence_nonzero) > 0:
        peak_cadence_avg = float(cadence_nonzero.mean())
    else:
        peak_cadence_avg = 0.0

    active_flags: np.ndarray = (df_win_obs.sort_values('MT_time')['cadence_spm'].fillna(0)
                                >= INACTIVE_CADENCE_THRESHOLD).astype(int).values
    bout_count: int = 0
    longest_bout: int = 0
    current_run: int = 0
    for flag in active_flags:
        if flag == 1:
            current_run += 1
        else:
            if current_run >= ACTIVE_BOUT_MIN_DURATION:
                bout_count += 1
                longest_bout = max(longest_bout, current_run)
            current_run = 0
    if current_run >= ACTIVE_BOUT_MIN_DURATION:
        bout_count += 1
        longest_bout = max(longest_bout, current_run)

    return {'active_min_per_obs_hr': round(active_min_per_obs_hr, 4),
            'steps_per_obs_hr': round(steps_per_obs_hr, 4),
            'peak_cadence_avg': round(peak_cadence_avg, 4),
            'active_bout_count': bout_count,
            'longest_active_bout_min': longest_bout}


########################################################################################################################
# Main loop: extract daily features per record
########################################################################################################################
empty_acc: pd.DataFrame = df_acc_obs.iloc[0:0]
results: list[dict] = []
for _, row in df_record.iterrows():
    record_id: int = int(row['Record_ID'])
    beiwe_id: str = row['BeiweId']
    daily_start: pd.Timestamp = row['daily_start']
    daily_end: pd.Timestamp = row['daily_end']

    df_acc_beiwe: pd.DataFrame = acc_by_beiwe.get(beiwe_id, empty_acc)
    df_win_obs: pd.DataFrame = df_acc_beiwe[(df_acc_beiwe['MT_time'] >= daily_start) &
                                            (df_acc_beiwe['MT_time'] < daily_end)]

    n_observed: int = len(df_win_obs)
    coverage_frac: float = round(n_observed / MAX_SAMPLES_PER_DAY, 4)

    record: dict = {'Record_ID': record_id, 'coverage_frac': coverage_frac}
    if n_observed >= MIN_COVERAGE_SAMPLES:
        record.update(detect_sleep(df_win_obs, daily_start, daily_end))
    else:
        record.update({'sleep_detected': None, 'n_sleep_periods': None,
                       'longest_sleep_min': np.nan, 'most_recent_sleep_min': np.nan})
    record.update(extract_activity_features(df_win_obs))
    results.append(record)

df_daily: pd.DataFrame = pd.DataFrame(results)
df_daily['sleep_detected'] = df_daily['sleep_detected'].map({True: 1, False: 0, None: np.nan})
logger.info(f'Extracted daily features for {len(df_daily)} records')
logger.print('-'*100)

########################################################################################################################
# Summary statistics
########################################################################################################################
logger.info('Coverage:')
logger.info(f'  >= {MIN_COVERAGE_FRAC:.0%}: {int((df_daily["coverage_frac"] >= MIN_COVERAGE_FRAC).sum())} '
            f'({(df_daily["coverage_frac"] >= MIN_COVERAGE_FRAC).mean()*100:.1f}%)')
logger.info(f'  <  {MIN_COVERAGE_FRAC:.0%}: {int((df_daily["coverage_frac"] < MIN_COVERAGE_FRAC).sum())} '
            f'({(df_daily["coverage_frac"] < MIN_COVERAGE_FRAC).mean()*100:.1f}%)')

n_suf: int = int(df_daily['sleep_detected'].notna().sum())
n_det: int = int((df_daily['sleep_detected'] == 1).sum())
logger.info(f'Sleep detected: {n_det}/{n_suf} sufficient-coverage records ({n_det/max(n_suf, 1)*100:.1f}%)')
detected: pd.Series = df_daily.loc[df_daily['sleep_detected'] == 1, 'longest_sleep_min']
if len(detected) > 0:
    logger.info(f'Longest sleep (min): p10={np.percentile(detected, 10):.0f} '
                f'p50={np.percentile(detected, 50):.0f} p90={np.percentile(detected, 90):.0f}')

for col in ['active_min_per_obs_hr', 'steps_per_obs_hr', 'peak_cadence_avg',
            'active_bout_count', 'longest_active_bout_min']:
    non_null: pd.Series = df_daily[col].dropna()
    logger.info(f'{col}: N={len(non_null)}, median={non_null.median():.2f}, '
                f'p10={np.percentile(non_null, 10):.2f}, p90={np.percentile(non_null, 90):.2f}')
logger.print('-'*100)

########################################################################################################################
# Append @daily suffix and broadcast to Features_v1 -> Features_v2
########################################################################################################################
daily_cols: list[str] = [c for c in df_daily.columns if c != 'Record_ID']
df_daily = df_daily.rename(columns={c: f'{c}@daily' for c in daily_cols})

n_rows_before: int = df.shape[0]
df_out: pd.DataFrame = df.merge(df_daily, on='Record_ID', how='left')
assert df_out.shape[0] == n_rows_before, f'row count changed: {n_rows_before} -> {df_out.shape[0]}'
df_out = df_out.sort_values(['Record_ID', 'bin_index']).reset_index(drop=True)
logger.info(f'Merged shape: {df_out.shape} (added {len(daily_cols)} @daily columns)')

########################################################################################################################
# Save
########################################################################################################################
out_path: str = os.path.join(IN_DIR, 'Features_v2.parquet')
df_out.to_parquet(out_path, index=False)
logger.info(f'Wrote {df_out.shape[0]} rows x {df_out.shape[1]} cols to {out_path}')
logger.print('-'*100)
