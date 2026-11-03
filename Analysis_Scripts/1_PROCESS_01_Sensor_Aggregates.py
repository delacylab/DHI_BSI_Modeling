########################################################################################################################
# Build a fixed-length (96 x 15-min) long-format sensor-feature table per survey record, over a dynamic 24h lookback,
# at four aggregation resolutions broadcast onto a shared 15-min grid. Output: a single wide-per-resolution parquet.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os
import pandas as pd
import warnings
from datetime import timedelta
from typing import Optional
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Accelerometer_Aggregate import aggregate_acc
from _Helper_Scripts.GPS_Aggregate import aggregate_gps
from _Helper_Scripts.PowerState_Aggregate import aggregate_ps
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Constants
########################################################################################################################
LOOKBACK_MAX: str = '24h'
BIN_WIDTH: str = '15min'
RESOL_LIST: list[str] = ['15min', '1h', '6h', '24h']
N_BINS: int = 96
OUTCOME: str = 'BSI_GSI_Case'

OUTCOME_PATH: str = '_Helper_Files/Outcome_v0.csv'
HOME_PATH: str = '_Helper_Files/Home_Coordinates.csv'
RAW_DIR: str = '../00_Data/01_Cleaned_Data/D3_Beiwe_Raw_Data'
OUT_DIR: str = f'_Helper_Files/{BIN_WIDTH}_Data'
OUT_PATH: str = os.path.join(OUT_DIR, 'Features_v0.parquet')

ACC_FEATS: list[str] = ['steps', 'walking_time_s', 'cadence_spm', 'n_samples', 'longest_gap_s',
                        'mean_x', 'std_x', 'mean_y', 'std_y', 'mean_z', 'std_z',
                        'mean_vm', 'median_vm', 'max_vm', 'std_vm', 'iqr_vm', 'energy_vm', 'entropy_vm']
GPS_FEATS: list[str] = ['n_points', 'duration_sec', 'total_path_length_m', 'max_step_distance_m',
                        'mean_speed_mps', 'max_speed_mps', 'centroid_distance_mean_m', 'centroid_distance_max_m',
                        'mean_dist_home_m', 'max_dist_home_m', 'home_fraction']
PS_FEATS: list[str] = ['unlock_count', 'lock_count', 'screen_on_fraction', 'plugged_fraction',
                       'plug_event_count', 'unplug_event_count', 'state_changes_count',
                       'battery_level_mean', 'n_events']

# PS coarse-resolution rollup classes (exact reconstruction from the 15-min pass)
PS_SUM: list[str] = ['unlock_count', 'lock_count', 'plug_event_count', 'unplug_event_count',
                     'state_changes_count', 'n_events']
PS_NMIN_WEIGHTED: list[str] = ['screen_on_fraction', 'plugged_fraction']
PS_NEVENTS_WEIGHTED: list[str] = ['battery_level_mean']

STREAM_COLS: dict[str, list[str]] = {'accelerometer': ['x', 'y', 'z'],
                                     'gps': ['latitude', 'longitude'],
                                     'power_state': ['event', 'level']}

########################################################################################################################
# Validate resolutions: each a multiple of BIN_WIDTH and a divisor of LOOKBACK_MAX
########################################################################################################################
_BW: pd.Timedelta = pd.Timedelta(BIN_WIDTH)
_LB: pd.Timedelta = pd.Timedelta(LOOKBACK_MAX)
assert int(_LB / _BW) == N_BINS, 'N_BINS must equal LOOKBACK_MAX / BIN_WIDTH'
for _r in RESOL_LIST:
    _rt: pd.Timedelta = pd.Timedelta(_r)
    assert (_rt / _BW).is_integer(), f'{_r} is not a multiple of {BIN_WIDTH}'
    assert (_LB / _rt).is_integer(), f'{_r} does not divide {LOOKBACK_MAX}'


########################################################################################################################
# Build the ordered column manifest (Groups A-G of the spec)
########################################################################################################################
def build_columns() -> list[str]:
    """
    :return: list[str]. The 181 output columns in on-disk order.
    """
    cols: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                       'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case', 'with_home_address']
    cols += ['bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']
    cols += ['ACC#availability', 'GPS#availability', 'PS#availability']
    for f in ACC_FEATS:
        cols += [f'ACC#{f}^{r}' for r in RESOL_LIST]
    for f in GPS_FEATS:
        cols += [f'GPS#{f}^{r}' for r in RESOL_LIST]
    for f in PS_FEATS:
        cols += [f'PS#{f}^{r}' for r in RESOL_LIST]
    for s in ['ACC', 'GPS', 'PS']:
        cols += [f'{s}#n_min^{r}' for r in RESOL_LIST]
    return cols


COLUMNS: list[str] = build_columns()
assert len(COLUMNS) == 181, f'expected 181 columns, got {len(COLUMNS)}'
FEATURE_COLS: list[str] = [c for c in COLUMNS if '^' in c and '#n_min^' not in c]
NMIN_COLS: list[str] = [c for c in COLUMNS if '#n_min^' in c]
AVAIL_COLS: list[str] = ['ACC#availability', 'GPS#availability', 'PS#availability']


########################################################################################################################
# Grid / window helpers
########################################################################################################################
def get_considered_start(survey_t_cur: pd.Timestamp, survey_t_prior: Optional[pd.Timestamp]) -> pd.Timestamp:
    """
    :param survey_t_cur: pd.Timestamp. Current survey time (grid right-anchor).
    :param survey_t_prior: Optional[pd.Timestamp]. Latest survey on the previous day, or None.
    :return: pd.Timestamp. Grid-aligned start of the considered window (<= raw start, keeps <=1 pre-cutoff bin).
    """
    lookback: pd.Timedelta = _LB
    if survey_t_prior is not None:
        lookback = min(lookback, survey_t_cur - survey_t_prior)
    k: int = int(np.ceil(lookback / _BW))
    k = min(k, N_BINS)
    return survey_t_cur - k * _BW


def block_broadcast(values: np.ndarray, m: int) -> np.ndarray:
    """
    :param values: np.ndarray. Per-block values, ordered oldest->newest, length N_BINS // m.
    :param m: int. Bins per block (resol / BIN_WIDTH).
    :return: np.ndarray. Length N_BINS, each block value repeated across its m sub-bins.
    """
    idx: np.ndarray = np.arange(N_BINS) // m
    return values[idx]


def rollup_sum(vals15: np.ndarray, m: int) -> np.ndarray:
    """
    :param vals15: np.ndarray. 15-min values, length N_BINS.
    :param m: int. Bins per block.
    :return: np.ndarray. Length N_BINS, block sum broadcast across sub-bins.
    """
    blocks: np.ndarray = np.nansum(vals15.reshape(N_BINS // m, m), axis=1)
    return block_broadcast(blocks, m)


def rollup_weighted(vals15: np.ndarray, weights15: np.ndarray, m: int) -> np.ndarray:
    """
    :param vals15: np.ndarray. 15-min values, length N_BINS (may contain NaN where weight is 0).
    :param weights15: np.ndarray. 15-min weights, length N_BINS.
    :param m: int. Bins per block.
    :return: np.ndarray. Length N_BINS, weight-weighted mean per block broadcast across sub-bins; NaN if block weight 0.
    """
    v: np.ndarray = vals15.reshape(N_BINS // m, m)
    w: np.ndarray = weights15.reshape(N_BINS // m, m)
    num: np.ndarray = np.nansum(np.where(np.isnan(v), 0.0, v) * w, axis=1)
    den: np.ndarray = np.nansum(w, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        blocks: np.ndarray = np.where(den > 0, num / den, np.nan)
    return block_broadcast(blocks, m)


def stream_nmin(first_ts: Optional[pd.Timestamp], last_ts: Optional[pd.Timestamp],
                considered_start: pd.Timestamp, survey_t_cur: pd.Timestamp,
                grid_start: pd.Timestamp, resol: str) -> np.ndarray:
    """
    Nominal covered minutes per block for ACC / GPS: overlap of each block with the considered window and the
    stream's [first_ts, last_ts] sample span. Duty-cycle gaps between first and last are counted as covered.
    :param first_ts: Optional[pd.Timestamp]. First in-window raw sample, or None if the stream is absent.
    :param last_ts: Optional[pd.Timestamp]. Last in-window raw sample, or None.
    :param considered_start: pd.Timestamp. Start of the considered window.
    :param survey_t_cur: pd.Timestamp. End of the considered window (exclusive).
    :param grid_start: pd.Timestamp. Oldest master-grid edge (survey_t_cur - 24h).
    :param resol: str. Aggregation resolution.
    :return: np.ndarray. Length N_BINS, covered minutes broadcast across each block's sub-bins.
    """
    nmin: np.ndarray = np.zeros(N_BINS, dtype=float)
    if first_ts is None or last_ts is None:
        return nmin
    r_td: pd.Timedelta = pd.Timedelta(resol)
    m: int = int(r_td / _BW)
    lo_win: pd.Timestamp = max(considered_start, first_ts)
    hi_win: pd.Timestamp = min(survey_t_cur, last_ts)
    blocks: np.ndarray = np.zeros(N_BINS // m, dtype=float)
    for j in range(N_BINS // m):
        b_start: pd.Timestamp = grid_start + j * r_td
        b_end: pd.Timestamp = b_start + r_td
        lo: pd.Timestamp = max(b_start, lo_win)
        hi: pd.Timestamp = min(b_end, hi_win)
        cover_s: float = (hi - lo).total_seconds()
        blocks[j] = max(0.0, cover_s) / 60.0
    return block_broadcast(blocks, m)


########################################################################################################################
# Raw stream loading (UTC -> Mountain wall-clock, tz-naive)
########################################################################################################################
def load_stream(beiwe_id: str, stream: str) -> pd.DataFrame:
    """
    :param beiwe_id: str. Participant id.
    :param stream: str. One of 'accelerometer', 'gps', 'power_state'.
    :return: pd.DataFrame. Columns ['MT_time'] + stream value columns; empty if the file is missing.
    """
    value_cols: list[str] = STREAM_COLS[stream]
    path: str = os.path.join(RAW_DIR, stream, f'{beiwe_id}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['MT_time'] + value_cols)
    d: pd.DataFrame = pd.read_parquet(path)
    d['MT_time'] = pd.to_datetime(d['UTC time'], utc=True).dt.tz_convert('America/Denver').dt.tz_localize(None)
    d = d[['MT_time'] + value_cols].sort_values('MT_time').drop_duplicates('MT_time').reset_index(drop=True)
    return d


########################################################################################################################
# Per-record assembly
########################################################################################################################
def process_record(rec: pd.Series,
                   acc_df: pd.DataFrame,
                   gps_df: pd.DataFrame,
                   ps_df: pd.DataFrame,
                   home_coords: Optional[tuple[float, float]]) -> pd.DataFrame:
    """
    :param rec: pd.Series. One outcome record (Record_ID, BeiweId, Survey_Time, survey_t_prior, label, ...).
    :param acc_df: pd.DataFrame. Participant accelerometer stream (MT_time, x, y, z).
    :param gps_df: pd.DataFrame. Participant GPS stream (MT_time, latitude, longitude).
    :param ps_df: pd.DataFrame. Participant power-state stream (MT_time, event, level).
    :param home_coords: Optional[tuple[float, float]]. Participant home (lat, lon) or None.
    :return: pd.DataFrame. 96 rows x 181 columns in COLUMNS order.
    """
    survey_t_cur: pd.Timestamp = rec['Survey_Time']
    survey_t_prior: Optional[pd.Timestamp] = rec['survey_t_prior']
    grid_start: pd.Timestamp = survey_t_cur - _LB
    considered_start: pd.Timestamp = get_considered_start(survey_t_cur, survey_t_prior)
    bin_start: pd.DatetimeIndex = pd.date_range(grid_start, periods=N_BINS, freq=BIN_WIDTH)

    # Confine ACC / GPS to the considered window; PS keeps full history (its ffill needs pre-window events)
    acc_win: pd.DataFrame = acc_df[(acc_df['MT_time'] >= considered_start) & (acc_df['MT_time'] < survey_t_cur)]
    gps_win: pd.DataFrame = gps_df[(gps_df['MT_time'] >= considered_start) & (gps_df['MT_time'] < survey_t_cur)]
    acc_first: Optional[pd.Timestamp] = acc_win['MT_time'].min() if len(acc_win) else None
    acc_last: Optional[pd.Timestamp] = acc_win['MT_time'].max() if len(acc_win) else None
    gps_first: Optional[pd.Timestamp] = gps_win['MT_time'].min() if len(gps_win) else None
    gps_last: Optional[pd.Timestamp] = gps_win['MT_time'].max() if len(gps_win) else None

    data: dict[str, np.ndarray] = {}

    # ACC: per-resolution aggregate, right-anchored (anchor at grid_start; 24h divisible by resol), then broadcast
    for r in RESOL_LIST:
        m: int = int(pd.Timedelta(r) / _BW)
        acc_r: pd.DataFrame = aggregate_acc(acc_win, start_time=grid_start, end_time=survey_t_cur, resol=r)
        for f in ACC_FEATS:
            data[f'ACC#{f}^{r}'] = block_broadcast(acc_r[f].to_numpy(dtype=float), m)
        if r == BIN_WIDTH:
            data['ACC#availability'] = acc_r['acc_availability'].to_numpy()
        data[f'ACC#n_min^{r}'] = stream_nmin(acc_first, acc_last, considered_start, survey_t_cur, grid_start, r)

    # GPS: same pattern, with home coordinates
    for r in RESOL_LIST:
        m = int(pd.Timedelta(r) / _BW)
        gps_r: pd.DataFrame = aggregate_gps(gps_win, home_coords=home_coords,
                                            start_time=grid_start, end_time=survey_t_cur, resol=r)
        for f in GPS_FEATS:
            data[f'GPS#{f}^{r}'] = block_broadcast(gps_r[f].to_numpy(dtype=float), m)
        if r == BIN_WIDTH:
            data['GPS#availability'] = gps_r['gps_availability'].to_numpy()
        data[f'GPS#n_min^{r}'] = stream_nmin(gps_first, gps_last, considered_start, survey_t_cur, grid_start, r)

    # PS: single 15-min pass anchored at considered_start, aligned to the master grid, coarse resolutions rolled up
    ps_15: pd.DataFrame = aggregate_ps(ps_df, start_time=considered_start, end_time=survey_t_cur, resol=BIN_WIDTH)
    ps_15 = ps_15.set_index('MT_time').reindex(bin_start)
    data['PS#availability'] = ps_15['ps_availability'].fillna(0).to_numpy(dtype=int)
    ps_nmin15: np.ndarray = ps_15['n_min'].fillna(0).to_numpy(dtype=float)
    ps_nevents15: np.ndarray = ps_15['n_events'].fillna(0).to_numpy(dtype=float)
    for f in PS_FEATS:
        base: np.ndarray = ps_15[f].to_numpy(dtype=float)
        for r in RESOL_LIST:
            m = int(pd.Timedelta(r) / _BW)
            if f in PS_SUM:
                data[f'PS#{f}^{r}'] = rollup_sum(np.nan_to_num(base, nan=0.0), m)
            elif f in PS_NMIN_WEIGHTED:
                data[f'PS#{f}^{r}'] = rollup_weighted(base, ps_nmin15, m)
            else:
                data[f'PS#{f}^{r}'] = rollup_weighted(base, ps_nevents15, m)
    for r in RESOL_LIST:
        m = int(pd.Timedelta(r) / _BW)
        data[f'PS#n_min^{r}'] = rollup_sum(ps_nmin15, m)

    # is_pad: nothing observed on any stream in the bin (PS observed == non-stale ffilled state present ==
    # PS#n_min^15min > 0). This set is a superset of the structural pad region (bins older than considered_start,
    # where every stream is empty by construction), so masking on is_pad also blanks that region.
    is_pad: np.ndarray = ((data['ACC#availability'] == 0) &
                          (data['GPS#availability'] == 0) &
                          (data['PS#n_min^15min'] == 0))

    # Blank every unobserved bin: features -> NaN, exposure -> 0, availability -> 0
    for c in FEATURE_COLS:
        data[c][is_pad] = np.nan
    for c in NMIN_COLS:
        data[c] = np.where(is_pad, 0.0, data[c])
    for c in AVAIL_COLS:
        data[c] = np.where(is_pad, 0, data[c])
    data['is_pad'] = is_pad.astype(int)

    # Round nominal-minute counts to integers
    for c in NMIN_COLS:
        data[c] = np.rint(data[c]).astype(int)

    # Grid position and record-level columns (broadcast across the 96 bins)
    data['bin_index'] = np.arange(N_BINS)
    data['Bin_Start_Time'] = bin_start
    data['Bin_End_Time'] = bin_start + _BW
    data['Record_ID'] = np.full(N_BINS, rec['Record_ID'])
    data['BeiweId'] = np.full(N_BINS, rec['BeiweId'], dtype=object)
    data['AppleId'] = np.full(N_BINS, rec['AppleId'], dtype=object)
    data['Sex'] = np.full(N_BINS, rec['Sex'], dtype=object)
    data['Survey_Day'] = np.full(N_BINS, rec['Survey_Day'])
    data['Survey_Time'] = np.full(N_BINS, np.datetime64(survey_t_cur))
    data['BSI_GSI_prorate'] = np.full(N_BINS, rec['BSI_GSI_prorate'])
    data['BSI_GSI_T'] = np.full(N_BINS, rec['BSI_GSI_T'])
    data['BSI_GSI_Case'] = np.full(N_BINS, rec['BSI_GSI_Case'])
    data['with_home_address'] = np.full(N_BINS, int(home_coords is not None))

    return pd.DataFrame(data)[COLUMNS]


########################################################################################################################
# Load outcome, assign Record_ID, precompute prior-day survey time
########################################################################################################################
df: pd.DataFrame = pd.read_csv(OUTCOME_PATH)
df['Survey_Time'] = pd.to_datetime(df['Survey_Time'])
df = df.loc[df[OUTCOME].notna()].copy()
df['Survey_Date'] = df['Survey_Time'].dt.date
df = df.sort_values(['BeiweId', 'Survey_Time']).reset_index(drop=True)
df['Record_ID'] = np.arange(1, len(df) + 1)

latest_by_day: dict[tuple[str, object], pd.Timestamp] = (
    df.groupby(['BeiweId', 'Survey_Date'])['Survey_Time'].max().to_dict())


def _prior(row: pd.Series) -> Optional[pd.Timestamp]:
    return latest_by_day.get((row['BeiweId'], row['Survey_Date'] - timedelta(days=1)))


df['survey_t_prior'] = df.apply(_prior, axis=1)
logger.info(f'Loaded {len(df)} records; {df["BeiweId"].nunique()} participants; '
            f'{int(df["survey_t_prior"].notna().sum())} with a prior-day survey')

########################################################################################################################
# Load home coordinates
########################################################################################################################
df_home: pd.DataFrame = pd.read_csv(HOME_PATH)
df_home = df_home[df_home['Status'].isin(['Complete', 'No_Geo'])]
assert 'BeiweId' in df_home.columns, 'Home_Coordinates.csv must contain a BeiweId column'
home_map: dict[str, tuple[float, float]] = {r['BeiweId']: (float(r['Home_Latitude_Fuzzed']),
                                                           float(r['Home_Longitude_Fuzzed']))
                                            for _, r in df_home.iterrows()}

########################################################################################################################
# Validate the raw-data location (fail fast rather than silently emit an all-pad table)
########################################################################################################################
for stream in STREAM_COLS:
    stream_dir: str = os.path.join(RAW_DIR, stream)
    if not os.path.isdir(stream_dir):
        raise FileNotFoundError(f'Raw stream directory not found: {os.path.abspath(stream_dir)} '
                                f'(RAW_DIR={RAW_DIR!r}). Check the path.')
    n_files: int = len([f for f in os.listdir(stream_dir) if f.endswith('.parquet')])
    logger.info(f'{stream}: {n_files} parquet files at {os.path.abspath(stream_dir)}')
    if n_files == 0:
        raise FileNotFoundError(f'No parquet files under {os.path.abspath(stream_dir)}. Check the path.')

########################################################################################################################
# Main loop: per participant, load streams once; per record, assemble 96 rows
########################################################################################################################
frames: list[pd.DataFrame] = []
for beiwe_id, df_p in df.groupby('BeiweId', sort=False):
    acc_df: pd.DataFrame = load_stream(beiwe_id, 'accelerometer')
    gps_df: pd.DataFrame = load_stream(beiwe_id, 'gps')
    ps_df: pd.DataFrame = load_stream(beiwe_id, 'power_state')
    home_coords: Optional[tuple[float, float]] = home_map.get(beiwe_id)
    logger.info(f'{beiwe_id}: {len(df_p)} records; '
                f'acc={len(acc_df)} gps={len(gps_df)} ps={len(ps_df)}; home={home_coords is not None}')
    for _, rec in df_p.iterrows():
        frames.append(process_record(rec, acc_df, gps_df, ps_df, home_coords))

########################################################################################################################
# Concatenate, validate, write
########################################################################################################################
out: pd.DataFrame = pd.concat(frames, ignore_index=True)
assert list(out.columns) == COLUMNS, 'output column order does not match the manifest'
assert out.shape == (N_BINS * len(df), len(COLUMNS)), f'unexpected output shape {out.shape}'
os.makedirs(OUT_DIR, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)
logger.info(f'Wrote {out.shape[0]} rows x {out.shape[1]} cols to {OUT_PATH}')
logger.print('-' * 100)