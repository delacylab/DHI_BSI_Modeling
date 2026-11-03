########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from typing import Optional

########################################################################################################################
# Define the function to compute aggregate gait metrics from raw accelerometer data at a specified resolution
########################################################################################################################


def aggregate_acc(df: pd.DataFrame,
                  time_col: str = 'MT_time',
                  cols_xyz: tuple[str, str, str] = ('x', 'y', 'z'),
                  sampling_freq: float = 10.0,       # Known Beiwe within-burst frequency
                  burst_gap_sec: float = 5.0,        # Gap separating bursts
                  lowpass_cutoff_hz: float = 2.5,    # Low-pass filter cutoff
                  peak_min_dist_s: float = 0.35,     # Minimum time between steps (~170 spm max)
                  peak_prominence: float = 0.05,     # Very relaxed — allows low-movement bursts
                  n_entropy_bins: int = 20,          # Number of bins for Shannon entropy of VM
                  resol: str = '1min',               # Aggregation bin width (e.g. '1min', '5min', '1h', '2h')
                  start_time: Optional[pd.Timestamp] = None,    # Timestamps to filter the df
                  end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Aggregate and summarize raw accelerometer data into bin-level (gait) metrics at the specified resolution.

    When start_time / end_time are provided, the lookback window is [start_time, end_time), and bins are anchored
    at start_time and step forward by resol: bin starts are {start_time + k * resol : k = 0, 1, ...}, giving
    exactly (end_time - start_time) / resol bins. When start_time is None, bins use absolute wall-clock alignment
    (legacy standalone behaviour).

    Burst-spanning policy: bursts that straddle bin boundaries are filtered and peak-detected as a single contiguous
    signal (to preserve filter context and step detection accuracy), but the resulting samples and detected steps
    are attributed to bins by their own timestamps. A bin with any samples in it has acc_availability = 1 and a
    full set of computed features, regardless of which bin the burst started in.

    :param df: Raw accelerometer DataFrame
    :param time_col: Name of the timestamp column
    :param cols_xyz: Tuple of (x, y, z) column names
    :param sampling_freq: Sampling frequency in Hz (default 10 Hz for Beiwe)
    :param burst_gap_sec: Inter-sample gap (seconds) that defines a new burst
    :param lowpass_cutoff_hz: Cutoff frequency for the low-pass Butterworth filter
    :param peak_min_dist_s: Minimum distance between peaks in seconds
    :param peak_prominence: Minimum prominence of a peak (very relaxed by default)
    :param n_entropy_bins: Number of bins used in computing Shannon entropy
    :param resol: Aggregation bin width as a pandas offset alias (default '1min')
    :param start_time: [Optional] Inclusive start of the lookback window
    :param end_time: [Optional] Exclusive end of the lookback window
    :return:
    A pandas.DataFrame recording the bin-based summary and gait metrics computed from the raw accelerometer data
    """
    if (start_time is None) != (end_time is None):
        raise ValueError('start_time and end_time must both be provided or both be None.')

    # Window-anchored bin helper
    resol_td = pd.Timedelta(resol)
    anchor = pd.Timestamp(start_time) if start_time is not None else None

    def _bin_start(x):
        if anchor is None:
            if isinstance(x, pd.Series):
                return x.dt.floor(resol)
            return x.floor(resol)
        return anchor + ((x - anchor) // resol_td) * resol_td

    # Define a helper function to reindex the output DataFrame
    def _reindex(out_: pd.DataFrame, grid_start_: pd.Timestamp, grid_end_: pd.Timestamp):
        full_index = pd.date_range(start=grid_start_, end=grid_end_, freq=resol, inclusive='left')
        result = out_.set_index(time_col).reindex(full_index).rename_axis(time_col).reset_index()
        result['acc_availability'] = result['acc_availability'].fillna(0).astype(int)
        return result

    # Define the feature columns
    feat_cols: list[str] = ['acc_availability',
                            'steps', 'walking_time_s', 'cadence_spm',
                            'n_samples', 'longest_gap_s',
                            'mean_x', 'std_x',
                            'mean_y', 'std_y',
                            'mean_z', 'std_z',
                            'mean_vm', 'median_vm', 'max_vm', 'std_vm', 'iqr_vm',
                            'energy_vm', 'entropy_vm']

    # Define an empty output
    empty_out = pd.DataFrame(columns=[time_col] + feat_cols)

    # Immediate exit if the input DataFrame (df) is empty
    if start_time is not None:
        grid_start = pd.Timestamp(start_time)
        grid_end = pd.Timestamp(end_time)
        if len(df) == 0:
            return _reindex(empty_out, grid_start, grid_end)
    elif len(df) == 0:
        return pd.DataFrame()

    # Extract the needed columns of df
    x_col, y_col, z_col = cols_xyz
    required_cols = [time_col, x_col, y_col, z_col]
    df = df[required_cols].copy()

    # Numeric coercion
    for c in cols_xyz:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Parse and sort timestamps
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # Subset to time window if provided ([start_time, end_time), half-open)
    if start_time is not None:
        df = df[(df[time_col] >= start_time) & (df[time_col] < end_time)].reset_index(drop=True)

    # Another immediate exit after timestamp filtering
    if df.empty:
        if start_time is not None:
            return _reindex(empty_out, grid_start, grid_end)
        return pd.DataFrame()

    # If not start_time or end_time, derive grid from data
    if start_time is None:
        grid_start = df[time_col].min().normalize()
        grid_end = df[time_col].max().normalize() + pd.Timedelta(days=1)

    # Burst segmentation
    dt = df[time_col].diff().dt.total_seconds().fillna(0)
    df['seg_id'] = (dt > burst_gap_sec).cumsum()

    # Low-pass Butterworth filter coefficients
    nyq = sampling_freq / 2.0
    b, a = butter(N=4, Wn=lowpass_cutoff_hz / nyq, btype='low')
    min_dist_samples = max(1, int(peak_min_dist_s * sampling_freq))

    # Helper function to compute Shannon entropy
    def _shannon_entropy(arr: np.ndarray, bins: int) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return np.nan
        hist, _ = np.histogram(arr, bins=bins, density=False)
        s = hist.sum()
        if s <= 0:
            return np.nan
        p = hist.astype(float) / s
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    # Tag every sample with its target bin BEFORE iterating bursts — used for sample-level attribution
    df['_bin'] = _bin_start(df[time_col])

    # Each (burst, bin) overlap will produce one partial record; multi-burst overlaps in the same bin
    # are reconciled in the final groupby below.
    minute_records = []

    for _, seg in df.groupby('seg_id'):
        seg = seg.reset_index(drop=True)

        # Burst-level signal: vector magnitude, low-pass filter, peak detection — done on the full burst
        # so the filter has its full context and peak detection isn't disrupted by bin boundaries.
        x = seg[x_col].to_numpy(dtype=float)
        y = seg[y_col].to_numpy(dtype=float)
        z = seg[z_col].to_numpy(dtype=float)
        vm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        try:
            vm_f = filtfilt(b, a, vm)
        except Exception:
            vm_f = vm
        peak_indices, _ = find_peaks(vm_f, distance=min_dist_samples, prominence=peak_prominence)

        # Attribute each detected peak to a bin via its sample's timestamp
        if len(peak_indices) > 0:
            peak_bins = seg['_bin'].iloc[peak_indices].to_numpy()
        else:
            peak_bins = np.array([], dtype='datetime64[ns]')

        # Attach VM to the burst frame so per-bin groupby can pull it
        seg = seg.assign(_vm=vm)

        # Split the burst by bin and compute per-bin features from samples in each bin
        for bin_label, sub in seg.groupby('_bin'):

            # Time-based stats from samples in this bin
            t_sub = sub[time_col].to_numpy(dtype='datetime64[ns]')
            dt_sub = np.diff(t_sub.astype('int64')) / 1e9
            longest_gap_s = float(np.max(dt_sub)) if dt_sub.size > 0 else np.nan
            rec = {time_col: bin_label, 'longest_gap_s': longest_gap_s}

            # Axis stats
            for col_name, axis_name in [(x_col, 'x'), (y_col, 'y'), (z_col, 'z')]:
                a_arr = sub[col_name].to_numpy(dtype=float)
                a_arr = a_arr[np.isfinite(a_arr)]
                rec[f'mean_{axis_name}'] = float(np.mean(a_arr)) if a_arr.size >= 1 else np.nan
                rec[f'std_{axis_name}'] = float(np.std(a_arr, ddof=1)) if a_arr.size >= 2 else np.nan

            # VM stats from samples in this bin
            vm_sub = sub['_vm'].to_numpy(dtype=float)
            vm_clean = vm_sub[np.isfinite(vm_sub)]
            n = int(vm_clean.size)
            rec['n_samples'] = n

            if n == 0:
                rec.update({'mean_vm': np.nan, 'median_vm': np.nan, 'max_vm': np.nan,
                            'std_vm': np.nan, 'iqr_vm': np.nan,
                            'energy_vm': np.nan, 'entropy_vm': np.nan})
            else:
                q25, q75 = np.nanpercentile(vm_clean, [25, 75])
                rec.update({'mean_vm': float(np.mean(vm_clean)),
                            'median_vm': float(np.median(vm_clean)),
                            'max_vm': float(np.max(vm_clean)),
                            'std_vm': float(np.std(vm_clean, ddof=1)) if n >= 2 else np.nan,
                            'iqr_vm': float(q75 - q25),
                            'energy_vm': float(np.mean(vm_clean ** 2)),
                            'entropy_vm': _shannon_entropy(vm_clean, bins=n_entropy_bins)})

            # Steps in this bin: count peaks whose sample falls in this bin
            steps_bin = int(np.sum(peak_bins == bin_label)) if peak_bins.size > 0 else 0

            # Walking time in this bin: time span of samples in this bin
            if steps_bin > 0 and len(sub) >= 2:
                walking_time_s = (sub[time_col].iloc[-1] - sub[time_col].iloc[0]).total_seconds()
                cadence_spm = 60.0 * steps_bin / walking_time_s if walking_time_s > 0 else 0.0
            else:
                walking_time_s, cadence_spm = 0.0, 0.0

            rec.update({'steps': steps_bin, 'walking_time_s': walking_time_s, 'cadence_spm': cadence_spm})
            minute_records.append(rec)

    if not minute_records:
        return _reindex(empty_out, grid_start, grid_end)

    # Aggregate (burst, bin) records that share a bin
    out = (pd.DataFrame(minute_records)
           .groupby(time_col, as_index=False)
           .agg(steps=('steps', 'sum'),
                walking_time_s=('walking_time_s', 'sum'),
                n_samples=('n_samples', 'sum'),
                longest_gap_s=('longest_gap_s', 'max'),
                mean_x=('mean_x', 'mean'),
                std_x=('std_x', 'mean'),
                mean_y=('mean_y', 'mean'),
                std_y=('std_y', 'mean'),
                mean_z=('mean_z', 'mean'),
                std_z=('std_z', 'mean'),
                mean_vm=('mean_vm', 'mean'),
                median_vm=('median_vm', 'mean'),
                max_vm=('max_vm', 'max'),
                std_vm=('std_vm', 'mean'),
                iqr_vm=('iqr_vm', 'mean'),
                energy_vm=('energy_vm', 'mean'),
                entropy_vm=('entropy_vm', 'mean')))
    out['cadence_spm'] = np.where(out['walking_time_s'] > 0, 60.0 * out['steps'] / out['walking_time_s'], 0)
    out['acc_availability'] = 1
    out = out[[time_col] + feat_cols]
    return _reindex(out, grid_start, grid_end)
