########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
from typing import Optional

########################################################################################################################
# Define the function to compute aggregate metrics from raw power state data at a specified resolution
########################################################################################################################


def aggregate_ps(df: pd.DataFrame,
                 time_col: str = 'MT_time',
                 event_col: str = 'event',
                 level_col: str = 'level',
                 resol: str = '1min',
                 state_max_stale: str = '12h',
                 start_time: Optional[pd.Timestamp] = None,
                 end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Aggregate and summarize raw power-state data into bin-level metrics at the specified resolution.

    When start_time / end_time are provided, the lookback window is [start_time, end_time), and bins are anchored
    at start_time and step forward by resol: bin starts are {start_time + k * resol : k = 0, 1, ...}, giving
    exactly (end_time - start_time) / resol bins. When start_time is None, bins use absolute wall-clock alignment
    (legacy standalone behaviour).

    :param df: Raw power-state DataFrame
    :param time_col: Name of the timestamp column
    :param event_col: Name of the event column
    :param level_col: Name of the battery level column
    :param resol: Aggregation bin width as a pandas offset alias (default '1min')
    :param state_max_stale: Max time since the last observed event for ffilled screen / plug states to remain trusted
                            (default '12h'); beyond this, screen_on_fraction / plugged_fraction are returned as NaN
    :param start_time: [Optional] Inclusive start of the lookback window
    :param end_time: [Optional] Exclusive end of the lookback window
    :return:
    A pandas.DataFrame recording the bin-based summary metrics computed from the raw power-state data
    """

    # Validate start_time / end_time are provided as a pair
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

    # Define the feature columns
    feat_cols: list[str] = ['ps_availability',
                            'unlock_count',
                            'lock_count',
                            'screen_on_fraction',
                            'plugged_fraction',
                            'plug_event_count',
                            'unplug_event_count',
                            'state_changes_count',
                            'battery_level_mean',
                            'n_events',
                            'n_min']

    # Define a helper function to reindex the output DataFrame
    def _reindex(out_: pd.DataFrame, grid_start_: pd.Timestamp, grid_end_: pd.Timestamp) -> pd.DataFrame:
        full_index = pd.date_range(start=grid_start_, end=grid_end_, freq=resol, inclusive='left')
        result = out_.set_index(time_col).reindex(full_index).rename_axis(time_col).reset_index()
        result['ps_availability'] = result['ps_availability'].fillna(0).astype(int)
        result['n_min'] = result['n_min'].fillna(0).astype(int)
        return result[[time_col] + feat_cols]

    # Define an empty output
    empty_out = pd.DataFrame(columns=[time_col] + feat_cols)

    # Immediate exit if the input DataFrame (df) is empty
    if start_time is not None:
        grid_start = pd.Timestamp(start_time)
        grid_end = pd.Timestamp(end_time)
        if df is None or len(df) == 0:
            return _reindex(empty_out, grid_start, grid_end)
    elif df is None or len(df) == 0:
        return pd.DataFrame()

    # Extract the needed columns of df and clean.
    # NOTE: we do NOT filter to the lookback window here — we keep the participant's full pre-window history
    # so that ffilled screen / plug states at the start of the window carry over from earlier events.
    # We do drop events at or after end_time, since they can't inform the (half-open) window.
    d = df[[time_col, event_col, level_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors='coerce')
    d[level_col] = pd.to_numeric(d[level_col], errors='coerce')
    d = d.dropna(subset=[time_col]).sort_values(time_col)
    d.reset_index(drop=True, inplace=True)
    if start_time is not None:
        d = d[d[time_col] < end_time].reset_index(drop=True)

    # Another immediate exit after timestamp filtering
    if d.empty:
        if start_time is not None:
            return _reindex(empty_out, grid_start, grid_end)
        return pd.DataFrame()

    # If not start_time or end_time, derive grid from data
    if start_time is None:
        grid_start = d[time_col].min().normalize()
        grid_end = d[time_col].max().normalize() + pd.Timedelta(days=1)

    # Index by time
    d = d.set_index(time_col).sort_index()

    # Forward-fill screen and plug states from events
    screen_flag = d[event_col].map({'Unlocked': 1, 'Locked': 0}).ffill().fillna(0).astype(int)
    d['screen_state'] = np.where(screen_flag == 1, 'Unlocked', 'Locked')

    plug_flag = d[event_col].map({'Charging': 1, 'Full': 1, 'Unplugged': 0}).ffill().fillna(0).astype(int)
    d['plug_state'] = np.where(plug_flag == 1, 'Plugged', 'Unplugged')

    # Build the 1-min internal state grid (kept at 1-min regardless of resol so that fractions remain true fractions).
    # Extend the grid forward to end_time so the lookback window is fully covered after the final crop.
    d_grid = d[~d.index.duplicated(keep='last')].copy() if d.index.has_duplicates else d
    d_1min = d_grid.resample('1min').ffill()
    if start_time is not None and d_1min.index.max() < pd.Timestamp(end_time):
        extended_idx = pd.date_range(d_1min.index.min(), pd.Timestamp(end_time), freq='1min')
        d_1min = d_1min.reindex(extended_idx, method='ffill')
    d_1min.index.name = 'minute'

    # Staleness: time since the most recent actual raw event at each 1-min row
    last_event_at = pd.Series(d_grid.index, index=d_grid.index).reindex(d_1min.index, method='ffill')
    staleness = d_1min.index.to_series() - last_event_at
    stale_mask = staleness > pd.Timedelta(state_max_stale)
    d_1min['_nonstale'] = (~stale_mask).astype(int)

    # Event-level transitions (on full-history d, so transitions across the window boundary are correctly counted)
    d_evt = d.copy()
    prev_screen = d_evt['screen_state'].shift(1)
    prev_plug = d_evt['plug_state'].shift(1)

    d_evt['unlock_event'] = ((d_evt[event_col] == 'Unlocked') & (prev_screen == 'Locked')).astype(int)
    d_evt['lock_event'] = ((d_evt[event_col] == 'Locked') & (prev_screen == 'Unlocked')).astype(int)
    d_evt['plug_event'] = (d_evt[event_col].isin(['Charging', 'Full']) & (prev_plug == 'Unplugged')).astype(int)
    d_evt['unplug_event'] = ((d_evt[event_col] == 'Unplugged') & (prev_plug == 'Plugged')).astype(int)

    state_change = ((d_evt['screen_state'] != prev_screen) | (d_evt['plug_state'] != prev_plug)).astype(int)
    if len(state_change) > 0:
        state_change.iloc[0] = 0
    d_evt['state_change'] = state_change

    # Bin events to resol (window-anchored)
    d_evt['minute'] = _bin_start(d_evt.index)
    events_per_bin = (d_evt.groupby('minute', sort=True)
                      [['unlock_event', 'lock_event', 'plug_event', 'unplug_event', 'state_change']].sum().astype(int))

    # Raw counts per resol bin (battery level mean + event counts)
    d['minute'] = _bin_start(d.index)
    raw_per_bin = d.groupby('minute', sort=True).agg(battery_level_mean=(level_col, 'mean'),
                                                     n_events=(event_col, 'count'))

    # Compute screen / plug fractions; .map preserves NaN for rows with no prior event (honest "unknown");
    # then the staleness mask further blanks rows whose last event is too old to trust.
    d_1min['screen_on_flag'] = d_1min['screen_state'].map({'Unlocked': 1.0, 'Locked': 0.0})
    d_1min['plugged_flag'] = d_1min['plug_state'].map({'Plugged': 1.0, 'Unplugged': 0.0})
    d_1min.loc[stale_mask, 'screen_on_flag'] = np.nan
    d_1min.loc[stale_mask, 'plugged_flag'] = np.nan

    # Attach event counts and raw counts
    d_1min = d_1min.join(events_per_bin, how='left')
    d_1min = d_1min.join(raw_per_bin, how='left')
    for c in ['unlock_event', 'lock_event', 'plug_event', 'unplug_event', 'state_change']:
        d_1min[c] = d_1min[c].fillna(0).astype(int)

    # Crop d_1min to the lookback window (half-open) before final aggregation
    if start_time is not None:
        d_1min = d_1min[(d_1min.index >= start_time) & (d_1min.index < end_time)]

    if d_1min.empty:
        if start_time is not None:
            return _reindex(empty_out, grid_start, grid_end)
        return pd.DataFrame()

    # Final aggregation: 1-min flags averaged into true fractions over each resol bin, event counts summed.
    # When windowed, anchor the resample at start_time so bins align with the window-anchored grid.
    if anchor is not None:
        g = d_1min.resample(resol, origin=anchor)
    else:
        g = d_1min.resample(resol)

    out = pd.DataFrame({'unlock_count': g['unlock_event'].sum(),
                        'lock_count': g['lock_event'].sum(),
                        'screen_on_fraction': g['screen_on_flag'].mean(),
                        'plugged_fraction': g['plugged_flag'].mean(),
                        'plug_event_count': g['plug_event'].sum(),
                        'unplug_event_count': g['unplug_event'].sum(),
                        'state_changes_count': g['state_change'].sum(),
                        'battery_level_mean': g['battery_level_mean'].mean(),
                        'n_events': g['n_events'].sum(),
                        'n_min': g['_nonstale'].sum()})

    out['ps_availability'] = out.index.isin(raw_per_bin.index).astype(int)

    out = out.reset_index().rename(columns={out.index.name or 'minute': time_col})
    return _reindex(out, grid_start, grid_end)
