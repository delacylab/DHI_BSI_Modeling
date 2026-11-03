########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
from typing import Optional

########################################################################################################################
# Define the function to compute aggregate metrics from raw GPS data at a specified resolution
########################################################################################################################


def aggregate_gps(df: pd.DataFrame,
                  home_coords: Optional[tuple[float, float]] = None,
                  time_col: str = 'MT_time',
                  lat_col: str = 'latitude',
                  lon_col: str = 'longitude',
                  radius_km: float = 0.05,
                  resol: str = '1min',
                  max_leg_gap: str = '10min',
                  start_time: Optional[pd.Timestamp] = None,
                  end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Aggregate and summarize raw GPS data into bin-level metrics at the specified resolution.

    When start_time / end_time are provided, the lookback window is [start_time, end_time), and bins are anchored
    at start_time and step forward by resol: bin starts are {start_time + k * resol : k = 0, 1, ...}, giving
    exactly (end_time - start_time) / resol bins. When start_time is None, bins use absolute wall-clock alignment.

    Per-ping features (centroid stats, home metrics, n_points, duration_sec, gps_availability) are computed only
    from raw pings that fall within a bin.

    Path/speed features (total_path_length_m, mean_speed_mps, max_speed_mps, max_step_distance_m) are computed by
    treating every consecutive ping pair (a "leg") with constant-speed interpolation, then splitting the leg
    proportionally by time across all bins it crosses. This way movement that physically traversed a bin is
    credited to that bin even when no ping landed there. Legs with inter-ping gap > max_leg_gap are excluded.

    :param df: Raw GPS DataFrame
    :param home_coords: Optional tuple of (latitude, longitude) for the participant's home.
    :param time_col: Name of the timestamp column
    :param lat_col: Name of the latitude column
    :param lon_col: Name of the longitude column
    :param radius_km: Radius in km used to define "at home" for the home_fraction metric. Default 0.05 (50 metres).
    :param resol: Aggregation bin width as a pandas offset alias (default '1min')
    :param max_leg_gap: Max inter-ping gap (pandas offset alias) above which the leg is treated as untrustworthy
                        and excluded from path/speed metrics. Default '10min'.
    :param start_time: [Optional] Inclusive start of the lookback window
    :param end_time: [Optional] Exclusive end of the lookback window
    :return:
     A pandas.DataFrame recording the bin-based summary and aggregate metrics computed from the raw GPS data
    """
    if (start_time is None) != (end_time is None):
        raise ValueError('start_time and end_time must both be provided or both be None.')

    # Binary flag for the availability of home coordinates
    with_home = int(home_coords is not None)

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
        result['gps_availability'] = result['gps_availability'].fillna(0).astype(int)
        result['with_home_address'] = with_home
        return result

    # Define the feature columns
    feat_cols: list[str] = ['gps_availability',
                            'with_home_address',
                            'n_points',
                            'duration_sec',
                            'total_path_length_m',
                            'max_step_distance_m',
                            'mean_speed_mps',
                            'max_speed_mps',
                            'centroid_distance_mean_m',
                            'centroid_distance_max_m',
                            'mean_dist_home_m',
                            'max_dist_home_m',
                            'home_fraction']

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

    # Extract the needed columns of df and clean
    df = df[[time_col, lat_col, lon_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[time_col, lat_col, lon_col]).sort_values(time_col).drop_duplicates(subset=time_col)
    df.reset_index(drop=True, inplace=True)

    # Build df_ext (for leg computation with one ping of edge context on each side) and df_in (in-window pings only)
    if start_time is not None:
        pre = df[df[time_col] < start_time]
        post = df[df[time_col] >= end_time]
        df_in = df[(df[time_col] >= start_time) & (df[time_col] < end_time)].reset_index(drop=True)
        pieces = []
        if len(pre) > 0:
            pieces.append(pre.iloc[[-1]])
        if len(df_in) > 0:
            pieces.append(df_in)
        if len(post) > 0:
            pieces.append(post.iloc[[0]])
        df_ext = pd.concat(pieces, ignore_index=True) if pieces else df_in
    else:
        df_in = df
        df_ext = df

    # Exit early if there is no in-window data AND no edge legs to compute
    if df_in.empty and len(df_ext) < 2:
        if start_time is not None:
            return _reindex(empty_out, grid_start, grid_end)
        return pd.DataFrame()

    # If not start_time or end_time, derive grid from data
    if start_time is None:
        grid_start = df_in[time_col].min().normalize()
        grid_end = df_in[time_col].max().normalize() + pd.Timedelta(days=1)

    # Per-ping bin assignment for in-window pings (used by the per-ping feature loop)
    if not df_in.empty:
        df_in = df_in.copy()
        df_in['_minute'] = _bin_start(df_in[time_col])

    # Unpack home coordinates
    if home_coords is not None:
        home_lat = float(home_coords[0])
        home_lon = float(home_coords[1])
        meters_per_deg_lat_home = 111_320.0
        meters_per_deg_lon_home = meters_per_deg_lat_home * float(np.cos(np.deg2rad(home_lat)))

    ####################################################################################################################
    # Leg attribution: split every valid consecutive-pair leg proportionally by time across bins it crosses
    ####################################################################################################################
    sub_leg_records: list[dict] = []
    if len(df_ext) >= 2:
        ts_arr = df_ext[time_col].to_numpy()
        lat_arr = df_ext[lat_col].to_numpy()
        lon_arr = df_ext[lon_col].to_numpy()

        dt_sec_arr = (ts_arr[1:] - ts_arr[:-1]).astype('int64') / 1e9

        lat_mid = (lat_arr[:-1] + lat_arr[1:]) / 2.0
        mpdlat_pair = 111_320.0
        mpdlon_pair = mpdlat_pair * np.cos(np.deg2rad(lat_mid))
        dx_arr = (lon_arr[1:] - lon_arr[:-1]) * mpdlon_pair
        dy_arr = (lat_arr[1:] - lat_arr[:-1]) * mpdlat_pair
        leg_dist_arr = np.sqrt(dx_arr ** 2 + dy_arr ** 2)

        max_leg_gap_sec = pd.Timedelta(max_leg_gap).total_seconds()
        valid_legs = (dt_sec_arr >= 1.0) & (dt_sec_arr <= max_leg_gap_sec) & np.isfinite(leg_dist_arr)

        for i in np.where(valid_legs)[0]:
            t0 = pd.Timestamp(ts_arr[i])
            t1 = pd.Timestamp(ts_arr[i + 1])
            dist = float(leg_dist_arr[i])
            dt = float(dt_sec_arr[i])
            speed = dist / dt

            b0 = _bin_start(t0)
            b1 = _bin_start(t1)

            if b0 == b1:
                sub_leg_records.append({'_bin': b0, 'sub_dist': dist, 'sub_time': dt,
                                        'leg_speed': speed, 'leg_dist': dist})
            else:
                # First sub-leg: from t0 to end of bin b0
                time_in_b0 = ((b0 + resol_td) - t0).total_seconds()
                sub_leg_records.append({'_bin': b0,
                                        'sub_dist': dist * time_in_b0 / dt,
                                        'sub_time': time_in_b0,
                                        'leg_speed': speed,
                                        'leg_dist': dist})

                # Intermediate full bins
                current = b0 + resol_td
                sub_time_full = resol_td.total_seconds()
                sub_dist_full = dist * sub_time_full / dt
                while current < b1:
                    sub_leg_records.append({'_bin': current,
                                            'sub_dist': sub_dist_full,
                                            'sub_time': sub_time_full,
                                            'leg_speed': speed,
                                            'leg_dist': dist})
                    current = current + resol_td

                # Last sub-leg: from start of bin b1 to t1
                time_in_b1 = (t1 - b1).total_seconds()
                if time_in_b1 > 0:
                    sub_leg_records.append({'_bin': b1,
                                            'sub_dist': dist * time_in_b1 / dt,
                                            'sub_time': time_in_b1,
                                            'leg_speed': speed,
                                            'leg_dist': dist})

    if sub_leg_records:
        sub_legs_df = pd.DataFrame(sub_leg_records)
        path_per_bin = sub_legs_df.groupby('_bin').agg(total_path_length_m=('sub_dist', 'sum'),
                                                       sub_time_sum=('sub_time', 'sum'),
                                                       max_speed_mps=('leg_speed', 'max'),
                                                       max_step_distance_m=('leg_dist', 'max'))
        path_per_bin['mean_speed_mps'] = path_per_bin['total_path_length_m'] / path_per_bin['sub_time_sum']
    else:
        path_per_bin = pd.DataFrame()

    ####################################################################################################################
    # Per-bin records: ping-bins (with full per-ping features) + path-only bins (only interpolated path features)
    ####################################################################################################################
    minute_records: list[dict] = []

    # Ping-bins: iterate in-window pings grouped by bin
    if not df_in.empty:
        for minute_start, g in df_in.groupby('_minute'):
            g = g.sort_values(time_col).reset_index(drop=True)
            n = int(g.shape[0])

            lat = g[lat_col].to_numpy(dtype=float)
            lon = g[lon_col].to_numpy(dtype=float)
            t = g[time_col]

            # Time coverage (within bin)
            duration_sec = float((t.iloc[-1] - t.iloc[0]).total_seconds()) if n >= 2 else 0.0

            # Centroid stats from in-bin pings
            lat_c = float(np.mean(lat))
            lon_c = float(np.mean(lon))
            mpdlat = 111_320.0
            mpdlon = mpdlat * float(np.cos(np.deg2rad(lat_c)))

            x = (lon - lon_c) * mpdlon
            y = (lat - lat_c) * mpdlat
            d_centroid = np.sqrt(x ** 2 + y ** 2)
            centroid_distance_mean_m = float(np.mean(d_centroid))
            centroid_distance_max_m = float(np.max(d_centroid))

            # Per-ping home metrics
            if home_coords is not None:
                dx_home = (lon - home_lon) * meters_per_deg_lon_home
                dy_home = (lat - home_lat) * meters_per_deg_lat_home
                dist_home_per_ping = np.sqrt(dx_home ** 2 + dy_home ** 2)
                mean_dist_home_m = float(np.mean(dist_home_per_ping))
                max_dist_home_m = float(np.max(dist_home_per_ping))
                home_fraction = float(np.mean(dist_home_per_ping <= radius_km * 1000.0))
            else:
                mean_dist_home_m = np.nan
                max_dist_home_m = np.nan
                home_fraction = np.nan

            # Path/speed: look up from path_per_bin
            if not path_per_bin.empty and minute_start in path_per_bin.index:
                pp = path_per_bin.loc[minute_start]
                total_path_length_m = float(pp['total_path_length_m'])
                max_step_distance_m = float(pp['max_step_distance_m'])
                mean_speed_mps = float(pp['mean_speed_mps'])
                max_speed_mps = float(pp['max_speed_mps'])
            else:
                total_path_length_m = np.nan
                max_step_distance_m = np.nan
                mean_speed_mps = np.nan
                max_speed_mps = np.nan

            minute_records.append({time_col: minute_start,
                                   'gps_availability': 1,
                                   'with_home_address': with_home,
                                   'n_points': n,
                                   'duration_sec': duration_sec,
                                   'total_path_length_m': total_path_length_m,
                                   'max_step_distance_m': max_step_distance_m,
                                   'mean_speed_mps': mean_speed_mps,
                                   'max_speed_mps': max_speed_mps,
                                   'centroid_distance_mean_m': centroid_distance_mean_m,
                                   'centroid_distance_max_m': centroid_distance_max_m,
                                   'mean_dist_home_m': mean_dist_home_m,
                                   'max_dist_home_m': max_dist_home_m,
                                   'home_fraction': home_fraction})

    # Path-only bins: bins with interpolated leg coverage but no in-window pings
    ping_bins = set(df_in['_minute'].unique()) if not df_in.empty else set()
    if not path_per_bin.empty:
        for path_bin in path_per_bin.index:
            if path_bin in ping_bins:
                continue
            pp = path_per_bin.loc[path_bin]
            minute_records.append({time_col: path_bin,
                                   # gps_availability deliberately omitted — reindex fills with 0
                                   'with_home_address': with_home,
                                   'n_points': 0,
                                   'duration_sec': 0.0,
                                   'total_path_length_m': float(pp['total_path_length_m']),
                                   'max_step_distance_m': float(pp['max_step_distance_m']),
                                   'mean_speed_mps': float(pp['mean_speed_mps']),
                                   'max_speed_mps': float(pp['max_speed_mps']),
                                   'centroid_distance_mean_m': np.nan,
                                   'centroid_distance_max_m': np.nan,
                                   'mean_dist_home_m': np.nan,
                                   'max_dist_home_m': np.nan,
                                   'home_fraction': np.nan})

    if not minute_records:
        return _reindex(empty_out, grid_start, grid_end)

    out = pd.DataFrame(minute_records)
    return _reindex(out, grid_start, grid_end)
