########################################################################################################################
# Helper: Rate-of-change features from long-format sensor time-series
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
from scipy.stats import linregress

########################################################################################################################
# Helper function to compute rate-of-change via ordinary-least-squares (OLS)
########################################################################################################################


def _ols_slope(t_idx: np.ndarray,
               values: np.ndarray,
               min_obs_slope: int = 3) -> float:
    # Compute OLS slope of values over t_idx, ignoring NaNs.
    mask = ~np.isnan(values)
    if mask.sum() < min_obs_slope:
        return np.nan
    slope, *_ = linregress(t_idx[mask], values[mask])
    return float(slope)

########################################################################################################################
# Helper function to compute rate-of-change via first-to-last differences (FLDS)
########################################################################################################################


def _first_last_diff(t_idx: np.ndarray,
                     values: np.ndarray,
                     min_obs_flds: int = 2) -> float:
    """
    Compute (last_valid - first_valid) / (last_valid_t - first_valid_t).
    Normalized by time gap (in timestamp indices) to produce a rate per unit time.
    """
    mask = ~np.isnan(values)
    if mask.sum() < min_obs_flds:
        return np.nan
    valid_t = t_idx[mask]
    valid_v = values[mask]
    t_gap = valid_t[-1] - valid_t[0]
    if t_gap == 0:
        return np.nan
    return float((valid_v[-1] - valid_v[0]) / t_gap)


########################################################################################################################
# Helper function to compute incremental rate-of-change (Approach 2)
########################################################################################################################


def _incremental(t_idx: np.ndarray,
                 values: np.ndarray) -> np.ndarray:
    """
    Compute the incremental rate of change at each timestamp relative to the previous observed timestamp.
    At each non-missing timestamp t, computes:
        (value[t] - value[prev_valid_t]) / (t_idx[t] - t_idx[prev_valid_t])
    where prev_valid_t is the most recent non-missing timestamp before t.
    Returns NaN at the first observed timestamp (no previous) and at all missing timestamps.
    :return: np.ndarray of the same length as t_idx and values, with NaN where not computable.
    """
    n: int = len(values)
    inc: np.ndarray = np.full(n, np.nan)
    prev_idx: int = -1
    prev_val: float = np.nan
    for i in range(n):
        if np.isnan(values[i]):
            continue
        if prev_idx >= 0:
            t_gap = t_idx[i] - t_idx[prev_idx]
            if t_gap > 0:
                inc[i] = (values[i] - prev_val) / t_gap
        prev_idx = i
        prev_val = values[i]
    return inc


########################################################################################################################
# Generic function to compute rate of change
########################################################################################################################
def rate_change(t_idx: np.ndarray,
                values: np.ndarray,
                feat_name: str,
                min_obs_slope: int = 3,
                min_obs_flds: int = 2) -> dict:
    """
    Compute rate-of-change features for a single record and a single sensor feature.
    :param t_idx: np.ndarray. Integer timestamp indices for this record (e.g., 0 to 14 for a 15-timestamp window).
    :param values: np.ndarray. Sensor feature values corresponding to t_idx.
    :param feat_name: str. Name of the feature.
    :param min_obs_slope: int. Minimum number of non-missing observations required to compute OLS slope.
    :param min_obs_flds: int. Minimum number of non-missing observations required to compute first-to-last difference.
    :return: A dictionary of the form {'{feat_name}_slope': float, '{feat_name}_fld': float}.
    """
    # return {f'{feat_name}_slope': _ols_slope(t_idx, values, min_obs_slope),
    #         f'{feat_name}_fld': _first_last_diff(t_idx, values, min_obs_flds),
    #         f'{feat_name}$inc': _incremental(t_idx, values)}
    return {f'{feat_name}@roc': _incremental(t_idx, values)}

