########################################################################################################################
# Defines the main preprocessing pipeline before modeling (fusion-aware).
# Differences from Preprocessing.py:
#   - Adds a `modality` parameter ('unimodal' or 'multimodal'). When 'unimodal', @survey columns are dropped
#     inside preprocess_data so callers no longer filter them post-hoc.
#   - In the 3D path, empirically detects static (constant-across-bin) vs temporal columns on the training array
#     and returns two extra index lists: static_idxs and temporal_idxs.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from typing import Literal, Optional
from .Logger import create_log, Logger

########################################################################################################################
# Continuous features (from _Helper_Files/15min_Data/MetaData_v1.csv). 318 items.
########################################################################################################################
CONT_COLS: list[str] = [
    'ACC#steps^15min', 'ACC#steps^1h', 'ACC#steps^6h',
    'ACC#steps^24h', 'ACC#walking_time_s^15min', 'ACC#walking_time_s^1h',
    'ACC#walking_time_s^6h', 'ACC#walking_time_s^24h', 'ACC#cadence_spm^15min',
    'ACC#cadence_spm^1h', 'ACC#cadence_spm^6h', 'ACC#cadence_spm^24h',
    'ACC#n_samples^15min', 'ACC#n_samples^1h', 'ACC#n_samples^6h',
    'ACC#n_samples^24h', 'ACC#longest_gap_s^15min', 'ACC#longest_gap_s^1h',
    'ACC#longest_gap_s^6h', 'ACC#longest_gap_s^24h', 'ACC#mean_x^15min',
    'ACC#mean_x^1h', 'ACC#mean_x^6h', 'ACC#mean_x^24h',
    'ACC#std_x^15min', 'ACC#std_x^1h', 'ACC#std_x^6h',
    'ACC#std_x^24h', 'ACC#mean_y^15min', 'ACC#mean_y^1h',
    'ACC#mean_y^6h', 'ACC#mean_y^24h', 'ACC#std_y^15min',
    'ACC#std_y^1h', 'ACC#std_y^6h', 'ACC#std_y^24h',
    'ACC#mean_z^15min', 'ACC#mean_z^1h', 'ACC#mean_z^6h',
    'ACC#mean_z^24h', 'ACC#std_z^15min', 'ACC#std_z^1h',
    'ACC#std_z^6h', 'ACC#std_z^24h', 'ACC#mean_vm^15min',
    'ACC#mean_vm^1h', 'ACC#mean_vm^6h', 'ACC#mean_vm^24h',
    'ACC#median_vm^15min', 'ACC#median_vm^1h', 'ACC#median_vm^6h',
    'ACC#median_vm^24h', 'ACC#max_vm^15min', 'ACC#max_vm^1h',
    'ACC#max_vm^6h', 'ACC#max_vm^24h', 'ACC#std_vm^15min',
    'ACC#std_vm^1h', 'ACC#std_vm^6h', 'ACC#std_vm^24h',
    'ACC#iqr_vm^15min', 'ACC#iqr_vm^1h', 'ACC#iqr_vm^6h',
    'ACC#iqr_vm^24h', 'ACC#energy_vm^15min', 'ACC#energy_vm^1h',
    'ACC#energy_vm^6h', 'ACC#energy_vm^24h', 'ACC#entropy_vm^15min',
    'ACC#entropy_vm^1h', 'ACC#entropy_vm^6h', 'ACC#entropy_vm^24h',
    'GPS#n_points^15min', 'GPS#n_points^1h', 'GPS#n_points^6h',
    'GPS#n_points^24h', 'GPS#duration_sec^15min', 'GPS#duration_sec^1h',
    'GPS#duration_sec^6h', 'GPS#duration_sec^24h', 'GPS#total_path_length_m^15min',
    'GPS#total_path_length_m^1h', 'GPS#total_path_length_m^6h', 'GPS#total_path_length_m^24h',
    'GPS#max_step_distance_m^15min', 'GPS#max_step_distance_m^1h', 'GPS#max_step_distance_m^6h',
    'GPS#max_step_distance_m^24h', 'GPS#mean_speed_mps^15min', 'GPS#mean_speed_mps^1h',
    'GPS#mean_speed_mps^6h', 'GPS#mean_speed_mps^24h', 'GPS#max_speed_mps^15min',
    'GPS#max_speed_mps^1h', 'GPS#max_speed_mps^6h', 'GPS#max_speed_mps^24h',
    'GPS#centroid_distance_mean_m^15min', 'GPS#centroid_distance_mean_m^1h', 'GPS#centroid_distance_mean_m^6h',
    'GPS#centroid_distance_mean_m^24h', 'GPS#centroid_distance_max_m^15min', 'GPS#centroid_distance_max_m^1h',
    'GPS#centroid_distance_max_m^6h', 'GPS#centroid_distance_max_m^24h', 'GPS#mean_dist_home_m^15min',
    'GPS#mean_dist_home_m^1h', 'GPS#mean_dist_home_m^6h', 'GPS#mean_dist_home_m^24h',
    'GPS#max_dist_home_m^15min', 'GPS#max_dist_home_m^1h', 'GPS#max_dist_home_m^6h',
    'GPS#max_dist_home_m^24h', 'GPS#home_fraction^15min', 'GPS#home_fraction^1h',
    'GPS#home_fraction^6h', 'GPS#home_fraction^24h', 'PS#unlock_count^15min',
    'PS#unlock_count^1h', 'PS#unlock_count^6h', 'PS#unlock_count^24h',
    'PS#lock_count^15min', 'PS#lock_count^1h', 'PS#lock_count^6h',
    'PS#lock_count^24h', 'PS#screen_on_fraction^15min', 'PS#screen_on_fraction^1h',
    'PS#screen_on_fraction^6h', 'PS#screen_on_fraction^24h', 'PS#plugged_fraction^15min',
    'PS#plugged_fraction^1h', 'PS#plugged_fraction^6h', 'PS#plugged_fraction^24h',
    'PS#plug_event_count^15min', 'PS#plug_event_count^1h', 'PS#plug_event_count^6h',
    'PS#plug_event_count^24h', 'PS#unplug_event_count^15min', 'PS#unplug_event_count^1h',
    'PS#unplug_event_count^6h', 'PS#unplug_event_count^24h', 'PS#state_changes_count^15min',
    'PS#state_changes_count^1h', 'PS#state_changes_count^6h', 'PS#state_changes_count^24h',
    'PS#battery_level_mean^15min', 'PS#battery_level_mean^1h', 'PS#battery_level_mean^6h',
    'PS#battery_level_mean^24h', 'PS#n_events^15min', 'PS#n_events^1h',
    'PS#n_events^6h', 'PS#n_events^24h', 'ACC#n_min^15min',
    'ACC#n_min^1h', 'ACC#n_min^6h', 'ACC#n_min^24h',
    'GPS#n_min^15min', 'GPS#n_min^1h', 'GPS#n_min^6h',
    'GPS#n_min^24h', 'PS#n_min^15min', 'PS#n_min^1h',
    'PS#n_min^6h', 'PS#n_min^24h', 'ACC#steps^15min@roc',
    'ACC#steps^1h@roc', 'ACC#steps^6h@roc', 'ACC#walking_time_s^15min@roc',
    'ACC#walking_time_s^1h@roc', 'ACC#walking_time_s^6h@roc', 'ACC#cadence_spm^15min@roc',
    'ACC#cadence_spm^1h@roc', 'ACC#cadence_spm^6h@roc', 'ACC#n_samples^15min@roc',
    'ACC#n_samples^1h@roc', 'ACC#n_samples^6h@roc', 'ACC#longest_gap_s^15min@roc',
    'ACC#longest_gap_s^1h@roc', 'ACC#longest_gap_s^6h@roc', 'ACC#mean_x^15min@roc',
    'ACC#mean_x^1h@roc', 'ACC#mean_x^6h@roc', 'ACC#std_x^15min@roc',
    'ACC#std_x^1h@roc', 'ACC#std_x^6h@roc', 'ACC#mean_y^15min@roc',
    'ACC#mean_y^1h@roc', 'ACC#mean_y^6h@roc', 'ACC#std_y^15min@roc',
    'ACC#std_y^1h@roc', 'ACC#std_y^6h@roc', 'ACC#mean_z^15min@roc',
    'ACC#mean_z^1h@roc', 'ACC#mean_z^6h@roc', 'ACC#std_z^15min@roc',
    'ACC#std_z^1h@roc', 'ACC#std_z^6h@roc', 'ACC#mean_vm^15min@roc',
    'ACC#mean_vm^1h@roc', 'ACC#mean_vm^6h@roc', 'ACC#median_vm^15min@roc',
    'ACC#median_vm^1h@roc', 'ACC#median_vm^6h@roc', 'ACC#max_vm^15min@roc',
    'ACC#max_vm^1h@roc', 'ACC#max_vm^6h@roc', 'ACC#std_vm^15min@roc',
    'ACC#std_vm^1h@roc', 'ACC#std_vm^6h@roc', 'ACC#iqr_vm^15min@roc',
    'ACC#iqr_vm^1h@roc', 'ACC#iqr_vm^6h@roc', 'ACC#energy_vm^15min@roc',
    'ACC#energy_vm^1h@roc', 'ACC#energy_vm^6h@roc', 'ACC#entropy_vm^15min@roc',
    'ACC#entropy_vm^1h@roc', 'ACC#entropy_vm^6h@roc', 'GPS#n_points^15min@roc',
    'GPS#n_points^1h@roc', 'GPS#n_points^6h@roc', 'GPS#duration_sec^15min@roc',
    'GPS#duration_sec^1h@roc', 'GPS#duration_sec^6h@roc', 'GPS#total_path_length_m^15min@roc',
    'GPS#total_path_length_m^1h@roc', 'GPS#total_path_length_m^6h@roc', 'GPS#max_step_distance_m^15min@roc',
    'GPS#max_step_distance_m^1h@roc', 'GPS#max_step_distance_m^6h@roc', 'GPS#mean_speed_mps^15min@roc',
    'GPS#mean_speed_mps^1h@roc', 'GPS#mean_speed_mps^6h@roc', 'GPS#max_speed_mps^15min@roc',
    'GPS#max_speed_mps^1h@roc', 'GPS#max_speed_mps^6h@roc', 'GPS#centroid_distance_mean_m^15min@roc',
    'GPS#centroid_distance_mean_m^1h@roc', 'GPS#centroid_distance_mean_m^6h@roc', 'GPS#centroid_distance_max_m^15min@roc',
    'GPS#centroid_distance_max_m^1h@roc', 'GPS#centroid_distance_max_m^6h@roc', 'GPS#mean_dist_home_m^15min@roc',
    'GPS#mean_dist_home_m^1h@roc', 'GPS#mean_dist_home_m^6h@roc', 'GPS#max_dist_home_m^15min@roc',
    'GPS#max_dist_home_m^1h@roc', 'GPS#max_dist_home_m^6h@roc', 'GPS#home_fraction^15min@roc',
    'GPS#home_fraction^1h@roc', 'GPS#home_fraction^6h@roc', 'PS#unlock_count^15min@roc',
    'PS#unlock_count^1h@roc', 'PS#unlock_count^6h@roc', 'PS#lock_count^15min@roc',
    'PS#lock_count^1h@roc', 'PS#lock_count^6h@roc', 'PS#screen_on_fraction^15min@roc',
    'PS#screen_on_fraction^1h@roc', 'PS#screen_on_fraction^6h@roc', 'PS#plugged_fraction^15min@roc',
    'PS#plugged_fraction^1h@roc', 'PS#plugged_fraction^6h@roc', 'PS#plug_event_count^15min@roc',
    'PS#plug_event_count^1h@roc', 'PS#plug_event_count^6h@roc', 'PS#unplug_event_count^15min@roc',
    'PS#unplug_event_count^1h@roc', 'PS#unplug_event_count^6h@roc', 'PS#state_changes_count^15min@roc',
    'PS#state_changes_count^1h@roc', 'PS#state_changes_count^6h@roc', 'PS#battery_level_mean^15min@roc',
    'PS#battery_level_mean^1h@roc', 'PS#battery_level_mean^6h@roc', 'PS#n_events^15min@roc',
    'PS#n_events^1h@roc', 'PS#n_events^6h@roc', 'ACC#n_min^15min@roc',
    'ACC#n_min^1h@roc', 'ACC#n_min^6h@roc', 'GPS#n_min^15min@roc',
    'GPS#n_min^1h@roc', 'GPS#n_min^6h@roc', 'PS#n_min^15min@roc',
    'PS#n_min^1h@roc', 'PS#n_min^6h@roc', 'coverage_frac@daily',
    'n_sleep_periods@daily', 'longest_sleep_min@daily', 'most_recent_sleep_min@daily',
    'active_min_per_obs_hr@daily', 'steps_per_obs_hr@daily', 'peak_cadence_avg@daily',
    'active_bout_count@daily', 'longest_active_bout_min@daily', 'TOD_sin@clock',
    'TOD_cos@clock', 'Age@survey', 'DASS21_Anx@survey',
    'DASS21_Dep@survey', 'DASS21_Stress@survey', 'DASS21_Total@survey',
    'DERS_Total@survey', 'DaysSched_ActCent@survey', 'DaysSched_DayHosp@survey',
    'DaysSched_School@survey', 'DaysSched_Work@survey', 'EducYears_cleaned@survey',
    'LFQ_DutHome_Ave@survey', 'LFQ_DutWorkSch_Ave@survey', 'LFQ_LeisFam_Ave@survey',
    'LFQ_LeisFriend_Ave@survey', 'MonthlyInc_cleaned@survey', 'PSS4_Total@survey',
    'TAPS.1_Total@survey', 'DaysMiss_School@survey', 'DaysMiss_Work@survey',
]

########################################################################################################################
# Define the core pipeline as a function
########################################################################################################################


def preprocess_data(df_X_train: pd.DataFrame,
                    df_X_val: pd.DataFrame,
                    df_X_test: Optional[pd.DataFrame] = None,
                    bkgd_cols: list[str] = [],
                    groupby_col: str = 'Record_ID',
                    sd_away: float = 3,
                    impute: Literal['Mean', 'Median', 'Zero'] = 'Median',
                    dim: Literal['2D', '3D'] = '2D',
                    modality: Literal['unimodal', 'multimodal'] = 'multimodal',
                    logger: Optional[Logger] = None):
    """
    This pipeline assumes that stratified partitioning has been performed via Outcome_v1.csv.
    df_X_test is optional because no preprocessing on it is needed during the cross-validation phase.
    :param df_X_train: pd.DataFrame. The training dataset.
    :param df_X_val: pd.DataFrame. The validation dataset.
    :param df_X_test: pd.DataFrame or None. The test dataset.
    :param bkgd_cols: list of str. Columns to exclude from preprocessing (metadata etc.).
    :param groupby_col: str. The column used for mean aggregation when dim == '2D' and for grouping in 3D.
    :param sd_away: float. Number of standard deviations from the mean (per-feature) to winsorize outliers.
    :param impute: str. Imputation method in ['Mean', 'Median', 'Zero'] (2D only).
    :param dim: str. Output dimension in ['2D', '3D'].
    :param modality: str. 'unimodal' drops @survey columns internally; 'multimodal' keeps them.
    :param logger: Logger from _Helper_Scripts.Logger. Used for logging if not None.
    :return:
    dim=='2D', df_X_test is None      -> (feat_train_2D, feat_val_2D)
    dim=='2D', df_X_test is not None  -> (feat_train_2D, feat_val_2D, feat_test_2D)
    dim=='3D', df_X_test is None      -> (X_train_3D, X_val_3D, ids_train, ids_val,
                                          feat_cols, static_idxs, temporal_idxs)
    dim=='3D', df_X_test is not None  -> (X_train_3D, X_val_3D, X_test_3D, ids_train, ids_val, ids_test,
                                          feat_cols, static_idxs, temporal_idxs)
    """
    print_func = print if logger is None else logger.info

    ####################################################################################################################
    # Step 0. Data validation
    ####################################################################################################################
    train_cols: list[str] = df_X_train.columns.tolist()
    val_cols: list[str] = df_X_val.columns.tolist()
    assert train_cols == val_cols
    assert set(bkgd_cols).issubset(train_cols)
    if df_X_test is not None:
        test_cols: list[str] = df_X_test.columns.tolist()
        assert train_cols == test_cols

    df_X_train = df_X_train.copy()
    df_X_val = df_X_val.copy()
    if df_X_test is not None:
        df_X_test = df_X_test.copy()

    ####################################################################################################################
    # Step 1. Remove features with no sample or being a constant in df_X_train
    ####################################################################################################################
    feat_cols: list[str] = [col for col in train_cols if col not in bkgd_cols]
    print_func(f'X has {len(train_cols)} columns where {len(feat_cols)} are specified as features.')
    remove_train_cols: list[str] = [col for col in feat_cols
                                    if df_X_train[col].notna().sum() == 0
                                    or df_X_train[col].dropna().nunique() <= 1]
    if len(remove_train_cols) > 0:
        df_X_train.drop(columns=remove_train_cols, inplace=True)
        df_X_val.drop(columns=remove_train_cols, inplace=True)
        if df_X_test is not None:
            df_X_test.drop(columns=remove_train_cols, inplace=True)
    feat_cols = [col for col in feat_cols if col not in remove_train_cols]
    print_func(f'{len(remove_train_cols)} features removed because of all missing or constant.')
    print_func(f'{len(feat_cols)} features remain.')

    ####################################################################################################################
    # Step 1b. Modality filter — unimodal drops @survey columns; multimodal keeps them
    ####################################################################################################################
    if modality == 'unimodal':
        survey_cols: list[str] = [c for c in feat_cols if c.endswith('@survey')]
        if len(survey_cols) > 0:
            df_X_train.drop(columns=survey_cols, inplace=True)
            df_X_val.drop(columns=survey_cols, inplace=True)
            if df_X_test is not None:
                df_X_test.drop(columns=survey_cols, inplace=True)
        feat_cols = [c for c in feat_cols if c not in survey_cols]
        print_func(f'Unimodal filter: {len(survey_cols)} @survey features removed.')
        print_func(f'{len(feat_cols)} features remain after unimodal filter.')

    ####################################################################################################################
    # Step 2. Clipping (winsorizing) the continuous feature values
    ####################################################################################################################
    cont_feats: list[str] = [c for c in CONT_COLS if c in feat_cols]
    print_func(f'Winsorizing {len(cont_feats)} continuous features at +/- {sd_away} SD from the training mean.')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        cont_mu: np.ndarray = np.nanmean(df_X_train[cont_feats], axis=0)
        cont_std: np.ndarray = np.nanstd(df_X_train[cont_feats], axis=0)
        cont_upper: np.ndarray = cont_mu + sd_away * cont_std
        cont_lower: np.ndarray = cont_mu - sd_away * cont_std

    df_X_train[cont_feats] = np.clip(df_X_train[cont_feats], cont_lower, cont_upper)
    df_X_val[cont_feats] = np.clip(df_X_val[cont_feats], cont_lower, cont_upper)
    if df_X_test is not None:
        df_X_test[cont_feats] = np.clip(df_X_test[cont_feats], cont_lower, cont_upper)
    print_func('Winsorization completed.')

    ####################################################################################################################
    # Step 3. Scaling
    ####################################################################################################################
    scaler = MinMaxScaler()
    df_X_train[feat_cols] = scaler.fit_transform(df_X_train[feat_cols].values)
    df_X_val[feat_cols] = scaler.transform(df_X_val[feat_cols].values)
    if df_X_test is not None:
        df_X_test[feat_cols] = scaler.transform(df_X_test[feat_cols].values)
    print_func('Min-max scaling completed.')

    ####################################################################################################################
    # Step 4. Mean aggregation + imputation (2D) or direct array conversion (3D)
    ####################################################################################################################
    if dim == '2D':
        feat_train_2D: pd.DataFrame = df_X_train.groupby(groupby_col)[feat_cols].mean().reset_index()
        assert feat_train_2D.shape[0] == df_X_train[groupby_col].nunique(), \
            '[Train] record count mismatch after aggregation'

        feat_val_2D: pd.DataFrame = df_X_val.groupby(groupby_col)[feat_cols].mean().reset_index()
        assert feat_val_2D.shape[0] == df_X_val[groupby_col].nunique(), \
            '[Val] record count mismatch after aggregation'

        feat_test_2D = None
        if df_X_test is not None:
            feat_test_2D = df_X_test.groupby(groupby_col)[feat_cols].mean().reset_index()
            assert feat_test_2D.shape[0] == df_X_test[groupby_col].nunique(), \
                '[Test] record count mismatch after aggregation'
        print_func('Mean-aggregation across timestamps completed.')

        # Build missingness flags from training NaN pattern; schema fixed from training
        flag_data: dict[str, pd.Series] = {}
        for col in feat_cols:
            flag: pd.Series = feat_train_2D[col].isna().astype(int)
            if flag.sum() == 0 or flag.sum() == len(flag):
                continue
            flag_data[f'{col}!!NA'] = flag
        X_train_flags: pd.DataFrame = pd.DataFrame(flag_data, index=feat_train_2D.index)
        flag_cols: list[str] = X_train_flags.columns.tolist()

        def _make_flags(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {fc: df[fc.replace('!!NA', '')].isna().astype(int) for fc in flag_cols},
                index=df.index)

        X_val_flags: pd.DataFrame = _make_flags(feat_val_2D)
        X_test_flags: Optional[pd.DataFrame] = _make_flags(feat_test_2D) if df_X_test is not None else None

        # Imputation: fit fill values on training aggregated frame only
        if impute == 'Zero':
            fill_values: np.ndarray = np.zeros(len(feat_cols))
        elif impute == 'Mean':
            fill_values = np.nanmean(feat_train_2D[feat_cols].values, axis=0)
        else:
            fill_values = np.nanmedian(feat_train_2D[feat_cols].values, axis=0)
        fill_map: dict[str, float] = dict(zip(feat_cols, fill_values))

        feat_train_2D[feat_cols] = feat_train_2D[feat_cols].fillna(fill_map)
        feat_val_2D[feat_cols] = feat_val_2D[feat_cols].fillna(fill_map)
        if df_X_test is not None:
            feat_test_2D[feat_cols] = feat_test_2D[feat_cols].fillna(fill_map)

        feat_train_2D = pd.concat([feat_train_2D, X_train_flags], axis=1)
        feat_val_2D = pd.concat([feat_val_2D, X_val_flags], axis=1)
        if df_X_test is not None:
            feat_test_2D = pd.concat([feat_test_2D, X_test_flags], axis=1)

        assert not feat_train_2D[feat_cols].isna().any().any(), 'NaN remains in train after imputation'
        assert not feat_val_2D[feat_cols].isna().any().any(), 'NaN remains in val after imputation'
        if df_X_test is not None:
            assert not feat_test_2D[feat_cols].isna().any().any(), 'NaN remains in test after imputation'
        print_func(f'{impute}-imputation completed.')

        if df_X_test is not None:
            print_func(f'Returning 2D datasets with shape {feat_train_2D.shape} (TRAIN); '
                       f'{feat_val_2D.shape} (VAL); {feat_test_2D.shape} (TEST)...')
            return feat_train_2D, feat_val_2D, feat_test_2D
        print_func(f'Returning 2D datasets with shape {feat_train_2D.shape} (TRAIN); '
                   f'{feat_val_2D.shape} (VAL)...')
        return feat_train_2D, feat_val_2D

    else:  # 3D
        def _to_3D(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            df_sorted: pd.DataFrame = df.sort_values([groupby_col, 'bin_index'])
            record_ids: np.ndarray = df_sorted[groupby_col].unique()
            arr: np.ndarray = np.stack([
                df_sorted.loc[df_sorted[groupby_col] == rid, feat_cols].values
                for rid in record_ids
            ])  # (n_records, 96, n_features)
            assert arr.shape == (len(record_ids), 96, len(feat_cols)), f'unexpected 3D shape {arr.shape}'
            return arr, record_ids

        X_train_3D, ids_train = _to_3D(df_X_train)
        X_val_3D, ids_val = _to_3D(df_X_val)
        if df_X_test is not None:
            X_test_3D, ids_test = _to_3D(df_X_test)

        ################################################################################################################
        # Detect static (constant-across-bin) columns on training array. NaN is treated as a fixed sentinel so that an
        # all-NaN-across-bins column for a record still counts as constant.
        # A column is static iff it is constant across the 96 bins for EVERY training record.
        ################################################################################################################
        X_train_filled: np.ndarray = np.nan_to_num(X_train_3D, nan=-999999.0)
        col_min: np.ndarray = X_train_filled.min(axis=1)                # (N, F)
        col_max: np.ndarray = X_train_filled.max(axis=1)                # (N, F)
        is_const_per_record: np.ndarray = (col_min == col_max)          # (N, F)
        static_mask: np.ndarray = is_const_per_record.all(axis=0)       # (F,)
        static_idxs: list[int] = [i for i in range(len(static_mask)) if static_mask[i]]
        temporal_idxs: list[int] = [i for i in range(len(static_mask)) if not static_mask[i]]
        print_func(f'Static/temporal split: {len(static_idxs)} static (constant across bins), '
                   f'{len(temporal_idxs)} temporal.')

        if df_X_test is not None:
            print_func(f'Returning 3D datasets with shape {X_train_3D.shape} (TRAIN); '
                       f'{X_val_3D.shape} (VAL); {X_test_3D.shape} (TEST)...')
            print_func(f'Returning the {groupby_col} values, feature names, static and temporal indices too.')
            return X_train_3D, X_val_3D, X_test_3D, ids_train, ids_val, ids_test, \
                feat_cols, static_idxs, temporal_idxs
        print_func(f'Returning 3D datasets with shape {X_train_3D.shape} (TRAIN); {X_val_3D.shape} (VAL)...')
        print_func(f'Returning the {groupby_col} values, feature names, static and temporal indices too.')
        return X_train_3D, X_val_3D, ids_train, ids_val, feat_cols, static_idxs, temporal_idxs
