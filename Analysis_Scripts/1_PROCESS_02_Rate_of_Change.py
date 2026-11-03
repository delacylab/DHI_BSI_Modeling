########################################################################################################################
# Compute rate-of-change (RoC) derived features from the existing sensor features
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
from _Helper_Scripts.Rate_of_Change import rate_change
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Specify the input directory
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'

########################################################################################################################
# Load the latest feature dataset
########################################################################################################################
data_path: str = os.path.join(IN_DIR, 'Features_v0.parquet')
df: pd.DataFrame = pd.read_parquet(data_path)
logger.info(f'Feature dataset loaded with dimension: {df.shape}')
logger.info(f'Number of Beiwe IDs: {df.BeiweId.nunique()}')
logger.info(f'Number of records: {df.Record_ID.nunique()}')
logger.print('-'*100)

########################################################################################################################
# Print out column names
########################################################################################################################
for c_idx, c in enumerate(df.columns, 1):
    logger.debug(f'{c_idx:03}. {c}')
logger.print('-'*100)

########################################################################################################################
# Identify the background (non-feature) columns and sensor (feature) columns
########################################################################################################################
bkgd_cols: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day',
                        'Survey_Time', 'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case', 'with_home_address',
                        'bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']
excl_cols: list[str] = ([col for col in df.columns if col.endswith('^24h')]
                        + ['ACC#availability', 'GPS#availability', 'PS#availability'])
sensor_cols: list[str] = [col for col in df.columns if col not in bkgd_cols and col not in excl_cols]
logger.info(f'{len(sensor_cols)} sensor features identified for rate-of-change computation.')

########################################################################################################################
# Compute rate-of-change features per record for each restricted sensor column
########################################################################################################################
df = df.sort_values(['Record_ID', 'Bin_Start_Time']).reset_index(drop=True)
df['_t_idx'] = df.groupby('Record_ID').cumcount()

inc_records: dict[str, np.ndarray] = {}

for rid, grp in df.groupby('Record_ID'):
    t_idx_arr: np.ndarray = grp['_t_idx'].values
    row_idx: np.ndarray = grp.index.values
    for feat in sensor_cols:
        values: np.ndarray = grp[feat].values.astype(float)
        roc: dict = rate_change(t_idx=t_idx_arr, values=values, feat_name=feat)
        inc_col: str = f'{feat}@roc'
        if inc_col not in inc_records:
            inc_records[inc_col] = np.full(len(df), np.nan)
        inc_records[inc_col][row_idx] = roc[inc_col]

for col, arr in inc_records.items():
    df[col] = arr

df.drop(columns=['_t_idx'], inplace=True)
logger.info(f'{len(sensor_cols)} @roc columns added. Shape of df after merging: {df.shape}.')

########################################################################################################################
# Export
########################################################################################################################
out_path: str = os.path.join(IN_DIR, 'Features_v1.parquet')
df.to_parquet(out_path, index=False)
logger.info(f'Wrote {df.shape[0]} rows x {df.shape[1]} columns to {out_path}')
logger.print('-'*100)
