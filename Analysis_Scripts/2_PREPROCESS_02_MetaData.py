########################################################################################################################
# Identify the variable type for each feature for subsequent preprocessing
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
data_path: str = os.path.join(IN_DIR, 'Features_v4.parquet')
df: pd.DataFrame = pd.read_parquet(data_path)
logger.info(f'Loaded: {data_path}  Shape: {df.shape}')
logger.print('=' * 100)

########################################################################################################################
# Create df_X by including only the feature columns
########################################################################################################################
bkgd_cols: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex',
                        'Survey_Day', 'Survey_Time',
                        'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                        'bin_index', 'Bin_Start_Time', 'Bin_End_Time',
                        'is_pad']
df_X: pd.DataFrame = df.drop(columns=bkgd_cols)

########################################################################################################################
# For each column in df_X, identify the unique values and variable type
########################################################################################################################
all_records: list[dict] = []
for col_idx, col in enumerate(df_X.columns, 1):
    log_head: str = f'[{col_idx}. {col}] '
    val_min: float = df_X[col].min()
    val_max: float = df_X[col].max()
    n_val: int = df_X[col].dropna().nunique()
    var_type: str = 'Binary' if n_val == 2 and set(df_X[col].dropna().unique()) == {0, 1} else None
    logger.info(f'{log_head}n_val = {n_val}; range = [{val_min}, {val_max}]')
    record = {'feature': col,
              'n_values': n_val,
              'min': val_min,
              'max': val_max,
              'type': var_type}
    all_records.append(record)

########################################################################################################################
# Organize the records
########################################################################################################################
df_records: pd.DataFrame = pd.DataFrame(all_records)
df_records.to_csv(os.path.join(IN_DIR, 'MetaData_v0.csv'), index=False)
