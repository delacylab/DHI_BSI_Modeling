########################################################################################################################
# Preparing time-context features in terms of clock time and day of the week
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
# Define the constant 2 * pi
########################################################################################################################
TWO_PI: float = 2 * np.pi

########################################################################################################################
# Load the feature file
########################################################################################################################
data_path: str = os.path.join(IN_DIR, 'Features_v2.parquet')
df: pd.DataFrame = pd.read_parquet(data_path)
logger.print(f'Loaded: {data_path}  Shape: {df.shape}')
logger.print('=' * 100)

########################################################################################################################
# Helper functions
########################################################################################################################


def encode_time_of_day(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mins = ts.dt.hour * 60 + ts.dt.minute
    frac = mins / 1440
    return np.sin(TWO_PI * frac).to_numpy(), np.cos(TWO_PI * frac).to_numpy()


def encode_day_of_week(ts: pd.Series) -> pd.Series:
    return (ts.dt.dayofweek >= 5).astype(int)


########################################################################################################################
# Compute and append time-context features
########################################################################################################################
tod_sin, tod_cos = encode_time_of_day(df['Bin_Start_Time'])
df['TOD_sin@clock'] = tod_sin
df['TOD_cos@clock'] = tod_cos
df['Weekend@clock'] = encode_day_of_week(df['Bin_Start_Time'])

logger.print('Time-context features added: TOD_sin@clock, TOD_cos@clock, Weekend@clock')
logger.print(f'  TOD_sin  — mean={df["TOD_sin@clock"].mean():.4f}, std={df["TOD_sin@clock"].std():.4f}')
logger.print(f'  TOD_cos  — mean={df["TOD_cos@clock"].mean():.4f}, std={df["TOD_cos@clock"].std():.4f}')
logger.print(f'  Weekend  — {int(df["Weekend@clock"].sum())} timestamps flagged as weekend '
             f'({df["Weekend@clock"].mean()*100:.1f}%)')
logger.print('=' * 100)

########################################################################################################################
# Export
########################################################################################################################
out_path: str = os.path.join(IN_DIR, 'Features_v3.parquet')
df.to_parquet(out_path, index=False)
logger.print(f'Saved: {out_path}  Shape: {df.shape}  Records: {df["Record_ID"].nunique()}')
