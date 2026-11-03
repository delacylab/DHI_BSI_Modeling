########################################################################################################################
# Run transformer modeling on the 3D multimodal dataset via 4-fold internal validation.
# Parallelizes the 576-configuration grid across available GPUs (strided chunking).
# Each (fold, worker) writes an independent CSV; final aggregation concatenates all results.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import glob
import numpy as np
import os
import pandas as pd
import torch
import torch.multiprocessing as mp
import warnings
from itertools import product
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.Preprocessing import preprocess_data
from _Helper_Scripts.Transformer_Modeling import ArrayDataset, parallel_hpo_worker
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

########################################################################################################################
# Create the logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Modality, outcome, background columns
########################################################################################################################
MODALITY: str = 'multimodal'
OUTCOME: str = 'BSI_GSI_Case'
BKGD_COLS: list[str] = ['Record_ID', 'BeiweId', 'AppleId', 'Sex', 'Survey_Day', 'Survey_Time',
                        'BSI_GSI_prorate', 'BSI_GSI_T', 'BSI_GSI_Case',
                        'bin_index', 'Bin_Start_Time', 'Bin_End_Time', 'is_pad']

########################################################################################################################
# Fixed runtime parameters
########################################################################################################################
BATCH_SIZE: int = 32
DROPOUT: float = 0.1
MAX_EPOCHS: int = 300
SCHEDULER_PATIENCE: int = 20
RANDOM_STATE: int = 42

########################################################################################################################
# Hyperparameter grid (288 configurations)
########################################################################################################################
D_MODEL_GRID: tuple[int, ...] = (128, 256)
N_HEAD_GRID: tuple[int, ...] = (2, 4)
N_LAYERS_GRID: tuple[int, ...] = (2, 3)
N_UNITS_GRID: tuple[int, ...] = (128, 256, 512)
LR_GRID: tuple[float, ...] = (1e-3, 1e-4, 1e-5)
COST_MUL_GRID: tuple[int, ...] = (1, 2, 5, 10)
HANDLE_NAN_GRID: tuple[str, ...] = ('explicit',)
FACTOR_GRID: tuple[float, ...] = (0.5, 0.1)

FULL_GRID: list[tuple] = list(product(D_MODEL_GRID, N_HEAD_GRID, N_LAYERS_GRID, N_UNITS_GRID,
                                      LR_GRID, COST_MUL_GRID, HANDLE_NAN_GRID, FACTOR_GRID))
assert len(FULL_GRID) == 576, f'expected 576 configs, got {len(FULL_GRID)}'

########################################################################################################################
# Paths
########################################################################################################################
IN_DIR: str = '_Helper_Files/15min_Data/'
X_PATH: str = os.path.join(IN_DIR, 'Features_v4.parquet')
Y_PATH: str = os.path.join(IN_DIR, 'Outcome_v2.csv')
OUT_DIR: str = f'_Results_Validation/TF_{MODALITY}/'
os.makedirs(OUT_DIR, exist_ok=True)

########################################################################################################################
# Worker entry point for mp.spawn (must be module-level to be picklable)
########################################################################################################################


def worker_entry(rank: int,
                 configs_chunks: list[list[tuple]],
                 train_ds: ArrayDataset,
                 val_ds: ArrayDataset,
                 out_dir: str,
                 fold_id: str,
                 batch_size: int,
                 dropout: float,
                 max_epochs: int,
                 scheduler_patience: int,
                 random_state: int):
    parallel_hpo_worker(rank=rank,
                        configs=configs_chunks[rank],
                        train_ds=train_ds,
                        val_ds=val_ds,
                        out_dir=out_dir,
                        fold_id=fold_id,
                        batch_size=batch_size,
                        dropout=dropout,
                        max_epochs=max_epochs,
                        scheduler_patience=scheduler_patience,
                        random_state=random_state)


########################################################################################################################
# Strided chunking of configs across GPUs
########################################################################################################################


def chunk_configs(configs: list[tuple], n_chunks: int) -> list[list[tuple]]:
    return [configs[i::n_chunks] for i in range(n_chunks)]


########################################################################################################################
# Main
########################################################################################################################
if __name__ == '__main__':

    ####################################################################################################################
    # Multiprocessing start method — 'spawn' is required for CUDA
    ####################################################################################################################
    mp.set_start_method('spawn', force=True)

    ####################################################################################################################
    # Detect GPUs
    ####################################################################################################################
    N_GPUS: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f'Number of GPUs detected: {N_GPUS}')
    logger.info(f'Grid size: {len(FULL_GRID)} configurations')

    ####################################################################################################################
    # Load data
    ####################################################################################################################
    df_X: pd.DataFrame = pd.read_parquet(X_PATH)
    df_y: pd.DataFrame = pd.read_csv(Y_PATH)

    ####################################################################################################################
    # Restrict to non-TEST partition for internal validation
    ####################################################################################################################
    K_FOLD_STRS: list[str] = sorted([k for k in df_y['Partition_Validation'].unique().tolist() if k != 'TEST'])
    df_y = df_y[df_y['Partition_Validation'] != 'TEST'].copy()
    y_ids: list[int] = df_y['Record_ID'].unique().tolist()
    df_y = df_y.set_index('Record_ID').loc[y_ids].reset_index()
    df_X = df_X[df_X['Record_ID'].isin(y_ids)].copy().set_index('Record_ID').loc[y_ids].reset_index()
    logger.info(f'Full training feature dataset loaded with dimension = {df_X.shape}')
    logger.print('-' * 100)

    ####################################################################################################################
    # Loop over folds
    ####################################################################################################################
    for k_fold_str in K_FOLD_STRS:
        log_head: str = f'[{k_fold_str}] '

        # Split partitions relative to this fold
        df_y_train: pd.DataFrame = df_y[df_y['Partition_Validation'] != k_fold_str].copy()
        y_train_ids: list[int] = df_y_train['Record_ID'].unique().tolist()
        df_X_train: pd.DataFrame = df_X.set_index('Record_ID').loc[y_train_ids].reset_index()

        df_y_val: pd.DataFrame = df_y[df_y['Partition_Validation'] == k_fold_str].copy()
        y_val_ids: list[int] = df_y_val['Record_ID'].unique().tolist()
        df_X_val: pd.DataFrame = df_X.set_index('Record_ID').loc[y_val_ids].reset_index()

        # Preprocess
        logger.info(f'{log_head}Preprocessing...')
        X_train_3D, X_val_3D, ids_train, ids_val, feats = preprocess_data(df_X_train=df_X_train,
                                                                          df_X_val=df_X_val,
                                                                          bkgd_cols=BKGD_COLS,
                                                                          groupby_col='Record_ID',
                                                                          sd_away=3,
                                                                          dim='3D',
                                                                          logger=logger)
        assert list(ids_train) == y_train_ids, 'ids_train order mismatch'
        assert list(ids_val) == y_val_ids, 'ids_val order mismatch'
        assert len(feats) == X_train_3D.shape[2], 'feats count mismatch'

        # Filter to unimodal features
        if MODALITY == 'unimodal':
            unimodal_feats: list[str] = [f for f in feats if not f.endswith('@survey')]
            feat_idxs: list[int] = [feats.index(f) for f in unimodal_feats]
            X_train_3D = X_train_3D[:, :, feat_idxs]
            X_val_3D = X_val_3D[:, :, feat_idxs]
            feats = unimodal_feats
            assert len(feats) == X_train_3D.shape[2]
        logger.info(f'{log_head}Preprocessed: X_train {X_train_3D.shape}, X_val {X_val_3D.shape}')

        # Build ArrayDatasets
        y_train: np.ndarray = df_y_train[OUTCOME].to_numpy(dtype=np.float32)
        y_val: np.ndarray = df_y_val[OUTCOME].to_numpy(dtype=np.float32)
        train_ds: ArrayDataset = ArrayDataset(X_train_3D, y_train, np.array(ids_train))
        val_ds: ArrayDataset = ArrayDataset(X_val_3D, y_val, np.array(ids_val))

        # Chunk configs across GPUs (strided for load balancing)
        chunks: list[list[tuple]] = chunk_configs(FULL_GRID, N_GPUS)
        for i, chk in enumerate(chunks):
            logger.info(f'{log_head}Worker {i}: {len(chk)} configs')

        # Run HPO
        logger.info(f'{log_head}Starting HPO grid across {N_GPUS} GPU(s)...')
        if N_GPUS == 1:
            # Single-GPU debug path: call worker directly, no subprocess
            parallel_hpo_worker(rank=0,
                                configs=FULL_GRID,
                                train_ds=train_ds,
                                val_ds=val_ds,
                                out_dir=OUT_DIR,
                                fold_id=k_fold_str,
                                batch_size=BATCH_SIZE,
                                dropout=DROPOUT,
                                max_epochs=MAX_EPOCHS,
                                scheduler_patience=SCHEDULER_PATIENCE,
                                random_state=RANDOM_STATE)
        else:
            mp.spawn(worker_entry,
                     args=(chunks, train_ds, val_ds, OUT_DIR, k_fold_str,
                           BATCH_SIZE, DROPOUT, MAX_EPOCHS, SCHEDULER_PATIENCE, RANDOM_STATE),
                     nprocs=N_GPUS,
                     join=True)
        logger.info(f'{log_head}HPO grid completed.')
        logger.print('-' * 100)

    ####################################################################################################################
    # Aggregate results across all folds and workers
    ####################################################################################################################
    result_files: list[str] = sorted(glob.glob(os.path.join(OUT_DIR, 'Fold_*_Worker_*_Results.csv')))
    logger.info(f'Aggregating {len(result_files)} result files...')

    all_frames: list[pd.DataFrame] = []
    for fp in result_files:
        fold_id: str = os.path.basename(fp).split('_')[1]
        df: pd.DataFrame = pd.read_csv(fp)
        df.insert(1, 'k_fold', fold_id)
        all_frames.append(df)

    df_all: pd.DataFrame = pd.concat(all_frames, ignore_index=True)
    summary_path: str = os.path.join(OUT_DIR, f'TF_{MODALITY}_Validation.csv')
    df_all.to_csv(summary_path, index=False)
    logger.info(f'Summary saved to {summary_path}. Shape: {df_all.shape}')
