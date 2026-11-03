########################################################################################################################
# Plot the ROC curves across 8 models with averaged TPR & FPR values across seeds
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from _Helper_Scripts.binary_metrics import binary_metrics
from _Helper_Scripts.Logger import create_log
from itertools import product
from sklearn.metrics import auc, precision_recall_curve, roc_curve, auc
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Century Gothic"]

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Specify the scaling methods
########################################################################################################################
SEEDS: list[int] = [0, 1, 2, 42]

########################################################################################################################
# Load the predicted probabilities
########################################################################################################################
IN_DIR_2D: str = '_Results_Retraining_v2/'
model_dict: dict[str, pd.DataFrame] = {m: pd.read_csv(os.path.join(IN_DIR_2D, f'{m}_Probability_SEEDED.csv'))
                                       for m in ['ANN', 'EN', 'SVM', 'XGB', 'TF_unimodal', 'TF_multimodal',]}
IN_DIR_FUSION: str = '_Results_Retraining_Fusion_v2/'
model_dict |= {m: pd.read_csv(os.path.join(IN_DIR_FUSION, m, f'{m}_Probability_SEEDED.csv'))
               for m in ['TF_Fusion_unimodal', 'TF_Fusion_multimodal']}

model_order: list[str] = ['ANN', 'EN', 'SVM', 'XGB',
                          'TF_unimodal', 'TF_Fusion_unimodal',
                          'TF_multimodal', 'TF_Fusion_multimodal']
model_dict = {k: model_dict[k] for k in model_order}

########################################################################################################################
# Book-keeping for the estimated probabilities
########################################################################################################################
fpr_dict: dict = {k: [] for k in model_order}
tpr_dict: dict = {k: [] for k in model_order}

########################################################################################################################
# Obtain the estimated probability measures of the 8 models across 4 seeds
########################################################################################################################
for m, df_m in model_dict.items():
    y_true_base = None
    for seed in SEEDS:
        df_m_s: pd.DataFrame = df_m[df_m['seed'] == seed]
        if 'Model' in df_m_s.columns:
            df_m_s = df_m_s[df_m_s['Model'] == 'Raw']
        y_true: np.ndarray = df_m_s['y_test'].to_numpy()
        y_prob: np.ndarray = df_m_s['y_prob_platt'].to_numpy()
        if y_true_base is None:
            y_true_base = y_true
        else:
            assert np.all(y_true_base == y_true)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fpr_dict[m].append(fpr)
        tpr_dict[m].append(tpr)

logger.print('-'*100)

########################################################################################################################
# Interpolate each seed's TPR onto a common FPR grid, then average across seeds
########################################################################################################################
FPR_GRID: np.ndarray = np.linspace(0, 1, 201)
mean_tpr_dict: dict[str, np.ndarray] = {}
std_tpr_dict: dict[str, np.ndarray] = {}
mean_auc_dict: dict[str, float] = {}
std_auc_dict: dict[str, float] = {}

for m in model_order:
    interp_tprs: list[np.ndarray] = []
    seed_aucs: list[float] = []
    for fpr, tpr in zip(fpr_dict[m], tpr_dict[m]):
        interp_tpr: np.ndarray = np.interp(FPR_GRID, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        seed_aucs.append(auc(fpr, tpr))
    interp_arr: np.ndarray = np.vstack(interp_tprs)
    mean_tpr: np.ndarray = interp_arr.mean(axis=0)
    mean_tpr[-1] = 1.0
    mean_tpr_dict[m] = mean_tpr
    std_tpr_dict[m] = interp_arr.std(axis=0)
    mean_auc_dict[m] = float(np.mean(seed_aucs))
    std_auc_dict[m] = float(np.std(seed_aucs))
    logger.info(f'{m}: mean AUROC = {mean_auc_dict[m]:.3f} (SD = {std_auc_dict[m]:.3f})')

########################################################################################################################
# Plot
########################################################################################################################
########################################################################################################################
# Plot — one panel per model (row 0: baselines, row 1: transformer variants)
########################################################################################################################
N_ROWS, N_COLS = 2, 4
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, 8), sharex=True, sharey=True)

for idx, m in enumerate(model_order):
    row_idx: int = idx // N_COLS
    col_idx: int = idx % N_COLS
    ax = axes[row_idx, col_idx]

    # Mean curve only
    auroc: float = mean_auc_dict[m]
    ax.plot(FPR_GRID, mean_tpr_dict[m], color='tab:orange', linewidth=3, label=f'ROC curve\n(area = {auroc:.2f})')

    # Chance line
    ax.plot([0, 1], [0, 1], linestyle='--', color='tab:blue', linewidth=1.8, label='Chance level\n(area = 0.50)')

    if m in ['ANN', 'EN', 'SVM', 'XGB']:
        model_str = m
    else:
        model_str = {'TF_unimodal': 'Transformer\n(unimodal, standard)',
                     'TF_Fusion_unimodal': 'Transformer\n(unimodal, fusion)',
                     'TF_multimodal': 'Transformer\n(multimodal, standard)',
                     'TF_Fusion_multimodal': 'Transformer\n(multimodal, fusion)'}[m]

    ax.set_title(model_str, fontweight='bold', fontsize=20)
    ax.legend(loc='lower right', fontsize=14)

    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.5)

    if row_idx == N_ROWS - 1:
        ax.set_xlabel('False positive rate', fontsize=15)
    if col_idx == 0:
        ax.set_ylabel('True positive rate', fontsize=15)

# fig.suptitle('ROC Curves by Model (mean across 4 seeds, Platt-scaled probabilities)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

OUT_DIR: str = '_Final_Results/'
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, 'ROC_Curves.png'), dpi=300)
# plt.close(fig)