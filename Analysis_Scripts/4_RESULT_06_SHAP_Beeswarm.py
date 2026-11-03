########################################################################################################################
# Plot the SHAP beeswarm for the best transformer (fusion) model, using the seed-averaged attributions produced by
# 4_RESULT_04_SHAP_Organize.py. Three panels: top survey features, top daily sensor-summary features, and top
# temporal (fine-grained, per-bin) sensor features — each ranked by mean |SHAP| among features that clear a
# minimum-observed-fraction eligibility floor (MIN_OBSERVED_FRAC).
#
# Daily and temporal sensor features are kept as separate panels rather than merged into one 'Sensor' panel due to their
# difference in attribution magnitudes (by roughly 20x where daily dominates).
#
# Static features cover essentially all @survey items plus @daily and Weekend@time which carry attribution only at bin
# index 0, so their beeswarm points are one per observed test record. Temporal (sensor-stream) features can carry
# attribution at every observed bin, so their points are flattened across all observed (record, bin) cells.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from _Helper_Scripts.Logger import create_log
from _Helper_Scripts.SHAP_Helper import summarize_shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Century Gothic"]

########################################################################################################################
# Create logging file
########################################################################################################################
logger = create_log(f'_Logging_Files/{os.path.basename(__file__)}')

########################################################################################################################
# Runtime parameters
########################################################################################################################
MODALITY: str = 'multimodal'
TOP_K: int = 5

# Minimum fraction of test records a STATIC (survey/@daily) feature must be observed in to be eligible for the
# top-feature ranking. Derived from diagnostics: several skip-gated survey clusters (sleep detection,
# WorkFT_ReasonStop/Duration, EducYears, PsyTx_Inpat, WorkDiff_Reasons, Occup_code, TAPS.*) sit at or below
# ~136/210 (~65%) observed and were shown to produce unstable, outcome-skewed SHAP estimates across seeds; the
# next tier (MonthlyInc_*) sits at 158/210 (~75%). 140/210 falls in the gap between these tiers.
# Not applied to temporal (sensor-stream) features, which have a different, much larger effective denominator
# (N_test x L bins) and showed no analogous instability in the diagnostics.
MIN_OBSERVED_FRAC: float = 140 / 210

# Features excluded from the temporal-sensor panel despite being empirically classified as Temporal (i.e. not
# exactly constant across all 96 bins for every training record).
TEMPORAL_PANEL_EXCLUDE: set[str] = set()        # Not implemented

########################################################################################################################
# Paths
########################################################################################################################
IN_DIR: str = f'_Final_Results/_Results_SHAP/TF_Fusion_{MODALITY}/'
OUT_DIR: str = IN_DIR
npz_path: str = os.path.join(IN_DIR, f'SHAP_Averaged_{MODALITY}.npz')

########################################################################################################################
# Load the averaged SHAP tensor and associated arrays from 4_RESULT_04_SHAP_Organize.py
########################################################################################################################
data = np.load(npz_path, allow_pickle=True)
shap_3d_mean: np.ndarray = data['shap_3d_mean']             # (N_test, L, F)
X_test_3D: np.ndarray = data['X_test_3D']                   # (N_test, L, F)
M_test: np.ndarray = data['M_test']                         # (N_test, L, F)
feats: list[str] = list(data['feats'])
static_idxs: list[int] = list(data['static_idxs'])
temporal_idxs: list[int] = list(data['temporal_idxs'])

N_test: int = X_test_3D.shape[0]
static_set: set[int] = set(static_idxs)
feat_to_idx: dict[str, int] = {f: i for i, f in enumerate(feats)}
logger.info(f'Loaded SHAP data: shap_3d_mean {shap_3d_mean.shape}, N_test={N_test}, '
           f'static={len(static_idxs)}, temporal={len(temporal_idxs)}')
logger.print('-' * 100)

########################################################################################################################
# Recompute the feature-level summary from the averaged tensor
########################################################################################################################
df_summary: pd.DataFrame = summarize_shap(shap_3d_mean, M_test, feats, static_idxs, temporal_idxs)

########################################################################################################################
# Eligibility: static features must clear MIN_OBSERVED_FRAC of N_test observed at bin 0; temporal features have
# no floor applied (see rationale above). n_observed is logged for all features either way, for QA.
########################################################################################################################
min_observed_n: int = int(np.ceil(MIN_OBSERVED_FRAC * N_test))
logger.info(f'MIN_OBSERVED_FRAC={MIN_OBSERVED_FRAC:.3f} -> min_observed_n={min_observed_n}/{N_test}')

n_observed: list[int] = []
eligible: list[bool] = []
for f in df_summary['Feature']:
    idx: int = feat_to_idx[f]
    if idx in static_set:
        n_obs: int = int(M_test[:, 0, idx].sum())
        n_observed.append(n_obs)
        eligible.append(n_obs >= min_observed_n)
    else:
        n_obs = int(M_test[:, :, idx].sum())
        n_observed.append(n_obs)
        eligible.append(True)
df_summary['n_observed'] = n_observed
df_summary['eligible'] = eligible

n_excluded: int = int((~df_summary['eligible']).sum())
logger.info(f'{n_excluded} static features excluded by the observed-count floor '
            f'(of {len(static_idxs)} static features total)')
logger.print('-' * 100)

########################################################################################################################
# Categorize eligible features into three panels:
#   Survey   — @survey suffix
#   Daily    — non-survey STATIC features (daily-aggregate sensor summaries, Weekend@clock, with_home_address, ...)
#   Temporal — non-survey TEMPORAL features (raw ACC/GPS/PS streams at 15min/1h/6h resolution)
# then take the top TOP_K of each by mean |SHAP|
########################################################################################################################
df_eligible: pd.DataFrame = df_summary[df_summary['eligible']].copy()

# Features treated as Daily in the beeswarm despite being empirically classified as Temporal by the static/
# temporal detection (i.e. not exactly constant across all 96 bins for every record). Weekend@clock is
# conceptually a daily-level context feature — its rare within-record variation comes only from records whose
# 24h window crosses a weekday/weekend boundary, not from genuine within-day sensor dynamics.
DAILY_OVERRIDE: set[str] = {'Weekend@clock'}


def _categorize(row: pd.Series) -> str:
    if row['Feature'].endswith('@survey'):
        return 'Survey'
    if row['Feature'] in DAILY_OVERRIDE:
        return 'Daily'
    return 'Daily' if row['FeatureType'] == 'Static' else 'Temporal'


df_eligible['Category'] = df_eligible.apply(_categorize, axis=1)

top_survey: pd.DataFrame = (df_eligible[df_eligible['Category'] == 'Survey']
                            .sort_values('MeanAbsSHAP', ascending=False).head(TOP_K))
top_daily: pd.DataFrame = (df_eligible[df_eligible['Category'] == 'Daily']
                           .sort_values('MeanAbsSHAP', ascending=False).head(TOP_K))
top_temporal: pd.DataFrame = (df_eligible[(df_eligible['Category'] == 'Temporal')
                                          & (~df_eligible['Feature'].isin(TEMPORAL_PANEL_EXCLUDE))]
                              .sort_values('MeanAbsSHAP', ascending=False).head(TOP_K))

n_excl_temporal: int = int(df_eligible[(df_eligible['Category'] == 'Temporal')
                                       & (df_eligible['Feature'].isin(TEMPORAL_PANEL_EXCLUDE))].shape[0])
if n_excl_temporal > 0:
    logger.info(f'{n_excl_temporal} temporal feature(s) excluded from panel eligibility by name: '
               f'{sorted(TEMPORAL_PANEL_EXCLUDE)}')

for label, df_top in [('survey', top_survey), ('daily sensor', top_daily), ('temporal sensor', top_temporal)]:
    if len(df_top) < TOP_K:
        logger.info(f'WARNING: only {len(df_top)} eligible {label} features available (requested top {TOP_K})')

for label, df_top in [('Top survey features', top_survey),
                      ('Top daily sensor features', top_daily),
                      ('Top temporal sensor features', top_temporal)]:
    logger.info(f'{label}:')
    for _, row in df_top.iterrows():
        logger.info(f"  {row['Feature']}: MeanAbsSHAP={row['MeanAbsSHAP']:.4f}, n_observed={row['n_observed']}")
logger.print('-' * 100)

########################################################################################################################
# Extract (shap, value) point pairs per feature, respecting the static-vs-temporal extraction difference.
# Static: one point per observed test record, from bin 0. Temporal: one point per observed (record, bin) cell.
########################################################################################################################


def extract_points(feature: str) -> tuple[np.ndarray, np.ndarray]:
    """
    :param feature: str. Feature name.
    :return: (shap_vals, feat_vals). 1D arrays restricted to observed cells for this feature.
    """
    idx: int = feat_to_idx[feature]
    if idx in static_set:
        mask: np.ndarray = M_test[:, 0, idx].astype(bool)
        s: np.ndarray = shap_3d_mean[:, 0, idx][mask]
        v: np.ndarray = X_test_3D[:, 0, idx][mask]
    else:
        mask = M_test[:, :, idx].astype(bool)
        s = shap_3d_mean[:, :, idx][mask]
        v = X_test_3D[:, :, idx][mask]
    return s, v


survey_points: list[tuple[np.ndarray, np.ndarray]] = [extract_points(f) for f in top_survey['Feature']]
daily_points: list[tuple[np.ndarray, np.ndarray]] = [extract_points(f) for f in top_daily['Feature']]
temporal_points: list[tuple[np.ndarray, np.ndarray]] = [extract_points(f) for f in top_temporal['Feature']]

########################################################################################################################
# Human-readable feature labels — populate once the top features above are known (see the log for the exact
# names). Falls back to the raw feature name for anything not listed here.
########################################################################################################################
FEATURE_LABEL_DICT: dict[str, str] = {
    'WorkSituation_11@survey': 'Full-time student',
    'LFQ_LeisFriend_Impair@survey': 'Difficulty socializing with friends',
    'LFQ_DutWorkSch_Impair@survey': 'Difficulty fulfilling work or school duties',
    'LFQ_LeisFam_Impair@survey':  'Difficulty socializing with family',
    'Duties_Home_C2@survey': 'Difficulty completing household duties\nwithout conflict',

    'active_bout_count@daily': 'Number of sustained activity bouts',
    'active_min_per_obs_hr@daily': 'Active minutes per observed hour',
    'steps_per_obs_hr@daily': 'Steps per observed hour',
    'coverage_frac@daily': 'Accelerometer data coverage',
    'longest_active_bout_min@daily': 'Longest sustained activity duration',

    'PS#n_min^6h': 'Phone usage (6h)',
    'ACC#n_min^6h': 'Accelerometer coverage (6h)',
    'ACC#availability': 'Accelerometer active (15m)',
    'PS#screen_on_fraction^1h': 'Screen-on fraction (1h)',
    'PS#n_min^15min': 'Phone usage (15m)',
}

########################################################################################################################
# Beeswarm helpers
########################################################################################################################
ROW_SPACING: float = 1.5


def beeswarm_y(x_vals: np.ndarray, y_center: float,
               bin_width: float = 0.005, dot_size: float = 0.004) -> np.ndarray:
    """
    Compute vertical jitter for a beeswarm by binning x values and spreading points within each bin.
    :param x_vals: np.ndarray. SHAP values for one feature.
    :param y_center: float. Row center for this feature.
    :param bin_width: float. Bin width along the x-axis used to group nearby points for spreading.
    :param dot_size: float. Vertical spacing applied between spread points.
    :return: np.ndarray. y-position for each point in x_vals.
    """
    y_out: np.ndarray = np.full(len(x_vals), y_center, dtype=float)
    bins: np.ndarray = np.floor(x_vals / bin_width).astype(int)
    for b in np.unique(bins):
        mask: np.ndarray = bins == b
        count: int = int(mask.sum())
        if count <= 1:
            continue
        offsets: np.ndarray = np.linspace(-(count - 1) / 2, (count - 1) / 2, count) * dot_size
        offsets = np.clip(offsets, -ROW_SPACING * 0.35, ROW_SPACING * 0.35)
        y_out[np.where(mask)[0]] = y_center + offsets
    return y_out


def plot_beeswarm(ax: plt.Axes,
                  points: list[tuple[np.ndarray, np.ndarray]],
                  feat_names: list[str],
                  title: str,
                  cmap,
                  norm: Normalize,
                  dot_size: float = 15.0) -> None:
    """
    :param ax: plt.Axes. Target axes.
    :param points: list of (shap_vals, feat_vals) pairs, one per feature, already restricted to observed cells.
                   Lengths may differ across features (ragged) — this is expected: static features contribute up
                   to N_test points, temporal features up to N_test x L points.
    :param feat_names: list[str]. Feature names, same order as points, used for y-tick labels.
    :param title: str. Panel title.
    :param cmap: Colormap for feature-value coloring.
    :param norm: Normalize. Assumes feature values are already scaled to [0, 1] (true post-preprocessing).
    :param dot_size: float. Scatter marker size.
    """
    n_feats: int = len(points)
    y_positions: list[float] = [float(n_feats - 1 - fi) * ROW_SPACING for fi in range(n_feats)]

    for fi, (shap_vals, feat_vals) in enumerate(points):
        if len(shap_vals) == 0:
            continue
        y_center: float = y_positions[fi]
        ax.scatter(shap_vals, beeswarm_y(shap_vals, y_center),
                   c=cmap(norm(feat_vals)), s=dot_size, alpha=0.7, linewidths=0, zorder=3)

    for fi in range(n_feats - 1):
        mid: float = (y_positions[fi] + y_positions[fi + 1]) / 2
        ax.axhline(y=mid, color='lightgray', linewidth=0.8, linestyle='-', zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_LABEL_DICT.get(f, f) for f in feat_names], fontsize=14)
    ax.set_xlabel('SHAP value', fontsize=15, labelpad=10)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title(title, fontsize=17, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_ylim(y_positions[-1] - 0.8, y_positions[0] + 0.8)


########################################################################################################################
# Create the 3-panel beeswarm plot. Each panel has its own independent x-axis (no sharex), so the daily and
# temporal panels correctly render at their own SHAP-magnitude scale rather than being squashed by the other's
# range — this is what makes the magnitude gap between them visible in the figure.
########################################################################################################################
cmap = plt.cm.RdBu_r
norm = Normalize(vmin=0, vmax=1)    # feature values are already MinMax-scaled to [0, 1] by preprocess_data

fig = plt.figure(figsize=(11, 16))
gs = GridSpec(3, 1, figure=fig, right=0.88, hspace=0.3)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

plot_beeswarm(axes[0], survey_points, top_survey['Feature'].tolist(),
             title=f'Top {len(survey_points)} survey features', cmap=cmap, norm=norm)
plot_beeswarm(axes[1], daily_points, top_daily['Feature'].tolist(),
             title=f'Top {len(daily_points)} daily sensor features', cmap=cmap, norm=norm)
plot_beeswarm(axes[2], temporal_points, top_temporal['Feature'].tolist(),
             title=f'Top {len(temporal_points)} temporal sensor features', cmap=cmap, norm=norm)

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Feature value', fontsize=16)
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.ax.tick_params(labelsize=14)

fig_path: str = os.path.join(OUT_DIR, f'SHAP_Beeswarm_{MODALITY}_top{TOP_K}.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
logger.info(f'Beeswarm plot saved to {fig_path}')
logger.print('-' * 100)

########################################################################################################################
# Save the selected feature lists and eligibility stats for reference / manuscript tables
########################################################################################################################
pd.concat([top_survey.assign(Panel='Survey'),
          top_daily.assign(Panel='Daily'),
          top_temporal.assign(Panel='Temporal')], ignore_index=True).to_csv(
    os.path.join(OUT_DIR, f'SHAP_Beeswarm_TopFeatures_{MODALITY}.csv'), index=False)
df_summary.to_csv(os.path.join(OUT_DIR, f'SHAP_Summary_WithEligibility_{MODALITY}.csv'), index=False)
logger.info(f'Saved top-feature list and full eligibility summary to {OUT_DIR}')
