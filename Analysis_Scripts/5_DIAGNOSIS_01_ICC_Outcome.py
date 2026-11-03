########################################################################################################################
# Compute the ICC for the outcome variable
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

########################################################################################################################
# Load the outcome labels
########################################################################################################################
OUTCOME_PATH: str = '_Helper_Files/15min_Data/Outcome_v2.csv'
df_y: pd.DataFrame = pd.read_csv(OUTCOME_PATH, usecols=['Record_ID', 'BeiweId', 'BSI_GSI_Case'])

########################################################################################################################
# Descriptive summary
########################################################################################################################
n_records: int = len(df_y)
n_participants: int = df_y['BeiweId'].nunique()
prevalence: float = df_y['BSI_GSI_Case'].mean()
records_per_participant: pd.Series = df_y.groupby('BeiweId')['Record_ID'].count()
print(f'Records: {n_records}, Participants: {n_participants}, Prevalence: {prevalence:.2%}')
print(f'Records per participant: median={records_per_participant.median():.0f}, '
      f'range=({records_per_participant.min()}, {records_per_participant.max()})')

########################################################################################################################
# Approach 1: Linear mixed model ICC (treats binary outcome as continuous)
# BSI_GSI_Case ~ 1 + (1 | BeiweId)
# ICC = tau^2 / (tau^2 + sigma^2)
########################################################################################################################
lmm = smf.mixedlm('BSI_GSI_Case ~ 1', data=df_y, groups=df_y['BeiweId'])
lmm_fit = lmm.fit(reml=True)
print('\n--- Linear Mixed Model ---')
print(lmm_fit.summary())

tau2_lmm: float = float(lmm_fit.cov_re.iloc[0, 0])
sigma2_lmm: float = float(lmm_fit.scale)
icc_lmm: float = tau2_lmm / (tau2_lmm + sigma2_lmm)
print(f'\nLinear LMM ICC: {icc_lmm:.4f}')
print(f'  tau^2 (between): {tau2_lmm:.6f}')
print(f'  sigma^2 (within): {sigma2_lmm:.6f}')

########################################################################################################################
# Approach 2: Logistic GLMM ICC (proper latent-variable approach)
# Fits a logistic mixed model to get tau^2 on the logit scale
# ICC = tau^2 / (tau^2 + pi^2/3)
########################################################################################################################
glmm = BinomialBayesMixedGLM.from_formula(
    'BSI_GSI_Case ~ 1',
    vc_formulas={'BeiweId': '0 + C(BeiweId)'},
    data=df_y
)
glmm_fit = glmm.fit_vb()
print('\n--- Logistic GLMM (Bayesian VB) ---')
print(glmm_fit.summary())

# vcp_mean holds variance component parameters as log standard deviations
log_tau: float = float(glmm_fit.vcp_mean[0])
tau_logit: float = float(np.exp(log_tau))
tau2_logit: float = tau_logit ** 2
sigma2_latent: float = (np.pi ** 2) / 3
icc_logistic: float = tau2_logit / (tau2_logit + sigma2_latent)
print(f'\nLogistic GLMM ICC: {icc_logistic:.4f}')
print(f'  log(tau) (log SD): {log_tau:.6f}')
print(f'  tau (SD, logit scale): {tau_logit:.4f}')
print(f'  tau^2 (between, logit scale): {tau2_logit:.6f}')
print(f'  pi^2/3 (latent within): {sigma2_latent:.6f}')