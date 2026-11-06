<div align="right">
  Last update: 2026 November 3, 11:08 MT (by Wayne Lam)
</div>
<hr>

# Classical machine learning versus transformers for predicting daily psychological distress from passive smartphone sensing and self-report: a prospective cohort study

[![medRxiv_badge](https://img.shields.io/badge/medRxiv-2026.11.04.26367218-blue)](https://www.medrxiv.org/content/10.64898/2026.11.04.26367218v1)
[![license_badge](https://img.shields.io/badge/License-Apache_2.0-8A2BE2)](https://www.apache.org/licenses/LICENSE-2.0)

<div align="center">
  <img width="1024" height="338" alt="Gemini_Generated_Image_31xmyq31xmyq31xm" src="https://github.com/user-attachments/assets/41bff789-da31-4a25-afda-414f61dff6cd" />
</div>



# :crystal_ball: Abstract #

**Objective:** We test whether data collected from daily passive smartphone sensing and self-reports can predict daily significant psychological distress with meaningful potential clinical utility, comparing across best-in-class machine learning algorithms.

**Methods:** In this prospective cohort study, 169 community-dwelling adults completed baseline surveys followed by 7 days of passive smartphone sensing (accelerometer, GPS, power state) and end-of-day Brief Symptom Inventory-18 (BSI-18) assessments. Cases were defined as days (total n = 1,062) where participants' psychological distress measured with the sex-specific BSI-18 Global Severity Index T-score was ≥63 (prevalence 11.2%). Four classical models (artificial neural network, elastic net, support vector machine [SVM], extreme gradient boosting [XGB]) and four transformers (sensor-only or multimodal; standard or fusion architecture) were tuned, Platt-calibrated and evaluated on held-out participants (210 records, 25 cases) with participant-level bootstrap 95% confidence intervals (CIs).

**Results:** The SVM (area under the receiver operating characteristic curve [AUROC] 0.84; 95% CI 0.71–0.93), the multimodal fusion transformer (0.84; 0.69–0.95) and XGB (0.82; 0.63–0.94) discriminated best; sensor-only transformers performed at chance (0.49–0.50). At the default threshold of 0.50, elastic net logistic regression was the best classifier (sensitivity 0.56, positive predictive value 0.64). At 0.12 (near the outcome prevalence), the SVM detected all cases (negative predictive value 1.00), but positive predictive values were low for all models (0.13–0.26). Decision-curve analysis favored most models over treat-all and treat-none lines at thresholds between 0.05 and 0.25. Baseline survey items, mainly on functional impairment, had the largest feature importance.

**Conclusions:** Compact classical models performed as well as transformers, which were considerably more computationally intensive. Decision-curve analysis supports further clinical exploration of these tools for triaging daily psychological distress, though our results suggest the relative value of psychosocial assessments and digital data should be carefully analyzed. External validation is needed before clinical use.

# :file_folder: Repository Structure #

```
DHI_BSI_Modeling/
├── Analysis_Scripts/
│   ├── _Helper_Scripts/          # Shared utility modules (preprocessing, modeling, metrics, etc.)
│   ├── 1_PROCESS_*.py            # Feature engineering from raw sensor and survey data
│   ├── 2_PREPROCESS_*.py         # Data partitioning and metadata generation
│   ├── 3_MODEL_*.py              # Hyperparameter optimization and model retraining
│   ├── 4_RESULT_*.py             # Performance evaluation, figures, and SHAP analysis
│   ├── 5_DIAGNOSIS_*.py          # Intraclass correlation diagnostics
│   └── 6_ORGANIZE_*.py           # Data dictionary assembly with feature importance
├── Supplementary_Table_1_-_Data_Dictionary_v1.csv
└── README.md
```

# :page_with_curl: Script Description #

Scripts are numbered to reflect their execution order. All scripts are located in `Analysis_Scripts/`.

### Stage 1: Feature Engineering ###

| Script | Description |
|---|---|
| `1_PROCESS_00_Outcome_Identification.py` | Identifies the binary outcome (BSI-18 GSI T-score ≥ 63) and exports the base outcome file. |
| `1_PROCESS_01_Sensor_Aggregates.py` | Aggregates raw accelerometer, GPS, and power state data into 96 × 15-minute bin sensor features per participant-day. |
| `1_PROCESS_02_Rate_of_Change.py` | Computes rate-of-change derived features from sensor aggregates. |
| `1_PROCESS_03_Day_Sensor_Aggregates.py` | Computes day-level summary features from 15-minute bin sensor data. |
| `1_PROCESS_04_Clock_Time.py` | Adds clock time features encoding the time of day for each bin. |
| `1_PROCESS_05_Survey.py` | Merges baseline survey features with the sensor feature table. |

### Stage 2: Preprocessing ###

| Script | Description |
|---|---|
| `2_PREPROCESS_01_Partitioning.py` | Generates participant-level stratified training-validation-test splits using seed search to minimize prevalence drift. |
| `2_PREPROCESS_02_MetaData.py` | Generates a metadata summary file for the analytic sample. |

### Stage 3: Model Development ###

| Script | Description |
|---|---|
| `3_MODEL_01_2D_Hyperparameter_Optimization.py` | 4-fold cross-validated hyperparameter optimization for 2D classical models (EN, SVM, XGB, ANN). |
| `3_MODEL_02_2D_Hyperparameter_Retrain_Seeded.py` | Retrains 2D classical models with best hyperparameters across 4 random seeds. |
| `3_MODEL_03_3D_Unimodal_Hyperparameter_Optimization.py` | Hyperparameter optimization for the standard transformer (sensor-only). |
| `3_MODEL_04_3D_Unimodal_Hyperparameter_Optimization_Fusion.py` | Hyperparameter optimization for the fusion transformer (sensor-only). |
| `3_MODEL_05_3D_Multimodal_Hyperparameter_Optimization.py` | Hyperparameter optimization for the standard transformer (sensor + survey). |
| `3_MODEL_06_3D_Multimodal_Hyperparameter_Optimization_Fusion.py` | Hyperparameter optimization for the fusion transformer (sensor + survey). |
| `3_MODEL_07_3D_Unimodal_Hyperparameter_Retrain_Seeded.py` | Retrains the standard unimodal transformer across 4 random seeds. |
| `3_MODEL_08_3D_Multimodal_Hyperparameter_Retrain_Seeded.py` | Retrains the standard multimodal transformer across 4 random seeds. |
| `3_MODEL_09_3D_Unimodal_Hyperparameter_Retrain_Fusion_Seeded.py` | Retrains the fusion unimodal transformer across 4 random seeds. |
| `3_MODEL_10_3D_Multimodal_Hyperparameter_Retrain_Fusion_Seeded.py` | Retrains the fusion multimodal transformer across 4 random seeds. |

### Stage 4: Results and Figures ###

| Script | Description |
|---|---|
| `4_RESULT_01_Performance_Statistics_Pooled.py` | Computes point-estimate performance metrics from seed-pooled predicted probabilities across all 8 models and 4 calibration methods. |
| `4_RESULT_02_Performance_Statistics_Pooled_CI.py` | Computes 95% cluster-bootstrap confidence intervals (2,000 resamples by participant) on seed-pooled performance metrics. |
| `4_RESULT_03_ROC_Pooled.py` | Plots ROC curves from seed-pooled Platt-scaled probabilities for all 8 models. |
| `4_RESULT_04_DCA_Pooled.py` | Plots decision-curve analysis (standardized net benefit) from seed-pooled Platt-scaled probabilities. |
| `4_RESULT_05_SHAP_Organize.py` | Computes seed-averaged GradientSHAP attributions for the fusion multimodal transformer on the test partition. |
| `4_RESULT_06_SHAP_Beeswarm.py` | Generates a 3-panel SHAP beeswarm figure (survey / daily sensor / temporal sensor features). |

### Stage 5: Diagnostics ###

| Script | Description |
|---|---|
| `5_DIAGNOSIS_01_ICC_Outcome.py` | Computes outcome-level intraclass correlation coefficients (ICC) via Bayesian logistic GLMM. |
| `5_DIAGNOSIS_02_ICC_Sensor_Features.py` | Computes per-feature ICC for sensor features via linear random-intercept models. |

### Stage 6: Data Dictionary ###

| Script | Description |
|---|---|
| `6_ORGANIZE_01_2D_Importance.py` | Computes feature importance for 2D models (mean \|SHAP\| for ANN/XGB; signed coefficients for EN/SVM) and appends to the data dictionary. |
| `6_ORGANIZE_02_3D_Importance.py` | Computes GradientSHAP feature importance for all 4 transformer models and appends to the data dictionary. |

# :bar_chart: Study Data #

The study sample consisted of 400 Utah-based participants recruited using a stratified random sampling design with two crossed factors—difficulties in emotion regulation and perceived life stress—to balance participants at high and low risk of episodic mental health conditions. Eligibility required age ≥ 18 years, Utah residency, smartphone ownership, English fluency, and health care through University of Utah Health or Intermountain Health. After completing baseline questionnaires on psychiatric, medical, and psychosocial history, participants entered a 7-day in situ phase, providing continuous passive smartphone sensor data (accelerometer at 10 Hz, GPS at 1 Hz, and event-based power state logging via the Beiwe Research Platform) and daily self-report measures (BSI-18). Data were collected between June 2022 and June 2024. Of the 400 participants, 253 provided sensor data; the analysis was restricted to 189 iPhone users for device consistency, with a final analytic sample of 169 participants (1,062 participant-days) after applying outcome-based inclusion criteria.

Baseline questionnaire features (260 in total) included validated measures of emotion regulation (DERS), perceived stress (PSS-4), depression/anxiety/stress symptoms (DASS-21), substance use risk (TAPS), psychosocial functioning (LFQ), demographics, clinical history, and medical conditions.

The data supporting the findings of this study are currently under review for deposition in a public repository. Until they become publicly available, the dataset may be obtained from Brian Baucom (brian.baucom@utah.edu) upon reasonable request.

# :pencil2: Ethics #

This study was approved by the Institutional Review Board of the University of Utah (IRB #00149365; approved March 28, 2022). All participants provided written informed consent, obtained electronically, prior to enrollment. This study is reported in accordance with the STROBE statement and the TRIPOD+AI guideline.

# :globe_with_meridians: License #
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for details.
