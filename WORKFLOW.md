# Machine Learning Workflow (Predictive Model)

This document explains how `Predictive_model.ipynb` creates the model results, metrics, and plots.

## Inputs
- `data/data_RAS.xlsx` (Sheet1): required input containing RAS features and labels.
  - Required columns: `Accession number`, `Respond` (`SVR` or `Non SVR`)
  - Optional columns used as features:
    - Numeric columns (excluding `No`/`id`)
    - `Genotype` (one-hot encoded into `Genotype_*`)
    - Mutation list columns (name contains `mut`, `mutation`, `resistance`, or `ras`) converted to mutation counts
- Sequence file (optional): CSV/XLSX set via `sequence_path` when `use_sequence=True`
  - Required columns: `Accession number`, `Respond`, plus sequence columns:
    - Amino acids: `NS3_aa`, `NS5A_aa`, `NS5B_aa` (`seq_type="aa"`)
    - Nucleotides: `NS3_Nu`, `NS5A_Nu`, `NS5B_Nu` (`seq_type="dna"`)

## Workflow steps (ordered)
1. **(Optional) Analyze RAS significance**
   - `analyze_ras_significance(...)` runs Fisher's exact test over `all_resistance_muts_ns3/ns5a/ns5b`.
   - Prints mutations significantly associated with `Non SVR` (filters by `min_support` and `alpha`).
2. **Build features + targets**
   - `build_feature_matrix(...)` supports two modes:
     - `use_sequence=False` (current notebook default): uses RAS-only features; targets come from `data_RAS.xlsx:Respond`.
     - `use_sequence=True`: loads and encodes sequences, aligns with RAS on `Accession number`, and concatenates `seq_*` + RAS features.
   - Targets are built by `build_target`: `SVR -> 0`, `Non SVR -> 1` (raises if other label values exist).
   - NaN/±inf are replaced with 0 in `X`.
3. **Train/test split**
   - 60/40 split with stratification (`random_state=42`).
4. **Model pipelines**
   - Preprocessing: `VarianceThreshold` → `StandardScaler` → `SMOTE` → `FiniteClipper`.
   - SVM adds `PCA(n_components=0.95)` before the classifier.
5. **Model selection**
   - `GridSearchCV(cv=5, scoring="roc_auc")` over:
     - SVM (RBF), Elastic Net logistic regression, Random Forest, GBM, Decision Tree
     - Optional XGBoost (if importable and `include_xgboost=True`)
6. **Threshold selection + metrics**
   - `optimize_threshold(...)` uses Youden index (`tpr - fpr`) from the ROC curve.
   - Metrics: Accuracy, Precision, Recall, F1, AUC, plus the chosen `Threshold`.
7. **Save plots and tables**
   - Writes Excel summaries + comparison plots to `save_path` (the per-run `run_dir`).
8. **Optional analysis**
   - `plot_learning_curve(...)` saves an AUC learning curve for the SVM model (if trained).
   - `explain_with_shap(...)` runs SHAP on a small sample (if `shap` is installed) and produces genotype-focused plots.

## Outputs (per run)
Created in `outputs/Experiment_Model_<timestamp>/`:
- `ensemble_predictions_<label>.xlsx`: per-sample predictions/probabilities.
- `model_metrics_<label>.xlsx`: overall metrics for all models.
- `model_metric_comparison_<label>.png`: bar chart of core metrics.
- `roc_comparison_<label>.png`: ROC overlay for all models.
- `learning_curve_<label>.png`: SVM learning curve (if generated).
- `shap_values_<label>.xlsx`: SHAP values table (if generated).
- `shap_summary_<label>.png`: SHAP summary plot (genotype features only, if generated).
- `shap_heatmap_<label>.png`: SHAP heatmap (genotype features only, if generated).

## How the results are created
The final notebook cell sets:
- `sequence_path`, `ras_path`, and `seq_type`
- `run_dir = outputs/Experiment_Model_<timestamp>`
Then it calls:
- `analyze_ras_significance(...)` (prints Fisher-test results)
- `build_feature_matrix(...)` to assemble `X`, `y`, and `feature_names`
- `train_and_evaluate_models(...)` to train, score, and write Excel outputs
- `plot_model_comparisons(...)` to save the metric/ROC plots
- Optional SVM learning curve and SHAP explainability

## How to reproduce
1. Install dependencies from `requirements.txt`.
2. Launch Jupyter and open `Predictive_model.ipynb`.
3. Set `ras_path` (and optionally `sequence_path` + `use_sequence=True`) in the final cell.
4. Run all cells to generate the outputs in `outputs/Experiment_Model_<timestamp>/`.

To include XGBoost, install `xgboost` and leave `include_xgboost=True` in
`train_and_evaluate_models(...)`. If XGBoost is not installed, it is skipped.

Notes:
- The notebook also imports `imblearn` (SMOTE), `scipy` (Fisher test), and optionally `shap`. If they are missing in your environment, install them (e.g., `pip install imbalanced-learn scipy shap`).
