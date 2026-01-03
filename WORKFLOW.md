# Machine Learning Workflow (Predictive Model)

This document explains how `predictive_model.ipynb` creates the model results, metrics, and plots.

## Inputs
- `data/data_training_162_model.csv`: sequence data and labels.
- `data/data_RAS.xlsx` (Sheet1): RAS features and mutation lists.
- Required columns:
  - Sequence columns: `NS3_aa`, `NS5A_aa`, `NS5B_aa` for `seq_type="aa"` or `NS3_Nu`, `NS5A_Nu`, `NS5B_Nu` for `seq_type="dna"`.
  - Label column: `Respond` with values `SVR` or `Non SVR`.
  - Join key: `Accession number` (used to align sequence + RAS rows).

## Workflow steps (ordered)
1. **Load sequence data**
   - `load_sequence_data` reads CSV/XLSX.
   - Sequence columns depend on `seq_type` (`aa` or `dna`).
2. **Encode sequences**
   - Each sequence is mapped to integer tokens (amino acid or nucleotide dictionary).
   - Sequences are padded to the max length across all rows so tensors align.
3. **Load RAS features**
   - `load_ras_features` reads `data_RAS.xlsx`.
   - Numeric columns are kept (excluding `No`, `id`).
   - `Genotype` is one-hot encoded.
   - Mutation list columns (e.g., `all_resistance_muts_ns3`) are converted to mutation counts.
4. **Build the full feature matrix**
   - Sequence and RAS rows are aligned on `Accession number` (intersection only).
   - Sequence tensors are flattened and concatenated with RAS features.
   - NaN and infinite values are replaced with zeros.
5. **Build targets**
   - `Respond` is mapped: `SVR -> 0`, `Non SVR -> 1`.
6. **Train/test split**
   - 60/40 split with stratification on the label.
7. **Model pipelines**
   - Shared preprocessing: `VarianceThreshold` → `StandardScaler` → `SMOTE` → `FiniteClipper`.
   - SVM includes PCA (95% variance) before the classifier.
8. **Model selection**
   - Grid search with 5-fold CV, scoring by ROC-AUC.
   - Models: SVM, Elastic Net, Random Forest, GBM, Decision Tree, optional XGBoost.
9. **Threshold selection + metrics**
   - Uses Youden index on the ROC curve to pick the best probability threshold.
   - Metrics computed: Accuracy, Precision, Recall, F1, AUC.
10. **Save outputs**
   - Prediction files and metric tables are written to the run folder.
   - Comparison plots are generated (metric bars + multi-model ROC).
11. **Optional analysis**
   - `analyze_ras_significance` uses Fisher's exact test for RAS associations.
   - `plot_learning_curve` runs for the SVM model if available.
   - `explain_with_shap` runs SHAP on a small SVM sample if installed.

## Outputs (per run)
Created in `outputs/Experiment_Model_<timestamp>/`:
- `ensemble_predictions_<label>.xlsx`: per-sample predictions/probabilities.
- `model_metrics_<label>.xlsx`: overall metrics for all models.
- `model_metrics_<label>_others.xlsx`: metrics for non-SVM models.
- `svm_metrics_<label>.xlsx`: metrics for SVM only.
- `svm_predictions_<label>.xlsx`: SVM probabilities and labels.
- `model_metric_comparison_<label>.png`: bar chart of core metrics.
- `roc_comparison_<label>.png`: ROC overlay for all models.
- `learning_curve_<label>.png`: SVM learning curve (if generated).

## How the results are created
The final notebook cell sets:
- `sequence_path`, `ras_path`, and `seq_type`
- `run_dir = outputs/Experiment_Model_<timestamp>`
Then it calls:
- `build_feature_matrix(...)` to assemble `X` and `y`
- `train_and_evaluate_models(...)` to train, score, and write Excel outputs
- `plot_model_comparisons(...)` to save the metric/ROC plots
- Optional SVM learning curve and SHAP explainability

## How to reproduce
1. Install dependencies from `requirements.txt`.
2. Launch Jupyter and open `predictive_model.ipynb`.
3. Set paths and `seq_type` in the final cell.
4. Run all cells to generate the outputs in `outputs/Experiment_Model_<timestamp>/`.

To include XGBoost, install `xgboost` and leave `include_xgboost=True` in
`train_and_evaluate_models(...)`. If XGBoost is not installed, it is skipped.
