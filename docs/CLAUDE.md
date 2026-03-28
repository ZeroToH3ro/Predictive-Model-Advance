# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter lab
```

Open `Predictive_model.ipynb` and run all cells ("Restart Kernel and Run All"). Set `ras_path` in the final cell before running.

## Architecture Overview

This is a **notebook-driven ML pipeline** for predicting SVR (Sustained Virological Response) from hepatitis RAS (Resistance-Associated Substitution) features. There are no automated tests.

### Data Flow

```
data/data_RAS.xlsx  →  build_feature_matrix()  →  train_and_evaluate_models()  →  outputs/Experiment_Model_<timestamp>/
```

- Input: `data/data_RAS.xlsx` (Sheet1) with columns `Accession number`, `Respond` (`SVR`/`Non SVR`), numeric features, `Genotype`, and mutation columns (names containing `mut`, `mutation`, `resistance`, or `ras`).
- Target: `SVR → 0`, `Non SVR → 1` (binary classification, "Non SVR" is the positive/failure class).
- Outputs per run: Excel files (predictions, metrics) and PNGs (ROC overlay, metric bar chart, optional t-SNE/SHAP/learning curve).

### Shared Preprocessing Pipeline (all models)

`VarianceThreshold → StandardScaler → SMOTE → FiniteClipper → [PCA 95% for SVM] → model`

`FiniteClipper` (in `model_utils.py`) is a custom sklearn transformer that replaces non-finite values and clips to quantile bounds. It is the only non-notebook Python module.

### Models Trained

SVM (RBF + PCA), Elastic Net (logistic), Random Forest, GBM, Decision Tree, optional XGBoost. All are selected via `GridSearchCV(cv=5, scoring="roc_auc")`. Decision thresholds use the **Youden index** (`tpr - fpr`) from the ROC curve.

### Notebooks

- `Predictive_model.ipynb`: primary end-to-end pipeline (feature engineering → training → evaluation → outputs).
- `plot_comparison_combine.ipynb`: additional plotting and result aggregation across runs.

## Coding Conventions

- Python: PEP 8, 4-space indentation, `snake_case` for functions/variables.
- Keep all file paths relative (`data/...`, `outputs/...`).
- `outputs/` is git-ignored — do not commit run artifacts.
- Clear large notebook cell outputs before committing.

## Key Reference Files

- `WORKFLOW.md`: step-by-step pipeline description with all inputs, steps, and output files.
- `EVALUDATION.md`: benchmark results table and per-model hyperparameter rationale (2W1H format).
- `AGENTS.md`: repository structure and commit/PR guidelines.
