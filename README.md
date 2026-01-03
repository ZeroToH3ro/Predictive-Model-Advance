# Predictive-Model-Advance

## Setup

### Create a virtual environment (Python 3.9+)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
python -m pip install jupyter
jupyter lab
```

Open `Predictive_model_3_Seq.ipynb` (or `Predictive_model.ipynb`) and run the cells.

## Data

The dataset `data_training_162_model.csv` is included in the repo. Keep the notebook in the same folder so relative paths work.

## What the notebook produces

- Trains multiple classifiers (SVM, Elastic Net, Random Forest, GBM, Decision Tree, optional XGBoost) via `train_and_evaluate_models`.
- Saves per-model predictions/metrics to `outputs/Experiment_Model_<timestamp>/` (Excel files).
- Generates comparison visuals: a grouped-bar chart of core metrics and an overlay ROC chart for all models (PNG files in the same run directory).
- SHAP explainability runs on the best SVM model sample if available.

## Workflow details

See `WORKFLOW.md` for a step-by-step explanation of how the notebook builds features, trains models, selects thresholds, and writes outputs.
