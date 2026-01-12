# Repository Guidelines

## Project Structure
- `Predictive_model.ipynb`: primary end-to-end notebook (feature building → model training → metrics/plots).
- `plot_comparison_combine.ipynb`: additional plotting/aggregation notebook.
- `data/`: input files (e.g., `data/data_RAS.xlsx`; the workflow also references `data/data_training_162_model.csv`).
- `outputs/`: per-run results written to `outputs/Experiment_Model_<timestamp>/` (Excel + PNG).
- `requirements.txt`: Python dependencies.
- `WORKFLOW.md`: detailed pipeline description and expected inputs/outputs.

## Build, Test, and Development Commands
- Create env: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run locally: `python -m pip install jupyter && jupyter lab`
  - Open `Predictive_model.ipynb` and “Restart Kernel and Run All” before pushing changes.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8, `snake_case` for functions/variables, `CapWords` for classes.
- Notebooks:
  - Keep paths relative (e.g., `data/...`, `outputs/...`) so runs work on other machines.
  - Prefer adding reusable logic as functions (instead of duplicating cells).
  - Avoid committing large cell outputs; clear outputs if they are noisy or huge.

## Testing Guidelines
- This repo is notebook-driven and does not currently include automated tests.
- If you add Python modules, include `pytest` tests under `tests/` (name files `test_*.py`) and document the command in `README.md`.

## Commit & Pull Request Guidelines
- Current history uses short, informal subjects like `update ...` / `fix ...` (no enforced convention).
- Recommended: imperative, 1-line summary (e.g., `fix: align RAS join keys`, `update: ROC comparison plot`).
- PRs: include a brief description, what changed, and (if plots/metrics changed) attach the updated PNGs or describe the metric deltas.

## Data, Outputs, and Reproducibility
- Do not commit run artifacts: `outputs/` is ignored via `.gitignore`.
- If adding datasets, confirm they are safe to commit (no sensitive/proprietary data) and keep them under `data/`.
