# Model Evaluation (2W1H)

This evaluation is based on the latest saved run in `outputs/Experiment_Model_20260112_195923/` using **RAS-only features** from `data/data_RAS.xlsx` (Sheet1).

## Data & Setup

- Dataset: 162 samples, 100 columns (`SVR`: 101, `Non SVR`: 61).
- Target mapping: `SVR -> Success (0)`, `Non SVR -> Failure (1)`.
- Split: 60/40 stratified (`train=97`, `test=65`).
- Shared pipeline (all models): `VarianceThreshold` → `StandardScaler` → `SMOTE` → `FiniteClipper` → `model`.
- Training selection: `GridSearchCV(cv=5, scoring=roc_auc)` picks best hyperparameters.
- Decision threshold: selected with **Youden index** on the ROC curve (`tpr - fpr`) to turn probabilities into class labels.

## Results Summary (Test Set, n=65)

`Failure` is treated as the positive class in the confusion counts below.

| Model         | AUC   | F1    | Accuracy | Precision | Recall | Threshold | TP | FP | TN | FN |
| ------------- | ----- | ----- | -------- | --------- | ------ | --------- | -- | -- | -- | -- |
| GBM           | 0.944 | 0.863 | 0.892    | 0.815     | 0.917  | 0.352     | 22 | 5  | 36 | 2  |
| Random Forest | 0.940 | 0.852 | 0.877    | 0.767     | 0.958  | 0.284     | 23 | 7  | 34 | 1  |
| Elastic Net   | 0.914 | 0.851 | 0.892    | 0.870     | 0.833  | 0.602     | 20 | 3  | 38 | 4  |
| XGBoost       | 0.912 | 0.816 | 0.862    | 0.800     | 0.833  | 0.302     | 20 | 5  | 36 | 4  |
| SVM           | 0.834 | 0.750 | 0.846    | 0.938     | 0.625  | 0.506     | 15 | 1  | 40 | 9  |
| Decision Tree | 0.762 | 0.682 | 0.785    | 0.750     | 0.625  | 0.500     | 15 | 5  | 36 | 9  |

High-level takeaways:
- Best overall ranking by AUC: **GBM** (0.944) and **Random Forest** (0.940).
- Best “catch failures” behavior (Recall): **Random Forest** (0.958) at the cost of more false positives.
- Most conservative “only flag failures when confident” (Precision): **SVM** (0.938) but lower recall.

## Algorithm Evaluations (2W1H)

### SVM (RBF kernel + PCA 95%)
**What**  
A margin-based classifier using an RBF kernel (non-linear boundary). The pipeline adds PCA (keep 95% variance) before the SVM.

**Why**  
RAS features can interact non-linearly. SVMs are strong on small-to-medium datasets with many correlated features after scaling. In this run it produced very high precision (few false alarms) but missed more failures (lower recall).

**How**  
Tuned parameters:
- `C` (`[0.1, 1, 10]`): controls the margin vs. training errors. This is the main bias/variance knob for SVMs.
- `gamma` (`["scale", 0.1, 0.01]`): controls how “local” the RBF decision boundary is. Too high overfits; too low underfits.

Why these parameters:
- They directly control decision boundary complexity, which is usually the biggest driver of SVM performance.
- The selected range is a practical coarse grid for small datasets; you can expand it (e.g., `C` up to 100, `gamma` down to 1e-3) if you want a deeper search.

### Elastic Net (Logistic Regression)
**What**  
A linear probabilistic classifier with a mix of L1 and L2 regularization (“elastic net”).

**Why**  
This is the most interpretable option and often performs strongly when signals are mostly additive/linear. It also handles many features well via regularization. Here it achieved strong AUC and a good precision/recall balance.

**How**  
Tuned parameters:
- `C` (`[0.01, 0.1, 1, 10]`): inverse regularization strength. Smaller `C` means stronger shrinkage (less overfitting).
- `l1_ratio` (`[0.1, 0.5, 0.9]`): mixes L2 vs L1. Higher values push sparsity (feature selection) which can help with noisy/high-dimensional inputs.

Why these parameters:
- They control how much the model generalizes vs. memorizes and whether it prefers sparse vs. smooth solutions.
- On 162 samples, regularization strength is usually critical for stability.

### Random Forest
**What**  
An ensemble of decision trees trained on bootstrapped samples with feature randomness, averaged to reduce variance.

**Why**  
Works well on mixed / non-linear tabular signals and is relatively robust to feature scaling issues. In this run it produced the highest recall (few missed failures), but with more false positives.

**How**  
Tuned parameters:
- `n_estimators` (`[300, 500]`): more trees reduce variance and improve stability (at higher compute cost).
- `max_depth` (`[None, 20]`): controls tree complexity. Depth limits reduce overfitting.
- `min_samples_leaf` (`[1, 2]`): prevents very small leaf nodes, improving generalization on small datasets.

Why these parameters:
- They directly manage the bias/variance trade-off of forests and are the most impactful “first set” to tune.

### GBM (GradientBoostingClassifier)
**What**  
A boosted ensemble of shallow trees built sequentially, each correcting the previous model’s errors.

**Why**  
Often excels on structured/tabular data by capturing non-linearities and interactions. It achieved the best AUC and best overall F1 in this run.

**How**  
Tuned parameters:
- `n_estimators` (`[200, 400]`): number of boosting stages; more stages can improve fit but risk overfitting.
- `learning_rate` (`[0.05, 0.1]`): step size; smaller values typically require more estimators but generalize better.
- `max_depth` (`[2, 3]`): limits base tree complexity (keeps boosting “weak learners” weak).

Why these parameters:
- They define the core complexity of a GBM (how fast it learns and how complex each tree can be).
- The grid is deliberately small to avoid overfitting and to keep compute reasonable on repeated CV.

### Decision Tree
**What**  
A single tree that learns rules by splitting features to reduce impurity.

**Why**  
Highly interpretable but high-variance: small data changes can drastically change the tree. As expected, it underperformed ensemble methods here.

**How**  
Tuned parameters:
- `max_depth` (`[3, 5, None]`): limits complexity; shallow trees generalize better.
- `min_samples_leaf` (`[1, 2]`): reduces overfitting by requiring more samples per leaf.

Why these parameters:
- They are the simplest, most effective regularizers for a decision tree.

### XGBoost
**What**  
A gradient-boosted tree system with regularization and advanced sampling controls, typically stronger than basic GBM when tuned.

**Why**  
Usually competitive on tabular problems with complex interactions. Here it performed well, but slightly below GBM/RF on AUC.

**How**  
Tuned parameters:
- `n_estimators` (`[300, 500]`) and `learning_rate` (`[0.05, 0.1]`): control boosting strength and stage count (same trade-off as GBM).
- `max_depth` (`[3, 5]`): tree complexity per stage.
- `subsample` (`[0.8, 1.0]`) and `colsample_bytree` (`[0.8, 1.0]`): stochastic training reduces overfitting.
Fixed (but important) parameter:
- `scale_pos_weight = neg/pos` (computed from the training split): compensates for class imbalance for the positive class (`Failure`).

Why these parameters:
- They are the highest-impact knobs for bias/variance and overfitting control in boosted trees.
- Sampling parameters (`subsample`, `colsample_bytree`) are especially useful on small datasets to reduce variance.
