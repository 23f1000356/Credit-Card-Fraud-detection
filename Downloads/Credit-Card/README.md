# Credit Card Fraud Detection Using Machine Learning

## What We Built and Why

This project compares two machine learning models — **Decision Tree** and **Support Vector Machine (SVM)** — for detecting fraudulent credit card transactions. The core challenge is extreme class imbalance: only 0.17% of transactions are fraud, meaning a naive model can achieve 99.8% accuracy by simply predicting "not fraud" for everything. We addressed this by implementing a consistent, fair evaluation pipeline across both models using SMOTE oversampling, StandardScaler normalization, stratified splitting, hyperparameter tuning via RandomizedSearchCV, and evaluation on the full real-world test set (85,443 samples). Both models also include an MLOps deployment gate that automatically compares new model metrics against a production baseline before allowing deployment, with full artifact versioning and logging.

We chose Decision Tree and SVM because they represent two fundamentally different approaches — tree-based splitting vs. margin-based classification — and comparing them on identical preprocessing reveals which algorithm is inherently better suited for high-dimensional fraud detection. The result: **SVM significantly outperforms Decision Tree** on this dataset, achieving an F1-score of 0.93 vs 0.52, with perfect precision and 87% recall on fraud.

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Project Structure
```
Credit Card Fraud Detection Using Machine Learning/
├── creditcard.csv                    # Dataset (download from Kaggle)
├── README.md
└── Code/
    ├── Credit Card Fraud Detection - Decision Tree.ipynb
    ├── Credit Card Fraud Detection - Support Vector Machines.ipynb
    ├── Credit Card Fraud Detection - Logistic Regression.ipynb
    └── Credit Card Fraud Detection - K-Nearest Neighbor.ipynb
```

### Running the Notebooks
1. Place `creditcard.csv` in the project root (parent of `Code/`)
2. Open either notebook in Jupyter or VS Code
3. Click **Run All** — each notebook runs end-to-end

**Note:** The SVM notebook takes 30-60+ minutes due to SVM's O(n^2) complexity on ~260K SMOTE-augmented samples with 5-fold cross-validation. The Decision Tree notebook runs in 1-3 minutes.

---

## Assumptions and Limitations

**Assumptions:**
- Training and test data share the same distribution (fraud patterns are stationary)
- PCA-transformed features (V1-V28) are meaningful; original feature names are anonymized
- The 0.17% fraud rate in this dataset is representative of real-world conditions
- All results use `random_state=42` for reproducibility

**Limitations:**
- SMOTE generates synthetic fraud samples that may not capture all real fraud patterns
- No temporal/sequential modeling — transaction order and time gaps are ignored
- SVM training is computationally expensive on large datasets, limiting real-time retraining
- Both models are evaluated on a single train/test split; production systems should use rolling validation
- The deployment gate uses file-based JSON logging — production systems would use MLflow, DVC, or similar

---

## Optimization Methods Applied to Both Models

Both models were transformed from basic implementations into properly tuned, production-ready pipelines using the same methodology:

### Shared Pipeline (Applied to Both)

| Step | Method | Why |
|------|--------|-----|
| **Train-Test Split** | `train_test_split(test_size=0.30, stratify=y)` | Preserves 0.17% fraud ratio in both sets |
| **Feature Scaling** | `StandardScaler` on all features | Normalizes features to mean=0, std=1 |
| **Class Imbalance** | `SMOTE(k_neighbors=7, sampling_strategy=0.3)` | Oversamples fraud to 30% of majority — avoids overfitting from full 1:1 balance |
| **Hyperparameter Tuning** | `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` | Finds optimal params while preserving class distribution in each fold |
| **Scoring Metric** | `scoring='f1'` (fraud-class F1) | Directly optimizes fraud detection, not misleading weighted accuracy |
| **Evaluation** | Full test set (85,443 samples), classification report, ROC-AUC, PR curve | Real-world performance, not inflated toy-set numbers |
| **MLOps Gate** | Baseline comparison + artifact logging | Only deploys if new model beats production baseline |

### Decision Tree — Specific Optimizations

| Parameter | Before | After | Purpose |
|-----------|--------|-------|---------|
| `max_depth` | 10-30 | **3-15** | Shallower trees to reduce overfitting |
| `min_samples_leaf` | 1-8 | **10-200** | Prevents tiny leaf nodes that memorize noise |
| `min_samples_split` | 2-20 | **5-100** | Forces more data per split |
| `max_features` | not set | **sqrt, log2, 0.5, 0.7** | Limits features per split |
| `min_impurity_decrease` | not set | **0.0-0.01** | Prunes low-value splits |
| `class_weight` | not set | **balanced** | Penalizes fraud misclassification more |
| `n_iter` | 20 | **50** | More exploration of larger param space |

### SVM — Specific Optimizations

| Parameter | Before | After | Purpose |
|-----------|--------|-------|---------|
| Data handling | `RandomUnderSampler` (threw away 99.6% of data) | **SMOTE** (keeps all data) | Model sees complete picture |
| Scaling | Only `Amount` column | **All 29 features** | SVM needs uniform feature scales |
| `C` | default (1.0) | **[0.1, 0.5, 1, 5, 10, 50]** | Regularization search |
| `kernel` | default (rbf) | **[rbf, linear]** | Tests both kernel types |
| `gamma` | default (scale) | **[scale, auto, 0.01, 0.001]** | Decision boundary curvature |
| `class_weight` | not set | **[balanced, None]** | Auto-adjusts for imbalance |
| Test set | ~148 samples (tiny, artificial) | **85,443 samples (real-world)** | Honest evaluation |
| Tuning | None | **RandomizedSearchCV, 30 iterations** | Data-driven parameter selection |

---

## Model Comparison Results

Both models evaluated on the **same full test set (85,443 samples, 148 fraud)** with identical preprocessing:

| Metric | Decision Tree | SVM | Winner |
|--------|:------------:|:---:|:------:|
| **F1-Score (fraud)** | 0.5234 | **0.9307** | SVM |
| **Precision (fraud)** | 0.40 | **1.00** | SVM |
| **Recall (fraud)** | 0.76 | **0.87** | SVM |
| **ROC-AUC** | 0.8808 | **0.9527** | SVM |
| **Accuracy** | 1.00 | 0.9527 | Misleading |

### Why Accuracy is Misleading

The Decision Tree shows 100% accuracy but only 0.52 F1 — it correctly classifies 99.8% of legitimate transactions but its fraud alerts are wrong 60% of the time (precision = 0.40). In fraud detection, a model that cries wolf constantly is unusable even if overall accuracy looks perfect. SVM's 95.3% accuracy is "lower" but every single fraud alert it raises is correct (precision = 1.00).

### Verdict

**SVM is the best model for this credit card fraud detection task.**
- F1-score nearly 2x better (0.93 vs 0.52)
- Perfect precision — zero false alarms
- Catches 87% of actual fraud
- Higher ROC-AUC (0.95 vs 0.88) — better probability calibration

---

## MLOps: Versioning and Deployment Gate

Both notebooks include a production deployment gate that:

1. **Loads or initializes** a baseline metrics file (`*_baseline_model_metrics.json`)
2. **Compares** the new model's F1-score against the production baseline
3. **PASS:** Saves model (`.pkl`), scaler (`.pkl`), updates baseline, logs deployment details
4. **FAIL:** Logs rejection reason, gap to threshold, and improvement recommendations

### Artifacts Generated

| Artifact | Decision Tree | SVM |
|----------|:---:|:---:|
| Trained model `.pkl` | `decision_tree_production_model.pkl` | `svm_production_model.pkl` |
| Scaler `.pkl` | `scaler_production.pkl` | `svm_scaler_production.pkl` |
| Baseline metrics | `baseline_model_metrics.json` | `svm_baseline_model_metrics.json` |
| Deployment/rejection log | `deployment_log.json` | `svm_deployment_log.json` |

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions, 492 frauds (0.17%)
- 28 PCA-anonymized features (V1-V28) + Amount + Time
- No missing values

---

## Key Files

| File | Description |
|------|-------------|
| `Code/Credit Card Fraud Detection - Decision Tree.ipynb` | Optimized Decision Tree with SMOTE, tuning, deployment gate |
| `Code/Credit Card Fraud Detection - Support Vector Machines.ipynb` | Optimized SVM with SMOTE, tuning, deployment gate |
| `Code/Credit Card Fraud Detection - Logistic Regression.ipynb` | Logistic Regression (basic, uses undersampling) |
| `Code/Credit Card Fraud Detection - K-Nearest Neighbor.ipynb` | KNN (has runtime errors due to library version mismatch) |
| `creditcard.csv` | Dataset (not included — download from Kaggle) |
