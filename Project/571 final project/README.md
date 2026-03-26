# Bankruptcy Prediction — MGMT 571

**3rd Place out of 53 Teams** in a graduate-level machine learning competition at Purdue University.

## Overview

Binary classification task: predict whether a company will go bankrupt (`class = 1`) or not (`class = 0`) based on 64 financial attributes derived from corporate financial statements.

## Approach

### Data Preprocessing
- **Winsorization** (0.5th–99.5th percentile) to cap extreme outliers across all 64 features
- **KNN Imputation** (k=5) on winsorized data to handle missing values without data leakage

### Model
- **XGBoost** (`binary:logistic`) with tuned hyperparameters:
  - 1,000 estimators, learning rate 0.02, max depth 6
  - Subsample 0.9, column sampling 0.8
  - L1 (alpha=0.1) and L2 (lambda=1.0) regularization

### Evaluation
- **5-Fold Stratified Cross-Validation AUC: 0.907**
- Metric: ROC-AUC (area under the receiver operating characteristic curve)

| Fold | AUC    |
|------|--------|
| 1    | 0.9179 |
| 2    | 0.9359 |
| 3    | 0.8963 |
| 4    | 0.9116 |
| 5    | 0.8746 |
| **Mean** | **0.9073** |

## Repository Structure

```
├── 571_final.ipynb                  # Main notebook with full pipeline
├── train.csv                        # Training data (10,000 rows × 65 columns)
├── test.csv                         # Test data (8,000 rows × 65 columns)
├── bankruptcy_sample_submission.csv # Sample submission format
├── Output/
│   └── Best_AUC.csv                # Best submission file
└── README.md
```

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost
- **Visualization** — matplotlib, seaborn
- **Evaluation** — ROC-AUC, Stratified K-Fold CV

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook 571_final.ipynb
```

## Results

Achieved a **0.907 mean CV AUC** and placed **3rd out of 53 teams** in the competition. Key factors contributing to the result:
- Careful outlier handling via winsorization preserved signal while reducing noise
- KNN imputation maintained feature relationships better than simple mean/median fills
- Well-tuned XGBoost with regularization prevented overfitting on the imbalanced dataset
