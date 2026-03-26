# NCAA Tournament Prediction — Final Four Analytics Challenge 2026

**[Live Demo](https://daniel-kang-vs.github.io/ncaa-tournament-prediction/)**

## Overview

A two-stage machine learning pipeline predicting NCAA tournament selection and seeding using an 8-model stacked ensemble with 50 engineered features. Built for the Final Four Analytics Challenge 2026.

## Approach

### Two-Stage Pipeline
1. **Selection Model** (Classification) — Predicts whether a team makes the tournament (AUC = 0.966)
2. **Seed Model** (Regression) — Predicts tournament seed for selected teams (MAE optimized)

### 8-Model Stacked Ensemble
- **Base models:** HistGradientBoosting, GradientBoosting, ExtraTrees, RandomForest (classifier + regressor variants)
- **Meta-learners:** LogisticRegression (selection), Ridge (seed)
- **Validation:** 10-fold stratified cross-validation with out-of-fold predictions

### Feature Engineering (15 → 50 features)
- W-L string parsing into wins, losses, win% for 8 record columns
- Conference tier encoding, power 6 flags
- NET momentum, improvement signals
- Quality wins, bad losses, composite SOS metrics

## Key Results

| Metric | Value |
|--------|-------|
| Selection AUC | 0.966 |
| Optimal Threshold | 0.653 (PR-curve calibrated) |
| Teams Selected | 70 (vs 68 actual) |
| Features | 50 engineered |

## Tech Stack

Python, scikit-learn, pandas, NumPy, matplotlib, Chart.js
