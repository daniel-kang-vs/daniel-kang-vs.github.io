# Public project evidence guide

Updated November 2, 2026. This review checks committed artifacts; it does not retrain models or establish production adoption.

## Seed Planning Dashboard

- **Source:** [hubner_predictions-app/pipeline.py](https://github.com/daniel-kang-vs/seed-demand-forecasting/blob/main/hubner_predictions-app/pipeline.py)
- **Evidence:** docs/data/synthetic_cleaned.json and docs/data/synthetic_merged.json power the hosted descriptive dashboard.
- **Interpretation:** The public deliverable is an inspectable workflow and interactive planning interface. It demonstrates how a planner can narrow a product group, inspect its history, and question a forecast; it does not establish an inventory reduction or realized revenue lift.
- **Next validation:** Synthetic data demonstrates functionality, not business impact. Archived prediction images are separate artifacts, not metrics calculated from the synthetic filter selections. A deployment needs input-schema checks, configurable paths, and a time-based forecast evaluation.

## Athlete Scoring — Sogility

- **Source:** [README.md](https://github.com/daniel-kang-vs/sogility-athlete-scoring/blob/main/README.md)
- **Evidence:** Public RPS and APS Python modules, 13 functional checks, and synthetic sensitivity probes are available. Confidential player records and internal interfaces are excluded.
- **Interpretation:** The framework explains one athlete through a peer-relative score and a separate performance band. Parent views emphasize progress and five skills; trainer views emphasize entry, comparison, and history. The repository documents implementation, edge-case safeguards, and remaining methodology issues.
- **Next validation:** Synthetic stress probes expose a hard cohort-threshold discontinuity and lower-is-better APS calibration concerns. APS reference benchmarks need versioning for longitudinal comparisons. No match-outcome validation or production deployment is established.

## Packaging Standardization — BabyShip

- **Source:** [IP_modeling.ipynb](https://github.com/daniel-kang-vs/stroller-box-optimization/blob/main/IP_modeling.ipynb)
- **Evidence:** IP_modeling.ipynb contains the model and stored solver output. Running the optimization requires a suitable Gurobi license.
- **Interpretation:** Recorded output selects seven types with total billable weight of 1,718.016 lb, versus 1,613.312 lb for the unconstrained per-product benchmark. The added 104.704 lb is 6.49%. Lotus Travel Crib and City Mini GT2 account for 67.712 lb, or about 64.7% of that penalty.
- **Next validation:** This is a modeled, equally weighted product comparison. It does not include shipment frequency, negotiated carrier rates, box purchase costs, damage, or measured warehouse labor. A 17-product baseline does not establish that 17 distinct boxes were previously stocked.

## Bankruptcy Risk Scoring

- **Source:** [571_final.ipynb](https://github.com/daniel-kang-vs/bankruptcy-prediction/blob/main/571_final.ipynb)
- **Evidence:** 571_final.ipynb stores the five-fold scores for the raw-input model. Dashboard class counts and missingness are derived from train.csv.
- **Interpretation:** The stored fold AUCs are 0.918, 0.936, 0.896, 0.912 and 0.875, averaging 0.9073. The historical competition placement is reported as third of 53 teams; a leaderboard artifact is not bundled in the repository.
- **Next validation:** Cross-validation is not prospective validation. A deployment would need calibration, drift checks, and a threshold based on the cost of missed failures versus false alarms. No preprocessing lift or operational impact is established by the saved output.

## NCAA Selection & Seed Modeling

- **Source:** [FinalFourAnalyticsChallenge2026.ipynb](https://github.com/daniel-kang-vs/ncaa-tournament-prediction/blob/main/FinalFourAnalyticsChallenge2026.ipynb)
- **Evidence:** NCAA_Seed_Training_Set2.0.csv supports the dashboard’s dataset counts. Historical model scores remain unverified, not newly reproduced.
- **Interpretation:** The committed artifact demonstrates the modeling workflow and feature engineering. The earlier narrative reports 0.966 AUC and a 0.653 threshold, but the notebook contains no saved evaluation outputs to verify those figures. The dashboard therefore shows source-derived dataset analysis rather than invented model curves.
- **Next validation:** Imputation precedes cross-validation, and the meta-model is fitted and scored on the same out-of-fold feature matrix. The score is not an independent nested evaluation. Refit preprocessing within folds, evaluate the full stack on held-out seasons, and save predictions before claiming generalization.

## NYC Taxi Fleet Positioning

- **Source:** [Implementation and outputs](https://github.com/daniel-kang-vs/daniel-kang-vs.github.io/tree/main/Project/mgmt590-team03-project)
- **Evidence:** Public portfolio source: Project/mgmt590-team03-project. outputs/bakeoff_results.csv, outputs/task2_ABC.csv and outputs/ENGINE_AUDIT.md document the displayed comparisons.
- **Interpretation:** In the committed 35-scenario bakeoff, the empirical method has the lowest mean newsvendor cost: about $369,468 per scenario. Its mean fill rate is 55.4%. These are modeled evaluation outputs, not realized savings. The dashboard shows the exact committed comparison and three stakeholder slots.
- **Next validation:** The outputs are historical scenarios with modeled cost assumptions. Repositioning time, demand shifts and operational constraints require further validation. A local agent prototype exists in the source; this does not establish deployment or live dispatch use.


NCAA input-quality note: some win–loss fields contain date-like strings (for example `8-Sep`). Recover the original record notation and validate parsing before retraining; the dashboard counts rows and selection labels without interpreting those fields.
