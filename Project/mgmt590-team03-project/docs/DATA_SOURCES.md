# Dashboard data provenance

The model-comparison charts use `../outputs/bakeoff_results.csv` (35 scenarios). The slot comparison uses `../outputs/task2_ABC.csv`, following the source selection documented in `../outputs/ENGINE_AUDIT.md`.

`data.json` is a snapshot of those committed outputs. It does not rerun demand forecasting or dispatch optimization. Total cost C includes fleet-adjustment cost; comparing only its newsvendor cost would omit part of the objective.

The empirical row is the lowest-cost row in the global bakeoff, including the empirical benchmark. The historical audit also discusses selection among learned forecasting methods; those are distinct comparisons. Global means are not values for the selected stakeholder slot.
