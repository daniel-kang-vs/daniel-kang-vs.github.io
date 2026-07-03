# Stakeholder Response — NYC Yellow Taxi Demand Optimization

## Model Selection

Operational demand model selected via bake-off (SAA / GLM / LGBM): **empirical**.

Full ranking saved to `outputs/bakeoff_model_selection.csv`.


---

## Task 1 — Zone-Specific Overage Costs (Model A vs Model B)

Co_z = α_{b(z)} · Co_t.  Borough multipliers (higher outer-borough repositioning cost):
| Borough | α |
|---|---|
| Manhattan | 1.00 |
| Brooklyn | 1.25 |
| Queens (non-airport) | 1.40 |
| Bronx | 1.50 |
| Staten Island | 1.80 |
| JFK (zone 132) | 0.70 (airport override) |
| LGA (zone 138) | 0.70 (airport override) |

**Why outer boroughs have higher idle cost:** Repositioning an idle cab from Staten Island
or the Bronx back to a high-demand corridor takes longer → higher opportunity cost.

**Airport zones:** Lower idle cost because airports self-serve a queue; an extra cab there
has low repositioning burden → lower overage penalty → Model B allocates more to JFK/LGA.

### Slot: dow1_0600_0859_allwks

**Borough q\* totals (A flat vs B borough-specific):**

| borough | q_A | q_B | delta_q |
| --- | --- | --- | --- |
| Manhattan | 10678.3059 | 10678.3059 | 0.0 |
| Bronx | 128.3983 | 118.4433 | -10.0 |
| Brooklyn | 424.0 | 415.513 | -8.5 |
| Queens | 1663.8766 | 1632.7587 | -31.1 |
| Staten Island | 83.7705 | 78.0 | -5.8 |


**Top-5 τ-shift zones (A→B):**

| PULocationID | tau_A | tau_B | delta_tau |
| --- | --- | --- | --- |
| 250.0 | 0.5655 | 0.4196 | 0.1459 |
| 254.0 | 0.5806 | 0.4348 | 0.1459 |
| 6.0 | 0.5835 | 0.4377 | 0.1458 |
| 259.0 | 0.5872 | 0.4414 | 0.1458 |
| 110.0 | 0.5876 | 0.4418 | 0.1458 |


**Shadow price λ(F₀) at fixed fleet:**  Model A (flat Co) = 0.0000   Model B (borough Co) = 0.0000

*(λ = marginal NV cost reduction per extra cab; break-even rental in Task 2 equals Model B's λ)*


**Realized NV cost on test:**  A=43728  B=45658  Δ=+1930


**3-way solver cross-check under Model B:**

| solver | objective | fleet_used | feasible | duality_gap | rel_gap_% | status | runtime_ms | max_diff_vs_slsqp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slsqp | 55258.0251 | 12923 | True | -5.0783 | -0.0092 | optimal | 7709.43 | 0.0 |
| water_filling | 55263.1033 | 12923 | True | 0.0 | 0.0 | optimal | 8.4 | 1.5501 |
| cvxpy | 55257.8142 | 12927 | True | -5.2891 | -0.0096 | optimal | 304.71 | 2.4391 |

### Slot: dow5_1900_2359_allwks

**Borough q\* totals (A flat vs B borough-specific):**

| borough | q_A | q_B | delta_q |
| --- | --- | --- | --- |
| Manhattan | 9159.5779 | 9183.9914 | 24.4 |
| Bronx | 162.3043 | 160.9557 | -1.3 |
| Brooklyn | 859.8684 | 850.5818 | -9.3 |
| Queens | 2708.1162 | 2694.2832 | -13.8 |
| Staten Island | 110.1332 | 110.1879 | 0.1 |


**Top-5 τ-shift zones (A→B):**

| PULocationID | tau_A | tau_B | delta_tau |
| --- | --- | --- | --- |
| 223.0 | 0.5665 | 0.4206 | 0.1459 |
| 214.0 | 0.5845 | 0.4386 | 0.1458 |
| 110.0 | 0.5916 | 0.4459 | 0.1457 |
| 6.0 | 0.5474 | 0.4019 | 0.1455 |
| 259.0 | 0.5464 | 0.4009 | 0.1455 |


**Shadow price λ(F₀) at fixed fleet:**  Model A (flat Co) = 25.3108   Model B (borough Co) = 25.3108

*(λ = marginal NV cost reduction per extra cab; break-even rental in Task 2 equals Model B's λ)*


**Realized NV cost on test:**  A=667436  B=667398  Δ=-38


**3-way solver cross-check under Model B:**

| solver | objective | fleet_used | feasible | duality_gap | rel_gap_% | status | runtime_ms | max_diff_vs_slsqp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slsqp | 650889.5923 | 13000 | True | -133.0039 | -0.0204 | optimal | 50100.66 | 0.0 |
| water_filling | 651190.3254 | 13000 | True | 167.7292 | 0.0258 | optimal | 178.86 | 250.4067 |
| cvxpy | 650889.3747 | 13000 | True | -133.2215 | -0.0205 | optimal | 706.99 | 6.4804 |

### Slot: dow6_0900_1559_allwks

**Borough q\* totals (A flat vs B borough-specific):**

| borough | q_A | q_B | delta_q |
| --- | --- | --- | --- |
| Manhattan | 8423.955 | 8433.8881 | 9.9 |
| Bronx | 205.76 | 203.9529 | -1.8 |
| Brooklyn | 876.5296 | 872.5601 | -4.0 |
| Queens | 3343.0754 | 3341.1396 | -1.9 |
| Staten Island | 150.6801 | 148.4593 | -2.2 |


**Top-5 τ-shift zones (A→B):**

| PULocationID | tau_A | tau_B | delta_tau |
| --- | --- | --- | --- |
| 259.0 | 0.5739 | 0.428 | 0.1459 |
| 254.0 | 0.5644 | 0.4185 | 0.1459 |
| 110.0 | 0.5928 | 0.4472 | 0.1457 |
| 44.0 | 0.5528 | 0.4071 | 0.1457 |
| 214.0 | 0.5972 | 0.4516 | 0.1455 |


**Shadow price λ(F₀) at fixed fleet:**  Model A (flat Co) = 28.3072   Model B (borough Co) = 28.3072

*(λ = marginal NV cost reduction per extra cab; break-even rental in Task 2 equals Model B's λ)*


**Realized NV cost on test:**  A=765356  B=765196  Δ=-160


**3-way solver cross-check under Model B:**

| solver | objective | fleet_used | feasible | duality_gap | rel_gap_% | status | runtime_ms | max_diff_vs_slsqp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slsqp | 702197.2821 | 13000 | True | -22.5807 | -0.0032 | optimal | 19698.86 | 0.0 |
| water_filling | 702255.7387 | 13000 | True | 35.8758 | 0.0051 | optimal | 193.85 | 7.0002 |
| cvxpy | 702197.2265 | 13000 | False | -22.6363 | -0.0032 | optimal | 783.32 | 7.9967 |

## Task 2 — Elastic Fleet (Model C)

Fleet F is a continuous decision variable jointly optimized with q.  Adjustment cost: r·(F−F₀)⁺ − s·(F₀−F)⁺ with r=45, s=20, F₀=13000.

**Dead zone [20, 45]:** when s ≤ λ(F₀) ≤ r the fleet stays at F₀ (marginal NV saving doesn't justify rental).

**Shadow price interpretation:** λ(F) = marginal NV cost reduction per extra cab.
TLC should expand if λ(F₀) > r, contract if λ(F₀) < s.

**Break-even rental rate** = λ(F₀): the rental rate at which TLC is indifferent about expansion.

**Total cost basis:** all of A/B/C below are the *realized* newsvendor cost on the held-out test set (same basis as Task 1's "Realized NV cost on test"), plus the fleet adjustment cost for C — so A/B/C are directly comparable.

### Slot: dow1_0600_0859_allwks

| Metric | Value |
|---|---|
| F* | 8186 (optimal:contract) |
| λ_A (flat Co, fixed fleet) | 0.0000 |
| λ_B = break-even rental (borough Co, fixed fleet) | 0.0000 |
| Shadow price λ(F*) at optimal fleet | 20.0000 |
| Total cost A (flat,fixed) | 43728 |
| Total cost B (borough,fixed) | 45658 |
| Total cost C (borough,elastic) | 48435 |
| Fleet adj cost (C) | -96282 |
| Budget binds at F*? | yes |


**A→C cost decomposition:**
- A→B (spatial reallocation): +1930
- B→C (fleet right-sizing): +2777


**Decision-time (SAA) vs. realized (test-set) cost:**

| Basis | Total cost @ F0 (Model B's choice) | Total cost @ F* (Model C's choice) |
|---|---|---|
| SAA-expected (decision time, historical samples) | 55263 | 20768 |
| Realized (this test set) | 45658 | 48435 |

Based on the historical (SAA) demand distribution, contracting/expanding to F*=8186 looked **+34495 better** than staying at F0 — "stay at F0" is always a feasible option for Model C, so its SAA-optimal total cost can never be worse than Model B's. However, on this specific test set, Model C **underperformed** Model B by 2777 because realized demand differed from the historical samples the decision was based on. This is the expected decision-time-vs-realized-outcome gap of any SAA-based policy, not an error in the optimization.

### Slot: dow5_1900_2359_allwks

| Metric | Value |
|---|---|
| F* | 13000 (optimal:stay) |
| λ_A (flat Co, fixed fleet) | 25.3108 |
| λ_B = break-even rental (borough Co, fixed fleet) | 25.3108 |
| Shadow price λ(F*) at optimal fleet | 25.3108 |
| Total cost A (flat,fixed) | 667436 |
| Total cost B (borough,fixed) | 667398 |
| Total cost C (borough,elastic) | 667398 |
| Fleet adj cost (C) | 0 |
| Budget binds at F*? | yes |


**A→C cost decomposition:**
- A→B (spatial reallocation): -38
- B→C (fleet right-sizing): +0

### Slot: dow6_0900_1559_allwks

| Metric | Value |
|---|---|
| F* | 13000 (optimal:stay) |
| λ_A (flat Co, fixed fleet) | 28.3072 |
| λ_B = break-even rental (borough Co, fixed fleet) | 28.3072 |
| Shadow price λ(F*) at optimal fleet | 28.3072 |
| Total cost A (flat,fixed) | 765356 |
| Total cost B (borough,fixed) | 765196 |
| Total cost C (borough,elastic) | 765196 |
| Fleet adj cost (C) | 0 |
| Budget binds at F*? | yes |


**A→C cost decomposition:**
- A→B (spatial reallocation): -160
- B→C (fleet right-sizing): +0
