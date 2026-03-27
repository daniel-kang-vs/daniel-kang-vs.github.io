# Stroller Box Optimization — Integer Linear Programming

**[Live Demo](https://daniel-kang-vs.github.io/stroller-box-optimization/)**

## Overview

An integer linear programming model that finds the optimal set of shipping boxes for 17 popular US strollers, minimizing total shipping cost (dimensional weight) while limiting box variety to at most 7 types.

## Problem

Shipping carriers charge by **dimensional weight** (L×W×H / 250) when it exceeds actual product weight. Using a unique box for each product is impractical for warehouse operations, but oversized boxes inflate costs. This model finds the sweet spot.

## Approach

- **17 strollers** — dimensions and weights sourced from manufacturer specs
- **43 candidate boxes** — from Walmart, Uline (S-codes), and standard sizes
- **Gurobi optimizer** — solves the ILP in 0.02 seconds with 774 binary variables and 1,394 constraints
- **Fit matrix** — enforces physical fit with 0.5" padding on all dimensions

## Results

| Metric | Value |
|--------|-------|
| Optimal box types selected | 7 |
| Total optimized shipping weight | 1,718.0 lb |
| Perfect baseline (unique box per product) | 1,613.3 lb |
| Standardization penalty | 104.7 lb (6.5%) |

## Repository Structure

```
├── IP_modeling.ipynb      # Full Gurobi optimization notebook
├── docs/                  # Interactive demo website
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## Tech Stack

- **Python** — pandas, NumPy, Gurobi (gurobipy)
- **Visualization** — Chart.js, seaborn, matplotlib
- **Optimization** — Integer Linear Programming, Gurobi Optimizer
