# NYC Yellow Taxi — Management Scenario Brief

**To:** Fleet Optimization Agent  
**From:** Operations Planning  
**Date:** June 2025  
**Subject:** Tuesday peak allocation under reduced fleet and weather stress test

## Executive summary

Reduce the active yellow fleet to **11,000 vehicles** for the upcoming budget cycle.
Re-run the Tuesday midday peak allocation and confirm feasibility before the board meeting.

## Scenario parameters

| Parameter | Value |
|-----------|-------|
| Day of week | Tuesday |
| Time bucket | 09:00-15:59 (midday peak) |
| Week | 50 |
| Fleet size | 11,000 |

## Service-level constraints

1. **Bronx coverage floor:** Guarantee at least **3 cabs** in every Bronx pickup zone.
   Use floor overrides where the default alpha-based floor is below 3.

2. **Cap discipline:** Keep the standard cap multiplier (1.5× p95 demand) unless a zone
   hits the fleet binding constraint.

## Weather stress test

Model a **rainy-day scenario** for sensitivity analysis:

- Precipitation: **8.0 mm**
- Temperature: **12 °C** (apparent temperature **10 °C**)
- Relative humidity: **85%**

Apply these as a weather override before solving.

## Deliverables requested

1. Optimal zone allocation `q*` under the above constraints  
2. Feasibility check (total floors vs fleet size)  
3. Top 10 zones by allocation with brief rationale  
4. Comparison vs the 13,000-cab baseline if available  

## Notes

- Proxy aggregation should remain **borough-level** (do not switch to global).  
- Solver preference: **SLSQP** (fast re-solve acceptable after initial refit).  
- If infeasible due to Bronx floors, report the minimum fleet size required.

---

*Upload this file in the agent chat with the message: "Apply this scenario and run allocation."*
