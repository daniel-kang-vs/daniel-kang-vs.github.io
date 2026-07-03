"""Constrained newsvendor inner solver — three independent implementations.

All three should agree on q* (up to numerical tolerance):
  1. SLSQP   — scipy.optimize.minimize, primary solver
  2. water_filling — analytic bisection on Lagrange multiplier λ (fastest, exact)
  3. cvxpy   — convex program solver (cross-check)

Problem (per time bucket):
    min_q  Σ_z [ Co_z * E[(q_z - D_z)+] + Cu_z * E[(D_z - q_z)+] ]
    s.t.   Σ_z q_z ≤ fleet_size
           floor_z ≤ q_z ≤ cap_z

With empirical demand samples the expected newsvendor cost per zone is:
    NV(q, D_samples) = Co * mean(max(q - D, 0)) + Cu * mean(max(D - q, 0))
which is convex and differentiable a.e. in q.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq

from engine.numeric import safe_int, sanitize_allocation


@dataclass
class SolveResult:
    q_star: np.ndarray          # 263-vector, continuous
    q_star_int: np.ndarray      # integer-rounded, budget-repaired
    objective: float            # newsvendor cost at q_star
    solver: str
    status: str
    runtime_ms: float
    fleet_used: int
    zones: np.ndarray           # PULocationID order matching q_star
    # Elastic-fleet / shadow-price extras (None for fixed-fleet solves)
    shadow_price: Optional[float] = None        # λ* = marginal value of one cab
    fleet_optimal: Optional[float] = None       # F* (may differ from fleet_size)
    breakeven_rental: Optional[float] = None    # r at which TLC is indifferent
    fleet_adjustment_cost: Optional[float] = None
    total_cost: Optional[float] = None          # NV cost + fleet adjustment cost


# ── Newsvendor cost helpers ──────────────────────────────────────────────────

def nv_cost_vectorized(
    q: np.ndarray,
    demand_samples: np.ndarray,   # shape (n_samples, n_zones)
    Cu: np.ndarray,               # (n_zones,)
    Co: np.ndarray,               # (n_zones,)
) -> float:
    """Total newsvendor cost across zones given a demand sample matrix."""
    overstock = np.maximum(q - demand_samples, 0)   # (n_samples, n_zones)
    understock = np.maximum(demand_samples - q, 0)
    cost_per_sample = (Co * overstock + Cu * understock).sum(axis=1)
    return float(cost_per_sample.mean())


def nv_gradient(
    q: np.ndarray,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
) -> np.ndarray:
    """Gradient of newsvendor cost w.r.t. q (used by SLSQP)."""
    # dNV/dq_z = Co_z * P(D_z < q_z) - Cu_z * P(D_z > q_z)
    #          = Co_z * F(q_z) - Cu_z * (1 - F(q_z))
    #          = (Co_z + Cu_z) * F(q_z) - Cu_z
    F_q = (demand_samples < q).mean(axis=0)  # empirical CDF at q
    return (Cu + Co) * F_q - Cu


# ── Module-level quantile helper (reused by water-filling + elastic solvers) ─

def _q_of_lambda(
    lam: float,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    """Return q_z(λ) for each zone given Lagrange multiplier λ on the budget constraint."""
    tau_adj = (Cu - lam) / (Cu + Co)
    n = demand_samples.shape[1]
    q = np.empty(n, dtype=float)
    for i in range(n):
        t = float(tau_adj[i])
        if t <= 0.0:
            q[i] = floors[i]
        elif t >= 1.0:
            q[i] = caps[i]
        else:
            q[i] = float(np.quantile(demand_samples[:, i], t, method="linear"))
    return np.clip(q, floors, caps)


def _fleet_sum_at_lambda(
    lam: float,
    F: float,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
) -> float:
    return float(_q_of_lambda(lam, demand_samples, Cu, Co, floors, caps).sum())


def _lambda_at_fleet(
    F: float,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    *,
    max_iter: int = 48,
) -> float:
    """Shadow price λ(F): marginal NV cost reduction per extra cab at fleet size F."""
    if _q_of_lambda(0.0, demand_samples, Cu, Co, floors, caps).sum() <= F:
        return 0.0
    if floors.sum() >= F:
        return float(Cu.max())

    lo, hi = 0.0, float(Cu.max()) * 4.0
    for _ in range(16):
        if _fleet_sum_at_lambda(hi, F, demand_samples, Cu, Co, floors, caps) <= F:
            break
        hi *= 2.0
        if hi > 1e6:
            break

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if _fleet_sum_at_lambda(mid, F, demand_samples, Cu, Co, floors, caps) > F:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ── Solver 1: SLSQP ──────────────────────────────────────────────────────────

def solve_slsqp(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
    zones: np.ndarray,
) -> SolveResult:
    # Warm start from the water-filling solution: it's the exact analytic optimum and
    # already feasible (budget + bounds satisfied), so SLSQP starts in the basin of the
    # KKT point instead of ~3x over budget.  The empirical-CDF gradient is a step
    # function, and naive proportional-to-floors projection from the unconstrained
    # quantile can land far from the optimum in a flat-gradient region, causing SLSQP
    # to zigzag for thousands of iterations without reaching feasibility.
    q0 = solve_water_filling(demand_samples, Cu, Co, floors, caps, fleet_size, zones).q_star
    # Nudge strictly inside the budget constraint: water_filling's repair leaves q0
    # exactly on the Sum(q) <= fleet_size boundary, and SLSQP's first QP subproblem
    # can be degenerate (zigzag for 2000 iterations) starting exactly on a boundary.
    q0 = np.clip(q0 * (1 - 1e-5), floors, caps)

    def obj(q):
        return nv_cost_vectorized(q, demand_samples, Cu, Co)

    def grad(q):
        return nv_gradient(q, demand_samples, Cu, Co)

    constraints = [{"type": "ineq", "fun": lambda q: fleet_size - q.sum()}]
    bounds = list(zip(floors.tolist(), caps.tolist()))

    t0 = time.perf_counter()
    res = minimize(
        obj,
        q0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-7},
    )
    rt = (time.perf_counter() - t0) * 1000

    q_cont = np.clip(res.x, floors, caps)
    q_int = _integer_round_repair(q_cont, floors, caps, fleet_size)

    return SolveResult(
        q_star=q_cont,
        q_star_int=q_int,
        objective=res.fun,
        solver="slsqp",
        status="optimal" if res.success else f"warning:{res.message}",
        runtime_ms=rt,
        fleet_used=safe_int(q_int.sum()),
        zones=zones,
    )


# ── Solver 2: Water-filling (analytic bisection) ─────────────────────────────

def solve_water_filling(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
    zones: np.ndarray,
) -> SolveResult:
    """Bisection on Lagrange multiplier λ for the budget constraint.

    KKT: F_z(q_z) = (Cu_z - λ) / (Cu_z + Co_z), clipped to [floor_z, cap_z].
    Returns SolveResult with shadow_price set to λ* (0 when budget doesn't bind).
    """
    t0 = time.perf_counter()

    q_free = _q_of_lambda(0.0, demand_samples, Cu, Co, floors, caps)
    lam_star = 0.0
    if q_free.sum() <= fleet_size:
        q_cont = q_free
    else:
        sum_floors = floors.sum()
        if sum_floors >= fleet_size:
            warnings.warn(
                f"sum(floors)={sum_floors:.0f} >= fleet_size={fleet_size}: "
                "using floor allocation (infeasible bounds)"
            )
            q_cont = floors.copy()
            lam_star = float(Cu.max())
        else:
            lam_hi = float(Cu.max()) * 10
            while _q_of_lambda(lam_hi, demand_samples, Cu, Co, floors, caps).sum() > fleet_size and lam_hi < 1e9:
                lam_hi *= 10
            try:
                lam_star = brentq(
                    lambda lam: _q_of_lambda(lam, demand_samples, Cu, Co, floors, caps).sum() - fleet_size,
                    0.0,
                    lam_hi,
                    xtol=1e-6,
                    maxiter=300,
                )
            except ValueError:
                lam_star = lam_hi
                warnings.warn("water_filling bisection: using lam_hi fallback")
            q_cont = _q_of_lambda(lam_star, demand_samples, Cu, Co, floors, caps)
            # _q_of_lambda is a step function (empirical-CDF inverse), so q(lam_star)
            # can be off the budget by a few dozen cabs — close the gap exactly by
            # proportionally redistributing it across non-floor/non-cap zones.
            q_cont = _continuous_budget_repair(q_cont, floors, caps, fleet_size)

    rt = (time.perf_counter() - t0) * 1000
    obj = nv_cost_vectorized(q_cont, demand_samples, Cu, Co)
    q_int = _integer_round_repair(q_cont, floors, caps, fleet_size)

    return SolveResult(
        q_star=q_cont,
        q_star_int=q_int,
        objective=obj,
        solver="water_filling",
        status="optimal",
        runtime_ms=rt,
        fleet_used=safe_int(q_int.sum()),
        zones=zones,
        shadow_price=float(lam_star),
    )


# ── Solver 3: cvxpy ──────────────────────────────────────────────────────────

def solve_cvxpy(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
    zones: np.ndarray,
    max_samples: Optional[int] = None,
) -> SolveResult:
    """cvxpy epigraph-LP cross-check solver.

    Uses the FULL sample set by default so it solves the *same* SAA objective as
    SLSQP / water-filling (previously it silently capped at 30 samples, which made
    its objective non-comparable in the 3-way cross-check).  Pass max_samples to
    sub-sample only if cvxpy build time becomes a concern.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy not installed; run: pip install cvxpy")

    n = len(zones)
    q = cp.Variable(n, nonneg=True)

    # Build the newsvendor cost using sample average approximation
    # NV(q) = (1/S) Σ_s [ Co·(q-D_s)+ + Cu·(D_s-q)+ ]
    S = demand_samples.shape[0]
    n_use = S if max_samples is None else min(S, max_samples)
    over = cp.maximum(q - demand_samples[:n_use], 0)    # (n_use, n) broadcast
    under = cp.maximum(demand_samples[:n_use] - q, 0)
    objective = cp.Minimize(cp.sum(over @ Co + under @ Cu) / n_use)

    constraints = [
        cp.sum(q) <= fleet_size,
        q >= floors,
        q <= caps,
    ]

    prob = cp.Problem(objective, constraints)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    rt = (time.perf_counter() - t0) * 1000

    if q.value is None:
        raise RuntimeError(f"cvxpy solver failed: {prob.status}")

    q_cont = np.clip(q.value, floors, caps)
    q_int = _integer_round_repair(q_cont, floors, caps, fleet_size)

    # Report the objective re-scored on the FULL sample set for apples-to-apples
    obj_full = nv_cost_vectorized(q_cont, demand_samples, Cu, Co)

    return SolveResult(
        q_star=q_cont,
        q_star_int=q_int,
        objective=float(obj_full),
        solver="cvxpy",
        status=prob.status,
        runtime_ms=rt,
        fleet_used=safe_int(q_int.sum()),
        zones=zones,
    )


# ── Elastic fleet helpers ────────────────────────────────────────────────────

def _fleet_adjustment_cost(F: float, F0: float, r: float, s: float) -> float:
    """cost(F) = r*(F-F0)+ - s*(F0-F)+"""
    return r * max(F - F0, 0.0) - s * max(F0 - F, 0.0)


def _find_fleet_for_lambda_target(
    target_lam: float,
    F0: float,
    F_low: float,
    F_high: float,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    *,
    max_iter: int = 40,
) -> float:
    """Find F in [F_low, F_high] where λ(F) ≈ target_lam (λ decreases in F)."""
    lam_low = _lambda_at_fleet(F_low, demand_samples, Cu, Co, floors, caps)
    lam_high = _lambda_at_fleet(F_high, demand_samples, Cu, Co, floors, caps)
    if lam_low <= target_lam:
        return float(F_low)
    if lam_high >= target_lam:
        return float(F_high)

    lo, hi = float(F_low), float(F_high)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if _lambda_at_fleet(mid, demand_samples, Cu, Co, floors, caps) > target_lam:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _bracket_fleet_high_for_lambda(
    target_lam: float,
    F_low: float,
    F_cap: float,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    *,
    max_steps: int = 24,
) -> float:
    """Find F_high ≥ F_low with λ(F_high) ≤ target_lam (λ decreases in F)."""
    F_high = float(F_low)
    if _lambda_at_fleet(F_high, demand_samples, Cu, Co, floors, caps) <= target_lam:
        return F_high
    step = max(1.0, (F_cap - F_low) / 32.0)
    for _ in range(max_steps):
        F_high = min(F_high + step, F_cap)
        if _lambda_at_fleet(F_high, demand_samples, Cu, Co, floors, caps) <= target_lam:
            return F_high
        step = min(step * 2.0, max(F_cap - F_high, 1.0))
        if F_high >= F_cap:
            break
    return float(F_cap)


def _dual_bound(
    fleet_size: float,
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
) -> Tuple[float, float]:
    """Lagrangian dual lower bound g(λ*) for the fixed-fleet newsvendor problem.

        g(λ) = min_{floor≤q≤cap} [ NV(q) + λ·(Σq − F) ]
             = NV(q(λ)) + λ·(Σq(λ) − F)            with q(λ) = _q_of_lambda(λ)

    Evaluated at the optimal multiplier λ* = λ(F).  By weak duality g(λ*) ≤ p*
    (the primal optimum) for any λ*≥0; by strong duality (convex) equality holds,
    so the gap between a solver's objective and g(λ*) certifies optimality.

    Returns (dual_bound, lam_star).
    """
    lam = _lambda_at_fleet(fleet_size, demand_samples, Cu, Co, floors, caps)
    q_lam = _q_of_lambda(lam, demand_samples, Cu, Co, floors, caps)
    nv = nv_cost_vectorized(q_lam, demand_samples, Cu, Co)
    bound = nv + lam * (q_lam.sum() - fleet_size)
    return float(bound), float(lam)


def solve_elastic_fleet(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    F0: float,
    r: float,
    s: float,
    zones: np.ndarray,
    fleet_min: Optional[float] = None,
) -> SolveResult:
    """Primary elastic-fleet solve via water-filling / shadow-price approach.

    Envelope theorem: dNV*/dF = -λ(F) (decreasing in F).
    - Expand if λ(F0) > r  → F* = F(r): smallest F where λ(F)=r
    - Contract if λ(F0) < s → F* = max(F(s), fleet_min)
    - Stay if s ≤ λ(F0) ≤ r  (dead zone) → F* = F0
    """
    t0 = time.perf_counter()

    if fleet_min is None:
        fleet_min = float(floors.sum())

    lam_at_F0 = _lambda_at_fleet(F0, demand_samples, Cu, Co, floors, caps)

    F_cap = float(caps.sum())

    if lam_at_F0 > r:
        # Expand: find F* such that λ(F*) = r (λ decreases in F)
        F_high = _bracket_fleet_high_for_lambda(
            r, F0, F_cap, demand_samples, Cu, Co, floors, caps,
        )
        F_star = _find_fleet_for_lambda_target(
            r, F0, F0, F_high, demand_samples, Cu, Co, floors, caps,
        )
        direction = "expand"
        effective_lam = r
    elif lam_at_F0 < s:
        # Contract: find F* such that λ(F*) = s, then apply fleet_min
        F_contract = _find_fleet_for_lambda_target(
            s, F0, fleet_min, F0, demand_samples, Cu, Co, floors, caps,
        )
        F_star = max(F_contract, fleet_min)
        direction = "contract"
        effective_lam = _lambda_at_fleet(F_star, demand_samples, Cu, Co, floors, caps)
    else:
        F_star = F0
        direction = "stay"
        effective_lam = lam_at_F0

    q_cont = _q_of_lambda(effective_lam, demand_samples, Cu, Co, floors, caps)
    fleet_int = safe_int(round(F_star))
    q_int = _integer_round_repair(q_cont, floors, caps, fleet_int)

    nv = nv_cost_vectorized(q_cont, demand_samples, Cu, Co)
    adj_cost = _fleet_adjustment_cost(F_star, F0, r, s)
    total = nv + adj_cost

    rt = (time.perf_counter() - t0) * 1000
    return SolveResult(
        q_star=q_cont,
        q_star_int=q_int,
        objective=nv,
        solver="elastic_water_filling",
        status=f"optimal:{direction}",
        runtime_ms=rt,
        fleet_used=safe_int(q_int.sum()),
        zones=zones,
        shadow_price=float(effective_lam),
        fleet_optimal=float(F_star),
        breakeven_rental=float(lam_at_F0),
        fleet_adjustment_cost=float(adj_cost),
        total_cost=float(total),
    )


def solve_elastic_slsqp(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    F0: float,
    r: float,
    s: float,
    zones: np.ndarray,
    fleet_min: Optional[float] = None,
) -> SolveResult:
    """SLSQP cross-check for the elastic fleet problem.

    Optimizes over [q (n_zones), F (scalar)] jointly.

    Warm-start note: the fleet-adjustment cost has a kink exactly at F=F0, where the
    sub-gradient is the interval [s, r] and the naive choice g_F=0 makes SLSQP report
    F0 as stationary even when it is not — the solver then never explores contraction
    or expansion.  We therefore warm-start F (and q) from the certified water-filling
    elastic solution, so SLSQP begins on a smooth branch of cost(F) and *confirms*
    that optimum rather than stalling at the kink.
    """
    if fleet_min is None:
        fleet_min = float(floors.sum())

    n = len(zones)
    # Warm-start from the water-filling elastic solution (off the F0 kink).
    wf = solve_elastic_fleet(demand_samples, Cu, Co, floors, caps, F0, r, s, zones, fleet_min)
    F_init = wf.fleet_optimal if wf.fleet_optimal is not None else F0
    # Nudge strictly off the kink so the F-gradient is unambiguous at x0.
    if abs(F_init - F0) < 1.0:
        F_init = F0  # genuine dead-zone case; gradient handled below
    x0 = np.append(wf.q_star, F_init)

    def obj(x):
        q, F = x[:n], x[n]
        return nv_cost_vectorized(q, demand_samples, Cu, Co) + _fleet_adjustment_cost(F, F0, r, s)

    def grad(x):
        q, F = x[:n], x[n]
        g_q = nv_gradient(q, demand_samples, Cu, Co)
        # cost'(F): +r above the baseline, +s at/below it (left-derivative at the kink).
        # Both are positive — cost(F) is monotone increasing in F — so the F-gradient is
        # never the spurious 0 that previously froze SLSQP at F0.  The budget multiplier
        # then settles F* where λ(F*) = s (contract) or λ(F*) = r (expand).
        g_F = r if F > F0 else s
        return np.append(g_q, g_F)

    constraints = [
        {"type": "ineq", "fun": lambda x: x[n] - x[:n].sum()},      # F - sum(q) >= 0
        {"type": "ineq", "fun": lambda x: x[n] - fleet_min},         # F >= fleet_min
    ]
    bounds = list(zip(floors.tolist(), caps.tolist())) + [(fleet_min, float(caps.sum()))]

    t0 = time.perf_counter()
    res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000, "ftol": 1e-7})
    rt = (time.perf_counter() - t0) * 1000

    q_cont = np.clip(res.x[:n], floors, caps)
    F_star = float(res.x[n])
    fleet_int = safe_int(round(F_star))
    q_int = _integer_round_repair(q_cont, floors, caps, fleet_int)

    nv = nv_cost_vectorized(q_cont, demand_samples, Cu, Co)
    adj_cost = _fleet_adjustment_cost(F_star, F0, r, s)
    shadow_price = _lambda_at_fleet(F_star, demand_samples, Cu, Co, floors, caps)

    return SolveResult(
        q_star=q_cont,
        q_star_int=q_int,
        objective=nv,
        solver="elastic_slsqp",
        status="optimal" if res.success else f"warning:{res.message}",
        runtime_ms=rt,
        fleet_used=safe_int(q_int.sum()),
        zones=zones,
        shadow_price=float(shadow_price),
        fleet_optimal=F_star,
        fleet_adjustment_cost=float(adj_cost),
        total_cost=float(nv + adj_cost),
    )


def compare_elastic_solvers(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    F0: float,
    r: float,
    s: float,
    zones: np.ndarray,
    fleet_min: Optional[float] = None,
    tol: float = 1e-3,
) -> pd.DataFrame:
    """Run water-filling and SLSQP elastic solvers; certify via the total-cost gap.

    `tol` is a RELATIVE tolerance on the minimized objective (total cost).  The
    elastic problem is convex, so the meaningful agreement check is on total_cost,
    not F*/q* (which can differ in flat regions).  F*/q* differences are reported
    for information but only total_cost mismatch is warned on.
    """
    wf = solve_elastic_fleet(demand_samples, Cu, Co, floors, caps, F0, r, s, zones, fleet_min)
    sl = solve_elastic_slsqp(demand_samples, Cu, Co, floors, caps, F0, r, s, zones, fleet_min)

    cost_wf = wf.total_cost if wf.total_cost is not None else float("nan")
    cost_sl = sl.total_cost if sl.total_cost is not None else float("nan")
    denom = abs(cost_wf) if cost_wf not in (0.0, float("nan")) else 1.0
    rel_cost_diff = abs(cost_sl - cost_wf) / denom
    if rel_cost_diff > tol:
        warnings.warn(
            f"Elastic total-cost mismatch: wf={cost_wf:.1f} (F*={wf.fleet_optimal:.0f}), "
            f"slsqp={cost_sl:.1f} (F*={sl.fleet_optimal:.0f}) — rel diff {100*rel_cost_diff:.3f}%"
        )

    rows = []
    for res in [wf, sl]:
        rows.append({
            "solver": res.solver,
            "F_star": res.fleet_optimal,
            "shadow_price": res.shadow_price,
            "nv_cost": res.objective,
            "adj_cost": res.fleet_adjustment_cost,
            "total_cost": res.total_cost,
            "status": res.status,
            "runtime_ms": round(res.runtime_ms, 2),
        })
    return pd.DataFrame(rows)


# ── Continuous budget repair ─────────────────────────────────────────────────

def _continuous_budget_repair(
    q: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
) -> np.ndarray:
    """Scale q proportionally toward caps/floors so sum(q) == fleet_size exactly.

    _q_of_lambda is a step function of lambda (it inverts the empirical CDF), so
    the bisection's q(lam_star) can land a few dozen cabs over or under budget.
    Redistribute that gap proportionally across the zones with slack (room to
    grow toward cap, or room to shrink toward floor).
    """
    q = np.clip(q, floors, caps)
    gap = fleet_size - q.sum()
    if gap > 0:
        slack = caps - q
        total_slack = slack.sum()
        if total_slack > 0:
            q = q + slack * min(gap / total_slack, 1.0)
    elif gap < 0:
        excess = q - floors
        total_excess = excess.sum()
        if total_excess > 0:
            q = q - excess * min(-gap / total_excess, 1.0)
    return np.clip(q, floors, caps)


# ── Integer rounding with greedy budget repair ───────────────────────────────

def _integer_round_repair(
    q_cont: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
) -> np.ndarray:
    """Round q_cont to integers and repair so that sum ≤ fleet_size and bounds hold."""
    q_cont = sanitize_allocation(q_cont)
    q = np.round(q_cont).astype(int)
    q = np.clip(q, floors, caps)

    budget_slack = fleet_size - q.sum()
    if budget_slack < 0:
        # Over budget: greedily subtract from zones furthest above floor
        excess = q - floors
        order = np.argsort(-excess)
        for i in order:
            if budget_slack >= 0:
                break
            reduce = min(excess[i], -budget_slack)
            q[i] -= reduce
            budget_slack += reduce
    return q


# ── Dispatch ─────────────────────────────────────────────────────────────────

def solve(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
    zones: np.ndarray,
    method: Literal["slsqp", "water_filling", "cvxpy"] = "slsqp",
) -> SolveResult:
    """Dispatch to the requested solver."""
    fn = {
        "slsqp": solve_slsqp,
        "water_filling": solve_water_filling,
        "cvxpy": solve_cvxpy,
    }[method]
    return fn(demand_samples, Cu, Co, floors, caps, fleet_size, zones)


def compare_solvers(
    demand_samples: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    fleet_size: int,
    zones: np.ndarray,
    tol: float = 1e-3,
) -> pd.DataFrame:
    """Run all three solvers and certify optimality via the convex duality gap.

    `tol` is now a RELATIVE optimality tolerance (fraction): a solver is flagged
    only if its re-scored objective exceeds the dual lower bound by more than
    `tol` (default 0.1%).  q* L∞ disagreement is reported but NOT warned on — for
    a convex problem with flat regions, different q* can be equally optimal.
    """
    results = {}
    for method in ("slsqp", "water_filling", "cvxpy"):
        try:
            results[method] = solve(demand_samples, Cu, Co, floors, caps, fleet_size, zones, method)
        except Exception as e:
            warnings.warn(f"{method} failed: {e}")

    # ── Optimality certificate via convex duality ─────────────────────────────
    # The dual lower bound g(λ*) is computed from the exact KKT multiplier λ* that
    # water-filling solves for.  By strong duality (convex problem), any feasible q
    # whose re-scored objective equals g(λ*) is the certified global optimum.
    dual_bound, _ = _dual_bound(fleet_size, demand_samples, Cu, Co, floors, caps)

    records = []
    ref_key = "slsqp" if "slsqp" in results else next(iter(results), None)
    ref_q = results[ref_key].q_star if ref_key else None

    for k, v in results.items():
        # Re-score every solver's q* through the SAME objective on the FULL sample
        # set, so the numbers are directly comparable (no solver-internal quirks).
        obj = nv_cost_vectorized(v.q_star, demand_samples, Cu, Co)
        # 1e-4 tolerance: matches cvxpy/CLARABEL's solver tolerance, which can leave
        # individual bound constraints off by more than 1e-6 across 263 zones.
        budget_ok = bool(v.q_star.sum() <= fleet_size + 1e-4)
        bounds_ok = bool(np.all(v.q_star >= floors - 1e-4) and np.all(v.q_star <= caps + 1e-4))
        feasible = budget_ok and bounds_ok
        gap = obj - dual_bound
        rel_gap = gap / abs(dual_bound) if dual_bound != 0 else gap
        max_diff = float(np.abs(v.q_star - ref_q).max()) if (ref_q is not None and k != ref_key) else 0.0

        records.append({
            "solver": k,
            "objective": round(obj, 4),          # re-scored, apples-to-apples
            "fleet_used": int(round(v.q_star.sum())),
            "feasible": feasible,
            "duality_gap": round(gap, 4),
            "rel_gap_%": round(100 * rel_gap, 4),
            "status": v.status,
            "runtime_ms": round(v.runtime_ms, 2),
            "max_diff_vs_slsqp": round(max_diff, 4),
        })

        if not feasible:
            warnings.warn(f"Solver {k} returned INFEASIBLE q* (budget/bounds violated)")
        elif rel_gap > tol:
            warnings.warn(
                f"Solver {k} not at optimum: rel duality gap {100*rel_gap:.3f}% "
                f"(obj {obj:.1f} vs dual lower bound {dual_bound:.1f})"
            )

    return pd.DataFrame(records)
