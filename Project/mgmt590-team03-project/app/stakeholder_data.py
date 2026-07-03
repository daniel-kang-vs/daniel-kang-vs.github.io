"""Load stakeholder deliverables from outputs/ (run_stakeholder_tasks.py)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

BOROUGH_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

SLOT_DISPLAY = {
    "dow1_0600_0859_allwks": ("Monday morning", 1, "06:00-08:59"),
    "dow5_1900_2359_allwks": ("Friday evening", 5, "19:00-23:59"),
    "dow6_0900_1559_allwks": ("Saturday midday", 6, "09:00-15:59"),
}

# Required presentation slots (same as run_stakeholder_tasks.DEFAULT_SLOTS)
CANONICAL_SLOTS = list(SLOT_DISPLAY.keys())

TIME_BUCKETS = [
    "00:00-05:59",
    "06:00-08:59",
    "09:00-15:59",
    "16:00-18:59",
    "19:00-23:59",
]

BOROUGH_MULTIPLIERS = [
    ("Manhattan", 1.00),
    ("Brooklyn", 1.25),
    ("Queens (non-airport)", 1.40),
    ("Bronx", 1.50),
    ("Staten Island", 1.80),
    ("JFK (zone 132)", 0.70),
    ("LGA (zone 138)", 0.70),
]

TASK1_QUESTIONS = [
    "Which boroughs **gain** fleet and which **lose** under Model B? Does the direction match economic intuition?",
    "How do **airport zones** (JFK 132, LGA 138) behave? Does lower overage cost cause more or fewer cabs, and why?",
    "Does Model B produce a **lower realized newsvendor cost** than Model A on the test set, and by how much?",
    "Confirm the agent absorbs new cost parameters **without code restructuring** — what changed in the pipeline?",
]

TASK2_QUESTIONS = [
    "Does the agent **expand or contract** the fleet, and does the answer vary by slot? What drives the difference?",
    "How does the **r ≠ s asymmetry** ($45 rental vs $20 stand-down) affect decisions around the 13,000 baseline?",
    "At optimal F*, is the fleet capacity constraint **binding**? What does the shadow price tell TLC?",
    "How much of total cost reduction (A → C) comes from **spatial allocation** (Task 1) vs **fleet right-sizing** (Task 2)?",
]


def _read_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUTS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _parse_md_table(block: str) -> pd.DataFrame | None:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def _extract_slot_sections(md: str, section_header: str) -> dict[str, str]:
    """Return {slot_id: markdown_body} for subsections under a ## header."""
    pattern = rf"## {re.escape(section_header)}.*?(?=\n## |\Z)"
    m = re.search(pattern, md, re.DOTALL)
    if not m:
        return {}
    body = m.group(0)
    slots: dict[str, str] = {}
    parts = re.split(r"\n### Slot: ", body)
    for part in parts[1:]:
        slot_id, _, rest = part.partition("\n")
        slots[slot_id.strip()] = rest.strip()
    return slots


def _parse_task1_slot(slot_md: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    bor_m = re.search(
        r"\*\*Borough q\\\* totals.*?\*\*\s*\n(.*?)(?=\n\*\*Top-5|\n\*\*Shadow|\Z)",
        slot_md,
        re.DOTALL,
    )
    if bor_m:
        out["borough_table"] = _parse_md_table(bor_m.group(1))

    tau_m = re.search(
        r"\*\*Top-5 τ-shift zones.*?\*\*\s*\n(.*?)(?=\n\*\*Shadow|\n\*\*Realized|\Z)",
        slot_md,
        re.DOTALL,
    )
    if tau_m:
        out["tau_table"] = _parse_md_table(tau_m.group(1))

    sp_m = re.search(
        r"\*\*Shadow price λ\(F₀\).*?\*\*\s+Model A \(flat Co\) = ([\d.]+)\s+Model B \(borough Co\) = ([\d.]+)",
        slot_md,
    )
    if sp_m:
        out["shadow_price_A"] = float(sp_m.group(1))
        out["shadow_price_B"] = float(sp_m.group(2))

    nv_m = re.search(
        r"\*\*Realized NV cost on test:\*\*\s+A=([\d.]+)\s+B=([\d.]+)\s+Δ=([+-]?[\d.]+)",
        slot_md,
    )
    if nv_m:
        out["nv_cost_A"] = float(nv_m.group(1))
        out["nv_cost_B"] = float(nv_m.group(2))
        out["nv_delta"] = float(nv_m.group(3))

    cc_m = re.search(
        r"\*\*3-way solver cross-check under Model B:\*\*\s*\n(.*?)(?=\n### |\Z)",
        slot_md,
        re.DOTALL,
    )
    if cc_m:
        out["solver_check"] = _parse_md_table(cc_m.group(1))

    return out


def _parse_task2_slot(slot_md: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    table_m = re.search(r"\| Metric \| Value \|\s*\n\|---\|---\|\s*\n(.*?)(?=\n\*\*A→C|\Z)", slot_md, re.DOTALL)
    if table_m:
        rows = []
        for ln in table_m.group(1).strip().splitlines():
            if "|" in ln:
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if len(cells) == 2:
                    rows.append({"Metric": cells[0], "Value": cells[1]})
        out["metrics_table"] = pd.DataFrame(rows) if rows else None

    decomp_m = re.search(
        r"\*\*A→C cost decomposition:\*\*\s*\n- A→B \(spatial reallocation\): ([+-]?[\d.]+)\s*\n- B→C \(fleet right-sizing\): ([+-]?[\d.]+)",
        slot_md,
    )
    if decomp_m:
        out["decomp_AB"] = float(decomp_m.group(1))
        out["decomp_BC"] = float(decomp_m.group(2))

    return out


def load_stakeholder_bundle() -> dict[str, Any]:
    """Load CSVs, parsed markdown sections, and readiness flags."""
    md_path = OUTPUTS / "STAKEHOLDER_RESPONSE.md"
    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    winner = "—"
    wm = re.search(r"Operational demand model.*?\*\*(\w+)\*\*", md_text)
    if wm:
        winner = wm.group(1)

    task1_csv = _read_csv("task1_AvsB.csv")
    task2_csv = _read_csv("task2_ABC.csv")
    bakeoff_sel = _read_csv("bakeoff_model_selection.csv")
    bakeoff_orig = _read_csv("bakeoff_results.csv")

    csv_slots: list[str] = []
    if task2_csv is not None and not task2_csv.empty:
        csv_slots = list(task2_csv["slot"].unique())
    if task1_csv is not None and not task1_csv.empty:
        for sid in task1_csv["slot"].unique():
            if sid not in csv_slots:
                csv_slots.append(sid)
    slots = list(CANONICAL_SLOTS)
    for sid in csv_slots:
        if sid not in slots:
            slots.append(sid)

    slots_with_data: set[str] = set(csv_slots)
    if task1_csv is not None and not task1_csv.empty:
        slots_with_data |= set(task1_csv["slot"].unique())

    task1_md_slots = _extract_slot_sections(md_text, "Task 1 — Zone-Specific Overage Costs (Model A vs Model B)")
    task2_md_slots = _extract_slot_sections(md_text, "Task 2 — Elastic Fleet (Model C)")

    task1_parsed = {sid: _parse_task1_slot(body) for sid, body in task1_md_slots.items()}
    task2_parsed = {sid: _parse_task2_slot(body) for sid, body in task2_md_slots.items()}

    return {
        "ready": md_path.exists() and task1_csv is not None and task2_csv is not None,
        "md_path": str(md_path.relative_to(PROJECT_ROOT)) if md_path.exists() else None,
        "md_text": md_text,
        "winner": winner,
        "bakeoff_selection": bakeoff_sel,
        "bakeoff_original": bakeoff_orig,
        "task1_csv": task1_csv,
        "task2_csv": task2_csv,
        "slots": slots,
        "slots_with_data": slots_with_data,
        "slot_display": SLOT_DISPLAY,
        "task1_parsed": task1_parsed,
        "task2_parsed": task2_parsed,
    }


def slot_label(slot_id: str) -> str:
    if slot_id in SLOT_DISPLAY:
        name, dow, bucket = SLOT_DISPLAY[slot_id]
        return f"{name} (DOW {dow}, `{bucket}`)"
    return slot_id


def borough_q_for_slot(task1_csv: pd.DataFrame | None, slot_id: str) -> pd.DataFrame | None:
    if task1_csv is None or task1_csv.empty:
        return None
    sub = task1_csv[task1_csv["slot"] == slot_id].copy()
    if sub.empty:
        return None
    pivot = sub.pivot_table(index="borough", columns="model", values="total_q", aggfunc="first")
    if "A_flat" in pivot.columns and "B_borough" in pivot.columns:
        pivot["delta_q"] = pivot["B_borough"] - pivot["A_flat"]
        pivot = pivot.rename(columns={"A_flat": "q_A", "B_borough": "q_B"})
    return pivot.reset_index()


def slot_metrics(bundle: dict, slot_id: str) -> dict[str, Any]:
    """Unified per-slot metrics for baseline / Task 1 / Task 2 (same slot)."""
    t1 = bundle["task1_parsed"].get(slot_id, {})
    t2 = None
    if bundle["task2_csv"] is not None:
        sub = bundle["task2_csv"][bundle["task2_csv"]["slot"] == slot_id]
        if not sub.empty:
            t2 = sub.iloc[0]

    nv_a = t1.get("nv_cost_A")
    if t2 is not None and pd.notna(t2.get("nv_cost_A")):
        nv_a = float(t2["nv_cost_A"])

    return {
        "slot_id": slot_id,
        "slot_label": slot_label(slot_id),
        "nv_cost_A": nv_a,
        "nv_cost_B": t1.get("nv_cost_B") or (float(t2["nv_cost_B"]) if t2 is not None else None),
        "nv_cost_C": float(t2["nv_cost_C"]) if t2 is not None else None,
        "total_cost_A": float(t2["total_cost_A"]) if t2 is not None else nv_a,
        "total_cost_B": float(t2["total_cost_B"]) if t2 is not None else t1.get("nv_cost_B"),
        "total_cost_C": float(t2["total_cost_C"]) if t2 is not None else None,
        "shadow_price_A": t1.get("shadow_price_A"),
        "shadow_price_B": t1.get("shadow_price_B"),
        "F_star": float(t2["F_star"]) if t2 is not None else None,
    }
