"""Tests for natural-language scenario parsing (no engine deps)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.scenario_parse import merge_pending_clarification, parse_scenario_text


def test_user_example():
    text = "q* for Apr (week 2), dow is 5, year 2026, Time bucket 6-8:59"
    r = parse_scenario_text(text)
    assert r.patch.get("day_of_week") == 5, r.patch
    assert r.patch.get("time_bucket") == "06:00-08:59", r.patch
    week = r.patch.get("week")
    assert week is not None and week != 2, f"expected calendar Apr week 2, got ISO week {week}"
    assert r.run_requested is True
    print(f"OK  user example → {r.interpreted_summary}, week={week}")


def test_dow_and_iso_week():
    r = parse_scenario_text("Friday 6-8:59 ISO week 14 fleet 11500 run allocation")
    assert r.patch["day_of_week"] == 5
    assert r.patch["time_bucket"] == "06:00-08:59"
    assert r.patch["week"] == 14
    assert r.patch["fleet_size"] == 11500
    assert not r.needs_clarification
    print("OK  Friday + flexible bucket + ISO week")


def test_ambiguous_week_only():
    r = parse_scenario_text("week 2 run allocation")
    assert r.needs_clarification or r.patch.get("week") == 2
    print("OK  ambiguous week 2 handled")


def test_clarification_followup():
    partial = {"day_of_week": 5, "time_bucket": "06:00-08:59"}
    r = merge_pending_clarification(partial, "ISO week 14 fleet 11500 run")
    assert r.patch["week"] == 14
    assert r.patch["fleet_size"] == 11500
    assert not r.needs_clarification
    print("OK  clarification merge")


def test_stakeholder_preset():
    r = parse_scenario_text("Saturday midday week 50 run")
    assert r.patch["day_of_week"] == 6
    assert r.patch["time_bucket"] == "09:00-15:59"
    print("OK  stakeholder preset")


def test_task1_task2_options():
    r = parse_scenario_text("borough Co Task 1 Friday week 14 fleet 13000 run")
    assert r.patch.get("co_mode") == "borough"
    r2 = parse_scenario_text("elastic fleet Task 2 Model C fleet 13000 r=$50 s=$25 run")
    assert r2.patch.get("elastic_fleet") is True
    assert r2.patch.get("rental_rate") == 50.0
    assert r2.patch.get("standdown_saving") == 25.0
    print("OK  Task 1 co_mode + Task 2 elastic")


def main():
    test_user_example()
    test_dow_and_iso_week()
    test_ambiguous_week_only()
    test_clarification_followup()
    test_stakeholder_preset()
    test_task1_task2_options()
    print("\nAll scenario parse tests passed.")


if __name__ == "__main__":
    main()
