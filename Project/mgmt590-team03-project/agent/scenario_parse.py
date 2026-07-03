"""Parse natural-language scenario and constraint phrases into OptimizationConfig patches."""

from __future__ import annotations

import calendar
import datetime
import re
from dataclasses import dataclass, field
from typing import Any, Optional

TLC_BUCKETS = [
    "00:00-05:59",
    "06:00-08:59",
    "09:00-15:59",
    "16:00-18:59",
    "19:00-23:59",
]

DOW_NAMES = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}

STAKEHOLDER_PRESETS: dict[str, dict[str, Any]] = {
    "monday morning": {"day_of_week": 1, "time_bucket": "06:00-08:59"},
    "mon am": {"day_of_week": 1, "time_bucket": "06:00-08:59"},
    "monday am": {"day_of_week": 1, "time_bucket": "06:00-08:59"},
    "friday evening": {"day_of_week": 5, "time_bucket": "19:00-23:59"},
    "friday eve": {"day_of_week": 5, "time_bucket": "19:00-23:59"},
    "fri evening": {"day_of_week": 5, "time_bucket": "19:00-23:59"},
    "saturday midday": {"day_of_week": 6, "time_bucket": "09:00-15:59"},
    "sat midday": {"day_of_week": 6, "time_bucket": "09:00-15:59"},
}

BUCKET_ALIASES: list[tuple[str, str]] = [
    (r"\b00:00-05:59\b", "00:00-05:59"),
    (r"\b06:00-08:59\b", "06:00-08:59"),
    (r"\b09:00-15:59\b", "09:00-15:59"),
    (r"\b16:00-18:59\b", "16:00-18:59"),
    (r"\b19:00-23:59\b", "19:00-23:59"),
    (r"\b(?:overnight|late\s*night)\b", "00:00-05:59"),
    (r"\b(?:morning\s*rush|early\s*morning|am\s*rush)\b", "06:00-08:59"),
    (r"\b(?:midday|business\s*hours|daytime)\b", "09:00-15:59"),
    (r"\b(?:evening\s*rush|pm\s*rush)\b", "16:00-18:59"),
    (r"\b(?:late\s*evening|night\s*shift)\b", "19:00-23:59"),
    (r"\b(?:6|06)\s*[-:–—to]\s*(?:8|08)(?:\s*:\s*59)?\b", "06:00-08:59"),
    (r"\b(?:6|06)\s*:\s*00\s*[-–—to]\s*(?:8|08)\s*:\s*59\b", "06:00-08:59"),
    (r"\b(?:9|09)\s*[-:–—to]\s*(?:15|3)\b", "09:00-15:59"),
    (r"\b(?:4|16)\s*[-:–—to]\s*(?:6|18)\b", "16:00-18:59"),
    (r"\b(?:7|19)\s*[-:–—to]\s*(?:11|23)\b", "19:00-23:59"),
    (r"\bpeak\b", "09:00-15:59"),
    (r"\bnight\b", "19:00-23:59"),
]

MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass
class ScenarioParseResult:
    patch: dict[str, Any] = field(default_factory=dict)
    run_requested: bool = False
    ambiguities: list[str] = field(default_factory=list)
    clarification_question: Optional[str] = None
    interpreted_summary: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification_question)


def _iso_week_from_month_week(year: int, month: int, week_in_month: int, dow: int) -> int:
    """Map 'week N of month' + DOW to ISO week number."""
    # Anchor: first day of the Nth calendar week chunk (7-day blocks from day 1)
    start_day = min(1 + (week_in_month - 1) * 7, 28)
    for day in range(start_day, min(start_day + 7, calendar.monthrange(year, month)[1] + 1)):
        try:
            d = datetime.date(year, month, day)
        except ValueError:
            continue
        if d.isoweekday() == dow:
            return d.isocalendar()[1]
    # fallback: mid-month
    d = datetime.date(year, month, min(15, calendar.monthrange(year, month)[1]))
    return d.isocalendar()[1]


def _parse_year(text: str) -> Optional[int]:
    m = re.search(r"\b(?:year\s*)?(20\d{2})\b", text, re.I)
    return int(m.group(1)) if m else None


def _parse_month(text: str) -> Optional[int]:
    t = text.lower()
    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            return num
    return None


def _parse_dow(text: str) -> Optional[int]:
    t = text.lower()
    for phrase, preset in STAKEHOLDER_PRESETS.items():
        if phrase in t and "day_of_week" in preset:
            return preset["day_of_week"]

    dow_map = {
        "monday": 1,
        "mon": 1,
        "tuesday": 2,
        "tue": 2,
        "tues": 2,
        "wednesday": 3,
        "wed": 3,
        "thursday": 4,
        "thu": 4,
        "thur": 4,
        "thurs": 4,
        "friday": 5,
        "fri": 5,
        "saturday": 6,
        "sat": 6,
        "sunday": 7,
        "sun": 7,
    }
    for name, num in dow_map.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            return num

    for pat in (
        r"\bdow\s*(?:is\s*)?[=:]?\s*(\d)\b",
        r"\bday\s*(?:of\s*week\s*)?(?:is\s*)?[=:]?\s*(\d)\b",
        r"\bdow\s*(\d)\b",
    ):
        m = re.search(pat, t, re.I)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 7:
                return n
    return None


def _parse_time_bucket(text: str) -> Optional[str]:
    t = text.lower()
    for phrase, preset in STAKEHOLDER_PRESETS.items():
        if phrase in t and "time_bucket" in preset:
            return preset["time_bucket"]

    for bucket in TLC_BUCKETS:
        if bucket.lower() in t:
            return bucket

    found: list[str] = []
    for pattern, bucket in BUCKET_ALIASES:
        if re.search(pattern, t, re.I):
            found.append(bucket)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        return None  # conflict — caller handles
    return None


def _parse_week_iso(text: str) -> Optional[int]:
    for pat in (
        r"\biso\s*week\s*(\d{1,2})\b",
        r"\bwk\s*(\d{1,2})\b",
        r"\bweek\s*(\d{1,2})\b",
    ):
        m = re.search(pat, text, re.I)
        if m:
            w = int(m.group(1))
            if 1 <= w <= 52:
                return w
    return None


def _parse_month_week_phrase(text: str) -> tuple[Optional[int], Optional[int], bool]:
    """Return (week_in_month, iso_week_if_explicit, is_ambiguous)."""
    t = text.lower()
    ambiguous = False

    # "2nd week of April", "week 2 of apr"
    m = re.search(
        r"(?:week\s*(\d{1,2})\s*(?:of|in)\s*(?:the\s*)?(?:month\s*)?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*)",
        t,
    )
    if m:
        return int(m.group(1)), None, False

    m = re.search(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*)\s*(?:\(?\s*)?week\s*(\d{1,2})",
        t,
    )
    if m:
        return int(m.group(2)), None, True  # "Apr week 2" is ambiguous vs ISO

    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+week\s+of\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", t)
    if m:
        return int(m.group(1)), None, False

    return None, None, False


def _parse_fleet_size(text: str) -> Optional[int]:
    patterns = [
        r"(?:total\s+)?fleet(?:\s*size)?\s*(?:to|=|:|of|≤|<)?\s*(\d[\d,]*)",
        # "fleet size for ... reduced/changed/set to 9000" or "... is 11500" —
        # fleet keyword and target number separated by other scenario details
        # (day, week, slot, etc.). Excludes "of" to avoid matching years like
        # "week 16 of 2026".
        r"fleet[^.\n]*?\b(?:to|is|=|equals)\s+(\d[\d,]*)",
        r"(\d[\d,]*)\s*(?:taxis|vehicles|cabs)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def _wants_run(text: str) -> bool:
    t = text.lower()
    if re.search(r"\brun\b", t):
        return True
    return any(
        k in t
        for k in (
            "run allocation",
            "run ",
            " allocate",
            "allocation",
            "allocate",
            "optimize",
            "what-if",
            "what if",
            "q*",
            "q-star",
            "q star",
            "show q",
            "give me q",
            "report",
        )
    )


def _wants_scenario(text: str) -> bool:
    t = text.lower()
    return bool(
        _parse_dow(text)
        or _parse_time_bucket(text)
        or _parse_week_iso(text)
        or _parse_month(text)
        or any(p in t for p in STAKEHOLDER_PRESETS)
        or re.search(r"\bweek\s*\d", t)
        or re.search(r"\bdow\b", t)
    )


def format_scenario_summary(patch: dict[str, Any]) -> str:
    parts = []
    dow = patch.get("day_of_week")
    if dow:
        parts.append(DOW_NAMES.get(dow, f"DOW {dow}"))
    if patch.get("time_bucket"):
        parts.append(str(patch["time_bucket"]))
    if patch.get("week"):
        parts.append(f"ISO week {patch['week']}")
    if patch.get("year"):
        parts.append(f"year {patch['year']}")
    if patch.get("fleet_size"):
        parts.append(f"fleet {patch['fleet_size']:,}")
    if patch.get("co_mode"):
        parts.append(f"Co={patch['co_mode']}")
    if patch.get("elastic_fleet"):
        parts.append("elastic fleet (Task 2)")
    return ", ".join(parts) if parts else "(defaults unchanged)"


def parse_scenario_text(text: str, *, default_year: int = 2026) -> ScenarioParseResult:
    """Heuristic parse of user text into a config patch + clarification needs."""
    result = ScenarioParseResult()
    t = text.lower()
    patch: dict[str, Any] = {}

    result.run_requested = _wants_run(text)

    # Stakeholder presets (full scenario)
    for phrase, preset in STAKEHOLDER_PRESETS.items():
        if phrase in t:
            patch.update(preset)
            break

    dow = _parse_dow(text)
    if dow is not None:
        if patch.get("day_of_week") and patch["day_of_week"] != dow:
            result.ambiguities.append(
                f"Conflicting days: preset vs explicit DOW {dow}"
            )
        else:
            patch["day_of_week"] = dow

    bucket = _parse_time_bucket(text)
    if bucket is not None:
        if patch.get("time_bucket") and patch["time_bucket"] != bucket:
            result.ambiguities.append(
                f"Conflicting time buckets: {patch['time_bucket']} vs {bucket}"
            )
        else:
            patch["time_bucket"] = bucket
    elif re.search(r"\b(?:6|8|9|morning|evening|night|peak|bucket)\b", t):
        # hinted time but unresolved
        buckets_found = []
        for pattern, b in BUCKET_ALIASES:
            if re.search(pattern, t, re.I):
                buckets_found.append(b)
        if len(buckets_found) > 1:
            result.ambiguities.append(
                f"Multiple time buckets matched: {', '.join(sorted(set(buckets_found)))}"
            )

    explicit_year = _parse_year(text)
    year = explicit_year or default_year
    if explicit_year is not None:
        patch["year"] = explicit_year
    month = _parse_month(text)
    week_in_month, _, month_week_ambiguous = _parse_month_week_phrase(text)
    iso_week = _parse_week_iso(text)

    if week_in_month is not None and month is not None:
        dow_for_iso = patch.get("day_of_week") or 1
        resolved = _iso_week_from_month_week(year, month, week_in_month, dow_for_iso)
        patch["week"] = resolved
        month_name = calendar.month_name[month]
        result.notes.append(
            f"Interpreted **{month_name} week {week_in_month}** ({year}) "
            f"→ ISO week **{resolved}**."
        )
    elif iso_week is not None:
        patch["week"] = iso_week
        if month_week_ambiguous and month:
            result.notes.append(
                f"Using ISO week {iso_week} (ignoring ambiguous month-week phrase)."
            )
    elif week_in_month is not None and month is None:
        if month_week_ambiguous:
            result.ambiguities.append(
                f"'week {week_in_month}' without month — ISO week {week_in_month} (January) "
                f"or week {week_in_month} of which month?"
            )
        else:
            patch["week"] = week_in_month

    if month_week_ambiguous and month and week_in_month is not None and month is not None:
        # Resolved calendar week — no clarification needed
        pass
    elif month_week_ambiguous and month and "week" not in patch:
        month_name = calendar.month_name[month]
        result.clarification_question = (
            f"Did you mean **ISO week 2** (early January) or the "
            f"**2nd week of {month_name} {year}**? "
            f"Reply e.g. `ISO week 14` or `2nd week of {month_name} {year}, Friday 06:00-08:59, run`."
        )

    fleet = _parse_fleet_size(text)
    if fleet is not None:
        patch["fleet_size"] = fleet

    floor_m = re.search(r"floor[_\s-]?alpha\s*[=:]?\s*(0?\.\d+|\d+)", t)
    if floor_m:
        patch["floor_alpha"] = float(floor_m.group(1))

    cap_m = re.search(r"cap[_\s-]?multiplier\s*[=:]?\s*(0?\.\d+|\d+)", t)
    if cap_m:
        patch["cap_multiplier"] = float(cap_m.group(1))

    if any(k in t for k in ("borough overage", "zone-specific", "zone specific", "model b", "task 1")):
        patch["co_mode"] = "borough"
    if any(k in t for k in ("flat co", "baseline co", "model a", "uniform overage")):
        patch["co_mode"] = "flat"

    if any(k in t for k in ("elastic fleet", "model c", "task 2", "flexible fleet", "optimize fleet size")):
        patch["elastic_fleet"] = True
    if any(k in t for k in ("fixed fleet", "no elastic")):
        patch["elastic_fleet"] = False

    rental_m = re.search(r"rental(?:\s*rate)?\s*[=:]\s*(\d+(?:\.\d+)?)", t)
    if not rental_m:
        rental_m = re.search(r"(?<=\s)r\s*=\s*\$?(\d+(?:\.\d+)?)", t)
    if rental_m:
        patch["rental_rate"] = float(rental_m.group(1))
    stand_m = re.search(r"standdown(?:\s*saving)?\s*[=:]\s*(\d+(?:\.\d+)?)", t)
    if not stand_m:
        stand_m = re.search(r"(?<=\s)s\s*=\s*\$?(\d+(?:\.\d+)?)", t)
    if stand_m:
        patch["standdown_saving"] = float(stand_m.group(1))

    if patch.get("elastic_fleet"):
        patch["co_mode"] = "borough"
        if patch.get("fleet_size"):
            patch["fleet_baseline"] = patch["fleet_size"]

    if any(k in t for k in ("rain", "rainy", "precipitation", "weather stress")):
        patch["weather_override"] = {
            "precipitation": 8.0,
            "temperature_2m": 12.0,
            "apparent_temperature": 10.0,
            "relative_humidity_2m": 85.0,
        }

    if _parse_year(text) and month and week_in_month:
        result.notes.append(f"Resolved calendar context: {year}, month {month}.")

    # Clarification: scenario requested but missing critical fields
    scenario_partial = _wants_scenario(text) or result.run_requested
    missing = []
    if scenario_partial and result.run_requested:
        if "day_of_week" not in patch:
            missing.append("day of week (e.g. Friday or dow 5)")
        if "time_bucket" not in patch:
            missing.append("time bucket (e.g. 06:00-08:59 or morning rush)")
        if "week" not in patch and not result.clarification_question:
            missing.append("week (ISO week number, e.g. week 14)")

    if missing and not result.clarification_question:
        result.clarification_question = (
            "To run allocation I still need: **"
            + "**, **".join(missing)
            + "**. Current understanding: "
            + format_scenario_summary(patch)
            + "."
        )

    if result.ambiguities and not result.clarification_question:
        result.clarification_question = (
            "I found conflicting values: "
            + "; ".join(result.ambiguities)
            + ". Please clarify which scenario you want."
        )

    result.patch = patch
    result.interpreted_summary = format_scenario_summary(patch)
    return result


def merge_pending_clarification(pending_patch: dict[str, Any], new_text: str) -> ScenarioParseResult:
    """Merge a follow-up clarification message with a pending partial patch."""
    follow = parse_scenario_text(new_text)
    merged = dict(pending_patch)
    merged.update(follow.patch)
    follow.patch = merged
    follow.interpreted_summary = format_scenario_summary(merged)

    # Re-check missing fields after merge
    if follow.run_requested or _wants_run(new_text) or _wants_scenario(new_text):
        follow.run_requested = True
        missing = []
        if "day_of_week" not in merged:
            missing.append("day of week")
        if "time_bucket" not in merged:
            missing.append("time bucket")
        if "week" not in merged:
            missing.append("ISO week")
        if missing and not follow.clarification_question:
            follow.clarification_question = (
                "Still need: **" + "**, **".join(missing) + "**."
            )
        elif not missing:
            follow.clarification_question = None

    return follow
