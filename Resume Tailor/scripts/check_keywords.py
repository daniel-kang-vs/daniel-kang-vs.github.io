#!/usr/bin/env python3
"""Check ATS keyword coverage in resume text against a keyword list."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def count_occurrences(text: str, keyword: str) -> int:
    pattern = re.escape(keyword.lower())
    return len(re.findall(pattern, text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ATS keyword coverage")
    parser.add_argument("text_file", help="Plain text or .parse.txt from verify_parse.sh")
    parser.add_argument(
        "keywords_file",
        help="One keyword/phrase per line (comments with # allowed)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.70,
        help="Minimum fraction of keywords that must appear (default: 0.70)",
    )
    args = parser.parse_args()

    text_path = Path(args.text_file)
    kw_path = Path(args.keywords_file)
    raw_text = load_text(text_path)
    norm_text = normalize(raw_text)

    keywords = []
    for line in kw_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keywords.append(line)

    if not keywords:
        print("No keywords found.", file=sys.stderr)
        return 1

    matched = []
    missing = []
    density_flags = []

    for kw in keywords:
        count = count_occurrences(norm_text, kw)
        if count > 0:
            matched.append((kw, count))
        else:
            missing.append(kw)
        if count > 8:
            density_flags.append((kw, count))

    coverage = len(matched) / len(keywords)
    print(f"Keyword coverage: {len(matched)}/{len(keywords)} ({coverage:.0%})")
    print(f"Target: {args.min_coverage:.0%}")
    print()

    print("Matched:")
    for kw, count in matched:
        print(f"  + {kw} ({count}x)")

    if missing:
        print("\nMissing:")
        for kw in missing:
            print(f"  - {kw}")

    if density_flags:
        print("\nDensity warnings (>8 occurrences):")
        for kw, count in density_flags:
            print(f"  ! {kw} ({count}x)")

    if coverage < args.min_coverage:
        print("\nCoverage below target.", file=sys.stderr)
        return 1

    print("\nKeyword check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
