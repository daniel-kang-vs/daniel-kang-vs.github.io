#!/usr/bin/env bash
# End-to-end Resume Tailor workflow: copy baseline, compile, parse test, keyword check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
  echo "Usage: $0 <company_slug> <role_slug> [keywords.txt]" >&2
  echo "" >&2
  echo "Steps:" >&2
  echo "  1. Looks for jobs/{company}_{role}.txt (optional JD reference)" >&2
  echo "  2. Creates resume_{company}_{role}.tex from baseline (if missing)" >&2
  echo "  3. Compiles PDF, runs ATS parse test, checks keywords" >&2
  echo "" >&2
  echo "Before running, edit the .tex file to tailor content for the JD." >&2
  exit 1
}

[[ $# -ge 2 ]] || usage

COMPANY="$1"
ROLE="$2"
KEYWORDS="${3:-$ROOT_DIR/jobs/${COMPANY}_${ROLE}.keywords.txt}"
TEX="resume_${COMPANY}_${ROLE}.tex"
PDF="resume_${COMPANY}_${ROLE}.pdf"
JD="$ROOT_DIR/jobs/${COMPANY}_${ROLE}.txt"

cd "$ROOT_DIR"

if [[ ! -f "$TEX" ]]; then
  if [[ -f "$JD" ]]; then
    "$SCRIPT_DIR/new_tailored.sh" "$COMPANY" "$ROLE" "$JD"
  else
    "$SCRIPT_DIR/new_tailored.sh" "$COMPANY" "$ROLE"
  fi
  echo ""
  echo "Edit $TEX for JD-specific content, then re-run this script."
  exit 0
fi

echo "==> Compiling $TEX"
"$SCRIPT_DIR/compile.sh" "$TEX"

echo ""
echo "==> ATS parse test"
"$SCRIPT_DIR/verify_parse.sh" "$PDF"

if [[ -f "$KEYWORDS" ]]; then
  echo ""
  echo "==> Keyword coverage check"
  python3 "$SCRIPT_DIR/check_keywords.py" "${PDF%.pdf}.parse.txt" "$KEYWORDS"
else
  echo ""
  echo "No keywords file at $KEYWORDS (skipping keyword check)"
fi

echo ""
echo "Done: $ROOT_DIR/$PDF"
