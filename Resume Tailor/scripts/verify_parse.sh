#!/usr/bin/env bash
# ATS copy-paste parse test: extract PDF text and verify section order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <resume.pdf>" >&2
  exit 1
fi

PDF_FILE="$1"
if [[ ! "$PDF_FILE" = /* ]]; then
  PDF_FILE="$ROOT_DIR/$PDF_FILE"
fi

if [[ ! -f "$PDF_FILE" ]]; then
  echo "Error: file not found: $PDF_FILE" >&2
  exit 1
fi

TEXT_FILE="${PDF_FILE%.pdf}.parse.txt"
python3 "$SCRIPT_DIR/extract_pdf_text.py" "$PDF_FILE" -o "$TEXT_FILE" >/dev/null
TEXT="$(cat "$TEXT_FILE")"

FAIL=0
check_contains() {
  local label="$1"
  local pattern="$2"
  if printf '%s' "$TEXT" | grep -qi "$pattern"; then
    echo "  OK: $label"
  else
    echo "  FAIL: $label (missing: $pattern)" >&2
    FAIL=1
  fi
}

echo "Parse test: $PDF_FILE"
check_contains "Name" "Byungjoo"
check_contains "Contact email" "kang531@purdue.edu"
check_contains "Summary or Profile" "Summary\\|Profile\\|Business Intelligence"
check_contains "Experience section" "Experience\\|Jambo"
check_contains "Education section" "Education\\|Purdue"
check_contains "Skills or tools" "SQL\\|Python\\|Power BI"

NAME_POS=$(printf '%s' "$TEXT" | grep -ni "Byungjoo" | head -1 | cut -d: -f1 || true)
EXP_POS=$(printf '%s' "$TEXT" | grep -ni "Jambo" | head -1 | cut -d: -f1 || true)
if [[ -n "$NAME_POS" && -n "$EXP_POS" && "$NAME_POS" -lt "$EXP_POS" ]]; then
  echo "  OK: Name appears before experience"
else
  echo "  FAIL: Text order may be scrambled" >&2
  FAIL=1
fi

echo "Extracted text saved to: $TEXT_FILE"
if [[ "$FAIL" -eq 0 ]]; then
  echo "Parse test passed."
else
  echo "Parse test failed." >&2
  exit 1
fi
