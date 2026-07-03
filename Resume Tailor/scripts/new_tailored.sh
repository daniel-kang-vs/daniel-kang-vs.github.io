#!/usr/bin/env bash
# Create a new tailored resume copy from baseline with ATS structural fixes applied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BASELINE="$ROOT_DIR/resume.tex"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <company_slug> <role_slug> [job_description.txt]" >&2
  echo "Example: $0 acme data_analyst jobs/acme_data_analyst.txt" >&2
  exit 1
fi

COMPANY="$1"
ROLE="$2"
JD_FILE="${3:-}"
OUT="$ROOT_DIR/resume_${COMPANY}_${ROLE}.tex"
DATE="$(date +%Y-%m-%d)"

if [[ ! -f "$BASELINE" ]]; then
  echo "Error: baseline not found: $BASELINE" >&2
  exit 1
fi

cp "$BASELINE" "$OUT"

# ATS structural fixes (baseline stays unchanged).
sed -i '' \
  -e 's/\\section\*{Profile}/\\section\*{Summary}/' \
  -e 's/\\section\*{Professional Experience}/\\section\*{Experience}/' \
  -e 's/Al-assisted/AI-assisted/g' \
  -e 's/Azure Al Fundamentals/Azure AI Fundamentals/g' \
  -e 's/LinkedIn |/\\href{https:\/\/www.linkedin.com\/in\/daniel-byungjoo-kang\/}{LinkedIn} |/' \
  "$OUT"

# Move inline Skills line to a dedicated Skills section when present.
python3 - "$OUT" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

skills_match = re.search(
    r"\\\\\[4pt\]\s*\n\\textbf\{Skills:\}\s*(.+?)\s*\n\n% --- EDUCATION ---",
    text,
    re.DOTALL,
)
if skills_match:
    skills = skills_match.group(1).strip()
    text = re.sub(
        r"\\\\\[4pt\]\s*\n\\textbf\{Skills:\}\s*.+?\n\n% --- EDUCATION ---",
        "\n\n% --- EDUCATION ---",
        text,
        count=1,
        flags=re.DOTALL,
    )
    skills_block = (
        "\n% --- SKILLS ---\n"
        "\\section*{Skills}\n"
        f"{skills}\n"
    )
    text = text.replace("% --- EDUCATION ---", skills_block + "\n% --- EDUCATION ---", 1)

path.write_text(text, encoding="utf-8")
PY

{
  echo "% Tailored: $DATE | Company: $COMPANY | Role: $ROLE"
  if [[ -n "$JD_FILE" && -f "$JD_FILE" ]]; then
    echo "% JD source: $(basename "$JD_FILE")"
  fi
  echo "% ATS structure applied: Summary, Skills, Experience headers; AI typo fixes; LinkedIn URL"
  cat "$OUT"
} > "${OUT}.tmp" && mv "${OUT}.tmp" "$OUT"

echo "Created: $OUT"
if [[ -n "$JD_FILE" && -f "$JD_FILE" ]]; then
  echo "JD saved for reference at: $JD_FILE"
fi
echo "Next: edit content for JD keywords, then run scripts/compile.sh and scripts/verify_parse.sh"
