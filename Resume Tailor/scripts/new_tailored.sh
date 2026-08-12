#!/usr/bin/env bash
# Create a tailored resume copy under Resume Tailor/{company}/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BASELINE="$ROOT_DIR/resume.tex"

# shellcheck source=naming.sh
source "$SCRIPT_DIR/naming.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <company> <role> [YYYYMMDD]" >&2
  echo "Example: $0 TransUnion \"Sr Analyst\" 20260704" >&2
  exit 1
fi

COMPANY="$1"
ROLE="$2"
DATE="$(date +%Y%m%d)"

if [[ $# -ge 3 && "$3" =~ ^[0-9]{8}$ ]]; then
  DATE="$3"
fi

OUT="$(resume_tex_path "$ROOT_DIR" "$COMPANY" "$ROLE" "$DATE")"
OUT_DIR="$(dirname "$OUT")"
BASENAME="$(basename "$OUT" .tex)"
DISPLAY_DATE="$(date -j -f "%Y%m%d" "$DATE" "+%Y-%m-%d" 2>/dev/null || date -d "$DATE" "+%Y-%m-%d" 2>/dev/null || echo "$DATE")"

if [[ -f "$OUT" ]]; then
  echo "Error: already exists: $OUT" >&2
  exit 1
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "Error: baseline not found: $BASELINE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cp "$BASELINE" "$OUT"

sed -i '' \
  -e 's/\\section\*{Profile}/\\section\*{Summary}/' \
  -e 's/\\section\*{Professional Experience}/\\section\*{Experience}/' \
  -e 's/Al-assisted/AI-assisted/g' \
  -e 's/Azure Al Fundamentals/Azure AI Fundamentals/g' \
  -e 's/LinkedIn |/\\href{https:\/\/www.linkedin.com\/in\/daniel-byungjoo-kang\/}{LinkedIn} |/' \
  "$OUT"

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
  echo "% Tailored: $DISPLAY_DATE | Company: $COMPANY | Role: $ROLE"
  echo "% Filename: ${BASENAME}.tex"
  cat "$OUT"
} > "${OUT}.tmp" && mv "${OUT}.tmp" "$OUT"

echo "Created: $OUT"
echo "Next: edit for JD keywords, then run:"
echo "  ./scripts/tailor.sh \"$COMPANY\" \"$ROLE\" $DATE"
