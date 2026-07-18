#!/usr/bin/env bash
# Compile, verify, and optionally check keywords. Saves only .tex and .pdf under {company}/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=naming.sh
source "$SCRIPT_DIR/naming.sh"

usage() {
  echo "Usage: $0 <company> <role> [YYYYMMDD] [keywords.txt]" >&2
  echo "" >&2
  echo "Output: Resume Tailor/{company}/{company}_{date}_{role}_resume.tex|pdf" >&2
  exit 1
}

[[ $# -ge 2 ]] || usage

COMPANY="$1"
ROLE="$2"
DATE="$(date +%Y%m%d)"
KEYWORDS=""

if [[ $# -ge 3 ]]; then
  if [[ "$3" =~ ^[0-9]{8}$ ]]; then
    DATE="$3"
    KEYWORDS="${4:-}"
  else
    KEYWORDS="$3"
  fi
fi

TEX="$(resume_tex_path "$ROOT_DIR" "$COMPANY" "$ROLE" "$DATE")"
PDF="$(resume_pdf_path "$ROOT_DIR" "$COMPANY" "$ROLE" "$DATE")"
REL_TEX="${TEX#"$ROOT_DIR"/}"

if [[ ! -f "$TEX" ]]; then
  "$SCRIPT_DIR/new_tailored.sh" "$COMPANY" "$ROLE" "$DATE"
  echo ""
  echo "Edit $TEX for JD-specific content, then re-run this script."
  exit 0
fi

echo "==> Compiling $REL_TEX"
"$SCRIPT_DIR/compile.sh" "$TEX"

echo ""
echo "==> ATS parse test"
"$SCRIPT_DIR/verify_parse.sh" "$PDF"

if [[ -n "$KEYWORDS" && -f "$KEYWORDS" ]]; then
  PARSE_TMP="$(mktemp)"
  python3 "$SCRIPT_DIR/extract_pdf_text.py" "$PDF" -o "$PARSE_TMP" >/dev/null
  echo ""
  echo "==> Keyword coverage check"
  python3 "$SCRIPT_DIR/check_keywords.py" "$PARSE_TMP" "$KEYWORDS"
  rm -f "$PARSE_TMP"
fi

echo ""
echo "Done:"
echo "  $TEX"
echo "  $PDF"
