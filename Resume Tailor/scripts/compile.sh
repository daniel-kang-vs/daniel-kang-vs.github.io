#!/usr/bin/env bash
# Compile a .tex resume; leave only .tex and .pdf in the output folder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <resume.tex>" >&2
  exit 1
fi

TEX_FILE="$1"
if [[ ! "$TEX_FILE" = /* ]]; then
  TEX_FILE="$ROOT_DIR/$TEX_FILE"
fi

if [[ ! -f "$TEX_FILE" ]]; then
  echo "Error: file not found: $TEX_FILE" >&2
  exit 1
fi

WORK_DIR="$(dirname "$TEX_FILE")"
BASE="$(basename "$TEX_FILE" .tex)"
TECTONIC="$SCRIPT_DIR/bin/tectonic"

cd "$WORK_DIR"

if [[ -x "$TECTONIC" ]]; then
  "$TECTONIC" -X compile "$BASE.tex" --synctex --keep-logs --keep-intermediates >/dev/null
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null
  pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null
else
  echo "Error: no LaTeX compiler found." >&2
  exit 1
fi

rm -f "$BASE.aux" "$BASE.log" "$BASE.out" "$BASE.synctex.gz" \
      "$BASE.fls" "$BASE.fdb_latexmk" "$BASE.parse.txt"

echo "Compiled: $WORK_DIR/$BASE.pdf"
