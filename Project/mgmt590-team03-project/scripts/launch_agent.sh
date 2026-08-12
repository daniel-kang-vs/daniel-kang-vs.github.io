#!/usr/bin/env bash
# Shared launcher for the NYC Taxi Optimization Agent (Streamlit).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

pick_python() {
  if [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
    echo "$PYTHON"
    return
  fi
  local candidates=(
    "$HOME/anaconda3/bin/python3"
    "$HOME/miniconda3/bin/python3"
    "$HOME/mambaforge/bin/python3"
    "/opt/anaconda3/bin/python3"
  )
  for py in "${candidates[@]}"; do
    if [[ -x "$py" ]] && "$py" -c "import lightgbm" 2>/dev/null; then
      echo "$py"
      return
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "python3"
}

PYTHON_BIN="$(pick_python)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " NYC Taxi Optimization Agent"
echo " Project: $PROJECT_ROOT"
echo " Python:  $PYTHON_BIN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! "$PYTHON_BIN" -c "import streamlit" 2>/dev/null; then
  echo ""
  echo "Installing dependencies (first run only)…"
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if ! "$PYTHON_BIN" -c "import lightgbm" 2>/dev/null; then
  echo ""
  echo "Warning: LightGBM not available — optimization runs will fail."
  echo "Use Anaconda Python or: brew install libomp"
  echo ""
fi

echo ""
echo "Starting Streamlit at http://localhost:8501"
echo "Stop the server with Ctrl+C."
echo ""

exec "$PYTHON_BIN" -m streamlit run app/streamlit_app.py --server.headless false "$@"
