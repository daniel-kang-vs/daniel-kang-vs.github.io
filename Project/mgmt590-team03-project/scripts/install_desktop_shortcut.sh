#!/usr/bin/env bash
# Copy Launch NYC Taxi Agent.command to Desktop (macOS).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$PROJECT_ROOT/Launch NYC Taxi Agent.command"
DEST="$HOME/Desktop/Launch NYC Taxi Agent.command"
cp "$SRC" "$DEST"
chmod +x "$DEST"
echo "Installed: $DEST"
echo "Double-click to start Streamlit at http://localhost:8501"
