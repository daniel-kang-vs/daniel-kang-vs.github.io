#!/usr/bin/env bash
# Terminal launcher — same as double-clicking Launch NYC Taxi Agent.command
set -euo pipefail
cd "$(dirname "$0")"
exec bash scripts/launch_agent.sh "$@"
