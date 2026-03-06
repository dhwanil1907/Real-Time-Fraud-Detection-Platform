#!/usr/bin/env bash
# Run the fraud detection CLI. Uses python3 if available, else python.
set -e
cd "$(dirname "$0")"
if command -v python3 &>/dev/null; then
  exec python3 main.py "$@"
else
  exec python main.py "$@"
fi
