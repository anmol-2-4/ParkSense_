#!/bin/bash
# One-command demo: install deps, generate data, train, serve API + UI
set -e
cd "$(dirname "$0")"
if [ -d .venv ]; then
  PIP=".venv/bin/pip"
  PY=".venv/bin/python"
  UVICORN=".venv/bin/uvicorn"
else
  python3 -m venv .venv
  PIP=".venv/bin/pip"
  PY=".venv/bin/python"
  UVICORN=".venv/bin/uvicorn"
fi
echo "Installing dependencies..."
$PIP install -q -r requirements.txt
echo "Training model (generates synthetic data if needed)..."
$PY -m src.train.run
echo "Starting API and UI at http://127.0.0.1:8000"
$UVICORN src.api.app:app --host 0.0.0.0 --port 8000
