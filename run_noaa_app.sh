#!/usr/bin/env bash
# Starts the NOAA-18 / NOAA-19 SST gap-filling web app at http://localhost:8052
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
NOAA_APP_OPEN_BROWSER=1 exec "$PY" noaa_app.py
