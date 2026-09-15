#!/usr/bin/env bash

set -euo pipefail

python3 -m venv tests/venv
source tests/venv/bin/activate
python -m pip install uv
uv pip install msgpack-numpy pytest
