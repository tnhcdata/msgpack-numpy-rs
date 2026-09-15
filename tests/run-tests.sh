#!/usr/bin/env bash

set -euo pipefail

echo "Running Rust tests..."
cargo test --locked

echo "Compiling Rust benchmark..."
cargo bench --locked --bench bench_serialize_and_deserialize --no-run

echo "Testing deserializing in Python msgpacks created in Rust..."
source tests/venv/bin/activate
pytest tests/test_deserialize.py -s
