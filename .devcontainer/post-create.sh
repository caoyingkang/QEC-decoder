#!/usr/bin/env bash
set -euo pipefail

uv sync
(cd packages/qecdec && uvx maturin develop --release)
