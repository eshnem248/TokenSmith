#!/usr/bin/env bash
# Run multimodal ingestion with MPS fallback enabled.
# Must be called from project root.
#
# Usage:
#   bash scripts/run_ingest.sh
#   bash scripts/run_ingest.sh --pdf data/chapters/silberschatz.pdf --index-dir data/multimodal

set -e

PYTHON="/Users/sathwiknemani/miniforge3/envs/tokensmith/bin/python"

# PYTORCH_ENABLE_MPS_FALLBACK must be set before Python starts so that the
# PyTorch C extension initializes with MPS fallback mode.  Setting it inside
# Python (os.environ) is too late — the Metal device is already initialized.
export PYTORCH_ENABLE_MPS_FALLBACK=1

exec "$PYTHON" scripts/ingest_multimodal.py "$@"
