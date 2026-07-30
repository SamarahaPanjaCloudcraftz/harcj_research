#!/usr/bin/env bash
# Sets up a dedicated conda environment for the HARCJ research dashboard.
#
# Usage:
#   ./setup.sh          # create env + install deps
#   ./setup.sh --run    # also launch the app after setup
#
# This is fully separate from dashboard_new/ and its venv — no live code
# or environment is touched by anything in this folder.

set -euo pipefail

ENV_NAME="harcj_research"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

echo "Installing requirements into ${ENV_NAME}"
conda run -n "${ENV_NAME}" pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "Setup complete. To run the dashboard:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${SCRIPT_DIR}"
echo "  streamlit run app.py --server.port 8600"

if [[ "${1:-}" == "--run" ]]; then
    cd "${SCRIPT_DIR}"
    conda run -n "${ENV_NAME}" streamlit run app.py --server.port 8600
fi
