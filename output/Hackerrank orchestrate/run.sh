#!/usr/bin/env bash
# run.sh — one-shot launcher for the Support Triage Agent
# Works on Linux, macOS, and Git Bash / WSL on Windows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TICKETS="${1:-"$REPO_ROOT/support_tickets/support_tickets.csv"}"
OUTPUT="${2:-"$REPO_ROOT/support_tickets/output.csv"}"
LOG="${3:-"$REPO_ROOT/support_tickets/log.txt"}"

echo "=============================================="
echo " Support Triage Agent"
echo "=============================================="
echo " Tickets : $TICKETS"
echo " Output  : $OUTPUT"
echo " Log     : $LOG"
echo "=============================================="

cd "$SCRIPT_DIR"

# Install dependencies if needed
if ! python -c "import faiss" 2>/dev/null; then
    echo "[setup] Installing Python dependencies …"
    pip install -r requirements.txt
fi

python main.py \
    --tickets "$TICKETS" \
    --output  "$OUTPUT"  \
    --log     "$LOG"

echo ""
echo "Done."
echo "  Output CSV : $OUTPUT"
echo "  Log file   : $LOG"
