#!/usr/bin/env bash
# Run from YOUR LAPTOP (not the server) to push the current code to the
# server. Excludes secrets, local databases, and generated files -- those
# get transferred separately and deliberately (see README.md), never as
# part of a routine code sync.
#
# Usage: deploy/sync.sh <user>@<host>
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <user>@<host>"
    exit 1
fi

TARGET="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -avz --delete \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '*.pem' \
    --exclude '*.db' \
    --exclude '*.db-wal' \
    --exclude '*.db-shm' \
    --exclude 'logs/' \
    --exclude '.git' \
    --exclude 'app/build/' \
    --exclude 'build/' \
    --exclude '.gradle/' \
    --exclude '.kotlin/' \
    "$REPO_ROOT/" "$TARGET:/home/tradebot/TRADE_BOT/"

echo "==> Code synced. Secrets and local state were deliberately NOT touched."
