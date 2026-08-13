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

# Required, not cosmetic: diagnosed live 2026-08-06 when this broke every
# service (sqlite3.OperationalError: attempt to write a readonly
# database). rsync's archive mode (-a, implied by -avz) preserves the
# SOURCE's UID/GID whenever the remote side runs as root (true here --
# this SSH alias connects as root), so every prior sync had been silently
# overwriting /home/tradebot/TRADE_BOT's ownership with whatever local
# laptop user ran this script, locking the tradebot user out of writing to
# its own directory the moment SQLite needed to create a fresh -wal file
# there. The GNU-rsync-only `--chown` flag would fix this in the rsync
# call itself, but macOS's built-in rsync is openrsync (BSD), which
# doesn't support it -- an explicit chown after the fact works regardless
# of which rsync is installed locally.
ssh "$TARGET" "chown -R tradebot:tradebot /home/tradebot/TRADE_BOT"

echo "==> Code synced. Secrets and local state were deliberately NOT touched."
