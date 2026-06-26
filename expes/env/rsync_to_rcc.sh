#!/usr/bin/env bash
# Rsync the Ripsnet- repo to RCC Cloud Lustre storage.
#
# Usage:
#   bash expes/env/rsync_to_rcc.sh
#
# Prerequisites: SSH host rcc-cloud configured.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH_HOST="${RCC_SSH_HOST:-rcc-cloud}"

RCC_USER="$(ssh ${RCC_SSH_OPTS:-} "$SSH_HOST" 'whoami' 2>/dev/null | tr -d '\r')"
REMOTE_DIR="/hs/work0/home/users/${RCC_USER}/exp/ripsnet/Ripsnet-"

echo "### Syncing $REPO_ROOT -> $SSH_HOST:$REMOTE_DIR"
rsync -avz --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.vscode' \
  --exclude='*.pkl' \
  --exclude='*.pt' \
  -e "ssh ${RCC_SSH_OPTS:-}" \
  "$REPO_ROOT/" "$SSH_HOST:$REMOTE_DIR"

echo "### Done. Repo synced to $REMOTE_DIR"
