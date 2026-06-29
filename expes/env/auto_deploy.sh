#!/bin/bash
# auto_deploy.sh
# ==============
# Cron-triggered script: checks GitHub for new commits, and when found,
# pulls the changes, kills any running ripsnet-ucr job, and submits a
# fresh SLURM array job.
#
# Intended to run every 15 minutes via crontab on the cluster login node.
#
# Usage
# -----
#   bash expes/env/auto_deploy.sh          # normal run (checks, deploys if new)
#   bash expes/env/auto_deploy.sh --force  # deploy even without new commits
#
# Cron entry (run `crontab -e` on the login node):
#   */15 * * * * /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/env/auto_deploy.sh >> /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/env/logs/auto_deploy.log 2>&1

set -euo pipefail

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
SLURM_SCRIPT="${REPO_DIR}/expes/env/run_ucr_split.slurm"
LOG_DIR="${REPO_DIR}/expes/env/logs"
DEPLOY_LOG="${LOG_DIR}/auto_deploy.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=== $(date) auto_deploy: checking ==="

# 1. Fetch remote and compare HEAD
HEAD_BEFORE=$(git rev-parse HEAD)
git fetch origin main 2>&1 || { echo "  git fetch failed"; exit 1; }
HEAD_REMOTE=$(git rev-parse origin/main)

FORCE="${1:-}"
if [ "$HEAD_BEFORE" = "$HEAD_REMOTE" ] && [ "$FORCE" != "--force" ]; then
    echo "  No new commits (HEAD=${HEAD_BEFORE:0:8}) — nothing to do."
    exit 0
fi

echo "  New commit detected: ${HEAD_BEFORE:0:8} → ${HEAD_REMOTE:0:8}"

# 2. Pull (fast-forward only)
git merge --ff-only origin/main 2>&1 || {
    echo "  git merge failed — stashing local changes and retrying..."
    git stash
    git merge --ff-only origin/main 2>&1 || { echo "  Still failed, aborting"; exit 1; }
}

echo "  Pulled: $(git log --oneline -1)"

# 3. Kill any existing ripsnet-ucr jobs
EXISTING=$(squeue -u "$USER" --noheader -o "%i" -n ripsnet-ucr 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "  Killing existing job(s): $EXISTING"
    scancel $EXISTING
    sleep 5
fi

# 4. Submit new job
cd "$(dirname "$SLURM_SCRIPT")"
NEW_JOB=$(sbatch "$(basename "$SLURM_SCRIPT")" 2>&1 | tail -1)
echo "  Submitted: $NEW_JOB"

echo "=== $(date) auto_deploy: done ==="
