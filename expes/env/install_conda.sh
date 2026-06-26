#!/usr/bin/env bash
# Install the ripsnet conda environment on RCC Cloud (from laptop).
#
# Usage:
#   bash expes/env/install_conda.sh --gh200    # for qc-gh200 (aarch64)
#   bash expes/env/install_conda.sh --a100     # for qc-a100  (x86_64)
#
# Prerequisites:
#   1. rsync the Ripsnet- repo to RCC first (see rsync_to_rcc.sh)
#   2. SSH host rcc-cloud configured
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="gh200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gh200) TARGET="gh200"; shift ;;
    --a100)  TARGET="a100";  shift ;;
    -h|--help)
      echo "Usage: $0 [--gh200|--a100]" >&2
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

SSH_HOST="${RCC_SSH_HOST:-rcc-cloud}"
RCC_USER="$(ssh ${RCC_SSH_OPTS:-} "$SSH_HOST" 'whoami' 2>/dev/null | tr -d '\r')"
[[ -n "$RCC_USER" ]] || { echo "ERROR: SSH to $SSH_HOST failed" >&2; exit 3; }

REMOTE_REPO="${REMOTE_RIPSNET_REPO:-/hs/work0/home/users/${RCC_USER}/exp/ripsnet/Ripsnet-}"
REMOTE_INSTALLER="${REMOTE_REPO}/expes/env/_install_compute.sh"

if [[ "$TARGET" == "a100" ]]; then
  SLURM_PARTITION="${SLURM_PARTITION:-qc-a100}"
else
  SLURM_PARTITION="${SLURM_PARTITION:-qc-gh200}"
fi
TIMELIMIT="${SLURM_TIMELIMIT:-02:00:00}"

echo "### Target: $TARGET partition=$SLURM_PARTITION"
echo "### Remote installer: $REMOTE_INSTALLER"

ssh "${RCC_SSH_OPTS:-}" "$SSH_HOST" bash -s -- \
  "$SLURM_PARTITION" "$TIMELIMIT" "$REMOTE_INSTALLER" <<'REMOTE_SCRIPT'
set -euo pipefail
PARTITION="$1"; TIMELIMIT="$2"; INSTALLER="$3"

echo "[$(date -Iseconds)] Checking installer on remote..."
[[ -f "$INSTALLER" ]] || { echo "ERROR: $INSTALLER not found – rsync repo first." >&2; exit 2; }

echo "[$(date -Iseconds)] Submitting srun to $PARTITION..."
srun -N1 -p "$PARTITION" -t "$TIMELIMIT" --export=ALL bash "$INSTALLER"
echo "[$(date -Iseconds)] Done."
REMOTE_SCRIPT
