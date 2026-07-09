#!/bin/bash
# quota_watch.sh
# ==============
# Monitors SLURM partition quotas and submits waiting experiments
# when quota frees up. Designed to run from cron on the login node.
#
# Cron entry (run `crontab -e` on the login node):
#   */15 * * * * bash /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/env/quota_watch.sh >> /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/env/logs/quota_watch.log 2>&1

set -euo pipefail

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
LOG_DIR="${REPO_DIR}/expes/env/logs"
cd "$REPO_DIR/expes/env"
mkdir -p "$LOG_DIR"

echo "=== $(date) quota_watch: checking ==="

# Partitions we want to submit experiments to
# NOTE: b300 is excluded because PyTorch 2.5.1 lacks Blackwell (sm_120) kernels
#       and all CUDA operations fail. Add back once PyTorch >= 2.6 is deployed.
PARTITIONS=(
  "qc-a100"
  "ai-l40s"
)

# Experiments to submit (script, short label for logging)
EXPERIMENTS=(
  "run_synth_qc-a100.slurm"
  "run_tfn_sweep_qc-a100.slurm"
  "run_synth_ai-l40s.slurm"
  "run_tfn_sweep_ai-l40s.slurm"
  # Synth — retries every 15 min when quota frees up
  "run_synth.slurm"
  "run_synth_ng-dgx-m0.slurm"
  "run_synth_ng-dgx-m1.slurm"
  "run_synth_ng-dgx-m2.slurm"
  "run_synth_ng-dgx-m3.slurm"
  # End-to-end — retries every 15 min when quota frees up
  "run_ucr_e2e.slurm"
  "run_ucr_e2e_ai-l40s.slurm"
  "run_ucr_e2e_qc-a100.slurm"
  "run_ucr_e2e_h100.slurm"
  "run_ucr_e2e_ai-h200-br.slurm"
  "run_ucr_e2e_ng-dgx-m0.slurm"
  "run_ucr_e2e_ng-dgx-m1.slurm"
  "run_ucr_e2e_ng-dgx-m2.slurm"
  "run_ucr_e2e_ng-dgx-m3.slurm"
)

submit_if_free() {
  local script="$1"

  # Skip if script doesn't exist
  if [ ! -f "$script" ]; then
    echo "  SKIP $script: not found"
    return
  fi

  # Extract job name and partition from the script
  local jobname
  jobname=$(grep -m1 "^#SBATCH --job-name=" "$script" | sed 's/.*=//')
  local partition
  partition=$(grep -m1 "^#SBATCH --partition=" "$script" | sed 's/.*=//')

  if [ -z "$jobname" ] || [ -z "$partition" ]; then
    echo "  SKIP $script: could not parse job-name or partition"
    return
  fi

  # Check for existing jobs matching this partition AND job name
  local existing
  existing=$(squeue -u "$USER" --noheader -o "%j %P" 2>/dev/null | awk -v n="$jobname" -v p="$partition" '$1 == n && $2 == p {count++} END {print count+0}')
  if [ "$existing" -gt 0 ]; then
    echo "  SKIP $script ($jobname @ $partition): $existing job(s) already exist"
    return
  fi

  echo "  Submitting $script ($jobname @ $partition)..."
  sbatch "$script" 2>&1 || echo "  FAILED: $script"
}

for exp in "${EXPERIMENTS[@]}"; do
  submit_if_free "$exp"
done

echo "=== $(date) quota_watch: done ==="
