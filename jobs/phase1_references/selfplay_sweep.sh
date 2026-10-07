#!/bin/bash
# Sweep launcher for the phase-1 self-play references — submits one SLURM job per seed.
# Works for both R2 (CoinGame) and R3 (IPDITM); each job is a 10-minute, 1-GPU run.
#
# Usage:
#   bash selfplay_sweep.sh <R2|R3> <platform> [num_seeds] [start_seed]
#
# Examples:
#   bash selfplay_sweep.sh R2 fir            # R2 seeds 0-99 on Fir
#   bash selfplay_sweep.sh R3 fir            # R3 seeds 0-99 on Fir
#   bash selfplay_sweep.sh R2 tri 100        # R2 seeds 0-99 on Trillium
#   bash selfplay_sweep.sh R3 fir 50 100     # R3 seeds 100-149 on Fir

JOB=${1:?Usage: bash selfplay_sweep.sh <R2|R3> <platform> [num_seeds] [start_seed]}
PLATFORM=${2:-fir}
NUM_SEEDS=${3:-100}
START_SEED=${4:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$JOB" in
    R2|r2) JOB_SCRIPT="$SCRIPT_DIR/R2_cg_selfplay.sh" ;;
    R3|r3) JOB_SCRIPT="$SCRIPT_DIR/R3_ipditm_selfplay.sh" ;;
    *)
        echo "Unknown job '$JOB'. Use: R2 or R3"
        exit 1
        ;;
esac

END_SEED=$((START_SEED + NUM_SEEDS - 1))
echo "Submitting $JOB self-play: platform=$PLATFORM seeds=$START_SEED..$END_SEED"

for ((SEED = START_SEED; SEED <= END_SEED; SEED++)); do
    echo "--- seed $SEED ---"
    bash "$JOB_SCRIPT" "$PLATFORM" "$SEED"
    sleep 1   # be gentle with the scheduler
done

echo "Submitted $NUM_SEEDS jobs. Check with: squeue -u \$USER"
