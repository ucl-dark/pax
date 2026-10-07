#!/bin/bash
# Unified launcher for R3 IPDITM (InTheMatrix) self-play (independent naive learners)
# Produces the self-play reference returns (v_ref) for Phase 2b welfare shaper training.
#
# Usage:
#   bash R3_ipditm_selfplay.sh <platform> <seed>
#
# Platforms:
#   fir        — Fir cluster, 1×H100 MIG slice (naive learner needs little compute)
#   tri        — Trillium cluster, 1×H100
#   tri-debug  — Trillium, 1×H100, 30min, tiny run to test the pipeline
#
# Examples:
#   bash R3_ipditm_selfplay.sh fir 0
#   bash R3_ipditm_selfplay.sh tri 0
#   bash R3_ipditm_selfplay.sh tri-debug 0

PLATFORM=${1:-tri}
SEED=${2:-0}

# ──────────────────────────────────────────────────────────────────
# Auto-submit: if not already running under SLURM, sbatch ourselves
# ──────────────────────────────────────────────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    case "$PLATFORM" in
        fir)
            sbatch \
                --account=def-jtyao_gpu \
                --job-name=R3_ipditm_s${SEED} \
                --nodes=1 \
                --ntasks=1 \
                --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=4 \
                --mem=8G \
                --time=00:10:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=R3_ipditm_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=4 \
                --time=00:10:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=R3_ipditm_dbg_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=4 \
                --time=0:30:00 \
                --output=/scratch/lichenqi/debug_output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        *)
            echo "Unknown platform '$PLATFORM'. Use: fir, tri, or tri-debug"
            ;;
    esac
    exit $?
fi

# ──────────────────────────────────────────────────────────────────
# Actual job (running under SLURM from here)
# ──────────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3
module load cuda/12.6
module load python/3.11.5
source /project/def-jtyao/lichenqi/pax_env_py3.11.5/bin/activate

export TMPDIR="${SLURM_TMPDIR:-/tmp}"

export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

export WANDB_API_KEY="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"
mkdir -p "$TMPDIR/wandb" "$TMPDIR/wandb-cache" "$TMPDIR/wandb_config"

export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb_config"
export WANDB_SERVICE_TRANSPORT=tcp
export WANDB__SERVICE_WAIT=180
export WANDB_INIT_TIMEOUT=180
export WANDB_START_METHOD=thread

EXPERIMENT="ipditm=independent_learners"
RESULTS_DIR="/scratch/lichenqi/results/R3_ipditm_seed${SEED}"
HYDRA_DIR="$TMPDIR/hydra_output"
EXP_OUTPUT="$HYDRA_DIR/exp"
mkdir -p "$RESULTS_DIR"

start_time=$(date +%s)
echo "=== Platform: $PLATFORM | Seed: $SEED | $(date '+%Y-%m-%d %H:%M:%S') ==="

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            hydra.run.dir=$HYDRA_DIR
        ;;
    tri-debug|fir-debug)
        # Debug run — tiny, just to verify the pipeline end-to-end
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=20 \
            ++save_interval=20 \
            hydra.run.dir=$HYDRA_DIR
        ;;
esac

# ──────────────────────────────────────────────────────────────────
# Copy results to persistent storage
# ──────────────────────────────────────────────────────────────────
echo "Copying final results to $RESULTS_DIR ..."
cp -rL "$EXP_OUTPUT"/independentLearning-*/ "$RESULTS_DIR/" 2>/dev/null
mkdir -p /scratch/lichenqi/wandb_saved
cp -rL "$WANDB_DIR"/wandb/offline-run-* /scratch/lichenqi/wandb_saved/ 2>/dev/null || true

end_time=$(date +%s)
echo "=== Done: $PLATFORM seed=$SEED | Elapsed: $((end_time - start_time))s ==="
echo ">>> RECORD the final 'Average Reward per Timestep: (r1, r2)' for Phase 2b <<<"
