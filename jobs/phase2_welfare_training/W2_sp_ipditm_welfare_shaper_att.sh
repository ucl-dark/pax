#!/bin/bash
# Unified launcher for W2 IPDITM welfare shaper att (self-play)
#
# Usage:
#   bash W2_sp_ipditm_welfare_shaper_att.sh <platform> <seed>
#
# Platforms:
#   fir        — Fir cluster, 4×H100, ~3h wall time
#   tri        — Trillium cluster, 4×H100, 1d20h wall time
#   tri-debug  — Trillium, 1×H100, 1h, small params, runs twice to test resume
#
# Examples:
#   bash W2_sp_ipditm_welfare_shaper_att.sh fir 21
#   bash W2_sp_ipditm_welfare_shaper_att.sh tri 0
#   bash W2_sp_ipditm_welfare_shaper_att.sh tri-debug 21

PLATFORM=${1:-tri}
SEED=${2:-0}

# ──────────────────────────────────────────────────────────────────
# h100:4 for 400gen,9:30h;600gen,14h;
# ──────────────────────────────────────────────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    case "$PLATFORM" in
        fir)
            sbatch \
                --account=def-jtyao_gpu \
                --job-name=W2sp_ipditm_s${SEED} \
                --nodes=1 \
                --ntasks=1 \
                --gpus-per-node=h100:4 \
                --cpus-per-task=6 \
                --mem=30G \
                --time=9:30:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=W2sp_ipditm_s${SEED} \
                --gpus-per-node=h100:4 \
                --time=7:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=W2sp_ipditm_dbg_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=6 \
                --time=1:00:00 \
                --output=/scratch/lichenqi/debug_output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        *)
            echo "platform '$PLATFORM'. Use: fir, tri, tri-debug or fir-debug"
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

EXPERIMENT="ipditm=welfare_shaper_att"
RESUME_DIR="/scratch/lichenqi/resume/W2_sp_ipditm_seed${SEED}"
HYDRA_DIR="$TMPDIR/hydra_output"
# Hydra changes CWD to HYDRA_DIR inside Python, so save_dir ends up here:
EXP_OUTPUT="$HYDRA_DIR/exp"
mkdir -p "$RESUME_DIR"

start_time=$(date +%s)
echo "=== Platform: $PLATFORM | Seed: $SEED | $(date '+%Y-%m-%d %H:%M:%S') ==="

cd /project/def-jtyao/lichenqi/pax
case "$PLATFORM" in
    fir|tri)
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_devices=4 \
            ++welfare.resume_dir=$RESUME_DIR \
            hydra.run.dir=$HYDRA_DIR
        # ──────────────────────────────────────────────────────────────────
        # Copy results to persistent storage
        # ──────────────────────────────────────────────────────────────────
        echo "Copying final results to $RESUME_DIR ..."
        cp -rL "$EXP_OUTPUT"/welfare-*/ "$RESUME_DIR/" 2>/dev/null
        mkdir -p /scratch/lichenqi/wandb_saved
        cp -rL "$WANDB_DIR"/wandb/offline-run-* /scratch/lichenqi/wandb_saved/ 2>/dev/null || true

        end_time=$(date +%s)
        echo "=== Done: $PLATFORM seed=$SEED | Elapsed: $((end_time - start_time))s ==="
        ;;
    tri-debug|fir-debug)
        # Debug run — small params, run TWICE to test save/resume
        # Run 1: 6 generations (0-5), saves at 0 and 5
        # Run 2: 11 generations total, resumes from 5, trains 6-10
        echo "=== Debug run 1/2 (gen 0-5) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=11 \
            ++popsize=40 \
            ++num_outer_steps=20 \
            ++num_inner_steps=80 \
            ++num_devices=1 \
            ++save_interval=5 \
            ++welfare.resume_dir=$RESUME_DIR \
            hydra.run.dir=$HYDRA_DIR

        # Copy checkpoint to resume_dir so second run can find it
        echo "Copying checkpoints from $EXP_OUTPUT to $RESUME_DIR ..."
        cp -rL "$EXP_OUTPUT"/welfare-*/ "$RESUME_DIR/debug" 2>/dev/null
        echo "Resume dir contents:"
        find "$RESUME_DIR" -name "generation_*" | sort

        echo "=== Debug run 2/2 (gen 6-10, testing resume) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=21 \
            ++popsize=40 \
            ++num_outer_steps=20 \
            ++num_inner_steps=80 \
            ++num_devices=1 \
            ++save_interval=5 \
            ++welfare.resume_dir=$RESUME_DIR \
            hydra.run.dir=$HYDRA_DIR
        ;;
esac
