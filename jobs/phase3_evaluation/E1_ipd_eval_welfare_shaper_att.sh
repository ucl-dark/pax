#!/bin/bash
# Unified launcher for E1 IPD welfare shaper att vs Tabular (evaluation)
#
# Usage:
#   bash E1_ipd_eval_welfare_shaper_att.sh <platform> <seed>
#
# Platforms:
#   fir        — Fir cluster, 1g.10gb partial H100, 1h wall time
#   tri        — Trillium cluster, 1g.10gb partial H100, 1h wall time
#   tri-debug  — Trillium, 1g.10gb partial H100, 1h, smaller num_iters
#
# IMPORTANT: Update model_path and run_path in the eval yaml before running:
#   pax/conf/experiment/ipd/eval_welfare_shaper_att_v_tabular.yaml
#
# Examples:
#   bash E1_ipd_eval_welfare_shaper_att.sh fir 0
#   bash E1_ipd_eval_welfare_shaper_att.sh tri 0
#   bash E1_ipd_eval_welfare_shaper_att.sh tri-debug 0

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
                --job-name=E1_ipd_eval_s${SEED} \
                --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=2 \
                --mem=4G \
                --time=0:30:00 \
                --output=/scratch/lichenqi/eval/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=E1_ipd_eval_s${SEED} \
                --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=2 \
                --mem=4G \
                --time=1:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=E1_ipd_eval_dbg_s${SEED} \
                --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=2 \
                --mem=4G \
                --time=1:00:00 \
                --output=/scratch/lichenqi/debug_output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        *)
            echo "platform '$PLATFORM'. Use: fir, tri, tri-debug"
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

# "ipd=eval_welfare_shaper_att_v_tabular"
# "ipd=eval_mfos_att_v_tabular"
# "ipd=eval_shaper_att_v_tabular"
EXPERIMENT="ipd=eval_welfare_shaper_att_v_tabular"
HYDRA_DIR="$TMPDIR/hydra_output"
NUM_SEEDS=20 #20

start_time=$(date +%s)
echo "=== Platform: $PLATFORM | Seed: $SEED | $(date '+%Y-%m-%d %H:%M:%S') ==="

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        for ((offset=0; offset<NUM_SEEDS; offset++)); do
            run_seed=$((SEED + offset))
            run_start_time=$(date +%s)
            echo "=== Run $((offset + 1))/$NUM_SEEDS | seed=$run_seed | $(date '+%Y-%m-%d %H:%M:%S') ==="

            python -m pax.experiment +experiment/$EXPERIMENT \
                seed=$run_seed \
                hydra.run.dir="$HYDRA_DIR/seed_${run_seed}"

            run_status=$?
            run_end_time=$(date +%s)
            if [ $run_status -ne 0 ]; then
                echo "=== Failed: $PLATFORM seed=$run_seed | Elapsed: $((run_end_time - run_start_time))s ==="
                exit $run_status
            fi

            echo "=== Completed: $PLATFORM seed=$run_seed | Elapsed: $((run_end_time - run_start_time))s ==="
        done

        end_time=$(date +%s)
        echo "=== Done: $PLATFORM seeds=${SEED}-$((SEED + NUM_SEEDS - 1)) | Elapsed: $((end_time - start_time))s ==="
        ;;
    tri-debug| fir-debug)
        echo "=== Debug eval run (small num_iters) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_outer_steps=10 \
            hydra.run.dir=$HYDRA_DIR \
            ++project=debug_ipd_eval
        ;;
esac
