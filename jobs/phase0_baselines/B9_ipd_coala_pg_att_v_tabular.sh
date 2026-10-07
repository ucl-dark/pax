#!/bin/bash
set -euo pipefail
# Unified launcher for B9 IPD COALA-PG vs tabular
#
# COALA-PG baseline: Meulemans et al. (2025), "Multi-agent cooperation through
# learning-aware policy gradients", ICLR 2025 (arXiv:2410.18636).
#
# Usage:
#   bash B9_ipd_coala_pg_att_v_tabular.sh <platform> <seed> [wandb_mode] [experiment] [hydra_overrides...]
#
# Platforms:
#   fir        — Fir cluster, 1×H100 MIG slice
#   tri        — Trillium cluster, 1×H100
#   tri-debug  — Trillium, 1×H100, 1h, tiny params for a smoke test
#
# wandb_mode: offline (default) | online
#   KEEP OFFLINE on Alliance compute nodes — they generally have no outbound
#   internet, so `online` will stall until WANDB_INIT_TIMEOUT and then degrade.
#   Run offline, then `wandb sync` from a login node.
#
# Examples:
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0
#   bash B9_ipd_coala_pg_att_v_tabular.sh tri 0
#   bash B9_ipd_coala_pg_att_v_tabular.sh tri-debug 0
#
#   # Lagrangian constrained welfare (PID dual, Stooke et al. 2020).
#   # Floors are PER INNER EPISODE; the config defaults to -14 for both
#   # players (see the yaml header for why, and for the unconstrained ablation).
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular \
#       ++welfare.freeze_lam=True ++welfare.lam_init=0.0 \
#       ++wandb.group=unconstrained-CoalaPGLagrangian-vs-CoalaA2C   # unconstrained ablation
#   # RCPO dual-ascent baseline: same file, P and D gains zeroed.
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular ++welfare.kp=0 ++welfare.kd=0
#   # Static weighted-welfare ablation: both lam frozen at lam_init.
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular ++welfare.freeze_lam=True ++welfare.lam_init=1.0
#   # Constrain the whole meta-episode instead of the last K episodes.
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular ++welfare.constraint_window=0
#   # Sweep the reference values -- same key names as the constrained configs.
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular ++welfare.v_ref_shaper=-17.5 ++welfare.v_ref_opponent=-20
#   # DROP the co-player constraint, to show both constraints are needed.
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular ++welfare.constrain_opponent=False
#   # Per-player override: slower dual on the co-player only.
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0 offline lagrangian_coala_pg_v_tabular ++welfare.ki_opponent=0.002

PLATFORM=${1:-tri}
SEED=${2:-0}
# Backward-compatible argument parsing:
#   <platform> <seed> <experiment>                  -> offline experiment
#   <platform> <seed> <wandb_mode> <experiment>     -> explicit mode + experiment
#   <platform> <seed> <wandb_mode> <experiment> ... -> plus Hydra overrides
case "${3:-offline}" in
    online|offline|shared|disabled|dryrun|run)
        WANDB_MODE_ARG=${3:-offline}
        EXPERIMENT_NAME=${4:-coala_pg_v_tabular}
        EXTRA_OVERRIDES=("${@:5}")
        ;;
    *)
        WANDB_MODE_ARG=offline
        EXPERIMENT_NAME=${3:-coala_pg_v_tabular}
        EXTRA_OVERRIDES=("${@:4}")
        ;;
esac

# ──────────────────────────────────────────────────────────────────
# Auto-submit: if not already running under SLURM, sbatch ourselves
# ──────────────────────────────────────────────────────────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    case "$PLATFORM" in
        fir)
            sbatch \
                --account=def-jtyao_gpu \
                --job-name=B9_coala_ipd_s${SEED} \
                --nodes=1 \
                --ntasks=1 \
                --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=6 \
                --mem=16G \
                --time=00:15:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=B9_coala_ipd_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=6 \
                --time=08:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=B9_coala_ipd_dbg_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=6 \
                --time=1:00:00 \
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

EXPERIMENT="ipd=${EXPERIMENT_NAME}"
if [ "$EXPERIMENT_NAME" = "coala_pg_v_tabular" ]; then
    RESULTS_DIR="/scratch/lichenqi/results/B9_coala_ipd_seed${SEED}"
else
    # Fold the Hydra overrides into the results path. Without this, two runs
    # of the SAME config that differ only by override land in the same
    # directory, separated by timestamp alone -- and the overrides are not
    # copied with the results, so they are unrecoverable afterwards. This bit
    # the constrained vs unconstrained-welfare comparison, where
    #   lagrangian_coala_pg_v_tabular
    # and
    #   lagrangian_coala_pg_v_tabular ++welfare.freeze_lam=True ++welfare.lam_init=0.0
    # produce architecturally IDENTICAL 3-head checkpoints that cannot be told
    # apart from the files. Only the SLURM .out log distinguishes them.
    RUN_TAG=""
    if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
        RUN_TAG=$(printf '%s-' "${EXTRA_OVERRIDES[@]}"             | sed -e 's/++//g' -e 's/welfare\.//g' -e 's/ppo[12]\.//g'                   -e 's/[^A-Za-z0-9]/-/g' -e 's/---*/-/g' -e 's/-$//')
        RUN_TAG="_${RUN_TAG:0:72}"
    fi
    RESULTS_DIR="/scratch/lichenqi/results/B9_${EXPERIMENT_NAME}${RUN_TAG}_seed${SEED}"
fi
HYDRA_DIR="$TMPDIR/hydra_output"
EXP_OUTPUT="$HYDRA_DIR/exp"
mkdir -p "$RESULTS_DIR"

start_time=$(date +%s)
echo "=== B9 COALA-PG | Experiment: $EXPERIMENT_NAME | Platform: $PLATFORM | Seed: $SEED | wandb: $WANDB_MODE_ARG | $(date '+%Y-%m-%d %H:%M:%S') ==="
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
    echo "=== Extra Hydra overrides: ${EXTRA_OVERRIDES[*]} ==="
fi

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        # NOTE: COALA-PG is a policy-gradient runner. It has no ES population
        # and no resume support, so no popsize / resume_dir overrides here.
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++wandb.mode=$WANDB_MODE_ARG \
            hydra.run.dir=$HYDRA_DIR \
            "${EXTRA_OVERRIDES[@]}"
        ;;
    tri-debug|fir-debug)
        # Smoke test only: tiny M / T / B. These settings are far too small for
        # the estimator to shape anything (the co-player barely learns within a
        # meta-trajectory) — they only verify the pipeline runs end to end.
        echo "=== Debug smoke test (tiny params; not a learning run) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=10 \
            ++num_outer_steps=4 \
            ++num_inner_steps=10 \
            ++num_envs=4 \
            ++num_opps=2 \
            ++ppo1.num_minibatches=4 \
            ++save_interval=5 \
            ++wandb.mode=$WANDB_MODE_ARG \
            hydra.run.dir=$HYDRA_DIR \
            "${EXTRA_OVERRIDES[@]}"
        ;;
esac

# ──────────────────────────────────────────────────────────────────
# Copy results to persistent storage
# ──────────────────────────────────────────────────────────────────
# Hydra writes checkpoints to  $EXP_OUTPUT/<wandb.group>/<wandb.name>/<timestamp>/
# (save_dir is "./exp/${wandb.group}/${wandb.name}" in pax/conf/config.yaml), and
# wandb.group differs per experiment: "baseline-COALA-PG-*" for the selfish
# baseline, "constrained-welfare-COALA-PG-*" for the constrained variant, etc.
# The old hardcoded "baseline-COALA-PG-*" glob silently matched nothing for
# every experiment but coala_pg_v_tabular. Copy whatever group dirs exist, and
# FAIL LOUDLY: both copies used to hide errors behind 2>/dev/null plus
# "|| true", so a no-match printed a success message anyway.
shopt -s nullglob

echo "Copying final results to $RESULTS_DIR ..."
group_dirs=("$EXP_OUTPUT"/*/)
if [ ${#group_dirs[@]} -eq 0 ]; then
    echo "WARNING: no run directories under $EXP_OUTPUT � nothing to copy." >&2
    echo "         (expected $EXP_OUTPUT/<wandb.group>/<wandb.name>/<timestamp>/)" >&2
else
    for d in "${group_dirs[@]}"; do
        echo "  <- $d"
        cp -rL "$d" "$RESULTS_DIR/" || echo "WARNING: copy failed for $d" >&2
    done
    echo "Results now in $RESULTS_DIR:"
    ls -1 "$RESULTS_DIR"
fi

if [ "$WANDB_MODE_ARG" = "offline" ]; then
    WANDB_SAVED=/scratch/lichenqi/wandb_saved
    mkdir -p "$WANDB_SAVED"
    # wandb appends its own "wandb/" under WANDB_DIR, hence the doubled path.
    offline_runs=("$WANDB_DIR"/wandb/offline-run-*)
    if [ ${#offline_runs[@]} -eq 0 ]; then
        echo "WARNING: no offline-run-* under $WANDB_DIR/wandb � nothing to sync." >&2
        echo "         Contents of $WANDB_DIR:" >&2
        ls -lR "$WANDB_DIR" 2>&1 | head -40 >&2
    else
        for r in "${offline_runs[@]}"; do
            echo "  <- $r"
            cp -rL "$r" "$WANDB_SAVED/" || echo "WARNING: copy failed for $r" >&2
        done
        echo ">>> Sync later from a LOGIN node: wandb sync $WANDB_SAVED/offline-run-* <<<"
    fi
fi

end_time=$(date +%s)
echo "=== Done: B9 COALA-PG $PLATFORM seed=$SEED | Elapsed: $((end_time - start_time))s ==="
