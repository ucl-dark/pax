#!/bin/bash
#SBATCH --account=def-jtyao_gpu
#SBATCH --job-name=W3_ipd_welfare_mfos
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%N-%j.out

module load python/3.11.5
module load cuda/12.6
source /home/lichenqi/pax_env_py3.11.5/bin/activate

export WANDB_API_KEY="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"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR/wandb" "$TMPDIR/wandb-cache" "$TMPDIR/wandb_config"

export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb_config"
export WANDB_SERVICE_TRANSPORT=tcp
export WANDB__SERVICE_WAIT=180
export WANDB_INIT_TIMEOUT=180
export WANDB_START_METHOD=thread

SEED=${1:-0}
EXPERIMENT="ipd=welfare_shaper_v_tabular"
start_time=$(date +%s)
echo "Start W3 $EXPERIMENT (IR calibration): $(date '+%Y-%m-%d %H:%M:%S')"

cd /home/lichenqi/pax
python -m pax.experiment +experiment/$EXPERIMENT seed=$SEED

mkdir -p "$HOME/wandb_saved"
cp -r "$WANDB_DIR"/wandb/offline-run-* "$HOME/wandb_saved/" 2>/dev/null || true
end_time=$(date +%s)
echo "End W3 $EXPERIMENT: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Elapsed: $((end_time - start_time)) seconds"
