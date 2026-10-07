#!/bin/bash
# Launcher for Phase 0: Original Baselines
# Submits each experiment across the appropriate number of seeds:
#   IPD        -> 100 seeds (0..99)
#   CoinGame   ->  20 seeds (0..19)
#   InTheMatrix->  20 seeds (0..19)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- IPD baselines (100 seeds) ---
IPD_SCRIPTS=(
    B1_ipd_shaper_att_v_tabular.sh
    B4_ipd_mfos_att_v_tabular.sh
)
for script in "${IPD_SCRIPTS[@]}"; do
    for seed in $(seq 0 99); do
        sbatch --export=ALL "$SCRIPT_DIR/$script" $seed
    done
    echo "Submitted 100 jobs for $script"
done

# --- InTheMatrix / STORM baselines (20 seeds) ---
IPDITM_SCRIPTS=(
    B2_ipditm_shaper_att.sh
    B5_ipditm_mfos_es.sh
    B7_ipditm_mfos_rl.sh
)
for script in "${IPDITM_SCRIPTS[@]}"; do
    for seed in $(seq 0 19); do
        sbatch --export=ALL "$SCRIPT_DIR/$script" $seed
    done
    echo "Submitted 20 jobs for $script"
done

# --- CoinGame baselines (20 seeds) ---
CG_SCRIPTS=(
    B3_cg_shaper.sh
    B6_cg_mfos_es.sh
    B8_cg_mfos_rl.sh
)
for script in "${CG_SCRIPTS[@]}"; do
    for seed in $(seq 0 19); do
        sbatch --export=ALL "$SCRIPT_DIR/$script" $seed
    done
    echo "Submitted 20 jobs for $script"
done

echo "Phase 0 submission complete."
