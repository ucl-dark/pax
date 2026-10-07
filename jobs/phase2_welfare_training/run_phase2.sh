#!/bin/bash
# Launcher for Phase 2: Welfare Shaper Training
# Submits all variants (IR calibration, self-play ref, noIR ablation) across seeds:
#   IPD        -> 100 seeds (0..99)
#   CoinGame   ->  20 seeds (0..19)
#   InTheMatrix->  20 seeds (0..19)
#
# IMPORTANT: Before running Phase 2b (self-play ref) scripts, fill in
# the v_ref values from Phase 1 inside the W*_sp_*.sh scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Phase 2a: IR calibration ---

# IPD (100 seeds)
IPD_IR_SCRIPTS=(
    W1_ipd_welfare_shaper_att.sh
    W3_ipd_welfare_shaper_mfos.sh
    W5_ipd_welfare_shaper_rl.sh
)
for script in "${IPD_IR_SCRIPTS[@]}"; do
    for seed in $(seq 0 99); do
        sbatch --export=ALL "$SCRIPT_DIR/$script" $seed
    done
    echo "Submitted 100 jobs for $script"
done

# InTheMatrix (20 seeds)
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/W2_ipditm_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for W2_ipditm_welfare_shaper_att"

# CoinGame (20 seeds)
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/W4_cg_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for W4_cg_welfare_shaper_att"

# --- Phase 2b: Self-play references (fill v_ref in scripts first!) ---

# IPD (100 seeds)
IPD_SP_SCRIPTS=(
    W1_sp_ipd_welfare_shaper_att.sh
    W3_sp_ipd_welfare_shaper_mfos.sh
)
for script in "${IPD_SP_SCRIPTS[@]}"; do
    for seed in $(seq 0 99); do
        sbatch --export=ALL "$SCRIPT_DIR/$script" $seed
    done
    echo "Submitted 100 jobs for $script"
done

# InTheMatrix (20 seeds)
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/W2_sp_ipditm_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for W2_sp_ipditm_welfare_shaper_att"

# CoinGame (20 seeds)
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/W4_sp_cg_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for W4_sp_cg_welfare_shaper_att"

# --- Phase 2c: noIR ablation ---

# IPD (100 seeds)
for seed in $(seq 0 99); do
    sbatch --export=ALL "$SCRIPT_DIR/W1_noIR_ipd_welfare_shaper_att.sh" $seed
done
echo "Submitted 100 jobs for W1_noIR_ipd_welfare_shaper_att"

# InTheMatrix (20 seeds)
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/W2_noIR_ipditm_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for W2_noIR_ipditm_welfare_shaper_att"

echo "Phase 2 submission complete."
