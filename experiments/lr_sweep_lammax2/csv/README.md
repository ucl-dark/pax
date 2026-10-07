Per-run series (every logged iteration) for the kept conditions:

- `V9_vref14both/` constrained, `lam_max=2 v_ref_shaper=-14 v_ref_opponent=-14` (recommended, now the config default)
- `V4_vref14/`     constrained, `lam_max=2 v_ref_shaper=-14`
- `C_lr3e-4/`      constrained, `lam_max=2` with the old `v_ref=-20` floors
- `U_lr3e-4/`      unconstrained (`freeze_lam=True lam_init=0`)

Columns: it; s/o/w = meta-mean shaper/co-player/welfare; s20/o20 = final-episode
returns; lam_s/lam_c = multipliers; pXY = P(shaper cooperates | state XY);
vXY = state visitation. Failed variants are summarised in ../REPORT.md only.
