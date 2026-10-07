# Constrained COALA-PG (IPD): what makes cooperation reliable

75 runs, 5001 iterations each, CPU, seeds as listed in `stability.txt`.
Base config: `lagrangian_coala_pg_v_tabular.yaml` with `welfare.lam_max=2.0`.
Metric: fraction of the last 2000 iterations in which the FINAL inner episode has
both players >= -14 ("coop"), or shaper <= -22 and co-player >= -8 ("sucker").
Per-seed numbers: `stability.txt`; per-iteration series: `csv/`.

| variant | overrides on top of base | n | reached coop | >=50% of last 2k coop | mostly sucker |
|---|---|---|---|---|---|
| unconstrained | `unconstrained_lagrangian_coala_pg_v_tabular` | 4 | 1/4 | 0/4 | 3/4 (meta-mean ~(-27,-2)) |
| baseline (user's) | none | 2 (+ user's 6: 3 good) | 1/2 | 1/2 | 1/2 |
| V1 slow dual | `kp=0 kd=0 ki=4e-4 lam_init_shaper=1 lam_init_opponent=0` | 6 | 4/6 | 3/6 | 1/6 |
| V3 mid dual | same, `ki=1.5e-3` | 6 | 5/6 | 2/6 | 1/6 |
| V5 lam floor | V3 + `lam_min_shaper=1.0` (new code) | 6 | 4/6 | 2/6 | 1/6 |
| V6 frozen (2,1) | `freeze_lam=True lam_init_shaper=1 lam_init_opponent=0` | 4 | 2/4 | 1/4 | 1/4 |
| **V4 shaper floor -14** | `v_ref_shaper=-14` | 12 | 9/12 | **8/12** | **0/12** |
| **V9 both floors -14** | `v_ref_shaper=-14 v_ref_opponent=-14` | 12 | **10/12** | 7/12 | 1/12 |
| V10 | V4 + `lam_max=1.5` | 6 | 3/6 | 1/6 | 1/6 |
| entropy 0.05 (V7/V8/U) | `ppo1.entropy_coeff_start=0.05` | 13 | no gain; blurs the unconstrained baseline | | |

Every successful seed, whatever the variant, lands in the same state: final
episode ~(-10.5, -10.5), shaper policy P(C|CC)~1.0, P(C|CD)~0.1-0.3,
P(C|DD)~0.02 (generous tit-for-tat), co-player cooperating.

## Mechanism (from the per-iteration traces)

1. With v_ref = -20 the duals form a limit cycle (period ~800 it): lam_s hits the
   cap, the whole policy swings to defect, the constraint is satisfied, lam_s
   collapses to 0 in ~170 it, pure welfare pulls the policy back to the sucker
   basin. A lower policy learning rate makes this WORSE (dual relatively faster).
2. Both welfare and lam_s act UNIFORMLY on all four states; tit-for-tat needs
   state-dependence, which only the learning-aware term supplies. It wins only
   when the uniform forces roughly cancel (lam_s ~ 1 + 2 lam_o).
3. Runs that reach cooperation then DRIFT: welfare erodes P(C|CD) over ~1000 it,
   the co-player starts exploiting, the shaper slides from -12 to -16..-18.
   With v_ref = -20 the constraint is slack there and nothing resists.
   v_ref_shaper = -14 binds exactly where the drift happens; that is why V4/V9
   have ~0 "mostly sucker" seeds.
4. Remaining failures (2-3 of 12) are a saturated softmax early in training
   (P(C)=0.00 or 1.00 at every state by it ~500) after the (3,1) push overshoots;
   a lower cap (1.5) undershoots instead. Entropy 0.05 does not prevent it and
   makes the unconstrained control partially cooperative.

## Recommendation

    bash jobs/phase0_baselines/B9_ipd_coala_pg_att_v_tabular.sh fir $s offline \
        lagrangian_coala_pg_v_tabular \
        ++welfare.lam_max=2.0 ++welfare.v_ref_shaper=-14.0 ++welfare.v_ref_opponent=-14.0

(or `v_ref_opponent=-20` for V4: fewer total successes, fewer sucker seeds).
Report the cooperation fraction over the last 2000 iterations and the policy
signature, not only the final iteration; 12 seeds, not 6.
Framing: -14 per episode (= -1.4/step) is between mutual cooperation (-1) and
mutual defection (-2); it is a floor "both players do better than halfway
between the two symmetric outcomes", stricter than the security level -20.
