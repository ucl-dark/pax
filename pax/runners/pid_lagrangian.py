"""PID Lagrangian multiplier controller.

Stooke, Achiam & Abbeel, "Responsive Safety in Reinforcement Learning by PID
Lagrangian Methods", ICML 2020. Reference implementation: OmniSafe's
``PIDLagrangian``.

Plain dual ascent (RCPO, Tessler et al. 2019) is a pure integrator on the
constraint violation: lam only moves once a violation has already accumulated,
so it lags into the violation and then overshoots coming out. Adding a
proportional term (reacts to the current violation) and a derivative term
(reacts to the violation getting *worse*) makes lam respond before the primal
has been driven deep into the infeasible region.

Sign convention here. The constraint is a FLOOR on a return:

    R_s >= tau

Written as a cost constraint ``J_c <= d`` with ``J_c = -R_s`` and ``d = -tau``,
the violation is

    delta = J_c - d = tau - R_s          (positive == violated)

so a satisfied constraint gives delta < 0 and drains the integrator. This
module is deliberately free of JAX: lam is a scalar updated once per
meta-trajectory on the host, and keeping it in plain Python makes it trivial to
test, log, and checkpoint.

Setting ``kp = kd = 0.0`` reduces the update to ``lam <- max(0, lam + ki*delta)``,
which is exactly RCPO dual ascent with step size ``ki``. That is intentional:
the same code path provides the dual-ascent baseline for ablations.
"""

from typing import Any, Dict, List, Optional

# The players a constraint can be written on, in the order their reward streams
# appear in the environment's reward tuple. The config suffix is what a sweep
# override uses (``++welfare.v_ref_shaper=-15``); the display name is what the
# logs and wandb tags use.
#
#   (config suffix, display name, reward index, reference-value key)
CONSTRAINT_PLAYERS = (
    ("shaper", "shaper", 0, "v_ref_shaper"),
    ("opponent", "co-player", 1, "v_ref_opponent"),
)

# Per-constraint settings. Each may be given once for both players
# (``welfare.ki``) or per player (``welfare.ki_shaper``, ``welfare.ki_opponent``);
# the suffixed form wins. `window` additionally accepts the shared spelling
# `constraint_window`, to match the key name used by the ES configs.
_CONSTRAINT_KEYS = (
    "kp",
    "ki",
    "kd",
    "lam_init",
    "lam_min",
    "lam_max",
    "ema_beta",
    "window",
)

_DEFAULTS = {
    "kp": 0.0,
    "ki": 0.01,
    "kd": 0.0,
    "lam_init": 0.5,
    "lam_min": 0.0,
    "lam_max": None,
    "ema_beta": 0.0,
    "window": 0,
}


def _lookup(welfare_args, key: str, suffix: str, default: Any) -> Any:
    """``<key>_<suffix>`` if present, else ``<key>``, else ``default``.

    Lets a sweep override one player without touching the other, while shared
    settings need writing only once.
    """
    for candidate in (f"{key}_{suffix}", key):
        if candidate in welfare_args:
            value = welfare_args[candidate]
            if value is not None:
                return value
    return default


def parse_constraint_specs(welfare_args) -> List[Dict[str, Any]]:
    """Read the flat ``welfare.*`` keys into one spec dict per constraint.

    Reference values use the SAME key names as the constrained-welfare ES
    runner -- ``v_ref_shaper`` and ``v_ref_opponent`` -- so a reference value
    can be swept with a single override and means the same thing across
    runners:

        ++welfare.v_ref_shaper=-15 ++welfare.v_ref_opponent=-20

    A constraint can be dropped entirely with ``constrain_<player>: False``
    (e.g. ``++welfare.constrain_opponent=False``), which is the ablation
    showing both constraints are load-bearing.

    Shared by the runner and by the agent factory, which needs the count to
    size the critic -- so the two cannot disagree about how many heads exist.
    """
    specs = []
    for suffix, name, reward_index, vref_key in CONSTRAINT_PLAYERS:
        enabled = _lookup(welfare_args, "constrain", suffix, True)
        if not bool(enabled):
            continue
        if vref_key not in welfare_args or welfare_args[vref_key] is None:
            raise ValueError(
                f"welfare.{vref_key} is required for the {name} constraint. "
                f"Set it, or disable the constraint with "
                f"welfare.constrain_{suffix}=False."
            )
        spec = {
            "name": name,
            "suffix": suffix,
            "reward_index": reward_index,
            "tau": float(welfare_args[vref_key]),
        }
        for key in _CONSTRAINT_KEYS:
            value = _lookup(welfare_args, key, suffix, None)
            if value is None and key == "window":
                # `constraint_window` is the shared spelling used by the
                # configs; fall back to it before the hard default.
                value = _lookup(
                    welfare_args, "constraint_window", suffix, None
                )
            if value is None:
                value = _DEFAULTS[key]
            if key == "lam_max":
                spec[key] = (
                    None if value in (None, "null", "") else float(value)
                )
            elif key == "window":
                spec[key] = int(value)
            else:
                spec[key] = float(value)
        specs.append(spec)

    if not specs:
        raise ValueError(
            "Every constraint is disabled. For the unconstrained welfare "
            "objective keep both constraints and run with "
            "welfare.freeze_lam=True welfare.lam_init=0.0 instead, so the "
            "network and critic match the constrained condition."
        )
    return specs


def num_constraints(welfare_args) -> int:
    """How many constraints the flat config declares (sizes the critic)."""
    return len(parse_constraint_specs(welfare_args))


class PIDLagrangian:
    """Scalar multiplier for one inequality constraint ``R >= tau``."""

    def __init__(
        self,
        tau: float,
        kp: float = 0.0,
        ki: float = 0.01,
        kd: float = 0.0,
        lam_init: float = 0.5,
        lam_max: Optional[float] = None,
        ema_beta: float = 0.0,
        lam_min: float = 0.0,
    ):
        """
        Args:
          tau: the floor the constrained return must clear.
          kp, ki, kd: proportional / integral / derivative gains. kp = kd = 0
            recovers RCPO dual ascent with learning rate ki.
          lam_init: initial multiplier. Deliberately > 0 by default -- early in
            meta-training the constraint is often satisfied by accident, and a
            multiplier initialised at 0 decays further, so by the time real
            violations appear the shaper has already learnt a self-sacrificing
            policy and lam has to climb from nothing to undo it.
          lam_max: optional ceiling. None leaves lam unbounded, which is the
            textbook method; a finite value trades exact feasibility for a
            bound on how far the objective can be tilted.
          lam_min: floor on the multiplier (default 0, the textbook method).
            A floor at the shaper's myopic INDIFFERENCE POINT (1.0 when the
            co-player's multiplier is 0, see the IPD config) removes the
            within-episode pull toward unconditional cooperation entirely:
            below the floor every term of the primal gradient is the
            learning-aware future-episode term, and the constraint only ever
            ADDS a push toward defection when the shaper is actually
            sacrificing. Measured on IPD: with floor 0 the multiplier drains
            to ~0.4 after a violation and the policy falls back into the
            sucker basin before the co-player has responded.
          ema_beta: if > 0, the controller sees an exponential moving average
            of the measured return instead of the raw per-iteration value.
            COALA-PG meta-trajectory returns are noisy and a derivative term on
            raw noise is mostly noise, so smoothing matters more here than in
            single-agent safe RL.
        """
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        self.tau = float(tau)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.lam_max = None if lam_max is None else float(lam_max)
        self.lam_min = float(lam_min)
        if self.lam_max is not None and self.lam_min > self.lam_max:
            raise ValueError(
                f"lam_min ({self.lam_min}) must not exceed lam_max ({self.lam_max})"
            )
        self.ema_beta = float(ema_beta)

        # The integrator holds the multiplier that pure dual ascent would
        # produce; lam is that plus the P and D corrections.
        self._integral = max(self.lam_min, float(lam_init))
        self.lam = max(self.lam_min, float(lam_init))
        self._prev_return: Optional[float] = None
        self._smoothed_return: Optional[float] = None

        # Diagnostics for the last update.
        self.delta = 0.0
        self.p_term = 0.0
        self.d_term = 0.0

    def update(self, measured_return: float) -> float:
        """Consume one measurement of the constrained return; return new lam."""
        r = float(measured_return)

        if self.ema_beta > 0.0:
            if self._smoothed_return is None:
                self._smoothed_return = r
            else:
                self._smoothed_return = (
                    self.ema_beta * self._smoothed_return
                    + (1.0 - self.ema_beta) * r
                )
            r_eff = self._smoothed_return
        else:
            r_eff = r

        # Violation: positive when the return is below the floor.
        delta = self.tau - r_eff

        # Integral: the dual-ascent term. Clamped at zero so a long stretch of
        # satisfied constraint cannot bank negative multiplier "credit" that
        # would delay the response to the next violation.
        #
        # ANTI-WINDUP: also bound it ABOVE by lam_max. Without this the
        # integrator keeps accumulating while lam is already saturated at the
        # cap, so lam becomes a function of training HISTORY rather than of the
        # current violation, and cannot come down when the constraint is met.
        # Measured on IPD (seed 1402): I reached 43.9 with lam pinned at 1.0 --
        # draining that at ki*|slack| would take ~1750 iterations, longer than
        # the run. A second run (seed 1400) sat at lam_s = 1.0 purely on
        # leftover windup while its constraint was satisfied by 4.6, which also
        # drove BOTH multipliers to the cap at once: weights (2, 2) is just
        # 2 x welfare, and under Adam that is plain welfare, so the constraints
        # supplied no differential signal at all. Bounding the integrator is
        # standard PID practice and is what the OmniSafe PIDLagrangian
        # reference implementation does.
        hi = float("inf") if self.lam_max is None else self.lam_max
        self._integral = min(
            hi, max(self.lam_min, self._integral + self.ki * delta)
        )

        # Derivative on the COST, i.e. positive when the return is falling.
        # One-sided: we want lam to anticipate a developing violation, but not
        # to be dragged down merely because things are improving -- the
        # integral already handles the release.
        if self._prev_return is None:
            d_raw = 0.0
        else:
            d_raw = self._prev_return - r_eff
        self._prev_return = r_eff

        self.delta = delta
        self.p_term = self.kp * delta
        self.d_term = self.kd * max(0.0, d_raw)

        lam = self._integral + self.p_term + self.d_term
        lam = max(self.lam_min, lam)
        if self.lam_max is not None:
            lam = min(lam, self.lam_max)
        self.lam = lam
        return self.lam

    def state_dict(self) -> Dict[str, float]:
        """Everything needed to resume the controller mid-run."""
        return {
            "lam": self.lam,
            "integral": self._integral,
            "prev_return": (
                float("nan")
                if self._prev_return is None
                else self._prev_return
            ),
            "smoothed_return": (
                float("nan")
                if self._smoothed_return is None
                else self._smoothed_return
            ),
        }

    def metrics(self) -> Dict[str, float]:
        return {
            "lagrangian/lam": self.lam,
            "lagrangian/violation": self.delta,
            "lagrangian/integral": self._integral,
            "lagrangian/p_term": self.p_term,
            "lagrangian/d_term": self.d_term,
            "lagrangian/tau": self.tau,
        }
