"""COALA-PG: co-agent learning-aware policy gradients.

Reference implementation of the estimator from

    Meulemans, Kobayashi, von Oswald, Scherrer, Elmoznino, Richards, Lajoie,
    Aguera y Arcas & Sacramento (2025). "Multi-agent cooperation through
    learning-aware policy gradients." ICLR 2025. arXiv:2410.18636

The authors released no code, so this is a reimplementation from the paper.

-------------------------------------------------------------------------------
What COALA-PG is
-------------------------------------------------------------------------------
The co-player learning problem is recast as a *batched co-player shaping POMDP*:
B inner episodes are played in lockstep, and the co-player updates its
parameters at every inner-episode boundary using its minibatch of all B
trajectories. Because the ego-agent's action in trajectory b changes that
co-player update, it influences *all B* trajectories in every later episode.
The unbiased policy gradient (Theorem 3.1) is therefore

  grad J = E[ sum_b sum_l  grad log pi(a_l^b | h_l^b)
              * ( 1/B * sum_{l'=l}^{m_l T}  r_{l'}^{b}
                + 1/B * sum_{b'} sum_{l'=m_l T + 1}^{MT} r_{l'}^{b'} ) ]

i.e. own rewards to the end of the *current* inner episode, plus all *future*
episodes' rewards summed over every batch member. No derivative is ever taken
through the co-player update, which is what makes it higher-derivative-free.

Crucially the double sum is never materialised: it is produced by a single
reverse scan that, at each inner-episode boundary, resets the per-trajectory
accumulator to the cross-batch mean accumulator (Algorithm 1 in the paper).

-------------------------------------------------------------------------------
Deviations from the paper, deliberately documented
-------------------------------------------------------------------------------
* Architecture. The paper conditions on long histories with a Hawk
  (linear-recurrent) model. Pax ships GRU and attention torsos, so this uses a
  GRU whose hidden state is carried across inner-episode boundaries for the
  whole meta-trajectory. Same role (long-context conditioning on the co-player's
  learning trace), not an exact reproduction of the architecture.
* `init_acc`. Algorithm 1 as printed initialises the accumulator to the final
  value estimate. That is correct when the routine is fed rewards (value
  targets), but when it is fed TD errors to produce a GAE-style advantage the
  recursion must start from zero. This module therefore takes `init_acc`
  explicitly; see `coala_advantages` and `coala_value_targets`.
-------------------------------------------------------------------------------
Calibration: gamma, lambda and the horizon M*T are NOT free parameters
-------------------------------------------------------------------------------
Algorithm 1 applies a single `discount` (= gamma * lambda) at every step, so the
future-episode term reaches step l of the current episode attenuated by
discount**T. Two measured failure modes on IPD (M=16, T=100, B=16):

  gamma=0.96, lambda=0.95  -> discount 0.912, 0.912**100 ~ 1e-4. The
      future-episode term, i.e. the ENTIRE learning-aware signal, is attenuated
      10,012x. The shaper becomes myopically self-interested and converges to
      always-defect (P(cooperate | CC) == 0.0 exactly, DD state prob 0.86).

  gamma=1.0, lambda=1.0    -> discount 1.0, signal intact but ALL variance
      reduction gone. Theorem 3.1 puts 1/B on the own-episode term but not on
      the batch-summed future term, so the advantage is dominated by a ~-3000
      future-episode component against a ~-12.5 action-relevant one. The critic
      cannot fit a -3200-magnitude target fast enough, the advantage is noise,
      and the policy degenerates to uniform random (P(cooperate) ~ 0.5 in every
      state).

Note that decoupling the two discounts does NOT help: the future credit is
injected at an episode's final step and then still decays through the remaining
T-1 within-episode steps (verified: weight 1.04e-4, i.e. unchanged).

The operative constraint is therefore the HORIZON. discount must be 1.0 to keep
the estimator faithful, which forces M*T small enough that undiscounted returns
stay in a range the critic can fit. The paper's IPD is a finite-horizon
5-state game, consistent with a much shorter inner episode than the T=100 used
by the ES baselines in this repo.
"""

from typing import Any, Dict, NamedTuple, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import omegaconf
import optax

from pax import utils
from pax.runners.pid_lagrangian import num_constraints as count_constraints
from pax.agents.agent import AgentInterface
from pax.agents.ppo.networks import (
    make_GRU_coingame_network,
)
from pax.utils import MemoryState, TrainingState


class RMSNorm(hk.Module):
    """RMSNorm used by the paper's Hawk backbone."""

    def __init__(self, eps: float = 1e-6, name: str | None = None):
        super().__init__(name=name)
        self._eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = hk.get_parameter(
            "scale", shape=(x.shape[-1],), init=jnp.ones
        )
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self._eps)
        return x / rms * scale


class CoalaIpdTorso(hk.Module):
    """IPD recurrent torso matching the paper dimensions.

    The paper uses a Hawk block (LRU width 32, MLP expanded width 32, 2 heads).
    Pax does not ship Hawk/LRU layers, so this is a self-contained, diagonal
    gated recurrent fallback with the same embedding/recurrent width and a
    residual MLP block. Policy/value heads are still zero-initialized as in the
    paper.
    """

    def __init__(self, width: int = 32, num_heads: int = 2):
        super().__init__(name="coala_ipd_torso")
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self._width = width
        self._num_heads = num_heads

    def __call__(
        self, observations: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Linear(
            self._width,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            b_init=hk.initializers.Constant(0),
            name="obs_embedding",
        )(observations)

        # Two-head diagonal recurrent update. This keeps the paper's "2 heads"
        # structure without adding an unavailable Hawk dependency.
        head_dim = self._width // self._num_heads
        x_h = jnp.reshape(x, x.shape[:-1] + (self._num_heads, head_dim))
        s_h = jnp.reshape(state, state.shape[:-1] + (self._num_heads, head_dim))

        gate = jax.nn.sigmoid(
            hk.Linear(self._width, name="gate_x")(x)
            + hk.Linear(self._width, with_bias=False, name="gate_h")(state)
        )
        cand = jnp.tanh(
            hk.Linear(self._width, name="cand_x")(x)
            + hk.Linear(self._width, with_bias=False, name="cand_h")(state)
        )
        gate_h = jnp.reshape(gate, gate.shape[:-1] + (self._num_heads, head_dim))
        cand_h = jnp.reshape(cand, cand.shape[:-1] + (self._num_heads, head_dim))
        new_state_h = gate_h * s_h + (1.0 - gate_h) * cand_h
        new_state = jnp.reshape(new_state_h, state.shape)

        y = x + new_state
        mlp = hk.nets.MLP(
            [self._width, self._width],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            b_init=hk.initializers.Constant(0),
            activation=jax.nn.gelu,
            activate_final=False,
            name="mlp",
        )(y)
        y = RMSNorm(name="rms_norm")(y + mlp)
        return y, new_state


class ZeroInitCategoricalValueHead(hk.Module):
    """Policy/value readouts with zero initialization, per Appendix B.3."""

    def __init__(self, num_actions: int):
        super().__init__(name="zero_init_categorical_value_head")
        self._num_actions = num_actions

    def __call__(self, inputs: jnp.ndarray):
        logits = hk.Linear(
            self._num_actions,
            w_init=hk.initializers.Constant(0),
            b_init=hk.initializers.Constant(0),
            name="policy_logits",
        )(inputs)
        value = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0),
            b_init=hk.initializers.Constant(0),
            name="value",
        )(inputs)
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


def make_coala_ipd_network(num_actions: int, hidden_size: int = 32):
    """Build the shared IPD recurrent policy/value network from the paper."""
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        observations: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[distrax.Categorical, jnp.ndarray], jnp.ndarray]:
        embedding, state = CoalaIpdTorso(width=hidden_size)(observations, state)
        dist, value = ZeroInitCategoricalValueHead(num_actions)(embedding)
        return (dist, value), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


class ZeroInitMultiValueHead(hk.Module):
    """Policy readout plus ``num_value_heads`` value readouts, stacked last.

    Head 0 estimates the WELFARE return; heads 1..K estimate the K constraint
    returns. The Lagrangian baseline is ``V_W + sum_k lam_k * V_k``.

    Why separate heads rather than one critic trained on the composite return
    ``r_W + sum_k lam_k * r_k``: the multipliers move after every
    meta-trajectory, so a composite critic is stale by exactly the amount they
    just changed, and it is most stale precisely when a constraint starts
    binding (when its multiplier is moving fastest). Fitting each return
    separately and combining them with the current multipliers keeps the
    baseline unbiased for any lam.
    """

    def __init__(self, num_actions: int, num_value_heads: int = 2):
        super().__init__(name="zero_init_multi_value_head")
        self._num_actions = num_actions
        self._num_value_heads = num_value_heads

    def __call__(self, inputs: jnp.ndarray):
        logits = hk.Linear(
            self._num_actions,
            w_init=hk.initializers.Constant(0),
            b_init=hk.initializers.Constant(0),
            name="policy_logits",
        )(inputs)
        # Named so a checkpoint is readable: head 0 is welfare, the rest are
        # the constraints in the order the runner stacks them.
        head_names = ["value_welfare"] + [
            f"value_constraint_{k}" for k in range(self._num_value_heads - 1)
        ]
        heads = [
            hk.Linear(
                1,
                w_init=hk.initializers.Constant(0),
                b_init=hk.initializers.Constant(0),
                name=name,
            )(inputs)
            for name in head_names
        ]
        values = jnp.concatenate(heads, axis=-1)
        return distrax.Categorical(logits=logits), values


def make_coala_ipd_multi_value_network(
    num_actions: int, hidden_size: int = 32, num_value_heads: int = 2
):
    """`make_coala_ipd_network` with a multi-headed critic.

    Head 0 is welfare; the remaining ``num_value_heads - 1`` heads are the
    constraint returns. The torso is identical to the stock COALA-PG network,
    so the only difference between the Lagrangian agent and the baseline is the
    extra value readouts and how the advantage is combined -- which is what
    lets the three-way comparison (selfish / welfare / constrained) be
    attributed to the objective rather than to the architecture.
    """
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        observations: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[distrax.Categorical, jnp.ndarray], jnp.ndarray]:
        embedding, state = CoalaIpdTorso(width=hidden_size)(observations, state)
        dist, values = ZeroInitMultiValueHead(
            num_actions, num_value_heads
        )(embedding)
        return (dist, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    advantages: jnp.ndarray
    target_values: jnp.ndarray
    behavior_values: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    hiddens: jnp.ndarray


class Logger:
    metrics: dict


# -----------------------------------------------------------------------------
# Algorithm 1: batched lambda returns
# -----------------------------------------------------------------------------
def batch_lambda_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    init_acc: jnp.ndarray,
    discount: float,
    lam: float,
    inner_episode_length: int,
    average_future_episodes: bool,
    normalize_current_episode: bool,
) -> jnp.ndarray:
    """Algorithm 1 of the COALA-PG paper, as a single reverse ``lax.scan``.

    Args:
      rewards: ``[B, L]`` per-step signal. Raw rewards when computing value
        targets; TD errors when computing advantages.
      values: ``[B, L]`` value estimates aligned with ``rewards``.
      init_acc: ``[B]`` initial accumulator (see module docstring).
      discount: per-step discount. ``gamma`` for value targets,
        ``gamma * lambda`` for the GAE-style advantage.
      lam: mixing coefficient. ``1.0`` for the GAE-style advantage.
      inner_episode_length: ``T``, length of one inner episode.
      average_future_episodes: if True, reset the per-trajectory accumulator to
        the cross-batch mean at every inner-episode boundary. This single line
        is what injects the ``sum_{b'}`` future-episode coupling.
      normalize_current_episode: if True, divide the current-episode signal by
        ``B`` (the ``1/B`` factor on the own-trajectory term).

    Returns:
      ``[B, L]`` returns (or advantages).
    """
    batch_size, seq_len = rewards.shape
    normalization = float(batch_size) if normalize_current_episode else 1.0

    t_idx = jnp.arange(seq_len)
    episode_end = (t_idx % inner_episode_length) == (inner_episode_length - 1)

    def step(carry, x):
        acc, global_acc = carry
        r_t, v_t, is_end = x

        if average_future_episodes:
            # Crossing an inner-episode boundary: everything after this point is
            # a *future* episode, whose credit is shared across the whole batch.
            acc = jnp.where(is_end, global_acc, acc)

        new_acc = r_t / normalization + discount * (
            (1.0 - lam) * v_t + lam * acc
        )
        # The global accumulator is deliberately batch-unaware (no 1/B).
        new_global = jnp.mean(
            r_t + discount * ((1.0 - lam) * v_t + lam * global_acc)
        )
        return (new_acc, new_global), new_acc

    # Scan backwards over time: transpose to time-major, then flip.
    xs = (
        jnp.flip(rewards.T, axis=0),
        jnp.flip(values.T, axis=0),
        jnp.flip(episode_end, axis=0),
    )
    (_, _), out = jax.lax.scan(
        step, (init_acc, jnp.mean(init_acc)), xs, length=seq_len
    )
    # out is [L, B] in reverse time order.
    return jnp.flip(out, axis=0).T


def coala_advantages(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
    inner_episode_length: int,
) -> jnp.ndarray:
    """COALA-PG advantage: Algorithm 1 driven by TD errors.

    Per the paper, the GAE variant calls Algorithm 1 with the TD errors in place
    of rewards, ``discount = gamma * lambda``, ``lam = 1.0`` and both flags set.

    Args:
      rewards, values, dones: ``[B, L]``.
      bootstrap_value: ``[B]`` value of the state after the last step.
    Returns:
      ``[B, L]`` advantages.
    """
    # V_{t+1} with the bootstrap appended. Inner-episode `dones` are IGNORED,
    # as in the paper (App. B.2.1: "Crucially, the done signals from the inner
    # episodes are ignored"): the critic is the long-horizon value of Eq. 10,
    # which sums ALL remaining inner episodes, so the TD error at an
    # inner-episode boundary must bootstrap from the value at the start of
    # the next episode. Masking it would subtract the whole remaining
    # meta-episode value once per boundary instead of letting it telescope.
    # Verified against Eq. 13-14 on a toy problem (match to 1e-7 only with
    # dones ignored). NOTE: `iterated_matrix_game` only sets done at the end
    # of the meta-episode, so there this is a no-op; coin_game and
    # in_the_matrix emit inner-episode dones, where it matters. Only the
    # meta-trajectory's final step has no successor; the runner passes that
    # bootstrap as 0.
    del dones
    next_values = jnp.concatenate(
        [values[:, 1:], bootstrap_value[:, None]], axis=1
    )
    deltas = rewards + gamma * next_values - values

    return batch_lambda_returns(
        rewards=deltas,
        values=values,
        # GAE accumulates TD errors from zero, not from a bootstrap.
        init_acc=jnp.zeros_like(bootstrap_value),
        discount=gamma * gae_lambda,
        lam=1.0,
        inner_episode_length=inner_episode_length,
        average_future_episodes=True,
        normalize_current_episode=True,
    )


def coala_value_targets(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
    inner_episode_length: int,
) -> jnp.ndarray:
    """Regression target for the critic: the *batch-unaware* return.

    The paper trains one critic on the plain own-trajectory return-to-go over
    the whole meta-trajectory (no ``1/B``, no cross-batch averaging), which is
    what lets the same critic serve naive and learning-aware co-players.
    """
    return batch_lambda_returns(
        rewards=rewards,
        values=values,
        init_acc=bootstrap_value,
        discount=gamma,
        lam=gae_lambda,
        inner_episode_length=inner_episode_length,
        average_future_episodes=False,
        normalize_current_episode=False,
    )


class CoalaPG(AgentInterface):
    """Long-context PPO learner trained with the COALA-PG estimator.

    The PPO surrogate is unchanged; what differs from `pax.agents.ppo.ppo_gru`
    is that advantages come from `coala_advantages` (which couples credit across
    the co-player's lockstep batch) rather than from per-trajectory GAE.
    """

    def __init__(
        self,
        network: NamedTuple,
        initial_hidden_state: jnp.ndarray,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        gru_dim: int,
        obs_spec: Tuple,
        num_envs: int = 4,
        num_opps: int = 1,
        num_minibatches: int = 8,
        num_epochs: int = 2,
        num_inner_steps: int = 100,
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3_000_000,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.96,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        reward_rescaling: float = 1.0,
        player_id: int = 0,
    ):
        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            key, subkey = jax.random.split(state.random_key)
            (dist, values), hidden_state = network.apply(
                state.params, observation, mem.hidden
            )
            actions = dist.sample(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = dist.log_prob(actions)
            mem = mem._replace(hidden=hidden_state, extras=mem.extras)
            state = state._replace(random_key=key)
            return actions, state, mem

        def loss(
            params: hk.Params,
            timesteps: int,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            behavior_log_probs: jnp.ndarray,
            target_values: jnp.ndarray,
            advantages: jnp.ndarray,
            behavior_values: jnp.ndarray,
            hiddens: jnp.ndarray,
        ):
            """PPO clipped surrogate on externally supplied COALA advantages."""
            (distribution, values), _ = network.apply(
                params, observations, hiddens
            )
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            rhos = jnp.exp(log_prob - behavior_log_probs)
            clipped_ratios_t = jnp.clip(
                rhos, 1.0 - ppo_clipping_epsilon, 1.0 + ppo_clipping_epsilon
            )
            clipped_objective = jnp.fmin(
                rhos * advantages, clipped_ratios_t * advantages
            )
            # COALA-PG sums, rather than averages, policy-gradient terms over
            # the co-player minibatch B. Because minibatches are flattened for
            # PPO, multiplying the mean by B restores that estimator scaling.
            policy_loss = -jnp.mean(clipped_objective) * num_envs

            unclipped_value_loss = (target_values - values) ** 2
            if clip_value:
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -ppo_clipping_epsilon,
                    ppo_clipping_epsilon,
                )
                clipped_value_loss = (target_values - clipped_values) ** 2
                value_loss = jnp.mean(
                    jnp.fmax(unclipped_value_loss, clipped_value_loss)
                )
            else:
                value_loss = jnp.mean(unclipped_value_loss)

            if anneal_entropy:
                fraction = jnp.fmax(1 - timesteps / entropy_coeff_horizon, 0)
                entropy_cost = (
                    fraction * entropy_coeff_start
                    + (1 - fraction) * entropy_coeff_end
                )
            else:
                entropy_cost = entropy_coeff_start
            entropy_loss = -jnp.mean(entropy)

            total_loss = (
                policy_loss
                + entropy_cost * entropy_loss
                + value_loss * value_coeff
            )
            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
                "entropy_cost": entropy_cost,
            }

        @jax.jit
        def sgd_step(state: TrainingState, sample: NamedTuple):
            """One COALA-PG update.

            `sample` fields are time-major with the co-player batch kept
            separate: ``[L, num_opps, num_envs, ...]`` where ``L = M * T``.
            Keeping ``num_opps`` distinct from ``num_envs`` matters: the
            cross-batch coupling must average over ``num_envs`` (the minibatch a
            single co-player learns from) and never across independent
            co-players.
            """
            observations = sample.observations
            actions = sample.actions
            rewards = sample.rewards
            behavior_log_probs = sample.behavior_log_probs
            behavior_values = sample.behavior_values
            dones = sample.dones
            hiddens = sample.hiddens

            seq_len = rewards.shape[0]
            n_opps = rewards.shape[1]
            n_envs = rewards.shape[2]
            learning_rewards = rewards * reward_rescaling

            # -> [num_opps, num_envs, L] for Algorithm 1.
            to_alg = lambda x: jnp.transpose(x, (1, 2, 0))
            r_a = to_alg(learning_rewards)
            v_a = to_alg(behavior_values)
            d_a = to_alg(dones)

            # Bootstrap with the final value estimate, zeroed where terminal.
            bootstrap = v_a[:, :, -1] * (1.0 - d_a[:, :, -1])

            # vmap Algorithm 1 over independent co-players.
            adv_a = jax.vmap(
                coala_advantages, in_axes=(0, 0, 0, 0, None, None, None)
            )(
                r_a,
                v_a,
                d_a,
                bootstrap,
                gamma,
                gae_lambda,
                num_inner_steps,
            )
            tgt_a = jax.vmap(
                coala_value_targets, in_axes=(0, 0, 0, None, None, None)
            )(r_a, v_a, bootstrap, gamma, gae_lambda, num_inner_steps)
            tgt_a = jax.lax.stop_gradient(tgt_a)

            # Back to time-major, then flatten (L, opps, envs) -> batch.
            from_alg = lambda x: jnp.transpose(x, (2, 0, 1))
            advantages = from_alg(adv_a)
            target_values = from_alg(tgt_a)

            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                target_values=target_values,
                behavior_values=behavior_values,
                behavior_log_probs=behavior_log_probs,
                hiddens=hiddens,
            )

            batch_size = seq_len * n_opps * n_envs
            assert batch_size % num_minibatches == 0, (
                "num_minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[3:]), trajectories
            )

            grad_fn = jax.jit(jax.grad(loss, has_aux=True))

            def model_update_minibatch(carry, minibatch: Batch):
                params, opt_state, timesteps = carry
                if advantage_normalization:
                    advantages = (
                        minibatch.advantages
                        - jnp.mean(minibatch.advantages, axis=0)
                    ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
                else:
                    advantages = minibatch.advantages
                gradients, metrics = grad_fn(
                    params,
                    timesteps,
                    minibatch.observations,
                    minibatch.actions,
                    minibatch.behavior_log_probs,
                    minibatch.target_values,
                    advantages,
                    minibatch.behavior_values,
                    minibatch.hiddens,
                )
                updates, opt_state = optimizer.update(gradients, opt_state)
                params = optax.apply_updates(params, updates)
                metrics["norm_grad"] = optax.global_norm(gradients)
                metrics["norm_updates"] = optax.global_norm(updates)
                return (params, opt_state, timesteps), metrics

            def model_update_epoch(carry, unused_t):
                key, params, opt_state, timesteps, batch = carry
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled,
                )
                (params, opt_state, timesteps), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state, timesteps),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, timesteps, batch), metrics

            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (
                    state.random_key,
                    state.params,
                    state.opt_state,
                    state.timesteps,
                    batch,
                ),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(rewards)
            metrics["rewards_std"] = jnp.std(rewards)
            metrics["coala/advantage_mean"] = jnp.mean(advantages)
            metrics["coala/advantage_std"] = jnp.std(advantages)
            metrics["coala/target_value_mean"] = jnp.mean(target_values)
            metrics["coala/reward_rescaling"] = jnp.asarray(reward_rescaling)
            metrics["coala/policy_batch_scale"] = jnp.asarray(n_envs)

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps + batch_size,
            )
            # Memory is deliberately NOT rebuilt here. The runner owns the
            # memory's leading dimensions -- [num_opps, num_envs, ...] -- which
            # this closure cannot know. Rebuilding it as [num_envs, gru_dim]
            # silently drops the num_opps axis, and the next batch_reset then
            # vmaps over the wrong axis. `update` hands the caller's memory
            # straight back instead; the runner resets it per meta-trajectory.
            return new_state, metrics

        def make_initial_state(
            key: Any, initial_hidden_state: jnp.ndarray
        ) -> Tuple[TrainingState, MemoryState]:
            key, subkey = jax.random.split(key)
            if isinstance(obs_spec, dict):
                dummy_obs = {k: jnp.zeros(shape=v) for k, v in obs_spec.items()}
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(
                subkey, dummy_obs, initial_hidden_state
            )
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, initial_hidden_state.shape[-1])),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        self._state, self._mem = make_initial_state(
            random_key, initial_hidden_state
        )
        self.make_initial_state = make_initial_state
        self._sgd_step = sgd_step

        self._logger = Logger()
        self._total_steps = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
        }

        self.network = network
        self._policy = policy
        self.forward = network.apply
        self.player_id = player_id

        self._num_envs = num_envs
        self._num_opps = num_opps
        self._num_minibatches = num_minibatches
        self._num_epochs = num_epochs
        self._num_inner_steps = num_inner_steps
        self._gru_dim = gru_dim

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
            hidden=jnp.zeros((num_envs, self._gru_dim)),
        )
        return memory

    def update(
        self,
        traj_batch: NamedTuple,
        obs: jnp.ndarray,
        state: TrainingState,
        mem: MemoryState,
    ):
        """Update at the end of a full meta-trajectory.

        `traj_batch` must be time-major with the co-player batch intact:
        ``[M * T, num_opps, num_envs, ...]``. Unlike the stock PPO runners, the
        caller must NOT collapse ``num_opps`` into the batch dimension.

        `mem` is returned unchanged -- see the note in `sgd_step`.
        """
        state, metrics = self._sgd_step(state, traj_batch)
        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        for k in (
            "loss_total",
            "loss_policy",
            "loss_value",
            "loss_entropy",
            "entropy_cost",
        ):
            self._logger.metrics[k] = metrics[k]
        return state, mem, metrics



class LagrangianBatch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...].

    Differs from `Batch` in that `target_values` and `behavior_values` carry
    every critic head on a trailing axis of size ``1 + num_constraints``
    (welfare first), while `advantages` is already the lam-combined scalar.
    """

    observations: jnp.ndarray
    actions: jnp.ndarray
    advantages: jnp.ndarray
    target_values: jnp.ndarray
    behavior_values: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    hiddens: jnp.ndarray


class CoalaPGLagrangian(AgentInterface):
    """COALA-PG on the Lagrangian of a constrained welfare problem.

    Supports K individual-rationality constraints, one per player:

        max_theta  W(theta)
        s.t.       R_s(theta) >= tau_s        (the shaper)
                   R_o(theta) >= tau_o        (the co-player)

    with Lagrangian

        L = W + lam_s * (R_s - tau_s) + lam_o * (R_o - tau_o).

    Every term is an expectation over the SAME trajectories, so the Lagrangian
    gradient is just the COALA-PG gradient of a reshaped reward,

        r_mix = (r_s + r_o) + lam_s * r_s + lam_o * r_o
              = (1 + lam_s) * r_s + (1 + lam_o) * r_o,

    and the learning-aware estimator (`coala_advantages`, Algorithm 1) needs no
    modification whatsoever. That is the whole point: the comparison against
    the selfish and plain-welfare objectives isolates the objective, because
    the estimator, rollout structure and co-player are untouched.

    Note the two multipliers do different jobs, which is why one is not enough:
    ``lam_s`` pulls the shaper back from self-sacrifice, while ``lam_o``
    protects the co-player from being exploited. With only ``lam_s``, the
    welfare term is the co-player's sole defence, and welfare is indifferent
    between (-10, -30) and (-30, -10).

    Rather than reshaping the reward and running one critic, this class keeps
    the streams separate and exploits the fact that Algorithm 1 is LINEAR in
    (rewards, values):

        adv(r_W + sum_k lam_k r_k, V_W + sum_k lam_k V_k)
            == adv(r_W, V_W) + sum_k lam_k * adv(r_k, V_k)

    so combining per-stream advantages is algebraically identical to
    reshaping, while letting each critic head fit its own return. A single
    critic on the composite return would be biased by exactly the amount the
    multipliers last moved -- worst precisely when a constraint begins to bind.

    `update` therefore takes an extra argument: ``lam``, a vector of length
    ``num_constraints`` in the same order the runner stacks the streams.
    """

    def __init__(
        self,
        network: NamedTuple,
        initial_hidden_state: jnp.ndarray,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        gru_dim: int,
        obs_spec: Tuple,
        num_envs: int = 4,
        num_opps: int = 2,
        num_minibatches: int = 16,
        num_epochs: int = 4,
        num_inner_steps: int = 100,
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3000,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        reward_rescaling: float = 1.0,
        num_constraints: int = 2,
        player_id: int = 0,
    ):
        # Head 0 is welfare, heads 1..K the constraints.
        num_value_heads = 1 + num_constraints

        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            key, subkey = jax.random.split(state.random_key)
            # `values` is [..., 1 + num_constraints]:
            # (welfare head, constraint heads...).
            (dist, values), hidden_state = network.apply(
                state.params, observation, mem.hidden
            )
            actions = dist.sample(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = dist.log_prob(actions)
            mem = mem._replace(hidden=hidden_state, extras=mem.extras)
            state = state._replace(random_key=key)
            return actions, state, mem

        def loss(
            params: hk.Params,
            timesteps: int,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            behavior_log_probs: jnp.ndarray,
            target_values: jnp.ndarray,
            advantages: jnp.ndarray,
            behavior_values: jnp.ndarray,
            hiddens: jnp.ndarray,
        ):
            """PPO clipped surrogate; value loss summed over both critic heads."""
            (distribution, values), _ = network.apply(
                params, observations, hiddens
            )
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            rhos = jnp.exp(log_prob - behavior_log_probs)
            clipped_ratios_t = jnp.clip(
                rhos, 1.0 - ppo_clipping_epsilon, 1.0 + ppo_clipping_epsilon
            )
            clipped_objective = jnp.fmin(
                rhos * advantages, clipped_ratios_t * advantages
            )
            # As in `CoalaPG`: the estimator sums over the co-player minibatch
            # B, and minibatches are flattened for PPO, so scale by B.
            policy_loss = -jnp.mean(clipped_objective) * num_envs

            # Each head regresses its OWN return. No lam appears here -- that
            # is what keeps both baselines valid as lam moves.
            unclipped_value_loss = (target_values - values) ** 2
            if clip_value:
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -ppo_clipping_epsilon,
                    ppo_clipping_epsilon,
                )
                clipped_value_loss = (target_values - clipped_values) ** 2
                per_head_loss = jnp.fmax(
                    unclipped_value_loss, clipped_value_loss
                )
            else:
                per_head_loss = unclipped_value_loss
            # Mean over batch AND over heads. Summing over heads was wrong:
            # the heads share one torso, so the torso received the SUM of all
            # head gradients -- with 3 heads, ~3x the value gradient the
            # single-critic baseline puts into the same 32-unit GRU, crowding
            # the policy gradient out of the shared representation. Measured:
            # with both multipliers at 0 (pure welfare) this agent converged
            # to always-defect, while the single-critic agent on the identical
            # objective reached CC 0.99. The mean keeps the torso's
            # policy/value balance equal to the baseline's, which is what
            # "same architecture, different objective" requires. Each head's
            # own readout is zero-init linear under Adam, so the 1/heads
            # scale on its parameters is immaterial.
            value_loss = jnp.mean(per_head_loss)
            value_loss_welfare = jnp.mean(per_head_loss[..., 0])
            value_loss_cost = jnp.mean(per_head_loss[..., 1:])

            if anneal_entropy:
                fraction = jnp.fmax(1 - timesteps / entropy_coeff_horizon, 0)
                entropy_cost = (
                    fraction * entropy_coeff_start
                    + (1 - fraction) * entropy_coeff_end
                )
            else:
                entropy_cost = entropy_coeff_start
            entropy_loss = -jnp.mean(entropy)

            total_loss = (
                policy_loss
                + entropy_cost * entropy_loss
                + value_loss * value_coeff
            )
            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_value_welfare": value_loss_welfare,
                "loss_value_cost": value_loss_cost,
                "loss_entropy": entropy_loss,
                "entropy_cost": entropy_cost,
            }

        @jax.jit
        def sgd_step(
            state: TrainingState, sample: NamedTuple, lam: jnp.ndarray
        ):
            """One Lagrangian COALA-PG update.

            `sample` is time-major with the co-player batch intact,
            ``[L, num_opps, num_envs, ...]`` where ``L = M * T``, and carries
            TWO reward tensors:

              * ``rewards``      -- the welfare stream r_s + r_o, shape
                ``[L, num_opps, num_envs]``
              * ``cost_rewards`` -- the K constraint streams stacked on a
                trailing axis, ``[L, num_opps, num_envs, K]``. The runner has
                already masked each to its constraint window and rescaled it.

            ``behavior_values`` is ``[L, num_opps, num_envs, 1 + K]``, head 0
            welfare and heads 1..K matching ``cost_rewards``' last axis.

            ``lam`` is ``[K]``, in that same order.
            """
            observations = sample.observations
            actions = sample.actions
            rewards_w = sample.rewards
            rewards_c = sample.cost_rewards
            behavior_log_probs = sample.behavior_log_probs
            behavior_values = sample.behavior_values
            dones = sample.dones
            hiddens = sample.hiddens

            seq_len = rewards_w.shape[0]
            n_opps = rewards_w.shape[1]
            n_envs = rewards_w.shape[2]

            # All streams share one rescaling so their ratios -- and therefore
            # the meaning of lam -- are unaffected by it.
            learning_rewards_w = rewards_w * reward_rescaling
            learning_rewards_c = rewards_c * reward_rescaling

            # -> [num_opps, num_envs, L] for Algorithm 1.
            to_alg = lambda x: jnp.transpose(x, (1, 2, 0))
            r_w_a = to_alg(learning_rewards_w)
            d_a = to_alg(dones)
            v_w_a = to_alg(behavior_values[..., 0])
            boot_w = v_w_a[:, :, -1] * (1.0 - d_a[:, :, -1])

            adv_fn = jax.vmap(
                coala_advantages, in_axes=(0, 0, 0, 0, None, None, None)
            )
            tgt_fn = jax.vmap(
                coala_value_targets, in_axes=(0, 0, 0, None, None, None)
            )

            def _streams(r_a, v_a):
                boot = v_a[:, :, -1] * (1.0 - d_a[:, :, -1])
                adv = adv_fn(
                    r_a, v_a, d_a, boot, gamma, gae_lambda, num_inner_steps
                )
                tgt = jax.lax.stop_gradient(
                    tgt_fn(r_a, v_a, boot, gamma, gae_lambda, num_inner_steps)
                )
                return adv, tgt

            # Algorithm 1 run once per stream. Because it is LINEAR in
            # (rewards, values), adv_w + sum_k lam_k * adv_k is exactly the
            # advantage of the reshaped reward r_W + sum_k lam_k * r_k under
            # the combined baseline V_W + sum_k lam_k * V_k -- no
            # approximation, and no change to the estimator itself.
            adv_w, tgt_w = _streams(r_w_a, v_w_a)

            lam_vec = jnp.reshape(jnp.asarray(lam), (num_constraints,))
            adv_a = adv_w
            adv_c_list = []
            tgt_c_list = []
            for k in range(num_constraints):
                r_k_a = to_alg(learning_rewards_c[..., k])
                v_k_a = to_alg(behavior_values[..., 1 + k])
                adv_k, tgt_k = _streams(r_k_a, v_k_a)
                adv_a = adv_a + lam_vec[k] * adv_k
                adv_c_list.append(adv_k)
                tgt_c_list.append(tgt_k)

            # Back to time-major, then stack the heads on the last axis in the
            # same order the network emits them.
            from_alg = lambda x: jnp.transpose(x, (2, 0, 1))
            advantages = from_alg(adv_a)
            target_values = jnp.stack(
                [from_alg(tgt_w)] + [from_alg(x) for x in tgt_c_list], axis=-1
            )

            trajectories = LagrangianBatch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                target_values=target_values,
                behavior_values=behavior_values,
                behavior_log_probs=behavior_log_probs,
                hiddens=hiddens,
            )

            batch_size = seq_len * n_opps * n_envs
            assert batch_size % num_minibatches == 0, (
                "num_minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            # Collapse (L, opps, envs) into the batch axis, keeping any
            # trailing feature axes (including the 2-head value axis).
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[3:]), trajectories
            )

            grad_fn = jax.jit(jax.grad(loss, has_aux=True))

            def model_update_minibatch(carry, minibatch: LagrangianBatch):
                params, opt_state, timesteps = carry
                if advantage_normalization:
                    advantages_mb = (
                        minibatch.advantages
                        - jnp.mean(minibatch.advantages, axis=0)
                    ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
                else:
                    advantages_mb = minibatch.advantages
                gradients, metrics = grad_fn(
                    params,
                    timesteps,
                    minibatch.observations,
                    minibatch.actions,
                    minibatch.behavior_log_probs,
                    minibatch.target_values,
                    advantages_mb,
                    minibatch.behavior_values,
                    minibatch.hiddens,
                )
                updates, opt_state = optimizer.update(gradients, opt_state)
                params = optax.apply_updates(params, updates)
                metrics["norm_grad"] = optax.global_norm(gradients)
                metrics["norm_updates"] = optax.global_norm(updates)
                return (params, opt_state, timesteps), metrics

            def model_update_epoch(carry, unused_t):
                key, params, opt_state, timesteps, batch = carry
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled,
                )
                (params, opt_state, timesteps), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state, timesteps),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, timesteps, batch), metrics

            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (
                    state.random_key,
                    state.params,
                    state.opt_state,
                    state.timesteps,
                    batch,
                ),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(rewards_w)
            metrics["rewards_std"] = jnp.std(rewards_w)
            metrics["coala/advantage_mean"] = jnp.mean(advantages)
            metrics["coala/advantage_std"] = jnp.std(advantages)
            # Each advantage stream separately: if the welfare term dominates
            # by orders of magnitude, no constraint can bite whatever lam says.
            metrics["coala/advantage_welfare_std"] = jnp.std(adv_w)
            metrics["coala/target_value_welfare_mean"] = jnp.mean(tgt_w)
            for k in range(num_constraints):
                metrics[f"coala/advantage_cost{k}_std"] = jnp.std(
                    adv_c_list[k]
                )
                metrics[f"coala/target_value_cost{k}_mean"] = jnp.mean(
                    tgt_c_list[k]
                )
                metrics[f"lagrangian/lam{k}_used"] = lam_vec[k]
            metrics["coala/reward_rescaling"] = jnp.asarray(reward_rescaling)
            metrics["coala/policy_batch_scale"] = jnp.asarray(n_envs)

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps + batch_size,
            )
            # Memory deliberately not rebuilt -- see the note in `CoalaPG`.
            return new_state, metrics

        def make_initial_state(
            key: Any, initial_hidden_state: jnp.ndarray
        ) -> Tuple[TrainingState, MemoryState]:
            key, subkey = jax.random.split(key)
            if isinstance(obs_spec, dict):
                dummy_obs = {k: jnp.zeros(shape=v) for k, v in obs_spec.items()}
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(
                subkey, dummy_obs, initial_hidden_state
            )
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, initial_hidden_state.shape[-1])),
                extras={
                    # One critic head per return: welfare + K constraints.
                    "values": jnp.zeros((num_envs, num_value_heads)),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        self._state, self._mem = make_initial_state(
            random_key, initial_hidden_state
        )
        self.make_initial_state = make_initial_state
        self._sgd_step = sgd_step

        self._logger = Logger()
        self._total_steps = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
        }

        self.network = network
        self._policy = policy
        self.forward = network.apply
        self.player_id = player_id

        self._num_envs = num_envs
        self._num_opps = num_opps
        self._num_minibatches = num_minibatches
        self._num_epochs = num_epochs
        self._num_inner_steps = num_inner_steps
        self._gru_dim = gru_dim
        self._num_constraints = num_constraints
        self._num_value_heads = num_value_heads

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros((num_envs, self._num_value_heads)),
                "log_probs": jnp.zeros(num_envs),
            },
            hidden=jnp.zeros((num_envs, self._gru_dim)),
        )
        return memory

    def update(
        self,
        traj_batch: NamedTuple,
        obs: jnp.ndarray,
        state: TrainingState,
        mem: MemoryState,
        lam: jnp.ndarray = None,
    ):
        """Update at the end of a full meta-trajectory.

        `lam` is the vector of current Lagrange multipliers (one per
        constraint), supplied by the runner's controllers. `mem` is returned
        unchanged -- see the note in `sgd_step`.
        """
        if lam is None:
            lam = jnp.zeros((self._num_constraints,))
        state, metrics = self._sgd_step(state, traj_batch, lam)
        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        for k in (
            "loss_total",
            "loss_policy",
            "loss_value",
            "loss_entropy",
            "entropy_cost",
        ):
            self._logger.metrics[k] = metrics[k]
        return state, mem, metrics


class CoalaA2C(AgentInterface):
    """Naive A2C co-player used in the COALA-PG paper's IPD experiments."""

    def __init__(
        self,
        network: NamedTuple,
        initial_hidden_state: jnp.ndarray,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: Tuple,
        num_envs: int = 16,
        num_inner_steps: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.0,
        reward_rescaling: float = 0.05,
        advantage_normalization: bool = True,
        player_id: int = 0,
    ):
        hidden_size = initial_hidden_state.shape[-1]

        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            key, subkey = jax.random.split(state.random_key)
            (dist, values), hidden_state = network.apply(
                state.params, observation, mem.hidden
            )
            actions, log_probs = dist.sample_and_log_prob(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = log_probs
            mem = mem._replace(hidden=hidden_state, extras=mem.extras)
            state = state._replace(random_key=key)
            return actions, state, mem

        def a2c_returns(rewards: jnp.ndarray) -> jnp.ndarray:
            """Discounted length-T returns without value bootstrapping."""

            def step(acc, reward_t):
                acc = reward_t + gamma * acc
                return acc, acc

            _, out = jax.lax.scan(
                step,
                jnp.zeros_like(rewards[-1]),
                jnp.flip(rewards, axis=0),
            )
            return jnp.flip(out, axis=0)

        def loss(
            params: hk.Params,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            returns: jnp.ndarray,
            advantages: jnp.ndarray,
            hiddens: jnp.ndarray,
        ):
            (distribution, values), _ = network.apply(
                params, observations, hiddens
            )
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()

            policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages))
            value_loss = jnp.mean((returns - values) ** 2)
            entropy_loss = -jnp.mean(entropy)
            total_loss = (
                policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
            )
            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
                "entropy_cost": jnp.asarray(entropy_coeff),
            }

        @jax.jit
        def sgd_step(state: TrainingState, sample: NamedTuple):
            rewards = sample.rewards * reward_rescaling
            returns = a2c_returns(rewards)
            advantages = returns - sample.behavior_values
            if advantage_normalization:
                advantages = (
                    advantages - jnp.mean(advantages)
                ) / (jnp.std(advantages) + 1e-8)

            batch_size = returns.shape[0] * returns.shape[1]
            flat = Batch(
                observations=sample.observations.reshape(
                    (batch_size,) + sample.observations.shape[2:]
                ),
                actions=sample.actions.reshape((batch_size,) + sample.actions.shape[2:]),
                advantages=advantages.reshape((batch_size,)),
                target_values=returns.reshape((batch_size,)),
                behavior_values=sample.behavior_values.reshape((batch_size,)),
                behavior_log_probs=sample.behavior_log_probs.reshape((batch_size,)),
                hiddens=sample.hiddens.reshape(
                    (batch_size,) + sample.hiddens.shape[2:]
                ),
            )

            grad_fn = jax.grad(loss, has_aux=True)
            gradients, metrics = grad_fn(
                state.params,
                flat.observations,
                flat.actions,
                flat.target_values,
                flat.advantages,
                flat.hiddens,
            )
            updates, opt_state = optimizer.update(gradients, state.opt_state)
            params = optax.apply_updates(state.params, updates)
            metrics["norm_grad"] = optax.global_norm(gradients)
            metrics["norm_updates"] = optax.global_norm(updates)
            metrics["rewards_mean"] = jnp.mean(sample.rewards)
            metrics["a2c/return_mean"] = jnp.mean(returns)
            metrics["a2c/advantage_std"] = jnp.std(advantages)

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=state.random_key,
                timesteps=state.timesteps + batch_size,
            )
            new_mem = MemoryState(
                hidden=jnp.zeros((num_envs, hidden_size)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )
            return new_state, new_mem, metrics

        def make_initial_state(
            key: Any, initial_hidden: jnp.ndarray
        ) -> Tuple[TrainingState, MemoryState]:
            key, subkey = jax.random.split(key)
            if isinstance(obs_spec, dict):
                dummy_obs = {k: jnp.zeros(shape=v) for k, v in obs_spec.items()}
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            # The COALA runner vmaps agent2 initialization over random keys but
            # passes a shared hidden argument because older opponents ignored it.
            # Initialize parameters with the canonical single-batch hidden.
            initial_params = network.init(subkey, dummy_obs, initial_hidden_state)
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, hidden_size)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        self._state, self._mem = make_initial_state(
            random_key, initial_hidden_state
        )
        self.make_initial_state = make_initial_state
        self._policy = policy
        self._sgd_step = sgd_step
        self.network = network
        self.forward = network.apply
        self.player_id = player_id
        self._num_envs = num_envs
        self._num_inner_steps = num_inner_steps
        self._hidden_size = hidden_size

        self._logger = Logger()
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff,
        }

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        return memory._replace(
            hidden=jnp.zeros((num_envs, self._hidden_size)),
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
        )

    def update(
        self,
        traj_batch: NamedTuple,
        obs: jnp.ndarray,
        state: TrainingState,
        mem: MemoryState,
    ):
        state, mem, metrics = self._sgd_step(state, traj_batch)
        self._logger.metrics["sgd_steps"] += 1
        for k in (
            "loss_total",
            "loss_policy",
            "loss_value",
            "loss_entropy",
            "entropy_cost",
        ):
            self._logger.metrics[k] = metrics[k]
        return state, mem, metrics


def make_coala_pg_agent(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    num_iterations: int,
    player_id: int,
):
    """Build a COALA-PG agent for the environment named in ``args.env_id``."""
    if args.env_id in (
        "iterated_matrix_game",
        "iterated_tensor_game",
        "iterated_nplayer_tensor_game",
    ):
        network, initial_hidden_state = make_coala_ipd_network(
            action_spec, agent_args.hidden_size
        )
    elif args.env_id == "coin_game":
        network, initial_hidden_state = make_GRU_coingame_network(
            action_spec,
            agent_args.with_cnn,
            agent_args.hidden_size,
            agent_args.output_channels,
            agent_args.kernel_shape,
        )
    else:
        raise NotImplementedError(
            f"COALA-PG has no network for env_id={args.env_id}. "
            "Implemented for iterated_matrix_game (IPD) and coin_game."
        )

    gru_dim = initial_hidden_state.shape[1]

    if agent_args.lr_scheduling:
        # One sgd_step per meta-trajectory, mirroring the stock PPO schedule.
        scheduler = optax.linear_schedule(
            init_value=agent_args.learning_rate,
            end_value=0,
            transition_steps=max(int(num_iterations), 1),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.max_gradient_norm),
            optax.scale_by_adam(eps=agent_args.adam_epsilon),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.max_gradient_norm),
            optax.scale_by_adam(eps=agent_args.adam_epsilon),
            optax.scale(-agent_args.learning_rate),
        )

    random_key = jax.random.PRNGKey(seed=seed)

    return CoalaPG(
        network=network,
        initial_hidden_state=initial_hidden_state,
        optimizer=optimizer,
        random_key=random_key,
        gru_dim=gru_dim,
        obs_spec=obs_spec,
        num_envs=args.num_envs,
        num_opps=args.num_opps,
        num_minibatches=agent_args.num_minibatches,
        num_epochs=agent_args.num_epochs,
        num_inner_steps=args.num_inner_steps,
        clip_value=agent_args.clip_value,
        value_coeff=agent_args.value_coeff,
        anneal_entropy=agent_args.anneal_entropy,
        entropy_coeff_start=agent_args.entropy_coeff_start,
        entropy_coeff_end=agent_args.entropy_coeff_end,
        entropy_coeff_horizon=agent_args.entropy_coeff_horizon,
        ppo_clipping_epsilon=agent_args.ppo_clipping_epsilon,
        gamma=agent_args.gamma,
        gae_lambda=agent_args.gae_lambda,
        advantage_normalization=agent_args.get("advantage_normalization", True),
        reward_rescaling=agent_args.get("reward_rescaling", 1.0),
        player_id=player_id,
    )


def make_coala_pg_lagrangian_agent(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    num_iterations: int,
    player_id: int,
):
    """Build a Lagrangian COALA-PG shaper (one critic head per return).

    Identical to `make_coala_pg_agent` except for the network: the torso is the
    same, with one extra value readout per constraint.

    The critic is sized from the SAME parser the runner uses
    (`pax.runners.pid_lagrangian.num_constraints`), so the two cannot disagree
    about how many value heads exist. By default that is 2 -- the shaper's and
    the co-player's individual-rationality conditions -- and
    ``welfare.constrain_<player>: False`` drops one.
    """
    welfare_args = omegaconf.OmegaConf.select(args, "welfare", default=None)
    if welfare_args is None:
        raise ValueError(
            "Lagrangian COALA-PG needs a `welfare` config block with "
            "v_ref_shaper / v_ref_opponent. See "
            "pax/conf/experiment/ipd/lagrangian_coala_pg_v_tabular.yaml."
        )
    num_constraints = count_constraints(welfare_args)

    if args.env_id in (
        "iterated_matrix_game",
        "iterated_tensor_game",
        "iterated_nplayer_tensor_game",
    ):
        network, initial_hidden_state = make_coala_ipd_multi_value_network(
            action_spec,
            agent_args.hidden_size,
            num_value_heads=1 + num_constraints,
        )
    else:
        raise NotImplementedError(
            "Lagrangian COALA-PG has no network for "
            f"env_id={args.env_id}. Implemented for iterated_matrix_game "
            "(IPD); the coin_game network has a single value head."
        )

    gru_dim = initial_hidden_state.shape[1]

    if agent_args.lr_scheduling:
        scheduler = optax.linear_schedule(
            init_value=agent_args.learning_rate,
            end_value=0,
            transition_steps=max(int(num_iterations), 1),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.max_gradient_norm),
            optax.scale_by_adam(eps=agent_args.adam_epsilon),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.max_gradient_norm),
            optax.scale_by_adam(eps=agent_args.adam_epsilon),
            optax.scale(-agent_args.learning_rate),
        )

    random_key = jax.random.PRNGKey(seed=seed)

    return CoalaPGLagrangian(
        network=network,
        initial_hidden_state=initial_hidden_state,
        optimizer=optimizer,
        random_key=random_key,
        gru_dim=gru_dim,
        obs_spec=obs_spec,
        num_envs=args.num_envs,
        num_opps=args.num_opps,
        num_minibatches=agent_args.num_minibatches,
        num_epochs=agent_args.num_epochs,
        num_inner_steps=args.num_inner_steps,
        clip_value=agent_args.clip_value,
        value_coeff=agent_args.value_coeff,
        anneal_entropy=agent_args.anneal_entropy,
        entropy_coeff_start=agent_args.entropy_coeff_start,
        entropy_coeff_end=agent_args.entropy_coeff_end,
        entropy_coeff_horizon=agent_args.entropy_coeff_horizon,
        ppo_clipping_epsilon=agent_args.ppo_clipping_epsilon,
        gamma=agent_args.gamma,
        gae_lambda=agent_args.gae_lambda,
        advantage_normalization=agent_args.get("advantage_normalization", True),
        reward_rescaling=agent_args.get("reward_rescaling", 1.0),
        num_constraints=num_constraints,
        player_id=player_id,
    )


def make_coala_a2c_agent(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    player_id: int,
):
    """Build the paper-style naive A2C co-player for IPD."""
    if args.env_id != "iterated_matrix_game":
        raise NotImplementedError(
            "CoalaA2C is implemented only for iterated_matrix_game/IPD."
        )

    network, initial_hidden_state = make_coala_ipd_network(
        action_spec, agent_args.hidden_size
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(agent_args.max_gradient_norm),
        optax.scale_by_adam(eps=agent_args.adam_epsilon),
        optax.scale(-agent_args.learning_rate),
    )
    random_key = jax.random.PRNGKey(seed=seed)
    return CoalaA2C(
        network=network,
        initial_hidden_state=initial_hidden_state,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        num_envs=args.num_envs,
        num_inner_steps=args.num_inner_steps,
        gamma=agent_args.gamma,
        gae_lambda=agent_args.gae_lambda,
        value_coeff=agent_args.value_coeff,
        entropy_coeff=agent_args.entropy_coeff_start,
        reward_rescaling=agent_args.get("reward_rescaling", 0.05),
        advantage_normalization=agent_args.get("advantage_normalization", True),
        player_id=player_id,
    )
