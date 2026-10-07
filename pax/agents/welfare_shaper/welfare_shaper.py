"""Welfare Shaper agent.

Architecture is the same as M-FOS (three-stream GRU with meta shaping weights).
The key difference is in how the *fitness / reward signal* is constructed:

  augmented_reward = welfare + mu1 * (R_shaper - v_ref_shaper)
                              + mu2 * (R_opponent - v_ref_opponent)

where the dual multipliers mu1, mu2 are updated by the runner via
Lagrangian dual ascent.  The agent itself is a standard PPO learner
that maximises whatever reward signal it receives.
"""

from typing import Any, Dict, Mapping, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from pax import utils
from pax.agents.agent import AgentInterface
from pax.agents.welfare_shaper.networks import (
    make_welfare_network,
    make_welfare_avg_network,
    make_welfare_continuous_network,
)
from pax.utils import TrainingState, get_advantages


class MemoryState(NamedTuple):
    """State consists of network extras (to be batched)"""
    hidden: jnp.ndarray
    th: jnp.ndarray
    curr_th: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""
    observations: jnp.ndarray
    actions: jnp.ndarray
    advantages: jnp.ndarray
    target_values: jnp.ndarray
    behavior_values: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    hiddens: jnp.ndarray
    meta_actions: jnp.ndarray


class Logger:
    metrics: dict


class WelfareShaper(AgentInterface):
    """PPO agent with M-FOS-style meta shaping, designed for welfare maximisation
    with individual-rationality constraints enforced via Lagrangian dual ascent."""

    def __init__(
        self,
        network: NamedTuple,
        initial_hidden_state: jnp.ndarray,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        gru_dim: int,
        obs_spec: Tuple,
        batch_size: int = 2000,
        num_envs: int = 4,
        num_minibatches: int = 16,
        num_epochs: int = 4,
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3_000_000,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        player_id: int = 0,
    ):
        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            key, subkey = jax.random.split(state.random_key)
            (dist, values), (_current_th, hidden) = network.apply(
                state.params,
                (observation, mem.th),# used fixed th, and th does not get updated within episode. Meta step between episodes will copy curr_th to th.
                mem.hidden,
            )

            actions = dist.sample(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = dist.log_prob(actions)
#curr_th gets updated within an episode and stored in mem.  
# but the updated curr_th only get applied to the policy in the next episode by copying to th. 
# This is important for stability of training.
            mem = mem._replace(
                hidden=hidden, curr_th=_current_th, extras=mem.extras
            )
            state = state._replace(random_key=key)
            return actions, state, mem

        def gae_advantages(
            rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray
        ) -> jnp.ndarray:
            discounts = gamma * jnp.logical_not(dones)
            reverse_batch = (
                jnp.flip(values[:-1], axis=0),
                jnp.flip(rewards, axis=0),
                jnp.flip(discounts, axis=0),
            )
            _, advantages = jax.lax.scan(
                get_advantages,
                (
                    jnp.zeros_like(values[-1]),
                    values[-1],
                    jnp.ones_like(values[-1]) * gae_lambda,
                ),
                reverse_batch,
            )
            advantages = jnp.flip(advantages, axis=0)
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)
            return advantages, target_values

        def loss(
            params: hk.Params,
            timesteps: int,
            observations: jnp.ndarray,
            actions: jnp.array,
            behavior_log_probs: jnp.array,
            target_values: jnp.array,
            advantages: jnp.array,
            behavior_values: jnp.array,
            hiddens: jnp.ndarray,
            meta_actions: jnp.ndarray,
        ):
            (distribution, values), _ = network.apply(
                params, (observations, meta_actions), hiddens
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
            policy_loss = -jnp.mean(clipped_objective)

            value_cost = value_coeff
            unclipped_value_error = target_values - values
            unclipped_value_loss = unclipped_value_error ** 2

            if clip_value:
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -ppo_clipping_epsilon,
                    ppo_clipping_epsilon,
                )
                clipped_value_error = target_values - clipped_values
                clipped_value_loss = clipped_value_error ** 2
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
                + value_loss * value_cost
            )

            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
                "entropy_cost": entropy_cost,
            }

        def sgd_step(
            state: TrainingState, sample: NamedTuple
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            (
                observations,
                actions,
                rewards,
                behavior_log_probs,
                behavior_values,
                dones,
                hiddens,
                meta_actions,
            ) = (
                sample.observations,
                sample.actions,
                sample.rewards,
                sample.behavior_log_probs,
                sample.behavior_values,
                sample.dones,
                sample.hiddens,
                sample.meta_actions,
            )

            advantages, target_values = gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            behavior_values = behavior_values[:-1, :]
            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
                hiddens=hiddens,
                meta_actions=meta_actions,
            )
            assert len(target_values.shape) > 1
            num_envs = target_values.shape[1]
            num_steps = target_values.shape[0]
            batch_size = num_envs * num_steps
            assert batch_size % num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            grad_fn = jax.jit(jax.grad(loss, has_aux=True))

            def model_update_minibatch(carry, minibatch):
                params, opt_state, timesteps = carry
                advantages = (
                    minibatch.advantages
                    - jnp.mean(minibatch.advantages, axis=0)
                ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
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
                    minibatch.meta_actions,
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
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                (params, opt_state, timesteps), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state, timesteps),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, timesteps, batch), metrics

            params = state.params
            opt_state = state.opt_state
            timesteps = state.timesteps

            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (state.random_key, params, opt_state, timesteps, batch),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(
                jnp.abs(jnp.mean(rewards, axis=(0, 1)))
            )
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps + batch_size,
            )

            new_memory = MemoryState(
                hidden=jnp.zeros(shape=(self._num_envs,) + (gru_dim,)),
                th=jnp.ones(shape=(self._num_envs,) + (gru_dim // 3,)),
                curr_th=jnp.ones(shape=(self._num_envs,) + (gru_dim // 3,)),
                extras={
                    "log_probs": jnp.zeros(self._num_envs),
                    "values": jnp.zeros(self._num_envs),
                },
            )

            return new_state, new_memory, metrics

        def make_initial_state(key, initial_hidden_state):
            key, subkey = jax.random.split(key)

            if isinstance(obs_spec, dict):
                dummy_obs = {
                    "inventory": jnp.zeros(obs_spec["inventory"]),
                    "observation": jnp.zeros(obs_spec["observation"]),
                }
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)

            dummy_obs = utils.add_batch_dim(dummy_obs)
            hidden_size = initial_hidden_state.shape[-1]

            dummy_meta = jnp.zeros(shape=hidden_size // 3)
            dummy_meta = utils.add_batch_dim(dummy_meta)
            dummy_input = (dummy_obs, dummy_meta)

            initial_params = network.init(
                subkey, dummy_input, initial_hidden_state
            )
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, hidden_size)),
                th=jnp.ones((num_envs, hidden_size // 3)),
                curr_th=jnp.ones((num_envs, hidden_size // 3)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        @jax.jit
        def prepare_batch(traj_batch, done, action_extras):
            _value = jax.lax.select(
                done,
                jnp.zeros_like(action_extras["values"]),
                action_extras["values"],
            )
            _value = jax.lax.expand_dims(_value, [0])
            traj_batch = traj_batch._replace(
                behavior_values=jnp.concatenate(
                    [traj_batch.behavior_values, _value], axis=0
                )
            )
            return traj_batch

        # Initialise training state
        self._state, self._mem = make_initial_state(
            random_key, initial_hidden_state
        )

        self.make_initial_state = make_initial_state
        self.prepare_batch = prepare_batch
        self._sgd_step = jax.jit(sgd_step)

        self._logger = Logger()
        self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
        }

        self._policy = policy
        self.forward = network.apply
        self.player_id = player_id

        self._num_envs = num_envs
        self._num_minibatches = num_minibatches
        self._num_epochs = num_epochs
        self._gru_dim = gru_dim

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
            hidden=jnp.zeros((num_envs, self._gru_dim)),
            th=jnp.ones((num_envs, self._gru_dim // 3)),
            curr_th=jnp.ones((num_envs, self._gru_dim // 3)),
        )
        return memory

    def update(self, traj_batch, obs, state, mem):
        _, _, mem = self._policy(state, obs, mem)
        traj_batch = self.prepare_batch(
            traj_batch, traj_batch.dones[-1, ...], mem.extras
        )
        state, mem, metrics = self._sgd_step(state, traj_batch)

        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        self._logger.metrics["loss_total"] = metrics["loss_total"]
        self._logger.metrics["loss_policy"] = metrics["loss_policy"]
        self._logger.metrics["loss_value"] = metrics["loss_value"]
        self._logger.metrics["loss_entropy"] = metrics["loss_entropy"]
        self._logger.metrics["entropy_cost"] = metrics["entropy_cost"]
        return state, mem, metrics

    def meta_policy(self, mem: MemoryState):
        """Between-episode meta step: carry forward shaping weights."""
        mem = mem._replace(th=mem.curr_th)
        mem = mem._replace(
            hidden=jnp.zeros_like(mem.hidden),
            curr_th=jnp.ones_like(mem.curr_th),
        )
        return mem


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_welfare_shaper_agent(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    player_id: int,
    num_iterations: int,
):
    """Create a WelfareShaper agent."""
    if args.env_id in ["Cournot", "Fishery"] or args.env_id == "Rice-N":
        network, initial_hidden_state = make_welfare_continuous_network(
            action_spec, agent_args.hidden_size,
        )
    elif args.env_id == "iterated_matrix_game":
        att_type = getattr(args, "att_type", "nothing")
        if att_type == "avg":
            network, initial_hidden_state = make_welfare_avg_network(
                action_spec, agent_args.hidden_size,
            )
        else:
            network, initial_hidden_state = make_welfare_network(
                action_spec, agent_args.hidden_size,
            )
    else:
        network, initial_hidden_state = make_welfare_network(
            action_spec, agent_args.hidden_size,
        )

    gru_dim = initial_hidden_state.shape[1]

    transition_steps = (
        num_iterations * agent_args.num_epochs * agent_args.num_minibatches
    )

    if agent_args.lr_scheduling:
        scheduler = optax.linear_schedule(
            init_value=agent_args.learning_rate,
            end_value=0,
            transition_steps=transition_steps,
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

    agent = WelfareShaper(
        network=network,
        initial_hidden_state=initial_hidden_state,
        optimizer=optimizer,
        random_key=random_key,
        gru_dim=gru_dim,
        obs_spec=obs_spec,
        batch_size=None,
        num_envs=args.num_envs,
        num_minibatches=agent_args.num_minibatches,
        num_epochs=agent_args.num_epochs,
        clip_value=agent_args.clip_value,
        value_coeff=agent_args.value_coeff,
        anneal_entropy=agent_args.anneal_entropy,
        entropy_coeff_start=agent_args.entropy_coeff_start,
        entropy_coeff_end=agent_args.entropy_coeff_end,
        entropy_coeff_horizon=agent_args.entropy_coeff_horizon,
        ppo_clipping_epsilon=agent_args.ppo_clipping_epsilon,
        gamma=agent_args.gamma,
        gae_lambda=agent_args.gae_lambda,
        player_id=player_id,
    )
    return agent
