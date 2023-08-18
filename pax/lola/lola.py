from typing import Any, Mapping, NamedTuple, Tuple, Dict

from pax import utils

# from pax.lola.buffer import TrajectoryBuffer
from pax.lola.network import make_network
from pax.utils import MemoryState, TrainingState

from dm_env import TimeStep
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


class LOLASample(NamedTuple):
    obs_self: jnp.ndarray
    obs_other: jnp.ndarray
    actions_self: jnp.ndarray
    actions_other: jnp.ndarray
    dones: jnp.ndarray
    rewards_self: jnp.ndarray
    rewards_other: jnp.ndarray


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    advantages: jnp.ndarray

    # Target value estimate used to bootstrap the value function.
    target_values: jnp.ndarray

    # Value estimate and action log-prob at behavior time.
    behavior_values: jnp.ndarray
    behavior_log_probs: jnp.ndarray


def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))


class Logger:
    metrics: dict


class LOLA:
    """LOLA with the DiCE objective function."""

    def __init__(
        self,
        network: NamedTuple,
        inner_optimizer: optax.GradientTransformation,
        outer_optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        player_id: int,
        obs_spec: Tuple,
        env_params: Any,
        num_envs: int = 4,
        num_steps: int = 150,
        use_baseline: bool = True,
        gamma: float = 0.96,
    ):
        self._num_envs = num_envs  # number of environments

        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            """Agent policy to select actions and calculate agent specific information"""

            key, subkey = jax.random.split(state.random_key)
            dist, values = network.apply(state.params, observation)
            actions = dist.sample(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = dist.log_prob(actions)
            state = state._replace(random_key=key)
            return actions, state, mem

        def outer_loss(params, other_params, samples):
            """Used for the outer rollout"""
            # Unpack the samples
            obs_1 = samples.obs_self
            obs_2 = samples.obs_other
            rewards = samples.rewards_self
            actions_1 = samples.actions_self
            actions_2 = samples.actions_other

            # Get distribution and value using my network
            distribution, values = self.network.apply(params, obs_1)
            self_log_prob = distribution.log_prob(actions_1)

            # Get distribution and value using other player's network
            distribution, _ = self.other_network.apply(other_params, obs_2)
            other_log_prob = distribution.log_prob(actions_2)

            # apply discount:
            cum_discount = (
                jnp.cumprod(self.gamma * jnp.ones(rewards.shape), axis=0)
                / self.gamma
            )

            discounted_rewards = rewards * cum_discount
            discounted_values = values * cum_discount

            # stochastics nodes involved in rewards dependencies:
            dependencies = jnp.cumsum(self_log_prob + other_log_prob, axis=0)

            # logprob of each stochastic nodes:
            stochastic_nodes = self_log_prob + other_log_prob

            # dice objective:
            dice_objective = jnp.mean(
                jnp.sum(magic_box(dependencies) * discounted_rewards, axis=0)
            )

            if use_baseline:
                # variance_reduction:
                baseline_term = jnp.mean(
                    jnp.sum(
                        (1 - magic_box(stochastic_nodes)) * discounted_values,
                        axis=0,
                    )
                )
                dice_objective = dice_objective + baseline_term

            # want to minimize this value
            value_objective = jnp.mean((rewards - values) ** 2)

            # want to maximize this objective
            loss_total = -dice_objective + value_objective

            return loss_total, {
                "loss_total": -dice_objective + value_objective,
                "loss_policy": -dice_objective,
                "loss_value": value_objective,
            }

        def inner_loss(params, other_params, samples):
            """Used for the inner rollout"""
            obs_1 = samples.obs_self
            obs_2 = samples.obs_other
            rewards = samples.rewards_self
            actions_1 = samples.actions_self
            actions_2 = samples.actions_other

            # Get distribution and value using other player's network
            distribution, values = self.other_network.apply(params, obs_1)
            self_log_prob = distribution.log_prob(actions_1)

            # Get distribution and value using my network
            distribution, _ = self.network.apply(other_params, obs_2)
            other_log_prob = distribution.log_prob(actions_2)

            # apply discount:
            cum_discount = (
                jnp.cumprod(self.gamma * jnp.ones(rewards.shape), axis=0)
                / self.gamma
            )
            discounted_rewards = rewards * cum_discount
            discounted_values = values * cum_discount

            # stochastics nodes involved in rewards dependencies:
            dependencies = jnp.cumsum(self_log_prob + other_log_prob, axis=0)

            # logprob of each stochastic nodes:
            stochastic_nodes = self_log_prob + other_log_prob

            # dice objective:
            dice_objective = jnp.mean(
                jnp.sum(magic_box(dependencies) * discounted_rewards, axis=0)
            )

            if use_baseline:
                # variance_reduction:
                baseline_term = jnp.mean(
                    jnp.sum(
                        (1 - magic_box(stochastic_nodes)) * discounted_values,
                        axis=0,
                    )
                )
                dice_objective = dice_objective + baseline_term

            # want to minimize this value
            value_objective = jnp.mean((rewards - values) ** 2)

            # want to maximize this objective
            loss_total = -dice_objective + value_objective

            return loss_total, {
                "loss_total": -dice_objective + value_objective,
                "loss_policy": -dice_objective,
                "loss_value": value_objective,
            }

        def sgd_step(
            state: TrainingState, sample: NamedTuple
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            state = state
            # placeholders
            results = {
                "loss_total": 0,
                "loss_policy": 0,
                "loss_value": 0,
                "loss_entropy": 0,
                "entropy_cost": 0,
            }
            return state, results

        def make_initial_state(key: Any, hidden) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = outer_optimizer.init(initial_params)
            return TrainingState(
                params=initial_params,
                opt_state=initial_opt_state,
                random_key=key,
                timesteps=0,
            ), MemoryState(
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
                hidden=jnp.zeros((self._num_envs, 1)),
            )

        inner_rollout_rng, random_key = jax.random.split(random_key)
        self.inner_rollout_rng = inner_rollout_rng
        # Initialise training state (parameters, optimiser state, extras).
        self._state, self._mem = make_initial_state(random_key, obs_spec)
        self.make_initial_state = make_initial_state

        # Setup player id
        self.player_id = player_id

        self.env_params = env_params

        # # Initialize buffer and sgd
        # self._trajectory_buffer = TrajectoryBuffer(
        #     num_envs, num_steps, obs_spec
        # )

        self.grad_fn_inner = jax.jit(jax.grad(inner_loss, has_aux=True))
        self.grad_fn_outer = jax.jit(jax.grad(outer_loss, has_aux=True))
        # self.grad_fn_outer = jax.grad(outer_loss, has_aux=True)

        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
        }

        # Initialize functions
        self._policy = policy
        self.network = network
        self._sgd_step = sgd_step

        # initialize some variables
        self._inner_optimizer = inner_optimizer
        self._outer_optimizer = outer_optimizer
        self.gamma = gamma

        # Other useful hyperparameters
        self._num_steps = num_steps  # number of steps per environment
        self._batch_size = int(num_envs * num_steps)  # number in one batch
        self._obs_spec = obs_spec

    def select_action(self, t: TimeStep):
        """Selects action and updates info with PPO specific information"""
        actions, self._state = self._policy(
            self._state.params, t.observation, self._state
        )
        return utils.to_numpy(actions)

    def in_lookahead(self, env, env_rollout):
        """
        Performs a rollout using the current parameters of both agents
        and simulates a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        """

        # my state
        my_state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
        )
        my_state = jax.tree_map(lambda x: jnp.expand_dims(x, 0), my_state)
        my_mem = MemoryState(
            extras={
                "values": jnp.zeros((self._num_envs)),
                "log_probs": jnp.zeros((self._num_envs)),
            },
            hidden=jnp.zeros((self._num_envs, 1)),
        )

        other_state = TrainingState(
            params=self.other_state.params,
            opt_state=self.other_state.opt_state,
            random_key=self.other_state.random_key,
            timesteps=self.other_state.timesteps,
        )
        other_state = jax.tree_map(
            lambda x: jnp.expand_dims(x, 0), other_state
        )
        other_mem = MemoryState(
            extras={
                "values": jnp.zeros((self._num_envs)),
                "log_probs": jnp.zeros((self._num_envs)),
            },
            hidden=jnp.zeros((self._num_envs, 1)),
        )
        reset_rng, self.inner_rollout_rng = jax.random.split(
            self.inner_rollout_rng
        )
        reset_rngs = jnp.concatenate(
            [jax.random.split(reset_rng, self._num_envs)]
        ).reshape((self._num_envs, -1))
        # do a full rollout
        obs, env_state = env.reset(reset_rngs, self.env_params)
        rewards = [
            jnp.zeros((self._num_envs)),
            jnp.zeros((self._num_envs)),  # TOD is numopps 1?
        ]
        inner_rollout_rng, self.inner_rollout_rng = jax.random.split(
            self.inner_rollout_rng
        )
        inner_rollout_rngs = jnp.concatenate(
            [jax.random.split(inner_rollout_rng, self._num_envs)]
        ).reshape((1, self._num_envs, -1))
        # jax.debug.breakpoint()
        _, trajectories = jax.lax.scan(
            env_rollout,
            (
                inner_rollout_rngs,
                obs[0],
                obs[1],
                rewards[0],
                rewards[1],
                my_state,
                my_mem,
                other_state,
                other_mem,
                env_state,
                self.env_params,
            ),
            None,
            length=self._num_steps,  # num_inner_steps
        )

        # flip the order of the trajectories
        # assuming we're the other player
        sample = LOLASample(
            obs_self=trajectories[1].observations,
            obs_other=trajectories[0].observations,
            actions_self=trajectories[1].actions,
            actions_other=trajectories[0].actions,
            dones=trajectories[0].dones,
            rewards_self=trajectories[1].rewards,
            rewards_other=trajectories[0].rewards,
        )

        # get gradients of opponent
        gradients, _ = self.grad_fn_inner(
            self.other_state.params, my_state.params, sample
        )

        # Update the optimizer

        updates, opt_state = self._inner_optimizer.update(
            gradients, self.other_state.opt_state
        )

        # apply the optimizer updates
        params = optax.apply_updates(self.other_state.params, updates)

        # replace the other player's current parameters with a simulated update
        self.other_state = TrainingState(
            params=params,
            opt_state=opt_state,
            random_key=self.other_state.random_key,
            timesteps=self.other_state.timesteps,
        )
        self.other_mem = MemoryState(
            extras=self.other_state.extras,
            hidden=jnp.zeros((self._num_envs, 1)),
        )

    def out_lookahead(self, env, env_rollout):
        """
        Performs a real rollout using the current parameters of both agents
        and a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        other_agents: list, a list of objects of the other agents
        """
        # make my own state
        my_state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
            extras={
                "values": jnp.zeros(self._num_envs),
                "log_probs": jnp.zeros(self._num_envs),
            },
            hidden=jnp.zeros((self._num_envs, 1)),
        )
        # copy the other person's state
        other_state = TrainingState(
            params=self.other_state.params,
            opt_state=self.other_state.opt_state,
            random_key=self.other_state.random_key,
            timesteps=self.other_state.timesteps,
            extras={
                "values": jnp.zeros(self._num_envs),
                "log_probs": jnp.zeros(self._num_envs),
            },
            hidden=jnp.zeros((self._num_envs, 1)),
        )

        # do a full rollout
        t_init = env.reset()
        _, trajectories = jax.lax.scan(
            env_rollout,
            (t_init[0], t_init[1], my_state, other_state),
            None,
            length=env.episode_length,
        )

        # Now keep the same order.
        sample = LOLASample(
            obs_self=trajectories[0].observations,
            obs_other=trajectories[1].observations,
            actions_self=trajectories[0].actions,
            actions_other=trajectories[1].actions,
            dones=trajectories[0].dones,
            rewards_self=trajectories[0].rewards,
            rewards_other=trajectories[1].rewards,
        )
        # print("Before updating")
        # print("---------------------")
        # print("params", self._state.params)
        # print("opt_state", self._state.opt_state)
        # print()
        # calculate the gradients
        gradients, results = self.grad_fn_outer(
            my_state.params, other_state.params, sample
        )
        # print("Gradients", gradients)
        # print()

        # Update the optimizer
        updates, opt_state = self._outer_optimizer.update(
            gradients, my_state.opt_state
        )
        # print("Updates", updates)
        # print("Updated optimizer", opt_state)
        # print()

        # apply the optimizer updates
        params = optax.apply_updates(my_state.params, updates)

        # Update internal agent's timesteps
        self._total_steps += self._num_envs
        self._logger.metrics["total_steps"] += self._num_envs
        self._state._replace(timesteps=self._total_steps)

        # Logging
        self._logger.metrics["sgd_steps"] += 1
        self._logger.metrics["loss_total"] = results["loss_total"]
        self._logger.metrics["loss_policy"] = results["loss_policy"]
        self._logger.metrics["loss_value"] = results["loss_value"]

        # replace the player's current parameters with a real update
        self._state = TrainingState(
            params=params,
            opt_state=opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
        )
        self._mem = MemoryState(
            extras={"log_probs": None, "values": None},
            hidden=jnp.zeros((self._num_envs, 1)),
        )
        # print("After updating")
        # print("---------------------")
        # print("params", self._state.params)
        # print("opt_state", self._state.opt_state)
        # print()

    def reset_memory(self, memory, eval=False) -> TrainingState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
            hidden=jnp.zeros((self._num_envs, 1)),
        )
        return memory

    def update(self, traj_batch, t_prime: TimeStep, state, mem):
        """Update the agent -> only called at the end of a trajectory"""
        return state, mem  # TODO


def make_lola(
    args, obs_spec, action_spec, seed: int, player_id: int, env_params: Any
):
    """Make Naive Learner Policy Gradient agent"""
    # Create Haiku network
    network = make_network(action_spec)
    # Inner optimizer uses SGD
    inner_optimizer = optax.sgd(args.lola.lr_in)
    # Outer optimizer uses Adam
    outer_optimizer = optax.adam(args.lola.lr_out)
    # Random key
    random_key = jax.random.PRNGKey(seed=seed)

    return LOLA(
        network=network,
        inner_optimizer=inner_optimizer,
        outer_optimizer=outer_optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        env_params=env_params,
        player_id=player_id,
        num_envs=args.num_envs,
        num_steps=args.num_inner_steps,
        use_baseline=args.lola.use_baseline,
        gamma=args.lola.gamma,
    )


if __name__ == "__main__":
    pass
