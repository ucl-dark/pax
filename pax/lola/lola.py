from binascii import a2b_base64
from typing import Any, Mapping, NamedTuple, Tuple, Dict

from regex import R

from pax import utils
from pax.lola.buffer import TrajectoryBuffer
from pax.lola.network import make_network

from dm_env import TimeStep
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Sample(NamedTuple):
    """Object containing a batch of data"""

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


class TrainingState(NamedTuple):
    """Training state consists of network parameters, optimiser state, random key, timesteps, and extras."""

    params: hk.Params
    opt_state: optax.GradientTransformation
    random_key: jnp.ndarray
    timesteps: int
    extras: Mapping[str, jnp.ndarray]
    hidden: None


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
        num_envs: int = 4,
        num_steps: int = 150,
        num_minibatches: int = 1,
        num_epochs: int = 1,
        use_baseline: bool = True,
        gamma: float = 0.96,
    ):
        @jax.jit
        def policy(
            params: hk.Params, observation: TimeStep, state: TrainingState
        ):
            """Agent policy to select actions and calculate agent specific information"""
            key, subkey = jax.random.split(state.random_key)
            dist, values = network.apply(params, observation)
            actions = dist.sample(seed=subkey)
            state.extras["values"] = values
            state.extras["log_probs"] = dist.log_prob(actions)
            state = TrainingState(
                params=params,
                opt_state=state.opt_state,
                random_key=key,
                timesteps=state.timesteps,
                extras=state.extras,
                hidden=None,
            )
            return actions, state

        @jax.jit
        def prepare_batch(
            traj_batch: NamedTuple, t_prime: TimeStep, action_extras: dict
        ):
            # Rollouts complete -> Training begins
            # Add an additional rollout step for advantage calculation

            _value = jax.lax.select(
                t_prime.last(),
                action_extras["values"],
                jnp.zeros_like(action_extras["values"]),
            )

            _value = jax.lax.expand_dims(_value, [0])
            _reward = jax.lax.expand_dims(t_prime.reward, [0])
            _done = jax.lax.select(
                t_prime.last(),
                2 * jnp.ones_like(_value),
                jnp.zeros_like(_value),
            )

            # need to add final value here
            traj_batch = traj_batch._replace(
                behavior_values=jnp.concatenate(
                    [traj_batch.behavior_values, _value], axis=0
                )
            )
            traj_batch = traj_batch._replace(
                rewards=jnp.concatenate([traj_batch.rewards, _reward], axis=0)
            )
            traj_batch = traj_batch._replace(
                dones=jnp.concatenate([traj_batch.dones, _done], axis=0)
            )
            return traj_batch

        def loss(params, other_params, samples):
            # Stacks so that the dimension is now (num_envs, num_steps)

            obs_1 = samples.obs_self
            obs_2 = samples.obs_other

            rewards = samples.rewards_self
            # r_1 = samples.rewards_self
            # r_2 = samples.rewards_other

            actions_1 = samples.actions_self
            actions_2 = samples.actions_other

            # distribution, values_self = self.network.apply(params, obs_1)
            distribution, values = self.network.apply(params, obs_1)
            self_log_prob = distribution.log_prob(actions_1)

            distribution, values_others = self.network.apply(
                other_params, obs_2
            )
            other_log_prob = distribution.log_prob(actions_2)

            # apply discount:
            cum_discount = (
                jnp.cumprod(self.gamma * jnp.ones(rewards.shape), axis=1)
                / self.gamma
            )
            discounted_rewards = rewards * cum_discount
            discounted_values = values * cum_discount

            # stochastics nodes involved in rewards dependencies:
            dependencies = jnp.cumsum(self_log_prob + other_log_prob, axis=1)

            # logprob of each stochastic nodes:
            stochastic_nodes = self_log_prob + other_log_prob

            # dice objective:
            dice_objective = jnp.mean(
                jnp.sum(magic_box(dependencies) * discounted_rewards, axis=1)
            )

            if use_baseline:
                # variance_reduction:
                baseline_term = jnp.mean(
                    jnp.sum(
                        (1 - magic_box(stochastic_nodes)) * discounted_values,
                        axis=1,
                    )
                )
                dice_objective = dice_objective + baseline_term

            loss_value = jnp.mean((rewards - values) ** 2)
            # loss_total = -dice_objective + loss_value
            loss_total = dice_objective + loss_value

            # want to minimize -objective
            # return loss_total, {
            #     "loss_total": -dice_objective + loss_value,
            #     "loss_policy": -dice_objective,
            #     "loss_value": loss_value,
            # }
            return loss_total, {
                "loss_total": dice_objective + loss_value,
                "loss_policy": dice_objective,
                "loss_value": loss_value,
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

        def make_initial_state(key: Any, obs_spec: Tuple) -> TrainingState:
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
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
                hidden=None,
            )

        # Initialise training state (parameters, optimiser state, extras).
        self._state = make_initial_state(random_key, obs_spec)

        # Setup player id
        self.player_id = player_id

        # Initialize buffer and sgd
        self._trajectory_buffer = TrajectoryBuffer(
            num_envs, num_steps, obs_spec
        )

        # self.grad_fn = jax.grad(loss, has_aux=True)
        self.grad_fn = jax.jit(jax.grad(loss, has_aux=True))

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
        self._prepare_batch = jax.jit(prepare_batch)
        self._sgd_step = sgd_step

        # initialize some variables
        self._inner_optimizer = inner_optimizer
        self._outer_optimizer = outer_optimizer
        self.gamma = gamma

        # Other useful hyperparameters
        self._num_envs = num_envs  # number of environments
        self._num_steps = num_steps  # number of steps per environment
        self._batch_size = int(num_envs * num_steps)  # number in one batch
        self._num_minibatches = num_minibatches  # number of minibatches
        self._num_epochs = num_epochs  # number of epochs to use sample
        self._obs_spec = obs_spec

    def select_action(self, t: TimeStep):
        """Selects action and updates info with PPO specific information"""
        actions, self._state = self._policy(
            self._state.params, t.observation, self._state
        )
        return utils.to_numpy(actions)

    def in_lookahead(self, env, other_agents, env_rollout):
        """
        Performs a rollout using the current parameters of both agents
        and simulates a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        other_agents: list, a list of objects of the other agents
        """

        # get other agent
        other_agent = other_agents[0]

        # my state
        my_state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
            extras={
                "values": jnp.zeros(self._num_envs),
                "log_probs": jnp.zeros(self._num_envs),
            },
            hidden=None,
        )
        # other player's state
        other_state = other_agent.reset_memory()

        # do a full rollout
        t_init = env.reset()
        vals, trajectories = jax.lax.scan(
            env_rollout,
            (t_init[0], t_init[1], my_state, other_state),
            None,
            length=env.episode_length,
        )

        # update agent / add final trajectory
        # do an agent update based on trajectories
        # only need the other agent's state
        # unpack the values from the buffer

        # update agent / add final trajectory
        # TODO: Unclear if this is still needed when not calculating advantage?
        # final_t1 = vals[0]._replace(step_type=2)
        # a1_state = vals[2]
        # _, state = self._policy(my_state.params, final_t1.observation, a1_state)
        # traj_batch_0 = self._prepare_batch(trajectories[0], final_t1, state.extras)

        # final_t2 = vals[1]._replace(step_type=2)
        # a2_state = vals[3]
        # _, state = self._policy(other_state.params, final_t2.observation, a2_state)
        # traj_batch_1 = self._prepare_batch(trajectories[1], final_t2, other_state.extras)

        traj_batch_0 = trajectories[0]
        traj_batch_1 = trajectories[1]
        # flip the order of the trajectories
        # assuming we're the other player
        sample = Sample(
            obs_self=traj_batch_1.observations,
            obs_other=traj_batch_0.observations,
            actions_self=traj_batch_1.actions,
            actions_other=traj_batch_0.actions,
            dones=traj_batch_0.dones,
            rewards_self=traj_batch_1.rewards,
            rewards_other=traj_batch_0.rewards,
        )

        # get gradients of opponent
        gradients, _ = self.grad_fn(
            other_state.params, my_state.params, sample
        )

        # Update the optimizer
        updates, opt_state = self._inner_optimizer.update(
            gradients, other_state.opt_state
        )

        # apply the optimizer updates
        params = optax.apply_updates(other_state.params, updates)

        # replace the other player's current parameters with a simulated update
        self._other_state = TrainingState(
            params=params,
            opt_state=opt_state,
            random_key=other_state.random_key,
            timesteps=other_state.timesteps,
            extras=other_state.extras,
            hidden=None,
        )

    def out_lookahead(self, env, env_rollout):
        """
        Performs a real rollout using the current parameters of both agents
        and a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        other_agents: list, a list of objects of the other agents
        """
        # get a copy of the agent's state
        my_state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
            extras={
                "values": jnp.zeros(self._num_envs),
                "log_probs": jnp.zeros(self._num_envs),
            },
            hidden=None,
        )
        # get a copy of the other opponent's state
        # TODO: Do I need to reset this? Maybe...
        other_state = self._other_state

        # do a full rollout
        t_init = env.reset()
        vals, trajectories = jax.lax.scan(
            env_rollout,
            (t_init[0], t_init[1], my_state, other_state),
            None,
            length=env.episode_length,
        )

        traj_batch_0 = trajectories[0]
        traj_batch_1 = trajectories[1]

        # Now keep the same order.
        sample = Sample(
            obs_self=traj_batch_0.observations,
            obs_other=traj_batch_1.observations,
            actions_self=traj_batch_0.actions,
            actions_other=traj_batch_1.actions,
            dones=traj_batch_0.dones,
            rewards_self=traj_batch_0.rewards,
            rewards_other=traj_batch_1.rewards,
        )

        # calculate the gradients
        gradients, results = self.grad_fn(
            my_state.params, other_state.params, sample
        )

        # Update the optimizer
        updates, opt_state = self._outer_optimizer.update(
            gradients, my_state.opt_state
        )

        # apply the optimizer updates
        params = optax.apply_updates(my_state.params, updates)

        # Update internal agent's timesteps
        self._total_steps += self._num_envs
        self._logger.metrics["total_steps"] += self._num_envs
        self._state._replace(timesteps=self._total_steps)

        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        self._logger.metrics["loss_total"] = results["loss_total"]
        self._logger.metrics["loss_policy"] = results["loss_policy"]
        self._logger.metrics["loss_value"] = results["loss_value"]

        # replace the other player's current parameters with a simulated update
        self._state = TrainingState(
            params=params,
            opt_state=opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
            extras={"log_probs": None, "values": None},
            hidden=None,
        )

    def reset_memory(self) -> TrainingState:
        self._state = self._state._replace(
            extras={
                "values": jnp.zeros(self._num_envs),
                "log_probs": jnp.zeros(self._num_envs),
            }
        )
        return self._state

    def update(
        self,
        traj_batch,
        t_prime: TimeStep,
        state,
    ):
        """Update the agent -> only called at the end of a trajectory"""
        _, state = self._policy(state.params, t_prime.observation, state)

        traj_batch = self._prepare_batch(traj_batch, t_prime, state.extras)
        state, results = self._sgd_step(state, traj_batch)
        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        self._logger.metrics["loss_total"] = results["loss_total"]
        self._logger.metrics["loss_policy"] = results["loss_policy"]
        self._logger.metrics["loss_value"] = results["loss_value"]
        self._logger.metrics["loss_entropy"] = results["loss_entropy"]
        self._logger.metrics["entropy_cost"] = results["entropy_cost"]

        self._state = state
        return state


def make_lola(args, obs_spec, action_spec, seed: int, player_id: int):
    """Make Naive Learner Policy Gradient agent"""

    network = make_network(action_spec)

    inner_optimizer = optax.chain(
        optax.scale_by_adam(eps=args.lola.adam_epsilon),
        optax.scale(-args.lola.lr_in),
    )
    outer_optimizer = optax.chain(
        optax.scale_by_adam(eps=args.lola.adam_epsilon),
        optax.scale(-args.lola.lr_out),
    )

    # Random key
    random_key = jax.random.PRNGKey(seed=seed)

    return LOLA(
        network=network,
        inner_optimizer=inner_optimizer,
        outer_optimizer=outer_optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        player_id=player_id,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.ppo.num_minibatches,
        num_epochs=args.ppo.num_epochs,
        use_baseline=args.lola.use_baseline,
        gamma=args.lola.gamma,
    )


if __name__ == "__main__":
    pass
