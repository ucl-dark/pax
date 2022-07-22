# Adapted from https://github.com/deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py

from typing import Any, Mapping, NamedTuple, Tuple, Dict

from pax import utils
from pax.lola.buffer import TrajectoryBuffer
from pax.lola.network import make_network

from dm_env import TimeStep
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


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


class Hp:
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.lr_v = 0.1
        self.gamma = 0.96
        self.n_update = 200
        self.len_rollout = 150
        self.batch_size = 128
        self.use_baseline = True
        self.seed = 42


hp = Hp()


def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))


class Logger:
    metrics: dict


# class Memory:
#     def __init__(self):
#         self.self_logprobs = []
#         self.other_logprobs = []
#         self.values = []
#         self.rewards = []

#     def add(self, lp, other_lp, v, r):
#         self.self_logprobs.append(lp)
#         self.other_logprobs.append(other_lp)
#         self.values.append(v)
#         self.rewards.append(r)

#     def dice_objective(self):
#         self_logprobs = jnp.stack(self.self_logprobs, dim=1)
#         other_logprobs = jnp.stack(self.other_logprobs, dim=1)
#         values = jnp.stack(self.values, dim=1)
#         rewards = jnp.stack(self.rewards, dim=1)

#         # apply discount:
#         cum_discount = (
#             jnp.cumprod(hp.gamma * jnp.ones(*rewards.size()), dim=1) / hp.gamma
#         )
#         discounted_rewards = rewards * cum_discount
#         discounted_values = values * cum_discount

#         # stochastics nodes involved in rewards dependencies:
#         dependencies = jnp.cumsum(self_logprobs + other_logprobs, dim=1)

#         # logprob of each stochastic nodes:
#         stochastic_nodes = self_logprobs + other_logprobs

#         # dice objective:
#         dice_objective = jnp.mean(
#             jnp.sum(magic_box(dependencies) * discounted_rewards, dim=1)
#         )

#         if hp.use_baseline:
#             # variance_reduction:
#             baseline_term = jnp.mean(
#                 jnp.sum(
#                     (1 - magic_box(stochastic_nodes)) * discounted_values,
#                     dim=1,
#                 )
#             )
#             dice_objective = dice_objective + baseline_term

#         return -dice_objective  # want to minimize -objective

#     def value_loss(self):
#         values = jnp.stack(self.values, dim=1)
#         rewards = jnp.stack(self.rewards, dim=1)
#         return jnp.mean((rewards - values) ** 2)


# ipd = IPD(hp.len_rollout, hp.batch_size)


# def act(batch_states, theta, values):
#     batch_states = jnp.from_numpy(batch_states).long()
#     probs = jnp.sigmoid(theta)[batch_states]
#     m = Bernoulli(1 - probs)
#     actions = m.sample()
#     log_probs_actions = m.log_prob(actions)
#     return actions.numpy().astype(int), log_probs_actions, values[batch_states]


# def get_gradient(objective, theta):
#     # create differentiable gradient for 2nd orders:
#     grad_objective = torch.autograd.grad(
#         objective, (theta), create_graph=True
#     )[0]
#     return grad_objective


# def step(theta1, theta2, values1, values2):
#     # just to evaluate progress:
#     (s1, s2), _ = ipd.reset()
#     score1 = 0
#     score2 = 0
#     for t in range(hp.len_rollout):
#         a1, lp1, v1 = act(s1, theta1, values1)
#         a2, lp2, v2 = act(s2, theta2, values2)
#         (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
#         # cumulate scores
#         score1 += np.mean(r1) / float(hp.len_rollout)
#         score2 += np.mean(r2) / float(hp.len_rollout)
#     return (score1, score2)


class LOLA:
    """LOLA with the DiCE objective function."""

    def __init__(
        self,
        network: NamedTuple,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        player_id: int,
        obs_spec: Tuple,
        num_envs: int = 4,
        num_steps: int = 150,
        num_minibatches: int = 1,
        num_epochs: int = 1,
        gamma: float = 0.96,
    ):

        # @jax.jit
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
            )
            return actions, state

        def rollouts(
            buffer: TrajectoryBuffer,
            t: TimeStep,
            actions: np.array,
            t_prime: TimeStep,
            state: TrainingState,
        ) -> None:
            """Stores rollout in buffer"""
            log_probs, values = (
                state.extras["log_probs"],
                state.extras["values"],
            )
            buffer.add(t, actions, log_probs, values, t_prime)

        def loss(
            logprobs: jnp.ndarray,
            other_logprobs: jnp.ndarray,
            values: jnp.ndarray,
            rewards: jnp.ndarray,
        ):
            logprobs = jnp.stack(logprobs, dim=1)
            other_logprobs = jnp.stack(other_logprobs, dim=1)
            values = jnp.stack(self.values, dim=1)
            rewards = jnp.stack(self.rewards, dim=1)

            # apply discount:
            cum_discount = (
                jnp.cumprod(gamma * jnp.ones(*rewards.size()), dim=1) / gamma
            )
            discounted_rewards = rewards * cum_discount
            discounted_values = values * cum_discount

            # stochastics nodes involved in rewards dependencies:
            dependencies = jnp.cumsum(logprobs + other_logprobs, dim=1)

            # logprob of each stochastic nodes:
            stochastic_nodes = logprobs + other_logprobs

            # dice objective:
            dice_objective = jnp.mean(
                jnp.sum(magic_box(dependencies) * discounted_rewards, dim=1)
            )

            baseline_term = jnp.mean(
                jnp.sum(
                    (1 - magic_box(stochastic_nodes)) * discounted_values,
                    dim=1,
                )
            )
            dice_objective = dice_objective + baseline_term

            return -dice_objective, {}  # want to minimize -objective

        @jax.jit
        def sgd_step(
            state: TrainingState,
            other_agent_params: hk.Params,
            sample: NamedTuple,
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract data
            (
                observations,
                actions,
                rewards,
                behavior_log_probs,
                behavior_values,
                dones,
            ) = (
                sample.observations,
                sample.actions,
                sample.rewards,
                sample.behavior_log_probs,
                sample.behavior_values,
                sample.dones,
            )

            # vmap
            # TODO: REMOVE THIS INEA IFNEF IAEIF AE
            def gae_advantages():
                pass

            batch_gae_advantages = jax.vmap(gae_advantages, in_axes=0)
            advantages, target_values = batch_gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_envs, num_steps, ..]
            (
                observations,
                actions,
                behavior_log_probs,
                behavior_values,
            ) = jax.tree_map(
                lambda x: x[:, :-1],
                (observations, actions, behavior_log_probs, behavior_values),
            )

            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            # Concatenate all trajectories. Reshape from [num_envs, num_steps, ..]
            # to [num_envs * num_steps,..]
            assert len(target_values.shape) > 1
            num_envs = target_values.shape[0]
            num_steps = target_values.shape[1]
            batch_size = num_envs * num_steps
            assert batch_size % num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            # Compute gradients.
            grad_fn = jax.grad(loss, has_aux=True)

            def model_update_minibatch(
                carry: Tuple[hk.Params, optax.OptState, int],
                minibatch: Batch,
            ) -> Tuple[
                Tuple[hk.Params, optax.OptState, int], Dict[str, jnp.ndarray]
            ]:
                """Performs model update for a single minibatch."""
                params, opt_state, timesteps = carry
                # Normalize advantages at the minibatch level before using them.
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
                )

                # Apply updates
                updates, opt_state = optimizer.update(gradients, opt_state)
                params = optax.apply_updates(params, updates)

                metrics["norm_grad"] = optax.global_norm(gradients)
                metrics["norm_updates"] = optax.global_norm(updates)
                return (params, opt_state, timesteps), metrics

            def model_update_epoch(
                carry: Tuple[
                    jnp.ndarray, hk.Params, optax.OptState, int, Batch
                ],
                unused_t: Tuple[()],
            ) -> Tuple[
                Tuple[jnp.ndarray, hk.Params, optax.OptState, Batch],
                Dict[str, jnp.ndarray],
            ]:
                """Performs model updates based on one epoch of data."""
                key, params, opt_state, timesteps, batch = carry
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                shuffled_batch = jax.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_map(
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

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            # signature is scan(function, carry, tuple to iterate over, length)
            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (state.random_key, params, opt_state, timesteps, batch),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(
                jnp.abs(jnp.mean(rewards, axis=(0, 1)))
            )
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps,
                extras={"log_probs": None, "values": None},
            )

            return new_state, metrics

        def make_initial_state(key: Any, obs_spec: Tuple) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                params=initial_params,
                opt_state=initial_opt_state,
                random_key=key,
                timesteps=0,
                extras={"values": None, "log_probs": None},
            )

        # def in_lookahead(self, other_theta, other_values):
        #     (s1, s2), _ = ipd.reset()
        #     other_memory = Memory()
        #     for t in range(hp.len_rollout):
        #         a1, lp1, v1 = act(s1, self.theta, self.values)
        #         a2, lp2, v2 = act(s2, other_theta, other_values)
        #         (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
        #         other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        #     other_objective = other_memory.dice_objective()
        #     grad = get_gradient(other_objective, other_theta)
        #     return grad

        # Initialise training state (parameters, optimiser state, extras).
        self._state = make_initial_state(random_key, obs_spec)

        # Setup player id
        self.player_id = player_id

        # Initialize buffer and sgd
        self._trajectory_buffer = TrajectoryBuffer(
            num_envs, num_steps, obs_spec
        )
        self._sgd_step = sgd_step

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
        self._rollouts = rollouts

        # Other useful hyperparameters
        self._num_envs = num_envs  # number of environments
        self._num_steps = num_steps  # number of steps per environment
        self._batch_size = int(num_envs * num_steps)  # number in one batch
        self._num_minibatches = num_minibatches  # number of minibatches
        self._num_epochs = num_epochs  # number of epochs to use sample

    def select_action(self, t: TimeStep):
        """Selects action and updates info with PPO specific information"""
        actions, self._state = self._policy(
            self._state.params, t.observation, self._state
        )
        return utils.to_numpy(actions)

    # def lookahead(self, env, other_agents):
    #     """
    #     Performs a rollout using the current parameters of both agents
    #     and simulates a naive learning update step for the other agent

    #     INPUT:
    #     env: SequentialMatrixGame, an environment object of the game being played
    #     other_agents: list, a list of objects of the other agents
    #     """
    #     t = env.reset()
    #     other_memory = Memory()
    #     for t in range(hp.len_rollout):
    #         actions, self.w = self._policy(
    #             self._state.params, t.observation, self._state
    #         )
    #         a1, lp1, v1 = act(s1, self.theta, self.values)
    #         a2, lp2, v2 = act(s2, other_theta, other_values)
    #         (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
    #         other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

    #     other_objective = other_memory.dice_objective()
    #     grad = get_gradient(other_objective, other_theta)
    #     return grad

    # pass

    def update(
        self,
        t: TimeStep,
        actions: np.array,
        t_prime: TimeStep,
        other_agents: list = None,
    ):
        # Adds agent and environment info to buffer
        self._rollouts(
            buffer=self._trajectory_buffer,
            t=t,
            actions=actions,
            t_prime=t_prime,
            state=self._state,
        )

        # Log metrics
        self._total_steps += self._num_envs
        self._logger.metrics["total_steps"] += self._num_envs

        # Update internal state with total_steps
        self._state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._state.random_key,
            timesteps=self._total_steps,
            extras=self._state.extras,
        )

        # Update counter until doing SGD
        self._until_sgd += 1

        # Add params to buffer
        # doesn't change throughout the rollout
        # but it can't be before the return
        self._trajectory_buffer.params = self._state.params

        # Rollouts onging
        if self._until_sgd % (self._num_steps) != 0:
            return

        # Rollouts complete -> Training begins
        # Add an additional rollout step for advantage calculation
        _, self._state = self._policy(
            self._state.params, t_prime.observation, self._state
        )
        # print("Other agents params", other_agents[0]._trajectory_buffer.params)

        self._trajectory_buffer.add(
            timestep=t_prime,
            action=0,
            log_prob=0,
            value=self._state.extras["values"]
            if not t_prime.last()
            else jnp.zeros_like(self._state.extras["values"]),
            new_timestep=t_prime,
        )

        # other_agent = other_agents[0]
        # other_agent_params = other_agents[0]._trajectory_buffer.params
        # It needs to be able to take in the opponents parameters
        # and then do a rollout under those parameters
        # could do sgd here for other agent?
        # sample = self._trajectory_buffer.sample()
        # self._state, results = self._sgd_step(
        #     self._state, other_agent_params, sample
        # )
        # self._logger.metrics["sgd_steps"] += (
        #     self._num_minibatches * self._num_epochs
        # )
        # self._logger.metrics["loss_total"] = results["loss_total"]
        # self._logger.metrics["loss_policy"] = results["loss_policy"]
        # self._logger.metrics["loss_value"] = results["loss_value"]


def make_lola(args, obs_spec, action_spec, seed: int, player_id: int):
    """Make Naive Learner Policy Gradient agent"""

    # print(f"Making network for {args.env_id}")
    network = make_network(action_spec)

    # Optimizer
    # batch_size = int(args.num_envs * args.num_steps)
    # transition_steps = (
    #     args.total_timesteps
    #     / batch_size
    #     * args.ppo.num_epochs
    #     * args.ppo.num_minibatches
    # )

    optimizer = optax.chain(
        optax.scale_by_adam(eps=args.ppo.adam_epsilon),
        optax.scale(-args.ppo.learning_rate),
    )

    # Random key
    random_key = jax.random.PRNGKey(seed=seed)

    return LOLA(
        network=network,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        player_id=player_id,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.ppo.num_minibatches,
        num_epochs=args.ppo.num_epochs,
        gamma=args.ppo.gamma,
        gae_lambda=args.ppo.gae_lambda,
    )


if __name__ == "__main__":
    pass
