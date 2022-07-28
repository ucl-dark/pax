from binascii import a2b_base64
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
            )
            return actions, state

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
            loss_total = -dice_objective + loss_value

            # want to minimize -objective
            return loss_total, {
                "loss_total": -dice_objective + loss_value,
                "loss_policy": -dice_objective,
                "loss_value": loss_value,
            }

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
                extras={"values": None, "log_probs": None},
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

    def in_lookahead(self, env, other_agents):
        """
        Performs a rollout using the current parameters of both agents
        and simulates a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        other_agents: list, a list of objects of the other agents
        """

        # initialize buffer
        other_memory = TrajectoryBuffer(
            self._num_envs, self._num_steps, self._obs_spec
        )

        # get the other agent
        other_agent = other_agents[0]
        # get a copy of the agent's state
        my_state = self._state
        # create a temporary training state for the other agent for the simulated rollout
        # TODO: make a separate optimizer for the other agent
        other_state = TrainingState(
            params=other_agent._state.params,
            opt_state=other_agent._state.opt_state,
            random_key=other_agent._state.random_key,
            timesteps=other_agent._state.timesteps,
            extras=other_agent._state.extras,
        )

        # Perform a rollout and store S,A,R,S tuples
        # TODO: Replace with jax.lax.scan
        t = env.reset()
        for _ in range(self._num_steps):
            # I take an action using my parameters
            a1, my_state = self._policy(
                my_state.params, t[0].observation, my_state
            )
            # Opponent takes an action using their parameters
            a2, other_state = self._policy(
                other_state.params, t[1].observation, other_state
            )
            # The environment steps as usual
            t_prime = env.step((a1, a2))
            # We add to the buffer as if we are the other player
            # TODO: IS THIS THE RIGHT ORDER???????
            other_memory.add(t[1], t[0], a2, a1, t_prime[1], t_prime[0])
            t = t_prime

        # unpack the values from the buffer
        sample = other_memory.sample()

        # calculate the gradients
        gradients, _ = self.grad_fn(
            other_state.params, my_state.params, sample
        )

        # Update the optimizer
        updates, opt_state = self._inner_optimizer.update(
            gradients, other_state.opt_state
        )

        # apply the optimizer updates
        # params =
        params = optax.apply_updates(other_state.params, updates)
        # replace the other player's current parameters with a simulated update
        # print("Params before replacement", params)
        self._other_state = TrainingState(
            params=params,
            opt_state=opt_state,
            random_key=other_state.random_key,
            timesteps=other_state.timesteps,
            extras=other_state.extras,
        )
        # other_state._replace(params=params)
        # print("Params after replacement", other_state.params)
        # other_state._replace(opt_state=opt_state)
        # self._other_state = other_state
        # print("Params after self.", self._other_state.params)

    def out_lookahead(self, env, other_agents):
        """
        Performs a real rollout using the current parameters of both agents
        and a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        other_agents: list, a list of objects of the other agents
        """

        # initialize buffer
        memory = TrajectoryBuffer(
            self._num_envs, self._num_steps, self._obs_spec
        )

        # get a copy of the agent's state
        my_state = self._state
        # Perform a rollout and store S,A,R,S tuples
        t = env.reset()
        other_state = self._other_state
        # TODO: Replace with jax.lax.scan
        for _ in range(self._num_steps):
            # I take an action using my parameters
            a1, my_state = self._policy(
                my_state.params, t[0].observation, my_state
            )
            # Opponent takes an action using their parameters
            a2, other_state = self._policy(
                other_state.params, t[1].observation, other_state
            )
            # The environment steps as usual
            t_prime = env.step((a1, a2))
            # We add to the buffer
            # TODO: IS THIS THE RIGHT ORDER???????
            memory.add(t[0], t[1], a1, a2, t_prime[0], t_prime[1])
            t = t_prime

        # Update internal agent's timesteps
        self._total_steps += self._num_envs
        self._logger.metrics["total_steps"] += self._num_envs
        self._state._replace(timesteps=self._total_steps)

        # unpack the values from the buffer
        sample = memory.sample()

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
        )

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
        sample = self._trajectory_buffer.sample()
        self._state, results = self._sgd_step(
            self._state, self.other_params, sample
        )
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
