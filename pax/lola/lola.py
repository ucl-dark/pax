# Learning with Opponent-Learning Awareness (LOLA) implementation in JAX
# https://arxiv.org/pdf/1709.04326.pdf

from typing import NamedTuple, Any, Mapping
from pax.lola.buffer import Replay
from pax.lola.network import make_network
from dm_env import TimeStep
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class TrainingState(NamedTuple):
    params: hk.Params
    random_key: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class LOLA:
    """Implements LOLA with policy gradient"""

    # TODO
    # 1. implement policy gradient
    # 2. Implement loss function
    # 3. Implement storing parameters in the buffer so that both agents can access
    # 4. Second order LOLA term

    def __init__(
        self,
        network: hk.Params,
        random_key: jnp.ndarray,
        player_id: int,
        replay_capacity: int = 100000,
        min_replay_size: int = 1000,
        sgd_period: int = 1,
        batch_size: int = 256,
        obs_spec: tuple = (5,),
        num_envs: int = 4,
        num_steps: int = 500,
    ):
        def policy(
            params: hk.Params, observation: jnp.ndarray, state: TrainingState
        ):
            """Determines how to choose an action"""
            # key, subkey = jax.random.split(state.random_key)
            # logits = network.apply(params, observation)
            # actions = jax.random.categorical(subkey, logits)
            # state = state._replace(random_key=key)
            # return actions, state
            key, subkey = jax.random.split(state.random_key)
            dist, values = network.apply(params, observation)
            actions = dist.sample(seed=subkey)
            state.extras["values"] = values
            state.extras["log_probs"] = dist.log_prob(actions)
            state = TrainingState(
                params=params,
                random_key=key,
                extras=state.extras,
            )
            return actions, state

        def rollouts(
            buffer: Replay,
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

        def loss():
            """Loss function"""
            # see pg. 4 of the LOLA paper
            pass

        def sgd():
            """Stochastic gradient descent"""
            pass

        def make_initial_state(key: jnp.ndarray) -> TrainingState:
            """Make initial training state for LOLA"""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=(1, 5))
            params = network.init(subkey, dummy_obs)
            return TrainingState(
                params=params,
                random_key=key,
                extras={"values": None, "log_probs": None},
            )

        # init buffer
        self._buffer = Replay(num_envs, num_steps, replay_capacity, obs_spec)

        # init functions
        self._state = make_initial_state(random_key)
        self._policy = policy
        self._rollouts = rollouts
        self._sgd_step = sgd

        # init constants
        self.player_id = player_id
        self._min_replay_size = min_replay_size
        self._sgd_period = sgd_period
        self._batch_size = batch_size

        # init variables
        self._total_steps = 0

        # self._trajectory_buffer = TrajectoryBuffer(
        #     num_envs, num_steps, obs_spec
        # )
        # TODO: get the correct parameters for this.
        # self._trajectory_buffer = Replay(
        #     num_envs, num_steps, obs_spec
        # )

    def select_action(self, t: TimeStep):
        """Select action based on observation"""
        # Unpack
        params = self._state.params
        state = self._state
        action, self._state = self._policy(params, t.observation, state)
        return action

    def update(
        self,
        t: TimeStep,
        action: jnp.ndarray,
        t_prime: TimeStep,
        other_agents: list = None,
    ):

        self._rollouts(
            buffer=self._buffer,
            t=t,
            actions=action,
            t_prime=t_prime,
            state=self._state,
        )

        self._total_steps += 1
        if self._total_steps % self._sgd_period != 0:
            return

        print(self._buffer.size())
        if self._buffer.size() < self._min_replay_size:
            return

        # Do a batch of SGD.
        sample, key = self._buffer.sample(
            self._batch_size, self._state.random_key
        )
        print(sample)
        self._state = self._state._replace(random_key=key)
        # self._state = self._sgd_step(self._state, transitions)

        """Update agent"""
        # for agent in other_agents:
        #     pass
        # other_agent_parameters = agent._trajectory_buffer.parameters
        # # print(f"other_agent_parameters: {other_agent_parameters}")
        # print(
        #     f"other_agent_parameters shape: {other_agent_parameters.shape}"
        # )


# TODO: Add args argument
def make_lola(args, obs_spec, action_spec, seed: int, player_id: int) -> LOLA:
    """Instantiate LOLA"""
    random_key = jax.random.PRNGKey(seed)

    # def forward(inputs):
    #     """Forward pass for LOLA"""
    #     values = hk.Linear(2, with_bias=False)
    #     return values(inputs)

    # network = hk.without_apply_rng(hk.transform(forward))

    network = make_network(action_spec)

    return LOLA(
        network=network,
        random_key=random_key,
        player_id=player_id,
        replay_capacity=100000,  # args.lola.replay_capacity,
        min_replay_size=50,  # args.lola.min_replay_size,
        sgd_period=1,  # args.dqn.sgd_period,
        batch_size=2,  # args.dqn.batch_size
        obs_spec=(5,),  # obs_spec
        num_envs=2,  # num_envs=args.num_envs,
        num_steps=4,  # num_steps=args.num_steps,
    )


if __name__ == "__main__":
    lola = make_lola(seed=0)
    print(f"LOLA state: {lola._state}")
    timestep = TimeStep(
        step_type=0,
        reward=1,
        discount=1,
        observation=jnp.array([[1, 0, 0, 0, 0]]),
    )
    action = lola.select_action(timestep)
    print("Action", action)
    timestep = TimeStep(
        step_type=0, reward=1, discount=1, observation=jnp.zeros(shape=(1, 5))
    )
    action = lola.select_action(timestep)
    print("Action", action)
