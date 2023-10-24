from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int


@chex.dataclass
class EnvParams:
    payoff_table: chex.ArrayDevice


class IteratedTensorGameNPlayer(environment.Environment):
    env_id = "iterated_nplayer_tensor_game"
    """
    JAX Compatible version of tensor game environment.
    """

    def __init__(
        self, num_players: int, num_inner_steps: int, num_outer_steps: int
    ):
        super().__init__()
        self.num_players = num_players

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, ...],
            params: EnvParams,
        ):
            assert len(actions) == num_players
            inner_t, outer_t = state.inner_t, state.outer_t
            inner_t += 1
            reset_inner = inner_t == num_inner_steps

            # sum of 1s is number of defectors
            num_defect = sum(list(actions)).astype(jnp.int8)

            # calculate rewards
            # row of payoff table is number of defectors
            # column of payoff table is whether player defected or not
            payoff_array = jnp.array(params.payoff_table, dtype=jnp.float32)
            assert payoff_array.shape == (num_players + 1, 2)
            action_array = jnp.array(actions, dtype=jnp.int8)
            relevant_row = payoff_array[num_defect]

            rewards_array = relevant_row[action_array]
            all_rewards = tuple(rewards_array)
            # calculate states
            # we want these to be from first person perspective
            # eg as if players were sitting in a circle
            # and they start with theirs, then go clockwise

            # list of first person perspective actions
            # with each player starting with theirs
            fpp_actions = []
            for i in range(num_players):
                fpp_action = actions[i:] + actions[:i]
                fpp_action = jnp.array(fpp_action)
                fpp_actions.append(fpp_action)

            # Start is the last state eg 2**num_players_th
            start_state_idx = 2**num_players
            len_one_hot = 2**num_players + 1
            # binary to int conversion for decoding actions
            b2i = 2 ** jnp.arange(num_players - 1, -1, -1)
            all_obs = []
            for i in range(num_players):
                # first state is all c...cc
                # second state is all c...cd
                # 2**num_playerth state is d...dd
                # so we can just binary decode the actions to get state
                obs = (fpp_actions[i] * b2i).sum()
                # if first step then return START state.
                obs = jax.lax.select(
                    reset_inner,
                    start_state_idx * jnp.ones_like(obs),
                    obs,
                )
                # one hot encode
                obs = jax.nn.one_hot(obs, len_one_hot, dtype=jnp.int8)
                all_obs.append(obs)
            all_obs = tuple(all_obs)
            # out step keeping
            inner_t = jax.lax.select(
                reset_inner, jnp.zeros_like(inner_t), inner_t
            )
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            reset_outer = outer_t == num_outer_steps
            state = EnvState(inner_t=inner_t, outer_t=outer_t)

            return (
                all_obs,
                state,
                all_rewards,
                reset_outer,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple[chex.Array, ...], EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            # start state is the last one eg 2**num_players_th
            # and one hot vectors should be 2**num_players+1 long
            obs = jax.nn.one_hot(
                (2**num_players) * jnp.ones(()),
                2**num_players + 1,
                dtype=jnp.int8,
            )
            all_obs = tuple([obs for _ in range(num_players)])
            return all_obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "IteratedTensorGame-Nplayer"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        """Observation space of the environment."""
        obs_space = jnp.power(2, self.num_players) + 1
        return spaces.Discrete(obs_space)
