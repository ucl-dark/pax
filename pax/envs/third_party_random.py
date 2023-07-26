from enum import IntEnum
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int


@chex.dataclass
class EnvParams:
    payoff_table: chex.ArrayDevice
    punishment: chex.ArrayDevice
    intrinsic: chex.ArrayDevice
    punish_cost: chex.ArrayDevice


class Actions(IntEnum):
    # first opponent, second opponent, punishment of previous game
    C_no_P = 0
    C_P1 = 1
    C_P2 = 2
    C_both_P = 3

    D_no_P = 4
    D_P1 = 5
    D_P2 = 6
    D_both_P = 7


class ThirdPartyRandom(environment.Environment):
    """
    JAX Compatible version of tensor game environment.
    """

    def __init__(self, num_inner_steps: int, num_outer_steps: int):
        super().__init__()

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            prev_actions: Tuple[int, ...],  # previous actions
            player_selection: Tuple[
                int, ...
            ],  # player 1, player 2, punisher idx
            curr_actions: Tuple[int, ...],  # this is the new actions
            params: EnvParams,
        ):
            num_players = 3
            # 3 players
            assert len(prev_actions) == num_players
            assert len(curr_actions) == num_players
            inner_t, outer_t = state.inner_t, state.outer_t
            inner_t += 1
            reset_inner = inner_t == num_inner_steps

            prev_actions_arr = jnp.array(prev_actions)

            prev_cd_actions = (
                jnp.where(  # all previous cd actions, 0 for c, 1 for d
                    prev_actions_arr > 3, 1, 0
                )
            )

            # we care about the current punishment actions
            actions_arr = jnp.array(curr_actions)
            punish_actions = (
                actions_arr
            ) % 4  # 0 if no punishment, 1 if punish 1, 2 if punish 2, 3 if punish both
            curr_cd_actions = jnp.where(actions_arr > 3, 1, 0)
            # select order of players to be random
            new_player_selection = jax.random.permutation(key, jnp.arange(3))
            curr_game_actions = (
                curr_cd_actions[new_player_selection[0]],
                curr_cd_actions[new_player_selection[1]],
            )

            old_pl1, old_pl2, old_punisher = player_selection
            old_chosen_game_actions = (
                prev_cd_actions[old_pl1],
                prev_cd_actions[old_pl2],
            )

            ########## calculate rewards from IPD ##############
            # lazy thing as only 3 players
            # sum of 1s is number of defectors
            num_defect = sum(old_chosen_game_actions).astype(jnp.int8)

            # calculate rewards
            # row of payoff table is number of defectors
            # column of payoff table is whether player defected or not
            payoff_array = jnp.array(params.payoff_table, dtype=jnp.float32)
            relevant_row1 = payoff_array[num_defect]
            game_rewards_array = (
                relevant_row1[old_chosen_game_actions[0]],
                relevant_row1[old_chosen_game_actions[1]],
            )

            # rewards for pl1, pl2, pl3 respectively for previous game
            rewards = jnp.zeros((3,), dtype=jnp.int8)
            rewards = rewards.at[old_pl1].set(game_rewards_array[0])
            rewards = rewards.at[old_pl2].set(game_rewards_array[1])

            ######### calculate rewards from punishment ############
            got_punished = jnp.zeros((3,), dtype=jnp.int8)
            # pl1 punished if punisher gave 1 or 3
            got_punished.at[old_pl1].set(
                jnp.where(punish_actions[old_punisher] % 4 == 1, 1, 0)
            )
            # pl2 punished if punisher gave 2 or 3
            got_punished.at[old_pl2].set(
                jnp.where(punish_actions[old_punisher] > 1, 1, 0)
            )

            rewards = rewards + params.punishment * got_punished

            ####### calculate observations ##########
            bits = 2
            len_one_hot = 2**bits + 1
            start_state_idx = 2**bits
            b2i = 2 ** jnp.arange(bits - 1, -1, -1)
            # first state is cc
            # second state is cd
            # 2**bits state is dd
            # so we can just binary decode the actions to get state
            new_obs = (
                curr_game_actions[0] * b2i[0] + curr_game_actions[1] * b2i[1]
            )
            # if first step then return START state.
            new_obs = jax.lax.select(
                reset_inner,
                start_state_idx * jnp.ones_like(new_obs),
                new_obs,
            )
            # one hot encode
            new_obs = jax.nn.one_hot(new_obs, len_one_hot, dtype=jnp.int8)

            # if this was first step, then rewards is 0 bc it depends on previous actions
            rewards = jax.lax.select(
                inner_t == 1,
                jnp.zeros_like(rewards),
                rewards,
            )
            # out step keeping
            inner_t = jax.lax.select(
                reset_inner, jnp.zeros_like(inner_t), inner_t
            )
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            reset_outer = outer_t == num_outer_steps
            state = EnvState(inner_t=inner_t, outer_t=outer_t)
            jax.debug.breakpoint()

            return (
                player_selection,
                (new_obs,),
                state,
                tuple(rewards),
                reset_outer,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            # start state is the last one eg 2**2_th
            # and one hot vectors should be 2**2+1 long
            bits = 2
            obs = jax.nn.one_hot(
                (2**bits) * jnp.ones(()),
                2**bits + 1,
                dtype=jnp.int8,
            )
            return (obs,), state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "ThirdPartyRandom"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 8

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        """Observation space of the environment."""
        obs_space = jnp.power(2, 2) + 1
        return spaces.Discrete(obs_space)
