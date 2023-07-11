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


class Actions(IntEnum):
    # first opponent, second opponent, punishment of previous game
    CC_no_P = 0
    CC_P1 = 1
    CC_P2 = 2
    CC_both_P = 3

    CD_no_P = 4
    CD_P1 = 5
    CD_P2 = 6
    CD_both_P = 7

    DC_no_P = 8
    DC_P1 = 9
    DC_P2 = 10
    DC_both_P = 11

    DD_no_P = 12
    DD_P1 = 13
    DD_P2 = 14
    DD_both_P = 15


class ThirdPartyPunishment(environment.Environment):
    """
    JAX Compatible version of tensor game environment.
    """

    def __init__(self, num_inner_steps: int, num_outer_steps: int):
        super().__init__()

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[
                Tuple[int, ...], Tuple[int, ...]
            ],  # prev action and current action
            params: EnvParams,
        ):
            num_players = 3
            # 3 players
            assert len(actions[0]) == num_players
            assert len(actions[1]) == num_players
            assert len(actions) == 2
            inner_t, outer_t = state.inner_t, state.outer_t
            inner_t += 1
            reset_inner = inner_t == num_inner_steps

            # we care about the previous C/D actions
            prev_actions = jnp.array(actions[0])
            # actions players took in their first games
            prev_first_cd_actions = jnp.where(
                prev_actions > 7, 1, 0
            )  # 1 if D, 0 if C
            # actions players took in their second games
            prev_second_cd_actions = jnp.where(
                prev_actions % 8 > 3, 1, 0
            )  # 1 if D, 0 if C

            # we care about the current punishment actions
            curr_actions = jnp.array(actions[1])
            punish_actions = (
                curr_actions % 4
            )  # 0 if no punishment, 1 if punish 1, 2 if punish 2, 3 if punish both

            curr_first_cd_actions = jnp.where(
                curr_actions > 7, 1, 0
            )
            curr_second_cd_actions = jnp.where(
                curr_actions % 8 > 3, 1, 0
            )


            # match up punishment actions with CD actions
            # we have 3 games, order will be:
            # pl1 (1st game) vs pl2 (1st game), pl3 punishes
            # pl2 (2nd game) vs pl3 (1st games), pl1 punishes
            # pl3 (2nd game) vs pl1 (2nd game), pl2 punishes

            prev_game1_actions = jnp.array(
                [prev_first_cd_actions[0], prev_first_cd_actions[1], punish_actions[2]],
                dtype=jnp.int8,
            )
            prev_game2_actions = jnp.array(
                [prev_second_cd_actions[1], prev_first_cd_actions[2], punish_actions[0]],
                dtype=jnp.int8,
            )
            prev_game3_actions = jnp.array(
                [
                    prev_second_cd_actions[2],
                    prev_second_cd_actions[0],
                    punish_actions[1],
                ],
                dtype=jnp.int8,
            )

            curr_game1_actions = jnp.array(
                [curr_first_cd_actions[0], curr_first_cd_actions[1]],
                dtype=jnp.int8,
            )
            curr_game2_actions = jnp.array(
                [curr_second_cd_actions[1], curr_first_cd_actions[2]],
                dtype=jnp.int8,
            )
            curr_game3_actions = jnp.array(
                [curr_second_cd_actions[2], curr_second_cd_actions[0]],
                dtype=jnp.int8,
            )

            ########## calculate rewards from IPD ##############
            # lazy thing as only 3 players
            # sum of 1s is number of defectors
            num_defect1 = sum(prev_game1_actions[0:2]).astype(jnp.int8)
            num_defect2 = sum(prev_game2_actions[0:2]).astype(jnp.int8)
            num_defect3 = sum(prev_game3_actions[0:2]).astype(jnp.int8)

            # calculate rewards
            # row of payoff table is number of defectors
            # column of payoff table is whether player defected or not
            payoff_array = jnp.array(params.payoff_table, dtype=jnp.float32)
            relevant_row1 = payoff_array[num_defect1]
            relevant_row2 = payoff_array[num_defect2]
            relevant_row3 = payoff_array[num_defect3]
            game1_rewards_array = relevant_row1[prev_game1_actions[0:2]]
            game2_rewards_array = relevant_row2[prev_game2_actions[0:2]]
            game3_rewards_array = relevant_row3[prev_game3_actions[0:2]]

            # pl1 rewards are from game1 and game3
            # pl2 rewards are from game1 and game2
            # pl3 rewards are from game2 and game3
            pl1_rewards = game1_rewards_array[0] + game3_rewards_array[1]
            pl2_rewards = game1_rewards_array[1] + game2_rewards_array[0]
            pl3_rewards = game2_rewards_array[1] + game3_rewards_array[0]

            ######### calculate rewards from punishment ############
            # pl1 punished if pl3 gives 1 or 3
            # pl1 punished if pl2 gives 2 or 3
            pl1_punished = jnp.where(
                punish_actions[2] % 2 == 1, 1, 0
            ) + jnp.where(punish_actions[1] > 1, 1, 0)
            # pl2 punished if pl3 gives 2 or 3
            # pl2 punished if pl1 gives 1 or 3
            pl2_punished = jnp.where(punish_actions[2] > 1, 1, 0) + jnp.where(
                punish_actions[0] % 2 == 1, 1, 0
            )
            # pl3 punished if pl1 gives 2 or 3
            # pl3 punished if pl2 gives 1 or 3
            pl3_punished = jnp.where(punish_actions[0] > 1, 1, 0) + jnp.where(
                punish_actions[1] % 2 == 1, 1, 0
            )

            pl1_rewards += params.punishment * pl1_punished
            pl2_rewards += params.punishment * pl2_punished
            pl3_rewards += params.punishment * pl3_punished

            all_rewards = (pl1_rewards, pl2_rewards, pl3_rewards)

            ####### calculate observations ##########
            # we don't want to include information about the punishment, just ipd actions
            # agents should also observe the other players' game to punish it on the next round
            # we have observations for three games, each in cc, cd, dc, dd order -> 12 + 1 for start states

            # we want these to be from first person perspective
            # eg. pl2 doesn't know she's the second player in game 1 and the first player in game 2

            # so state should be:
            # player vs its 1st op, player vs its 2nd op, 1st op vs second op
            # pl1:  pl1 vs pl2, pl1 vs pl3, pl2 vs pl3
            # pl2:  pl2 vs pl3, pl2 vs pl1, pl3 vs pl1
            # pl3:  pl3 vs pl1, pl3 vs pl2, pl1 vs pl2

            # states are numbered as follows:
            # cccccc, cccccd, ccccdc, ccccdd... etc.

            # list of first person perspective actions for each player
            # with each player starting with theirs
            # eg if game was DC, we will

            # we need this for logging purposes only
            prev_pl1_binary = jnp.concatenate(
                (prev_game1_actions[0:2], jnp.flip(prev_game3_actions[0:2]), prev_game2_actions[0:2]),)
            prev_pl2_binary = jnp.concatenate(
                (prev_game2_actions[0:2], jnp.flip(prev_game1_actions[0:2]), prev_game3_actions[0:2]),)
            prev_pl3_binary = jnp.concatenate(
                (prev_game3_actions[0:2], jnp.flip(prev_game2_actions[0:2]), prev_game1_actions[0:2]),)
            
            # we need this for returning the state for observations
            curr_pl1_binary = jnp.concatenate(
                (curr_game1_actions[0:2], jnp.flip(curr_game3_actions[0:2]), curr_game2_actions[0:2]),)
            curr_pl2_binary = jnp.concatenate(
                (curr_game2_actions[0:2], jnp.flip(curr_game1_actions[0:2]), curr_game3_actions[0:2]),)
            curr_pl3_binary = jnp.concatenate(
                (curr_game3_actions[0:2], jnp.flip(curr_game2_actions[0:2]), curr_game1_actions[0:2]),)



            # Start is the last state, the 2**6=128th state
            start_state_idx = 2**6
            len_one_hot = 2**6 + 1
            bits = 6
            # binary to int conversion for decoding actions
            b2i = 2 ** jnp.arange(bits - 1, -1, -1)
            all_obs = []
            for pl_bin in [curr_pl1_binary, curr_pl2_binary, curr_pl3_binary]:
                # first state is all c...cc
                # second state is all c...cd
                # 2**bits state is d...dd
                # so we can just binary decode the actions to get state
                obs = (pl_bin * b2i).sum()
                # if first step then return START state.
                obs = jax.lax.select(
                    reset_inner,
                    start_state_idx * jnp.ones_like(obs),
                    obs,
                )
                # one hot encode
                obs = jax.nn.one_hot(obs, len_one_hot, dtype=jnp.int8)
                all_obs.append(obs)
            player_obs = tuple(all_obs)


            # we want to return the punishment actions corresponding to the previous 
            # IPD actions for logging purposes even if agents can't see it
            # this is only for logging, so we return the binary of the first player's perspective only
            # of the games:
            # pl1 vs pl2, pl1 vs pl3, pl2 vs pl3 of the prev step
            # and pl1 pl2 pl3 punishment actions (0 for no punish, 1,2 for punish first or second opponent, 2 for punish both)
            log_obs = (prev_pl1_binary, punish_actions)
            # out step keeping
            inner_t = jax.lax.select(
                reset_inner, jnp.zeros_like(inner_t), inner_t
            )
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            reset_outer = outer_t == num_outer_steps
            state = EnvState(inner_t=inner_t, outer_t=outer_t)

            return (
                (player_obs, log_obs),
                state,
                all_rewards,
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
            # start state is the last one eg 2**6_th
            # and one hot vectors should be 2**n6+1 long
            bits = 6
            obs = jax.nn.one_hot(
                (2**6) * jnp.ones(()),
                2**bits + 1,
                dtype=jnp.int8,
            )
            num_players = 3
            all_obs = tuple([obs for _ in range(num_players)])
            return all_obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "ThirdPartyPunishment"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 16

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        """Observation space of the environment."""
        obs_space = jnp.power(2, 6) + 1
        return spaces.Discrete(obs_space)
