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

    def __init__(
        self, num_inner_steps: int, num_outer_steps: int
    ):
        super().__init__()

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[Tuple[int, ...],Tuple[int, ...]], # prev action and current action
            params: EnvParams,
        ):
            num_players = 3
            # 3 players
            assert len(actions) == num_players
            inner_t, outer_t = state.inner_t, state.outer_t
            inner_t += 1
            reset_inner = inner_t == num_inner_steps


            # we care about the previous C/D actions
            prev_actions = jnp.array(actions[0])
            # actions players took in their first games
            first_cd_actions = jnp.where(prev_actions >7, 1, 0) # 1 if D, 0 if C
            # actions players took in their second games
            second_cd_actions = jnp.where(prev_actions % 8 > 3, 1, 0) # 1 if D, 0 if C

            # we care about the current punishment actions
            curr_actions = jnp.array(actions[1])
            punish_actions = curr_actions % 4 # 0 if no punishment, 1 if punish 1, 2 if punish 2, 3 if punish both

            # match up punishment actions with CD actions
            # we have 3 games, order will be:
            # pl1 (1st game) vs pl2 (1st game), pl3 punishes
            # pl2 (2nd game) vs pl3 (1st games), pl1 punishes
            # pl3 (2nd game) vs pl1 (2nd game), pl2 punishes

            game1_actions = jnp.array([first_cd_actions[0], first_cd_actions[1], punish_actions[2]], dtype=jnp.int8)
            game2_actions = jnp.array([second_cd_actions[1], first_cd_actions[2], punish_actions[0]], dtype=jnp.int8)
            game3_actions = jnp.array([second_cd_actions[2], second_cd_actions[0], punish_actions[1]], dtype=jnp.int8)

            ########## calculate rewards from IPD ##############
            # lazy thing as only 3 players
            # sum of 1s is number of defectors
            num_defect1 = sum(game1_actions[0:2]).astype(jnp.int8)
            num_defect2 = sum(game2_actions[0:2]).astype(jnp.int8)
            num_defect3 = sum(game3_actions[0:2]).astype(jnp.int8)

            # calculate rewards
            # row of payoff table is number of defectors
            # column of payoff table is whether player defected or not
            payoff_array = jnp.array(params.payoff_table, dtype=jnp.float32)
            assert payoff_array.shape == (num_players + 1, 2)
            relevant_row1 = payoff_array[num_defect1]
            relevant_row2 = payoff_array[num_defect2]
            relevant_row3 = payoff_array[num_defect3]
            game1_rewards_array = relevant_row1[game1_actions[0:2]]
            game2_rewards_array = relevant_row2[game2_actions[0:2]]
            game3_rewards_array = relevant_row3[game3_actions[0:2]]

            # pl1 rewards are from game1 and game3
            # pl2 rewards are from game1 and game2
            # pl3 rewards are from game2 and game3
            pl1_rewards = game1_rewards_array[0] + game3_rewards_array[1]
            pl2_rewards = game1_rewards_array[1] + game2_rewards_array[0]
            pl3_rewards = game2_rewards_array[1] + game3_rewards_array[0]
            
            ######### calculate rewards from punishment ############
            # pl1 punished in game1 if pl3 gives 1 or 3
            # pl1 punished in game3 if pl2 gives 2 or 3 
            pl1_punished = jnp.where(punish_actions[0] % 2 ==1, 1, 0) + jnp.where(punish_actions[2] >1, 1, 0)
            # pl2 punished in game1 if pl3 gives 2 or 3
            # pl2 punished in game2 if pl1 gives 1 or 3
            pl2_punished = jnp.where(punish_actions[0] >1, 1, 0) + jnp.where(punish_actions[1] % 2 ==1, 1, 0)
            # pl3 punished in game2 if pl1 gives 2 or 3
            # pl3 punished in game3 if pl2 gives 1 or 3
            pl3_punished = jnp.where(punish_actions[1] >1, 1, 0) + jnp.where(punish_actions[2] % 2 ==1, 1, 0)

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


            # if 3 games were: CC pl1 pl2, CD pl2 pl3, DD, pl3 pl1
            # then we want to return:
            # pl1: CC, CD, DD
            # UGGHHHHHH

            # list of first person perspective actions for each player
            # with each player starting with theirs
            # eg if game was DC, we will 
            fpp_actions_game1 = []
            fpp_actions_game2 = []
            fpp_actions_game3 = []
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
        ) -> Tuple[chex.Array, EnvState]:
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
