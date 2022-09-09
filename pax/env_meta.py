from typing import NamedTuple, Tuple
from xmlrpc.client import Boolean

import jax
import jax.numpy as jnp
from dm_env import Environment, TimeStep, specs, termination, transition


class MetaFiniteGame:
    def __init__(
        self,
        num_envs: int,
        payoff: list,
        inner_ep_length: int,
        num_steps: int,
    ) -> None:

        self.payoff = jnp.array(payoff)

        def step(actions, state):
            inner_t, outer_t, rng = state
            a1, a2 = actions
            inner_t += 1

            cc_p1 = self.payoff[0][0] * (a1 - 1.0) * (a2 - 1.0)
            cc_p2 = self.payoff[0][1] * (a1 - 1.0) * (a2 - 1.0)
            cd_p1 = self.payoff[1][0] * (1.0 - a1) * a2
            cd_p2 = self.payoff[1][1] * (1.0 - a1) * a2
            dc_p1 = self.payoff[2][0] * a1 * (1.0 - a2)
            dc_p2 = self.payoff[2][1] * a1 * (1.0 - a2)
            dd_p1 = self.payoff[3][0] * a1 * a2
            dd_p2 = self.payoff[3][1] * a1 * a2

            r1 = cc_p1 + dc_p1 + cd_p1 + dd_p1
            r2 = cc_p2 + dc_p2 + cd_p2 + dd_p2

            s1 = (
                0 * (1 - a1) * (1 - a2)
                + 1 * (1 - a1) * a2
                + 2 * a1 * (1 - a2)
                + 3 * (a1) * (a2)
            )

            s2 = (
                0 * (1 - a1) * (1 - a2)
                + 1 * a1 * (1 - a2)
                + 2 * (1 - a1) * a2
                + 3 * (a1) * (a2)
            )
            # if first step then return START state.
            done = inner_t % inner_ep_length == 0
            s1 = jax.lax.select(done, jnp.float32(4.0), jnp.float32(s1))
            s2 = jax.lax.select(done, jnp.float32(4.0), jnp.float32(s2))

            obs = jax.nn.one_hot(s1, 5), jax.nn.one_hot(s2, 5)
            # done = jax.lax.select(inner_t >= inner_ep_length, 2, 1)

            # out step keeping
            reset_inner = inner_t == inner_ep_length
            inner_t = jax.lax.select(reset_inner, 0.0, inner_t)
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)

            t1 = TimeStep(1, r1, 0, obs[0])
            t2 = TimeStep(1, r2, 0, obs[1])

            return (t1, t2), (inner_t, outer_t, rng)

        def runner_reset(ndims, rng):
            """Returns the first `TimeStep` of a new episode."""
            # ndims: [num_opps, num_envs] or [num_envs]

            rngs = jax.random.split(rng, ndims[-1])
            if len(ndims) != 1:
                rngs = jnp.tile(
                    jax.lax.expand_dims(rngs, [0]), (ndims[:-1] + (1, 1))
                )

            state = (jnp.zeros(ndims), jnp.zeros(ndims), rngs)
            obs = jax.nn.one_hot(4 * jnp.ones(ndims), 5)
            discount = jnp.zeros(ndims, dtype=int)
            step_type = jnp.zeros(ndims, dtype=int)
            rewards = jnp.zeros(ndims)
            return (
                TimeStep(step_type, rewards, discount, obs),
                TimeStep(step_type, rewards, discount, obs),
            ), state

        self.num_envs = num_envs
        self.inner_episode_length = inner_ep_length
        self.num_trials = int(num_steps / inner_ep_length)
        self.episode_length = num_steps
        self._reset_next_step = True

        # for runner
        self.runner_step = jax.jit(jax.vmap(step))
        self.batch_step = jax.jit(jax.vmap(jax.vmap(step)))
        self.runner_reset = runner_reset

    def step(self, actions):
        if self._reset_next_step:
            return self.reset()

        output, self.state = self.runner_step(actions, self.state)
        if (self.state[1] == self.num_trials).all():
            self._reset_next_step = True
            output = (
                TimeStep(
                    2 * jnp.ones(self.num_envs),
                    output[0].reward,
                    4,
                    output[0].observation,
                ),
                TimeStep(
                    2 * jnp.ones(self.num_envs),
                    output[1].reward,
                    4,
                    output[1].observation,
                ),
            )
        return output

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(num_values=5, name="previous turn")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(dtype=int, num_values=2, name="action")

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        t, self.state = self.runner_reset(
            (self.num_envs,), jax.random.PRNGKey(0)
        )
        self._reset_next_step = False
        return t


class CoinGameState(NamedTuple):
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    key: jnp.ndarray
    inner_t: int
    outer_t: int
    # stats
    red_coop: int
    red_defect: int
    blue_coop: int
    blue_defect: int


# class CoinGameJAX:
MOVES = jnp.array(
    [
        [0, 1],  # right
        [0, -1],  # left
        [1, 0],  # up
        [-1, 0],  # down
        [0, 0],  # stay
    ]
)


class CoinGame:
    def __init__(
        self,
        num_envs: int,
        inner_ep_length: int,
        num_steps: int,
        seed: int,
        cnn: Boolean,
    ):
        def _relative_position(state: CoinGameState) -> jnp.ndarray:
            """Assume canonical agent is red player"""
            # (x) redplayer at (2, 2)
            # (y) redcoin at   (0 ,0)
            #
            #  o o x        o o y
            #  o o o   ->   o x o
            #  y o o        o o o
            #
            # redplayer goes to (1, 1)
            # redcoing goes to  (2, 2)
            # offset = (-1, -1)
            # new_redcoin = (0, 0) + (-1, -1) = (-1, -1) mod3
            # new_redcoin = (2, 2)

            # agent_loc = jnp.array([state.red_pos[0], state.red_pos[1]])
            # ego_offset = jnp.ones(2, dtype=jnp.int8) - agent_loc

            # rel_other_player = (state.blue_pos + ego_offset) % 3
            # rel_red_coin = (state.red_coin_pos + ego_offset) % 3
            # rel_blue_coin = (state.blue_coin_pos + ego_offset) % 3

            # # create observation
            # obs = jnp.zeros((3, 3, 4), dtype=jnp.int8)
            # obs = obs.at[1, 1, 0].set(1)
            # obs = obs.at[rel_other_player[0], rel_other_player[1], 1].set(1)
            # obs = obs.at[rel_red_coin[0], rel_red_coin[1], 2].set(1)
            # obs = obs.at[rel_blue_coin[0], rel_blue_coin[1], 3].set(1)

            obs = jnp.zeros((3, 3, 4), dtype=jnp.int8)
            obs = obs.at[state.red_pos[0], state.red_pos[1], 0].set(1)
            obs = obs.at[state.blue_pos[0], state.blue_pos[1], 1].set(1)
            obs = obs.at[state.red_coin_pos[0], state.red_coin_pos[1], 2].set(
                1
            )
            obs = obs.at[
                state.blue_coin_pos[0], state.blue_coin_pos[1], 3
            ].set(1)
            return obs

        def _state_to_obs(state: CoinGameState) -> jnp.ndarray:
            obs1 = _relative_position(state)

            # flip red and blue coins for second agent
            obs2 = _relative_position(
                state._replace(
                    red_pos=state.blue_pos,
                    blue_pos=state.red_pos,
                    red_coin_pos=state.blue_coin_pos,
                    blue_coin_pos=state.red_coin_pos,
                )
            )

            if not self.cnn:
                return obs1.flatten(), obs2.flatten()
            return obs1, obs2

        def _reset(
            key: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, CoinGameState]:
            key, subkey = jax.random.split(key)
            all_pos = jax.random.randint(
                subkey, shape=(4, 2), minval=0, maxval=3
            )
            discount = 0
            step_type = 0
            rewards = 0
            inner_t = 0
            outer_t = 0

            state = CoinGameState(
                all_pos[0, :],
                all_pos[1, :],
                all_pos[2, :],
                all_pos[3, :],
                key,
                inner_t,
                outer_t,
                0,
                0,
                0,
                0,
            )
            obs1, obs2 = _state_to_obs(state)

            output = (
                TimeStep(step_type, rewards, discount, obs1),
                TimeStep(step_type, rewards, discount, obs2),
            )

            return output, state

        def _step(
            actions: Tuple[int, int], state: CoinGameState
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, CoinGameState]:
            action_0, action_1 = actions

            new_red_pos = state.red_pos + MOVES[action_0]
            new_blue_pos = state.blue_pos + MOVES[action_1]

            new_red_pos = jnp.clip(new_red_pos, a_min=0, a_max=3)
            new_blue_pos = jnp.clip(new_blue_pos, a_min=0, a_max=3)

            red_reward = 0
            blue_reward = 0

            red_red_matches = jnp.all(
                new_red_pos == state.red_coin_pos, axis=-1
            )
            red_blue_matches = jnp.all(
                new_red_pos == state.blue_coin_pos, axis=-1
            )

            blue_red_matches = jnp.all(
                new_blue_pos == state.red_coin_pos, axis=-1
            )
            blue_blue_matches = jnp.all(
                new_blue_pos == state.blue_coin_pos, axis=-1
            )

            red_reward = jnp.where(red_red_matches, red_reward + 1, red_reward)
            red_reward = jnp.where(
                red_blue_matches, red_reward + 1, red_reward
            )
            red_reward = jnp.where(
                blue_red_matches, red_reward - 2, red_reward
            )

            blue_reward = jnp.where(
                blue_red_matches, blue_reward + 1, blue_reward
            )
            blue_reward = jnp.where(
                blue_blue_matches, blue_reward + 1, blue_reward
            )
            blue_reward = jnp.where(
                red_blue_matches, blue_reward - 2, blue_reward
            )

            key, subkey = jax.random.split(state.key)
            new_random_coin_poses = jax.random.randint(
                subkey, shape=(2, 2), minval=0, maxval=3
            )
            new_red_coin_pos = jnp.where(
                jnp.logical_or(red_red_matches, blue_red_matches),
                new_random_coin_poses[0],
                state.red_coin_pos,
            )
            new_blue_coin_pos = jnp.where(
                jnp.logical_or(red_blue_matches, blue_blue_matches),
                new_random_coin_poses[1],
                state.blue_coin_pos,
            )

            next_state = CoinGameState(
                new_red_pos,
                new_blue_pos,
                new_red_coin_pos,
                new_blue_coin_pos,
                key,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                red_coop=state.red_coop + red_red_matches,
                red_defect=state.red_defect + red_blue_matches,
                blue_coop=state.blue_coop + blue_blue_matches,
                blue_defect=state.blue_defect + blue_red_matches,
            )

            obs1, obs2 = _state_to_obs(next_state)
            inner_t = next_state.inner_t
            outer_t = next_state.outer_t
            done = inner_t % inner_ep_length == 0

            # if inner episode is done, return start state for next game
            new_ep_outputs, new_ep_state = _reset(state.key)
            new_ep_state = new_ep_state._replace(
                outer_t=new_ep_state.outer_t + 1
            )

            next_state = CoinGameState(
                jnp.where(done, new_ep_state.red_pos, next_state.red_pos),
                jnp.where(done, new_ep_state.blue_pos, next_state.blue_pos),
                jnp.where(
                    done, new_ep_state.red_coin_pos, next_state.red_coin_pos
                ),
                jnp.where(
                    done, new_ep_state.blue_coin_pos, next_state.blue_coin_pos
                ),
                key,
                jnp.where(done, jnp.zeros_like(inner_t), next_state.inner_t),
                jnp.where(done, outer_t + 1, outer_t),
                next_state.red_coop,
                next_state.red_defect,
                next_state.blue_coop,
                next_state.blue_defect,
            )

            obs1 = jnp.where(done, new_ep_outputs[0].observation, obs1)
            obs2 = jnp.where(done, new_ep_outputs[1].observation, obs2)

            blue_reward = jnp.where(done, 0, blue_reward)
            red_reward = jnp.where(done, 0, red_reward)
            return (obs1, obs2), red_reward, blue_reward, next_state

        def runner_step(
            actions: Tuple[int, int],
            env_state: CoinGameState,
        ) -> Tuple[Tuple[TimeStep, TimeStep], CoinGameState]:

            obs, red_reward, blue_reward, new_state = _step(actions, env_state)
            output = [
                TimeStep(
                    step_type=jnp.ones_like(red_reward, dtype=int),
                    reward=red_reward,
                    discount=jnp.zeros_like(red_reward, dtype=int),
                    observation=obs[0],
                ),
                TimeStep(
                    step_type=jnp.ones_like(blue_reward, dtype=int),
                    reward=blue_reward,
                    discount=jnp.zeros_like(blue_reward, dtype=int),
                    observation=obs[1],
                ),
            ]

            return output, new_state

        self._reset = jax.vmap(_reset)
        self.key = jax.random.split(jax.random.PRNGKey(seed=seed), num_envs)

        self.num_envs = num_envs
        self.inner_episode_length = inner_ep_length
        self.num_trials = int(num_steps / inner_ep_length)
        self.episode_length = num_steps
        self.state = CoinGameState(0, 0, 0, 0, self.key, 0, 0, 0, 0, 0, 0)

        self.runner_step = jax.vmap(runner_step)
        self.batch_step = jax.jit(jax.vmap(jax.vmap(runner_step)))
        self.batch_reset = jax.jit(jax.vmap(jax.vmap(_reset)))

        self.cnn = cnn

    def runner_reset(
        self, ndims: Tuple[int], key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, CoinGameState]:

        rngs = jax.random.split(key, ndims[-1])
        if len(ndims) > 1:
            rngs = jnp.tile(
                jax.lax.expand_dims(rngs, [0]), (ndims[:-1] + (1, 1))
            )
        output, state = self.batch_reset(rngs)
        return output, state

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        output, self.state = self._reset(self.key)
        return output

    def step(self, actions: Tuple[int, int]) -> Tuple[TimeStep, TimeStep]:
        output, self.state = self.runner_step(actions, self.state)
        return output

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        if self.cnn:
            return specs.BoundedArray(
                shape=(3, 3, 4),
                minimum=0,
                maximum=1,
                name="obs",
                dtype=int,
            )
        else:
            return specs.BoundedArray(
                shape=(36,),
                minimum=0,
                maximum=1,
                name="obs",
                dtype=int,
            )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(num_values=5, name="actions")
