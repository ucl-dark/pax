from typing import Tuple
from typing import NamedTuple

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
            obs = obs = jax.nn.one_hot(4 * jnp.ones(ndims), 5)
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

        output, self._state = self.runner_step(actions, self._state)
        if (self._state[1] == self.num_trials).all():
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
        t, self._state = self.runner_reset(
            (self.num_envs,), jax.random.PRNGKey(0)
        )
        self._reset_next_step = False
        return t


class InfiniteMatrixGame(Environment):
    def __init__(
        self,
        num_envs: int,
        payoff: list,
        episode_length: int,
        gamma: float,
        seed: int,
    ) -> None:
        self.payoff = jnp.array(payoff)
        payout_mat_1 = jnp.array([[r[0] for r in self.payoff]])
        payout_mat_2 = jnp.array([[r[1] for r in self.payoff]])

        def _step(actions, state):
            theta1, theta2 = actions
            (t, rng) = state
            rng, _ = jax.random.split(rng, 2)
            theta1, theta2 = jax.nn.sigmoid(theta1), jax.nn.sigmoid(theta2)
            obs1 = jnp.concatenate([theta1, theta2])
            obs2 = jnp.concatenate([theta2, theta1])

            _th2 = jnp.array(
                [theta2[0], theta2[2], theta2[1], theta2[3], theta2[4]]
            )

            p_1_0 = theta1[4:5]
            p_2_0 = _th2[4:5]
            p = jnp.concatenate(
                [
                    p_1_0 * p_2_0,
                    p_1_0 * (1 - p_2_0),
                    (1 - p_1_0) * p_2_0,
                    (1 - p_1_0) * (1 - p_2_0),
                ]
            )
            p_1 = jnp.reshape(theta1[0:4], (4, 1))
            p_2 = jnp.reshape(_th2[0:4], (4, 1))
            P = jnp.concatenate(
                [
                    p_1 * p_2,
                    p_1 * (1 - p_2),
                    (1 - p_1) * p_2,
                    (1 - p_1) * (1 - p_2),
                ],
                axis=1,
            )
            M = jnp.matmul(p, jnp.linalg.inv(jnp.eye(4) - gamma * P))
            L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (4, 1)))
            L_2 = jnp.matmul(M, jnp.reshape(payout_mat_2, (4, 1)))
            return (L_1.sum(), L_2.sum(), obs1, obs2, M), (t + 1, rng)

        self._jit_step = jax.jit(jax.vmap(_step, (0, None), (0, None)))
        self.gamma = gamma

        self.num_envs = num_envs
        self.episode_length = episode_length
        self._num_steps = 0
        self._reset_next_step = True

        self._state = (0.0, jax.random.PRNGKey(seed=seed))
        self.num_trials = 1
        self.inner_episode_length = episode_length

        # for runner
        self.num_trials = 1
        self.inner_episode_length = episode_length
        self.batch_step = jax.jit(
            jax.vmap(self.runner_step, (0, None), (0, None))
        )

        # for runner
        self.num_trials = 1
        self.inner_episode_length = episode_length
        self.batch_step = jax.jit(
            jax.vmap(self.runner_step, (0, None), (0, None))
        )

    def step(
        self, actions: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[TimeStep, TimeStep]:
        """
        takes a tuple of batched policies and produce value functions from infinite game
        policy of form [B, 5]
        """
        if self._reset_next_step:
            return self.reset()

        action_1, action_2 = actions
        self._num_steps += 1
        assert action_1.shape == action_2.shape
        assert action_1.shape == (self.num_envs, 5)

        outputs, self._state = self._jit_step(actions, self._state)
        r1, r2, obs1, obs2, _ = outputs
        r1, r2 = (1 - self.gamma) * r1, (1 - self.gamma) * r2

        if self._num_steps == self.episode_length:
            self._reset_next_step = True
            return termination(reward=r1, observation=obs1), termination(
                reward=r2, observation=obs2
            )
        return transition(reward=r1, observation=obs1), transition(
            reward=r2, observation=obs2
        )

    def runner_step(
        self,
        actions: Tuple[jnp.ndarray, jnp.ndarray],
        env_state: Tuple[float, float],
    ) -> Tuple[Tuple[TimeStep, TimeStep], Tuple[float, float]]:

        (r1, r2, obs1, obs2, _), env_state = self._jit_step(actions, env_state)
        r1, r2 = (1 - self.gamma) * r1, (1 - self.gamma) * r2
        return (
            TimeStep(
                jnp.ones_like(r1, dtype=int),
                reward=r1,
                discount=jnp.zeros_like(r1, dtype=int),
                observation=obs1,
            ),
            TimeStep(
                jnp.ones_like(r1, dtype=int),
                reward=r2,
                discount=jnp.zeros_like(r1, dtype=int),
                observation=obs2,
            ),
        ), env_state

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(num_values=10, name="previous policy")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(self.num_envs, 5),
            minimum=0,
            maximum=1,
            name="action",
            dtype=float,
        )

    def runner_reset(self, ndims, rng):
        """Pure version of reset"""
        key, _ = jax.random.split(rng)
        new_state = (0, key)
        discount = jnp.zeros(ndims, dtype=int)
        step_type = jnp.zeros(ndims, dtype=int)
        obs = jax.nn.sigmoid(jax.random.uniform(key, ndims + (10,)))
        return (
            TimeStep(step_type, jnp.zeros(ndims), discount, obs),
            TimeStep(step_type, jnp.zeros(ndims), discount, obs),
        ), (new_state)

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._num_steps = 0
        t_init, self._state = self.runner_reset(
            (self.num_envs,), self._state[1]
        )
        self.key = self._state[1]
        return t_init


class CoinGameState(NamedTuple):
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    key: jnp.ndarray
    inner_t: jnp.ndarray
    outer_t: jnp.ndarray


# class CoinGameJAX:
MOVES = jax.device_put(
    jnp.array(
        [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
        ]
    )
)


class CoinGame:
    def __init__(
        self, num_envs: int, inner_ep_length: int, num_steps: int, seed: int
    ):
        def _state_to_obs(state: CoinGameState) -> jnp.ndarray:
            obs = jnp.zeros((4, 3, 3))
            obs = obs.at[0, state.red_pos[0], state.red_pos[1]].set(1.0)
            obs = obs.at[1, state.blue_pos[0], state.blue_pos[1]].set(1.0)
            obs = obs.at[2, state.red_coin_pos[0], state.red_coin_pos[1]].set(
                1.0
            )
            obs = obs.at[
                3, state.blue_coin_pos[0], state.blue_coin_pos[1]
            ].set(1.0)
            return obs.flatten()

        def _runner_reset(
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
            )
            obs = _state_to_obs(state)

            output = (
                TimeStep(step_type, rewards, discount, obs),
                TimeStep(step_type, rewards, discount, obs),
            )

            return output, state

        def _step(
            actions: Tuple[int, int], state: CoinGameState
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, CoinGameState]:
            action_0, action_1 = actions
            new_red_pos = (state.red_pos + MOVES[action_0]) % 3
            new_blue_pos = (state.blue_pos + MOVES[action_1]) % 3

            red_reward = 0
            blue_reward = 0

            red_red_matches = jnp.all(
                new_red_pos == state.red_coin_pos, axis=-1
            )
            red_blue_matches = jnp.all(
                new_red_pos == state.blue_coin_pos, axis=-1
            )

            red_reward = jnp.where(red_red_matches, red_reward + 1, red_reward)
            red_reward = jnp.where(
                red_blue_matches, red_reward + 1, red_reward
            )

            blue_red_matches = jnp.all(
                new_blue_pos == state.red_coin_pos, axis=-1
            )
            blue_blue_matches = jnp.all(
                new_blue_pos == state.blue_coin_pos, axis=-1
            )

            blue_reward = jnp.where(
                blue_red_matches, blue_reward + 1, blue_reward
            )
            blue_reward = jnp.where(
                blue_blue_matches, blue_reward + 1, blue_reward
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
            )

            obs = _state_to_obs(next_state)
            inner_t = next_state.inner_t
            outer_t = next_state.outer_t
            done = inner_t % inner_ep_length == 0

            # if inner episode is done, return start state for next game
            new_ep_outputs, new_ep_state = _runner_reset(state.key)
            new_ep_state = new_ep_state._replace(
                outer_t=new_ep_state.outer_t + 1
            )

            next_state = CoinGameState(
                jnp.where(done, new_red_pos, next_state.red_pos),
                jnp.where(done, new_blue_pos, next_state.blue_pos),
                jnp.where(done, new_red_coin_pos, next_state.red_coin_pos),
                jnp.where(done, new_blue_coin_pos, next_state.blue_coin_pos),
                key,
                jnp.where(done, jnp.zeros_like(inner_t), next_state.inner_t),
                jnp.where(done, outer_t + 1, outer_t),
            )

            obs = jnp.where(done, new_ep_outputs[0].observation, obs)
            blue_reward = jnp.where(done, 0, blue_reward)
            red_reward = jnp.where(done, 0, red_reward)

            return obs, red_reward, blue_reward, next_state

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
                    observation=obs,
                ),
                TimeStep(
                    step_type=jnp.ones_like(blue_reward, dtype=int),
                    reward=blue_reward,
                    discount=jnp.zeros_like(blue_reward, dtype=int),
                    observation=obs,
                ),
            ]

            return output, new_state

        self._jit_step = jax.jit(jax.vmap(_step))
        self.runner_reset = jax.vmap(_runner_reset)
        self.key = jax.random.split(jax.random.PRNGKey(seed=seed), num_envs)

        self.num_envs = num_envs
        self.inner_episode_length = inner_ep_length
        self.num_trials = int(num_steps / inner_ep_length)
        self.episode_length = num_steps
        self.state = CoinGameState(0, 0, 0, 0, self.key, 0, 0)

        self.batch_step = jax.jit(jax.vmap(jax.vmap(runner_step)))
        self.batch_reset = jax.jit(jax.vmap(jax.vmap(_runner_reset)))

    def init_state(
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
        self._state, output = self.runner_reset(self.key)
        return output

    def step(self, actions: Tuple[int, int]) -> Tuple[TimeStep, TimeStep]:
        output, self._state = self.runner_step(actions, self._state)
        return output

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=(36,),
            minimum=0,
            maximum=1,
            name="obs",
            dtype=int,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(num_values=4, name="actions")
