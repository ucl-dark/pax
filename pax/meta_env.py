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
            inner_t, outer_t = state
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

            return (t1, t2), (inner_t, outer_t)

        self.runner_step = jax.jit(jax.vmap(step, (0, None), (0, None)))
        self.num_envs = num_envs
        self.inner_episode_length = inner_ep_length
        self.num_trials = int(num_steps / inner_ep_length)
        self.episode_length = num_steps
        self.state = (0.0, 0.0)
        self._reset_next_step = True

    def step(self, actions):
        if self._reset_next_step:
            return self.reset()

        output, self.state = self.runner_step(actions, self.state)
        if self.state[1] == self.num_trials:
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
        self.state = (0.0, 0.0)
        self._reset_next_step = False
        obs = jax.nn.one_hot(4 * jnp.ones(self.num_envs), 5)
        discount = jnp.zeros(self.num_envs, dtype=int)
        step_type = jnp.zeros(self.num_envs, dtype=int)
        return TimeStep(
            step_type, jnp.zeros(self.num_envs), discount, obs
        ), TimeStep(step_type, jnp.zeros(self.num_envs), discount, obs)


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

        def _step(theta1, theta2):
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
            return L_1.sum(), L_2.sum(), obs1, obs2, M

        def _loss(theta1, theta2):
            theta1 = jax.nn.sigmoid(theta1)
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
            return L_1.sum(), M

        self._jit_step = jax.jit(jax.vmap(_step))
        self.grad = jax.jit(jax.vmap(jax.value_and_grad(_loss, has_aux=True)))
        self.gamma = gamma

        self.num_envs = num_envs
        self.episode_length = episode_length
        self.key = jax.random.PRNGKey(seed=seed)
        self._num_steps = 0
        self._reset_next_step = True

        # Dummy variables to work with meta_runner
        self.state = (0.0, 0.0)
        self.num_trials = 1
        self.inner_episode_length = episode_length

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

        r1, r2, obs1, obs2, _ = self._jit_step(
            action_1,
            action_2,
        )
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

        r1, r2, obs1, obs2, _ = self._jit_step(
            actions[0],
            actions[1],
        )
        r1, r2 = (1 - self.gamma) * r1, (1 - self.gamma) * r2
        return (
            transition(reward=r1, observation=obs1),
            transition(reward=r2, observation=obs2),
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

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._num_steps = 0
        self.key, _ = jax.random.split(self.key)
        obs = jax.nn.sigmoid(jax.random.uniform(self.key, (self.num_envs, 10)))
        return TimeStep(0, jnp.zeros(self.num_envs), 0.0, obs), TimeStep(
            0, jnp.zeros(self.num_envs), 0.0, obs
        )


class CoinGameState(NamedTuple):
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    steps_left: jnp.ndarray
    key: jnp.ndarray


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
            return obs

        def _step(
            actions: Tuple[int, int], state: CoinGameState
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, CoinGameState]:
            action_0, action_1 = actions
            new_red_pos = (state.red_pos + MOVES[action_0]) % 3
            new_blue_pos = (state.blue_pos + MOVES[action_1]) % 3

            red_reward = jnp.zeros(1)
            blue_reward = jnp.zeros(1)

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
            step_count = state.step_count + 1

            new_state = CoinGameState(
                new_red_pos,
                new_blue_pos,
                new_red_coin_pos,
                new_blue_coin_pos,
                step_count,
                key,
            )
            obs = _state_to_obs(new_state)

            return obs, red_reward, blue_reward, new_state

        self._jit_step = jax.jit(jax.vmap(_step))

        # key, subkey = jax.random.split(self.key)
        # all_pos = jax.random.randint(subkey, shape=(4, 2), minval=0, maxval=3)
        # step_count = jnp.zeros(1)
        # self._state = CoinGameState(all_pos[0], all_pos[1], all_pos[2], all_pos[3], step_count, key)
        def _runner_reset(
            key: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, CoinGameState]:
            key, subkey = jax.random.split(key)
            all_pos = jax.random.randint(
                subkey, shape=(4, 2), minval=0, maxval=3
            )
            discount = jnp.zeros(1)
            step_type = jnp.zeros(1)
            rewards = jnp.zeros(1)
            step_count = jnp.zeros(1)
            state = CoinGameState(
                all_pos[0, :],
                all_pos[1, :],
                all_pos[2, :],
                all_pos[3, :],
                step_count,
                key,
            )
            obs = _state_to_obs(state)

            output = [
                TimeStep(step_type, rewards, discount, obs),
                TimeStep(step_type, rewards, discount, obs),
            ]

            return state, output

        self.runner_reset = jax.vmap(_runner_reset)
        self.key = jax.random.split(jax.random.PRNGKey(seed=seed), num_envs)

        self.num_envs = num_envs
        self.inner_episode_length = inner_ep_length
        self.num_trials = int(num_steps / inner_ep_length)
        self.episode_length = num_steps
        self.state = CoinGameState(0, 0, 0, 0, 0, self.key)

    def init_state(
        self, ndims: Tuple[int], key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, CoinGameState]:
        # ndims: [num_opps x num_envs] or [num_envs]
        rngs = jax.random.split(key, ndims[-1])
        if len(ndims) > 1:
            rngs = jnp.tile(
                jax.lax.expand_dims(rngs, [0]), (ndims[:-1] + (1, 1))
            )
        state, output = self.runner_reset(rngs)
        return state, output

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        self._state, output = self.runner_reset(self.key)
        return output

    def runner_step(
        self,
        actions: Tuple[int, int],
        env_state: CoinGameState,
    ) -> Tuple[Tuple[TimeStep, TimeStep], CoinGameState]:

        obs, red_reward, blue_reward, new_state = self._jit_step(
            actions, env_state
        )
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

    def step(self, actions: Tuple[int, int]) -> Tuple[TimeStep, TimeStep]:
        output, self._state = self.runner_step(actions, self._state)
        return output

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=(self.num_envs, 4, 3, 3),
            minimum=0,
            maximum=1,
            name="obs",
            dtype=int,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(num_values=4, name="actions")
