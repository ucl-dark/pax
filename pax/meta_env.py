from typing import Tuple
import jax
import jax.numpy as jnp
from dm_env import specs, TimeStep
from pandas import Timestamp


class MetaFiniteGame:
    def __init__(
        self,
        num_envs: int,
        payoff: list,
        inner_ep_length: int,
        episode_length: int,
    ) -> None:

        self.payoff = jnp.array(payoff)

        def step(actions, state):
            inner_t, outer_t = state
            inner_t += 1
            a1, a2 = actions

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
            s1 = jax.lax.select(
                inner_t == 0.0, jnp.float32(4.0), jnp.float32(s1)
            )
            s2 = jax.lax.select(inner_t == 0.0, jnp.float32(4.0), s2)

            obs = jax.nn.one_hot(s1, 4), jax.nn.one_hot(s2, 4)
            done = jax.lax.select(inner_t >= inner_ep_length, 2, 1)

            # out step keeping
            reset_inner = inner_t == inner_ep_length
            inner_t = jax.lax.select(reset_inner, 0.0, inner_t)
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)

            t1 = TimeStep(done, r1, 0.0, obs[0])
            t2 = TimeStep(done, r2, 0.0, obs[1])

            return (t1, t2), (inner_t, outer_t)

        self.runner_step = jax.jit(jax.vmap(step, (0, None), (0, None)))
        self.num_envs = num_envs
        self.num_trials = episode_length
        self.state = (0.0, 0.0)
        self._reset_next_step = True

    def step(self, actions):
        if self._reset_next_step:
            return self.reset()

        output, self.state = self.runner_step(actions, self.state)
        if self.state[1] == self.num_trials:
            self._reset_next_step = True
            output = (
                TimeStep(2, output[0].reward, 0, output[0].observation),
                TimeStep(2, output[1].reward, 0, output[1].observation),
            )
        return output

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(num_values=len(5), name="previous turn")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(dtype=int, num_values=len(2), name="action")

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self.state = (0.0, 0.0)
        self._reset_next_step = False
        obs = obs = jax.nn.one_hot(jnp.ones(self.num_envs), 5)
        return TimeStep(0, jnp.zeros(self.num_envs), 0.0, obs), TimeStep(
            0, jnp.zeros(self.num_envs), 0.0, obs
        )
