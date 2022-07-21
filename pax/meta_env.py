import jax
import jax.numpy as jnp


class MetaFiniteGame:
    def __init__(
        self,
        payoff: list,
        episode_length: int,
    ) -> None:

        self.payoff = jnp.array(payoff)

        def step(actions, state):
            inner_t, outer_t = state
            a1, a2 = actions

            cc_p1 = self.payoff[0][0] * (a1 - 1.0) * (a2 - 1.0)
            cc_p2 = self.payoff[0][1] * (a1 - 1.0) * (a2 - 1.0)
            cd_p1 = self.payoff[1][0] * (1.0 - a1) * a2
            cd_p2 = self.payoff[1][1] * (1.0 - a1) * a2
            dc_p1 = self.payoff[2][0] * a1 * (1.0 - a2)
            dc_p2 = self.payoff[2][1] * a1 * (1.0 - a2)
            dd_p1 = self.payoff[3][0] * a1 * a2
            dd_p2 = self.payoff[3][1] * a1 * a2

            r1 = (cc_p1 + dc_p1 + cd_p1 + dd_p1,)
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

            print(f"state: {s1}")
            print(f"inner t: {inner_t}")

            # if first step then return START state.
            s1 = jax.lax.select(inner_t == 0, 4.0, s1)
            s2 = jax.lax.select(inner_t == 0, 4.0, s2)

            obs = jax.nn.one_hot(s1, 5), jax.nn.one_hot(s2, 5)
            done = jax.lax.select(inner_t >= episode_length, 1, 0)
            inner_t += 1

            # out step keeping
            reset_inner = inner_t == episode_length
            inner_t = jax.lax.select(reset_inner, 0.0, inner_t)
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            return ((r1, r2), obs, done), (inner_t, outer_t)

        self.step = jax.vmap(step, (0, None), (0, None))
