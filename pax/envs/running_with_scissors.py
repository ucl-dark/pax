from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    red_coin_pos: jnp.ndarray
    blue_coin_pos: jnp.ndarray
    inner_t: int
    outer_t: int


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice


MOVES = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # stay
        [0, 0, 0],  # fire
    ]
)

STEP = jnp.array(
    [
        [1, 0, 0],  # right
        [0, 1, 0],  # up
        [-1, 0, 0],  # left
        [0, -1, 0],  # down
    ]
)
# POS = [x, y, theta]
# right turn = 0  -> [x, y, theta + 1]
# left turn = 1 -> [x, y, theta - 1]
# theta 1 -> [1, 0, 0], theta 2 -> [0, 1, 0], theta 3 -> [-1, 0, 0], theta 4 -> [0, -1, 0]
# forward = POS + theta_n


class RunningWithScissors(environment.Environment):
    """
    JAX Compatible version of coin game environment.
    0. Make the environment larger (10x10)
    1. We make the environment 2D with orientation
    2. Then we add a fire action
    3. Then we add variable number of coins
    """

    def __init__(
        self,
        num_inner_steps: int,
        num_outer_steps: int,
        cnn: bool,
    ):

        super().__init__()

        def _relative_position(state: EnvState) -> jnp.ndarray:
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

            agent_loc = jnp.array([state.red_pos[0], state.red_pos[1]])
            ego_offset = jnp.ones(2, dtype=jnp.int8) - agent_loc

            rel_other_player = (state.blue_pos + ego_offset) % 3
            rel_red_coin = (state.red_coin_pos + ego_offset) % 3
            rel_blue_coin = (state.blue_coin_pos + ego_offset) % 3

            # create observation
            obs = jnp.zeros((5, 5, 4), dtype=jnp.int8)
            obs = obs.at[1, 1, 0].set(1)
            obs = obs.at[rel_other_player[0], rel_other_player[1], 1].set(1)
            obs = obs.at[rel_red_coin[0], rel_red_coin[1], 2].set(1)
            obs = obs.at[rel_blue_coin[0], rel_blue_coin[1], 3].set(1)
            return obs

        def _state_to_obs(state: EnvState) -> jnp.ndarray:
            obs1 = _relative_position(state)

            # flip red and blue coins for second agent
            obs2 = _relative_position(
                EnvState(
                    red_pos=state.blue_pos,
                    blue_pos=state.red_pos,
                    red_coin_pos=state.blue_coin_pos,
                    blue_coin_pos=state.red_coin_pos,
                    inner_t=0,
                    outer_t=0,
                )
            )

            if not cnn:
                return obs1.flatten(), obs2.flatten()
            return obs1, obs2

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            action_0, action_1 = actions

            # turning red
            new_red_pos = state.red_pos
            red_turn = (action_0 == 1) or (action_0 == 2)
            new_red_pos = jnp.where(
                red_turn, new_red_pos + MOVES[action_0], new_red_pos
            )

            # moving red
            red_step = action_0 == 3
            new_red_pos = jnp.where(
                red_step, new_red_pos + STEP[state.red_pos[2]], new_red_pos
            )
            new_red_pos = jnp.clip(new_red_pos, min=0, max=3)

            # turning blue
            new_blue_pos = state.blue_pos
            blue_turn = (action_1 == 1) or (action_1 == 2)
            new_blue_pos = jnp.where(
                blue_turn, new_blue_pos + MOVES[action_0], new_blue_pos
            )

            # moving blue
            blue_step = action_1 == 3
            new_blue_pos = jnp.where(
                blue_step, new_blue_pos + STEP[state.blue_pos[2]], new_blue_pos
            )
            new_blue_pos = jnp.clip(new_blue_pos, min=0, max=3)

            # stay
            red_reward, blue_reward = 0, 0

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

            ### [[1, 1, -2],[1, 1, -2]]
            _rr_reward = params.payoff_matrix[0][0]
            _rb_reward = params.payoff_matrix[0][1]
            _r_penalty = params.payoff_matrix[0][2]
            _br_reward = params.payoff_matrix[1][0]
            _bb_reward = params.payoff_matrix[1][1]
            _b_penalty = params.payoff_matrix[1][2]

            red_reward = jnp.where(
                red_red_matches, red_reward + _rr_reward, red_reward
            )
            red_reward = jnp.where(
                red_blue_matches, red_reward + _rb_reward, red_reward
            )
            red_reward = jnp.where(
                blue_red_matches, red_reward + _r_penalty, red_reward
            )

            blue_reward = jnp.where(
                blue_red_matches, blue_reward + _br_reward, blue_reward
            )
            blue_reward = jnp.where(
                blue_blue_matches, blue_reward + _bb_reward, blue_reward
            )
            blue_reward = jnp.where(
                red_blue_matches, blue_reward + _b_penalty, blue_reward
            )

            key, subkey = jax.random.split(key)
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

            next_red_coop = state.red_coop + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(red_red_matches)
            next_red_defect = state.red_defect + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(red_blue_matches)
            next_blue_coop = state.blue_coop + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(blue_blue_matches)
            next_blue_defect = state.blue_defect + jnp.zeros(
                num_outer_steps, dtype=jnp.int8
            ).at[state.outer_t].set(blue_red_matches)

            next_state = EnvState(
                red_pos=new_red_pos,
                blue_pos=new_blue_pos,
                red_coin_pos=new_red_coin_pos,
                blue_coin_pos=new_blue_coin_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                red_coop=next_red_coop,
                red_defect=next_red_defect,
                blue_coop=next_blue_coop,
                blue_defect=next_blue_defect,
            )

            obs1, obs2 = _state_to_obs(next_state)

            # now calculate if done for inner or outer episode
            inner_t = next_state.inner_t
            outer_t = next_state.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            reset_obs, reset_state = _reset(key, params)
            next_state = EnvState(
                red_pos=jnp.where(
                    reset_inner, reset_state.red_pos, next_state.red_pos
                ),
                blue_pos=jnp.where(
                    reset_inner, reset_state.blue_pos, next_state.blue_pos
                ),
                red_coin_pos=jnp.where(
                    reset_inner,
                    reset_state.red_coin_pos,
                    next_state.red_coin_pos,
                ),
                blue_coin_pos=jnp.where(
                    reset_inner,
                    reset_state.blue_coin_pos,
                    next_state.blue_coin_pos,
                ),
                inner_t=jnp.where(
                    reset_inner, jnp.zeros_like(inner_t), next_state.inner_t
                ),
                outer_t=jnp.where(reset_inner, outer_t + 1, outer_t),
                red_coop=next_state.red_coop,
                red_defect=next_state.red_defect,
                blue_coop=next_state.blue_coop,
                blue_defect=next_state.blue_defect,
            )

            obs1 = jnp.where(reset_inner, reset_obs[0], obs1)
            obs2 = jnp.where(reset_inner, reset_obs[1], obs2)

            blue_reward = jnp.where(reset_inner, 0.0, blue_reward)
            red_reward = jnp.where(reset_inner, 0.0, red_reward)
            return (
                (obs1, obs2),
                next_state,
                (red_reward, blue_reward),
                reset_inner,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            key, subkey = jax.random.split(key)
            all_pos = jax.random.randint(
                subkey, shape=(4, 3), minval=(0, 0), maxval=(10, 4)
            )
            state = EnvState(
                red_pos=all_pos[0, :],
                blue_pos=all_pos[1, :],
                red_coin_pos=all_pos[2, :],
                blue_coin_pos=all_pos[3, :],
                inner_t=0,
                outer_t=0,
            )
            obs1, obs2 = _state_to_obs(state)
            return (obs1, obs2), state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)
        self.cnn = cnn

        self.step = _step
        self.reset = _reset

    @property
    def name(self) -> str:
        """Environment name."""
        return "CoinGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        # TODO: add zap bitches!
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        _shape = (5, 5, 4) if self.cnn else (100,)
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        _shape = (10, 10, 4) if self.cnn else (400,)
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def render(self, state: EnvState):
        import numpy as np
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as FigureCanvas,
        )
        from matplotlib.figure import Figure
        from PIL import Image

        """Small utility for plotting the agent's state."""
        fig = Figure((5, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(121)
        ax.imshow(
            np.zeros((3, 3)),
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=[0, 3, 0, 3],
        )
        ax.set_aspect("equal")

        # ax.margins(0)
        ax.set_xticks(jnp.arange(1, 4))
        ax.set_yticks(jnp.arange(1, 4))
        ax.grid()
        red_pos = jnp.squeeze(state.red_pos)
        blue_pos = jnp.squeeze(state.blue_pos)
        red_coin_pos = jnp.squeeze(state.red_coin_pos)
        blue_coin_pos = jnp.squeeze(state.blue_coin_pos)
        ax.annotate(
            "R",
            fontsize=20,
            color="red",
            xy=(red_pos[0], red_pos[1]),
            xycoords="data",
            xytext=(red_pos[0] + 0.5, red_pos[1] + 0.5),
        )
        ax.annotate(
            "B",
            fontsize=20,
            color="blue",
            xy=(blue_pos[0], blue_pos[1]),
            xycoords="data",
            xytext=(blue_pos[0] + 0.5, blue_pos[1] + 0.5),
        )
        ax.annotate(
            "Rc",
            fontsize=20,
            color="red",
            xy=(red_coin_pos[0], red_coin_pos[1]),
            xycoords="data",
            xytext=(red_coin_pos[0] + 0.3, red_coin_pos[1] + 0.3),
        )
        ax.annotate(
            "Bc",
            color="blue",
            fontsize=20,
            xy=(blue_coin_pos[0], blue_coin_pos[1]),
            xycoords="data",
            xytext=(
                blue_coin_pos[0] + 0.3,
                blue_coin_pos[1] + 0.3,
            ),
        )

        ax2 = fig.add_subplot(122)
        ax2.text(0.0, 0.95, "Timestep: %s" % (state.inner_t))
        ax2.text(0.0, 0.75, "Episode: %s" % (state.outer_t))
        ax2.text(
            0.0, 0.45, "Red Coop: %s" % (state.red_coop[state.outer_t].sum())
        )
        ax2.text(
            0.6,
            0.45,
            "Red Defects : %s" % (state.red_defect[state.outer_t].sum()),
        )
        ax2.text(
            0.0, 0.25, "Blue Coop: %s" % (state.blue_coop[state.outer_t].sum())
        )
        ax2.text(
            0.6,
            0.25,
            "Blue Defects : %s" % (state.blue_defect[state.outer_t].sum()),
        )
        ax2.text(
            0.0,
            0.05,
            "Red Total: %s"
            % (
                state.red_defect[state.outer_t].sum()
                + state.red_coop[state.outer_t].sum()
            ),
        )
        ax2.text(
            0.6,
            0.05,
            "Blue Total: %s"
            % (
                state.blue_defect[state.outer_t].sum()
                + state.blue_coop[state.outer_t].sum()
            ),
        )
        ax2.axis("off")
        canvas.draw()
        image = Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )
        return image


if __name__ == "__main__":
    action = 1
    rng = jax.random.PRNGKey(0)
    env = RunningWithScissors(8, 16, True, False)

    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    obs, state = env.reset(rng, params)
    pics = []

    for _ in range(16):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        a1 = jax.random.randint(rng1, (), minval=0, maxval=4)
        a2 = jax.random.randint(rng2, (), minval=0, maxval=4)
        obs, state, reward, done, info = env.step(
            rng, state, (a1 * action, a2 * action), params
        )
        img = env.render(state)
        pics.append(img)

    pics[0].save(
        "test1.gif",
        format="gif",
        save_all=True,
        append_images=pics[1:],
        duration=300,
        loop=0,
    )
