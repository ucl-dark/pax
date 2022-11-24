from ast import Num
from turtle import st
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


ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # stay
        [0, 0, 0],  # fire
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [0, 1, 0],  # up
        [1, 0, 0],  # right
        [0, -1, 0],  # down
        [-1, 0, 0],  # left
    ],
    dtype=jnp.int8,
)

GRID_SIZE = 10
OBS_SIZE = 5
PADDING = OBS_SIZE - 1
NUM_TYPES = 5  # red, blue, red coin, blue coin, wall

GRID = jnp.zeros(
    (GRID_SIZE + 2 * PADDING, GRID_SIZE + 2 * PADDING),
    dtype=jnp.int8,
)


# First layer of Padding is walls
GRID = GRID.at[PADDING, :].set(5)
GRID = GRID.at[:, PADDING].set(5)
GRID = GRID.at[GRID_SIZE + PADDING, :].set(5)
GRID = GRID.at[:, GRID_SIZE + PADDING].set(5)


class RunningWithScissors(environment.Environment):
    """
    JAX Compatible version of coin game environment.
    0. Make the environment larger (10x10)
    1. We make the environment 2D with orientation
    2. Agents can not stand or move onto same square (coins can respawn there tho)
    2. Then we add a fire action
    3. Then we add variable number of coins
    """

    def __init__(
        self,
        num_inner_steps: int,
        num_outer_steps: int,
    ):

        super().__init__()

        pos_choice = jnp.array(
            [[(i, j) for i in range(GRID_SIZE)] for j in range(GRID_SIZE)],
            dtype=jnp.int8,
        ).reshape(-1, 2)

        def _state_to_obs(state: EnvState) -> jnp.ndarray:
            """Assume canonical agent is red player"""
            # create state
            obs = GRID
            obs = obs.at[
                state.red_pos[0] + PADDING, state.red_pos[1] + PADDING
            ].set(1)
            obs = obs.at[
                state.blue_pos[0] + PADDING, state.blue_pos[1] + PADDING
            ].set(2)
            obs = obs.at[
                state.red_coin_pos[0] + PADDING,
                state.red_coin_pos[1] + PADDING,
            ].set(3)
            obs = obs.at[
                state.blue_coin_pos[0] + PADDING,
                state.blue_coin_pos[1] + PADDING,
            ].set(4)

            angle = jnp.zeros(
                (GRID_SIZE + 2 * PADDING, GRID_SIZE + 2 * PADDING),
                dtype=jnp.int8,
            )
            angle = angle.at[
                state.red_pos[0] + PADDING, state.red_pos[1] + PADDING
            ].set(state.red_pos[2])

            angle = angle.at[
                state.blue_pos[0] + PADDING, state.blue_pos[1] + PADDING
            ].set(state.blue_pos[2])

            obs = jnp.stack([obs, angle], axis=-1)

            # crop
            startx = (state.red_pos[0] + PADDING) // 2 - (OBS_SIZE // 2)
            starty = state.red_pos[1] + PADDING
            obs1 = jax.lax.dynamic_slice(
                obs,
                start_indices=(startx, starty, jnp.int8(0)),
                slice_sizes=(OBS_SIZE, OBS_SIZE, 2),
            )
            # rotate
            obs1 = jnp.where(
                state.red_pos[2] == 1, jnp.rot90(obs1, k=1, axes=(0, 1)), obs1
            )
            obs1 = jnp.where(
                state.red_pos[2] == 2, jnp.rot90(obs1, k=2, axes=(0, 1)), obs1
            )
            obs1 = jnp.where(
                state.red_pos[2] == 3, jnp.rot90(obs1, k=3, axes=(0, 1)), obs1
            )

            # one-hot (drop first channel as its empty blocks)
            obs1 = jax.nn.one_hot(obs1[:, :, 0], NUM_TYPES + 1)[:, :, 1:]
            angle1 = jax.nn.one_hot(obs1[:, :, 1], 4)
            obs1 = jnp.concatenate([obs1, angle1], axis=-1)

            startx = (state.blue_pos[0] + PADDING) // 2 - (OBS_SIZE // 2)
            starty = state.blue_pos[1] + PADDING
            obs2 = jax.lax.dynamic_slice(
                obs,
                start_indices=(startx, starty, jnp.int8(0)),
                slice_sizes=(OBS_SIZE, OBS_SIZE, 2),
            )

            obs2 = jnp.where(
                state.red_pos[2] == 1, jnp.rot90(obs2, k=1, axes=(0, 1)), obs2
            )
            obs2 = jnp.where(
                state.red_pos[2] == 2, jnp.rot90(obs2, k=2, axes=(0, 1)), obs2
            )
            obs2 = jnp.where(
                state.red_pos[2] == 3, jnp.rot90(obs2, k=3, axes=(0, 1)), obs2
            )
            obs2 = jax.nn.one_hot(obs2[:, :, 0], NUM_TYPES + 1)[:, :, 1:]
            angle2 = jax.nn.one_hot(obs2[:, :, 1], 4)
            obs2 = jnp.concatenate([obs2, angle2], axis=-1)

            _obs2 = obs2.at[:, :, 0].set(obs2[:, :, 1])
            _obs2 = obs2.at[:, :, 1].set(obs2[:, :, 0])
            _obs2 = obs2.at[:, :, 2].set(obs2[:, :, 3])
            _obs2 = obs2.at[:, :, 3].set(obs2[:, :, 2])
            return obs1, _obs2

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            action_0, action_1 = actions
            # turning red
            red_pos = jnp.int8(
                (state.red_pos + ROTATIONS[action_0])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4])
            )

            # moving red
            red_move = action_0 == 2
            new_red_pos = jnp.where(
                red_move, red_pos + STEP[state.red_pos[2]], red_pos
            )
            # grid boundaries
            new_red_pos = jnp.clip(
                new_red_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array(
                    [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                ),
            )

            # turning blue
            blue_pos = jnp.int8(
                (state.blue_pos + ROTATIONS[action_0])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4], dtype=jnp.int8)
            )

            # moving blue
            blue_move = action_1 == 2
            new_blue_pos = jnp.where(
                blue_move, blue_pos + STEP[state.blue_pos[2]], blue_pos
            )
            new_blue_pos = jnp.clip(
                new_blue_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array(
                    [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                ),
            )

            # if collision, priority to whoever didn't move
            collision = jnp.all(new_red_pos[:2] == new_blue_pos[:2])
            red_pos = jnp.where(
                collision * red_move * (1 - blue_move), red_pos, new_red_pos
            )
            blue_pos = jnp.where(
                collision * (1 - red_move) * blue_move, blue_pos, new_blue_pos
            )

            # if both moved, then randomise
            red_takes_square = jax.random.choice(key, jnp.array([0, 1]))
            red_pos = jnp.where(
                collision * blue_move * red_move * red_takes_square,
                new_red_pos,
                red_pos,
            )
            blue_pos = jnp.where(
                collision * (1 - red_takes_square), new_blue_pos, blue_pos
            )

            # rewards
            red_reward, blue_reward = 0.0, 0.0

            red_red_matches = jnp.all(
                red_pos[:2] == state.red_coin_pos, axis=-1
            )
            red_blue_matches = jnp.all(
                red_pos[:2] == state.blue_coin_pos, axis=-1
            )
            blue_red_matches = jnp.all(
                blue_pos[:2] == state.red_coin_pos, axis=-1
            )
            blue_blue_matches = jnp.all(
                blue_pos[:2] == state.blue_coin_pos, axis=-1
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
            new_object_pos = jax.random.choice(
                subkey,
                pos_choice,
                shape=(2,),
                replace=False,
            )

            new_red_coin_pos = jnp.where(
                jnp.logical_or(red_red_matches, blue_red_matches),
                new_object_pos[0],
                state.red_coin_pos,
            )
            new_blue_coin_pos = jnp.where(
                jnp.logical_or(red_blue_matches, blue_blue_matches),
                new_object_pos[1],
                state.blue_coin_pos,
            )

            state_nxt = EnvState(
                red_pos=red_pos,
                blue_pos=blue_pos,
                red_coin_pos=new_red_coin_pos,
                blue_coin_pos=new_blue_coin_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key, params)
            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree_map(
                lambda x, y: jax.lax.select(reset_inner, x, y),
                state_re,
                state_nxt,
            )

            obs = _state_to_obs(state)
            blue_reward = jnp.where(reset_inner, 0, blue_reward)
            red_reward = jnp.where(reset_inner, 0, red_reward)
            return (
                obs,
                state,
                (red_reward, blue_reward),
                reset_inner,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset_state(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            key, subkey = jax.random.split(key)

            object_pos = jax.random.choice(
                subkey, pos_choice, shape=(4,), replace=False
            )
            player_rot = jax.random.randint(
                subkey, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [object_pos[:2, 0], object_pos[:2, 1], player_rot]
            ).T
            coin_pos = object_pos[2:]

            return EnvState(
                red_pos=player_pos[0, :],
                blue_pos=player_pos[1, :],
                red_coin_pos=coin_pos[0, :],
                blue_coin_pos=coin_pos[1, :],
                inner_t=0,
                outer_t=0,
            )

        def reset(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            state = _reset_state(key, params)
            obs = _state_to_obs(state)
            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(reset)

        # for debugging
        self.step = _step
        self.reset = reset
        self.cnn = True

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
        _shape = (
            (OBS_SIZE, OBS_SIZE, NUM_TYPES + 4)
            if self.cnn
            else (OBS_SIZE**2 * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (GRID_SIZE, GRID_SIZE, NUM_TYPES + 4)
            if self.cnn
            else (GRID_SIZE**2 * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    def render(self, state: EnvState):
        import numpy as np
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as FigureCanvas,
        )
        from matplotlib.figure import Figure
        from PIL import Image

        """Small utility for plotting the agent's state."""
        fig = Figure((15, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(121)
        ax.imshow(
            np.zeros((GRID_SIZE + 1, GRID_SIZE + 1)),
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
        ax.set_xticks(jnp.arange(1, GRID_SIZE + 1))
        ax.set_yticks(jnp.arange(1, GRID_SIZE + 1))
        ax.grid()
        red_pos = jnp.squeeze(state.red_pos[:2])
        blue_pos = jnp.squeeze(state.blue_pos[:2])
        red_coin_pos = jnp.squeeze(state.red_coin_pos)
        blue_coin_pos = jnp.squeeze(state.blue_coin_pos)

        ax.annotate(
            "▲",
            fontsize=14,
            color="red",
            xy=(red_pos[0], red_pos[1]),
            xycoords="data",
            xytext=(red_pos[0] + 0.5, red_pos[1] + 0.5),
            rotation=float(-90 * state.red_pos[2]),
        )
        ax.annotate(
            "▲",
            fontsize=14,
            color="blue",
            xy=(blue_pos[0], blue_pos[1]),
            xycoords="data",
            xytext=(blue_pos[0] + 0.5, blue_pos[1] + 0.5),
            rotation=float(-90 * state.blue_pos[2]),
        )
        ax.annotate(
            "C",
            fontsize=14,
            color="red",
            xy=(red_coin_pos[0], red_coin_pos[1]),
            xycoords="data",
            xytext=(red_coin_pos[0] + 0.3, red_coin_pos[1] + 0.3),
        )
        ax.annotate(
            "C",
            color="blue",
            fontsize=14,
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
        # ax2.text(
        #     0.0, 0.45, "Red Coop: %s" % (state.red_coop[state.outer_t].sum())
        # )
        # ax2.text(
        #     0.6,
        #     0.45,
        #     "Red Defects : %s" % (state.red_defect[state.outer_t].sum()),
        # )
        # ax2.text(
        #     0.0, 0.25, "Blue Coop: %s" % (state.blue_coop[state.outer_t].sum())
        # )
        # ax2.text(
        #     0.6,
        #     0.25,
        #     "Blue Defects : %s" % (state.blue_defect[state.outer_t].sum()),
        # )
        # ax2.text(
        #     0.0,
        #     0.05,
        #     "Red Total: %s"
        #     % (
        #         state.red_defect[state.outer_t].sum()
        #         + state.red_coop[state.outer_t].sum()
        #     ),
        # )
        # ax2.text(
        #     0.6,
        #     0.05,
        #     "Blue Total: %s"
        #     % (
        #         state.blue_defect[state.outer_t].sum()
        #         + state.blue_coop[state.outer_t].sum()
        #     ),
        # )
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
    env = RunningWithScissors(100, 100)

    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    obs, state = env.reset(rng, params)
    pics = []

    for _ in range(100):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        a1 = jax.random.randint(rng1, (), minval=0, maxval=3)
        a2 = jax.random.randint(rng2, (), minval=0, maxval=3)
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
