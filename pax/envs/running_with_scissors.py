from ast import Num
import math
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from gymnax.environments import environment, spaces

from pax.envs.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)


GRID_SIZE = 3
OBS_SIZE = 5
PADDING = OBS_SIZE - 1
NUM_TYPES = 5  # red, blue, red coin, blue coin, wall
NUM_OBJECTS = 4  # red, blue, red coin, blue coin


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

GRID = jnp.zeros(
    (GRID_SIZE + 2 * PADDING, GRID_SIZE + 2 * PADDING),
    dtype=jnp.int8,
)


# First layer of Padding is Wall
GRID = GRID.at[PADDING - 1, :].set(5)
GRID = GRID.at[GRID_SIZE + PADDING, :].set(5)
GRID = GRID.at[:, PADDING - 1].set(5)
GRID = GRID.at[:, GRID_SIZE + PADDING].set(5)

COORDS = jnp.array(
    [[(j, i) for i in range(GRID_SIZE)] for j in range(GRID_SIZE)],
    dtype=jnp.int8,
).reshape(-1, 2)


class RunningWithScissors(environment.Environment):
    """
    JAX Compatible version of coin game environment.
    0. Make the environment larger (10x10)
    1. We make the environment 2D with orientation
    2. Agents can not stand or move onto same square (coins can respawn there tho)
    2. Then we add a fire action
    3. Then we add variable number of coins
    """

    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps: int,
        num_outer_steps: int,
    ):

        super().__init__()

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

            angle = jnp.copy(GRID)
            angle = angle.at[
                state.red_pos[0] + PADDING, state.red_pos[1] + PADDING
            ].set(state.red_pos[2])

            angle = angle.at[
                state.blue_pos[0] + PADDING, state.blue_pos[1] + PADDING
            ].set(state.blue_pos[2])

            obs = jnp.stack([obs, angle], axis=-1)

            startx = (state.red_pos[0] + PADDING) - (OBS_SIZE // 2)
            starty = (state.red_pos[1] + PADDING) - (OBS_SIZE // 2)
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

            startx = (state.blue_pos[0] + PADDING) - (OBS_SIZE // 2)
            starty = (state.blue_pos[1] + PADDING) - (OBS_SIZE // 2)
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
            new_red_pos = jnp.clip(
                new_red_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array(
                    [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                ),
            )

            # turning blue
            blue_pos = jnp.int8(
                (state.blue_pos + ROTATIONS[action_1])
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
                collision * blue_move * red_move * (1 - red_takes_square),
                new_blue_pos,
                blue_pos,
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

            # respawn coins

            # find free space
            red_idx = jnp.array([GRID_SIZE * red_pos[0] + red_pos[1]])
            blue_idx = jnp.array([GRID_SIZE * blue_pos[0] + blue_pos[1]])
            rc_idx = jnp.array(
                [GRID_SIZE * state.red_coin_pos[0] + state.red_coin_pos[1]]
            )
            bc_idx = jnp.array(
                [GRID_SIZE * state.blue_coin_pos[0] + state.blue_coin_pos[1]]
            )

            free_space = jnp.zeros((GRID_SIZE * GRID_SIZE,), dtype=jnp.int8)
            free_space = free_space.at[red_idx].set(1)
            free_space = free_space.at[blue_idx].set(1)
            free_space = free_space.at[rc_idx].set(1)
            free_space = free_space.at[bc_idx].set(1)
            free_space = jnp.logical_not(free_space)
            empty_idx = free_space.nonzero(size=GRID_SIZE**2 - NUM_OBJECTS)[
                0
            ]

            # sample new coins
            key, subkey = jax.random.split(key)
            new_object_pos = jax.random.choice(
                subkey,
                empty_idx,
                shape=(2,),
                replace=False,
            )
            # both new coins guaranteed to be different locations and not on existing coins!
            new_red_coin_pos = jnp.where(
                jnp.logical_or(red_red_matches, blue_red_matches),
                COORDS[new_object_pos[0]],
                state.red_coin_pos,
            )
            new_blue_coin_pos = jnp.where(
                jnp.logical_or(red_blue_matches, blue_blue_matches),
                COORDS[new_object_pos[1]],
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
                subkey, COORDS, shape=(4,), replace=False
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
        return 3

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

    @classmethod
    def render_tile(
        cls,
        obj: int,
        agent_dir: Union[int, None] = None,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        if obj:
            key = (obj, 0, 0) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = onp.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            dtype=onp.uint8,
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj == 1:
            # Draw the agent 1
            agent_color = (255, 0, 0)
        elif obj == 2:
            # Draw agent 2
            agent_color = (0, 0, 255)
        elif obj == 3:
            # Draw the red coin
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (255, 0, 0))
        elif obj == 4:
            # Draw the blue coin
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (0, 0, 255))
        elif obj == 5:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (225, 193, 110))

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        state: EnvState,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(GRID))

        x = state.red_pos[0] + PADDING
        y = state.red_pos[1] + PADDING

        startx = x - (OBS_SIZE // 2)
        starty = y - (OBS_SIZE // 2)
        highlight_mask[
            startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
        ] = True

        x = state.blue_pos[0] + PADDING
        y = state.blue_pos[1] + PADDING

        startx = x - (OBS_SIZE // 2)
        starty = y - (OBS_SIZE // 2)
        highlight_mask[
            startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
        ] = True

        # Compute the total grid size

        width_px = GRID.shape[0] * tile_size
        height_px = GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(GRID)
        grid[state.red_pos[0] + PADDING, state.red_pos[1] + PADDING] = 1
        grid[state.blue_pos[0] + PADDING, state.blue_pos[1] + PADDING] = 2
        grid[
            state.red_coin_pos[0] + PADDING, state.red_coin_pos[1] + PADDING
        ] = 3
        grid[
            state.blue_coin_pos[0] + PADDING, state.blue_coin_pos[1] + PADDING
        ] = 4
        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                red_agent_here = cell == 1
                blue_agent_here = cell == 2

                agent_dir = None
                agent_dir = (
                    state.red_pos[2].item() if red_agent_here else agent_dir
                )
                agent_dir = (
                    state.blue_pos[2].item() if blue_agent_here else agent_dir
                )

                tile_img = RunningWithScissors.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        return img[
            (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
            (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
            :,
        ]


if __name__ == "__main__":
    from PIL import Image

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
    pics = [Image.fromarray(img) for img in pics]

    pics[0].save(
        "test1.gif",
        format="gif",
        save_all=True,
        append_images=pics[1:],
        duration=300,
        loop=0,
    )
