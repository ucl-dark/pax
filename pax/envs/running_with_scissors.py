from enum import IntEnum
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


GRID_SIZE = 8
OBS_SIZE = 5
PADDING = OBS_SIZE - 1
NUM_TYPES = 5  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COINS = 8  # per type
NUM_COIN_TYPES = 2
NUM_OBJECTS = (
    2 + NUM_COIN_TYPES * NUM_COINS + 1
)  # red, blue, 2 red coin, 2 blue coin


@chex.dataclass
class EnvState:
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    red_inventory: jnp.ndarray
    blue_inventory: jnp.ndarray


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    interact = 3
    stay = 4


class Items(IntEnum):
    empty = 0
    red_agent = 1
    blue_agent = 2
    red_coin = 3
    blue_coin = 4
    wall = 5
    interact = 6


ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # zap
        [0, 0, 0],  # stay
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

        def _get_obs_point(x: int, y: int, dir: int) -> jnp.ndarray:
            x, y = x + PADDING, y + PADDING
            x = jnp.where(dir == 0, x - (OBS_SIZE // 2), x)
            x = jnp.where(dir == 2, x - (OBS_SIZE // 2), x)
            x = jnp.where(dir == 3, x - (OBS_SIZE - 1), x)

            y = jnp.where(dir == 1, y - (OBS_SIZE // 2), y)
            y = jnp.where(dir == 2, y - (OBS_SIZE - 1), y)
            y = jnp.where(dir == 3, y - (OBS_SIZE // 2), y)
            return x, y

        def _get_obs(state: EnvState) -> jnp.ndarray:
            # create state
            obs = jnp.pad(
                state.grid,
                ((PADDING, PADDING), (PADDING, PADDING)),
                constant_values=Items.wall,
            )
            angle = jnp.copy(obs)
            angle = angle.at[
                state.red_pos[0] + PADDING, state.red_pos[1] + PADDING
            ].set(state.red_pos[2])

            angle = angle.at[
                state.blue_pos[0] + PADDING, state.blue_pos[1] + PADDING
            ].set(state.blue_pos[2])

            obs = jnp.stack([obs, angle], axis=-1)

            x, y = _get_obs_point(
                state.red_pos[0], state.red_pos[1], state.red_pos[2]
            )

            obs1 = jax.lax.dynamic_slice(
                obs,
                start_indices=(x, y, jnp.int8(0)),
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
            obs1 = jax.nn.one_hot(obs1[:, :, 0], len(Items), dtype=jnp.int8)[
                :, :, 1:
            ]
            angle1 = (obs1[:, :, 1] - state.red_pos[3]) % 4
            angle1 = jax.nn.one_hot(angle1, 4)
            obs1 = jnp.concatenate([obs1, angle1], axis=-1)

            x, y = _get_obs_point(
                state.blue_pos[0], state.blue_pos[1], state.blue_pos[2]
            )

            obs2 = jax.lax.dynamic_slice(
                obs,
                start_indices=(x, y, jnp.int8(0)),
                slice_sizes=(OBS_SIZE, OBS_SIZE, 2),
            )

            obs2 = jnp.where(
                state.blue_pos[2] == 1, jnp.rot90(obs2, k=1, axes=(0, 1)), obs2
            )
            obs2 = jnp.where(
                state.blue_pos[2] == 2, jnp.rot90(obs2, k=2, axes=(0, 1)), obs2
            )
            obs2 = jnp.where(
                state.blue_pos[2] == 3, jnp.rot90(obs2, k=3, axes=(0, 1)), obs2
            )
            obs2 = jax.nn.one_hot(obs2[:, :, 0], len(Items), dtype=jnp.int8)[
                :, :, 1:
            ]
            angle2 = (obs2[:, :, 1] - state.blue_pos[3]) % 4
            angle2 = jax.nn.one_hot(angle2, 4)
            obs2 = jnp.concatenate([obs2, angle2], axis=-1)

            _obs2 = obs2.at[:, :, 0].set(obs2[:, :, 1])
            _obs2 = obs2.at[:, :, 1].set(obs2[:, :, 0])
            _obs2 = obs2.at[:, :, 2].set(obs2[:, :, 3])
            _obs2 = obs2.at[:, :, 3].set(obs2[:, :, 2])
            return {"observation": obs1, "inventory": state.red_inventory}, {
                "observation": _obs2,
                "inventory": state.blue_inventory,
            }

        def _get_reward(state: EnvState, params: EnvParams) -> jnp.ndarray:
            inv1 = state.red_inventory / state.red_inventory.sum()
            inv2 = state.blue_inventory / state.blue_inventory.sum()
            r1 = inv1 @ params.payoff_matrix @ inv2.T
            r2 = inv1 @ params.payoff_matrix.T @ inv2.T
            return r1, r2

        def _interact(
            state: EnvState, actions: Tuple[int, int], params: EnvParams
        ) -> Tuple[float, float, EnvState]:
            # if interact
            a0, a1 = actions
            red_zap = a0 == Actions.interact
            blue_zap = a1 == Actions.interact
            interact_idx = jnp.int8(Items.interact)

            # remove old interacts
            state.grid = jnp.where(
                state.grid == interact_idx, jnp.int8(Items.empty), state.grid
            )

            # check 1 ahead
            red_target = state.red_pos + STEP[state.red_pos[2]]
            blue_target = state.blue_pos + STEP[state.blue_pos[2]]

            red_interact = (
                state.grid[red_target[0], red_target[1]] == Items.blue_agent
            )
            blue_interact = (
                state.grid[blue_target[0], blue_target[1]] == Items.red_agent
            )

            # check 2 ahead
            red_target_ahead = jnp.clip(
                state.red_pos + 2 * STEP[state.red_pos[2]], 0, GRID_SIZE
            )
            blue_target_ahead = jnp.clip(
                state.blue_pos + 2 * STEP[state.blue_pos[2]], 0, GRID_SIZE
            )

            red_interact_ahead = (
                state.grid[red_target_ahead[0], red_target_ahead[1]]
                == Items.blue_agent
            )
            blue_interact_ahead = (
                state.grid[blue_target_ahead[0], blue_target_ahead[1]]
                == Items.red_agent
            )

            # check to your right
            red_target_right = red_target + STEP[(state.red_pos[2] + 1) % 4]
            blue_target_right = blue_target + STEP[(state.blue_pos[2] + 1) % 4]

            red_interact_right = (
                state.grid[red_target_right[0], red_target_right[1]]
                == Items.blue_agent
            )
            blue_interact_right = (
                state.grid[blue_target_right[0], blue_target_right[1]]
                == Items.red_agent
            )

            # check to your left
            red_target_left = red_target + STEP[(state.red_pos[2] - 1) % 4]
            blue_target_left = blue_target + STEP[(state.blue_pos[2] - 1) % 4]

            red_interact_left = (
                state.grid[red_target_left[0], red_target_left[1]]
                == Items.blue_agent
            )
            blue_interact_left = (
                state.grid[blue_target_left[0], blue_target_left[1]]
                == Items.red_agent
            )

            red_interact = jnp.logical_or(
                red_interact,
                jnp.logical_or(
                    red_interact_ahead,
                    jnp.logical_or(red_interact_right, red_interact_left),
                ),
            )

            # update grid with red zaps
            aux_grid = jnp.copy(state.grid)

            item = jnp.where(
                state.grid[red_target[0], red_target[1]],
                state.grid[red_target[0], red_target[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target[0], red_target[1]].set(item)

            item = jnp.where(
                state.grid[red_target_ahead[0], red_target_ahead[1]],
                state.grid[red_target_ahead[0], red_target_ahead[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                red_target_ahead[0], red_target_ahead[1]
            ].set(item)

            item = jnp.where(
                state.grid[red_target_right[0], red_target_right[1]],
                state.grid[red_target_right[0], red_target_right[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                red_target_right[0], red_target_right[1]
            ].set(item)

            item = jnp.where(
                state.grid[red_target_left[0], red_target_left[1]],
                state.grid[red_target_left[0], red_target_left[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target_left[0], red_target_left[1]].set(
                item
            )

            state.grid = jnp.where(red_zap, aux_grid, state.grid)

            # update grid with blue zaps
            aux_grid = jnp.copy(state.grid)
            blue_interact = jnp.logical_or(
                blue_interact,
                jnp.logical_or(
                    blue_interact_ahead,
                    jnp.logical_or(blue_interact_right, blue_interact_left),
                ),
            )

            item = jnp.where(
                state.grid[blue_target[0], blue_target[1]],
                state.grid[blue_target[0], blue_target[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[blue_target[0], blue_target[1]].set(item)

            item = jnp.where(
                state.grid[blue_target_ahead[0], blue_target_ahead[1]],
                state.grid[blue_target_ahead[0], blue_target_ahead[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                blue_target_ahead[0], blue_target_ahead[1]
            ].set(item)

            item = jnp.where(
                state.grid[blue_target_right[0], blue_target_right[1]],
                state.grid[blue_target_right[0], blue_target_right[1]],
                interact_idx,
            )
            state.grid = aux_grid.at[
                blue_target_right[0], blue_target_right[1]
            ].set(item)

            item = jnp.where(
                state.grid[blue_target_left[0], blue_target_left[1]],
                state.grid[blue_target_left[0], blue_target_left[1]],
                interact_idx,
            )
            state.grid = state.grid.at[
                blue_target_left[0], blue_target_left[1]
            ].set(item)

            # rewards
            red_reward, blue_reward = 0.0, 0.0
            _r_reward, _b_reward = _get_reward(state, params)

            red_reward = jnp.where(
                red_zap * red_interact, red_reward + _r_reward, red_reward
            )
            blue_reward = jnp.where(
                blue_zap * blue_interact, blue_reward + _b_reward, blue_reward
            )
            return red_reward, blue_reward, state

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            action_0, action_1 = actions
            # turning red
            new_red_pos = jnp.int8(
                (state.red_pos + ROTATIONS[action_0])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4])
            )

            # moving red
            red_move = action_0 == Actions.forward
            new_red_pos = jnp.where(
                red_move, new_red_pos + STEP[state.red_pos[2]], new_red_pos
            )
            new_red_pos = jnp.clip(
                new_red_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array(
                    [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                ),
            )

            # turning blue
            new_blue_pos = jnp.int8(
                (state.blue_pos + ROTATIONS[action_1])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4], dtype=jnp.int8)
            )

            # moving blue
            blue_move = action_1 == Actions.forward
            new_blue_pos = jnp.where(
                blue_move, new_blue_pos + STEP[state.blue_pos[2]], new_blue_pos
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
            new_red_pos = jnp.where(
                collision * red_move * (1 - blue_move),
                new_red_pos,
                new_red_pos,
            )
            new_blue_pos = jnp.where(
                collision * (1 - red_move) * blue_move,
                new_blue_pos,
                new_blue_pos,
            )

            # if both moved, then randomise
            red_takes_square = jax.random.choice(key, jnp.array([0, 1]))
            new_red_pos = jnp.where(
                collision * blue_move * red_move * red_takes_square,
                new_red_pos,
                new_red_pos,
            )
            new_blue_pos = jnp.where(
                collision * blue_move * red_move * (1 - red_takes_square),
                new_blue_pos,
                new_blue_pos,
            )
            # update inventories
            red_red_matches = (
                state.grid[new_red_pos[0], new_red_pos[1]] == Items.red_coin
            )
            red_blue_matches = (
                state.grid[new_red_pos[0], new_red_pos[1]] == Items.blue_coin
            )
            blue_red_matches = (
                state.grid[new_blue_pos[0], new_blue_pos[1]] == Items.red_coin
            )
            blue_blue_matches = (
                state.grid[new_blue_pos[0], new_blue_pos[1]] == Items.blue_coin
            )

            state.red_inventory = state.red_inventory + jnp.array(
                [red_red_matches, red_blue_matches]
            )
            state.blue_inventory = state.blue_inventory + jnp.array(
                [blue_blue_matches, blue_red_matches]
            )

            # update grid
            state.grid = state.grid.at[
                (state.red_pos[0], state.red_pos[1])
            ].set(0)
            state.grid = state.grid.at[
                (state.blue_pos[0], state.blue_pos[1])
            ].set(0)
            state.grid = state.grid.at[(new_red_pos[0], new_red_pos[1])].set(1)
            state.grid = state.grid.at[(new_blue_pos[0], new_blue_pos[1])].set(
                2
            )
            state.red_pos = new_red_pos
            state.blue_pos = new_blue_pos

            # if interact
            red_reward, blue_reward, state = _interact(state, actions, params)

            state_nxt = EnvState(
                red_pos=state.red_pos,
                blue_pos=state.blue_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                red_inventory=state.red_inventory,
                blue_inventory=state.blue_inventory,
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

            obs = _get_obs(state)
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
                subkey, COORDS, shape=(NUM_OBJECTS,), replace=False
            )
            player_dir = jax.random.randint(
                subkey, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [object_pos[:2, 0], object_pos[:2, 1], player_dir]
            ).T
            coin_pos = object_pos[2:]
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)
            grid = grid.at[player_pos[0, 0], player_pos[0, 1]].set(1)
            grid = grid.at[player_pos[1, 0], player_pos[1, 1]].set(2)
            for i in range(NUM_COINS):
                grid = grid.at[coin_pos[i, 0], coin_pos[i, 1]].set(3)

            for i in range(NUM_COINS):
                grid = grid.at[coin_pos[NUM_COINS + i, 0], coin_pos[i, 1]].set(
                    4
                )
            return EnvState(
                red_pos=player_pos[0, :],
                blue_pos=player_pos[1, :],
                inner_t=0,
                outer_t=0,
                grid=grid,
                red_inventory=jnp.ones(2),
                blue_inventory=jnp.ones(2),
            )

        def reset(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            state = _reset_state(key, params)
            obs = _get_obs(state)
            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(reset)
        self.get_obs_point = _get_obs_point

        # for debugging
        self.step = _step
        self.reset = reset
        self.get_obs = _get_obs
        self.cnn = True

    @property
    def name(self) -> str:
        """Environment name."""
        return "CoinGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """Observation space of the environment."""
        _shape = (
            (OBS_SIZE, OBS_SIZE, len(Items) - 1 + 4)
            if self.cnn
            else (OBS_SIZE**2 * (len(Items) - 1 + 4),)
        )

        return {
            "observation": spaces.Box(
                low=0, high=1, shape=_shape, dtype=jnp.uint8
            ),
            "inventory": spaces.Box(
                low=0, high=NUM_COINS, shape=NUM_COIN_TYPES, dtype=jnp.uint8
            ),
        }

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

        if obj == Items.red_agent:
            # Draw the agent 1
            agent_color = (31.0, 119.0, 180.0)
        elif obj == Items.blue_agent:
            # Draw agent 2
            agent_color = (255.0, 127.0, 14.0)
        elif obj == Items.red_coin:
            # Draw the red coin
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (44.0, 160.0, 44.0)
            )
        elif obj == Items.blue_coin:
            # Draw the blue coin
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

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

    def render_agent_view(
        self, state: EnvState, agent: int
    ) -> Tuple[onp.ndarray]:
        """
        Render the observation for each agent"""

        tile_size = 32
        obs = self.get_obs(state)

        grid = onp.array(obs[agent]["observation"])
        empty_space_channel = onp.zeros((OBS_SIZE, OBS_SIZE, 1))
        grid = onp.concatenate((empty_space_channel, grid), axis=-1)
        grid = onp.argmax(grid.reshape(-1, grid.shape[-1]), axis=1)
        grid = grid.reshape(OBS_SIZE, OBS_SIZE)

        # Compute the total grid size
        width_px = grid.shape[0] * tile_size
        height_px = grid.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)

        red_dir = state.red_pos[2].item() if agent == 1 else 0
        blue_dir = state.blue_pos[2].item() if agent == 0 else 0

        print(env.action_space().n)

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                red_agent_here = cell == 1
                blue_agent_here = cell == 2

                agent_dir = None

                agent_dir = red_dir if red_agent_here else agent_dir
                agent_dir = blue_dir if blue_agent_here else agent_dir

                tile_img = RunningWithScissors.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    highlight=None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return onp.rot90(img, 2, axes=(0, 1))

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

        startx, starty = self.get_obs_point(
            state.red_pos[0], state.red_pos[1], state.red_pos[2]
        )
        highlight_mask[
            startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
        ] = True

        startx, starty = self.get_obs_point(
            state.blue_pos[0], state.blue_pos[1], state.blue_pos[2]
        )
        highlight_mask[
            startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
        ] = True

        # Compute the total grid size
        width_px = GRID.shape[0] * tile_size
        height_px = GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        grid = onp.pad(
            grid, ((PADDING, PADDING), (PADDING, PADDING)), constant_values=5
        )
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

        return onp.rot90(
            img[
                (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
                (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
                :,
            ],
            2,
        )


if __name__ == "__main__":
    from PIL import Image

    action = 1
    rng = jax.random.PRNGKey(0)
    env = RunningWithScissors(100, 100)
    num_actions = env.action_space().n
    params = EnvParams(payoff_matrix=jnp.array([[3, 0], [5, 1]]))
    obs, state = env.reset(rng, params)
    pics = []
    pics1 = []
    pics2 = []

    img = env.render(state)
    img1 = env.render_agent_view(state, agent=0)
    img2 = env.render_agent_view(state, agent=1)
    pics.append(img)

    int_action = {
        0: "left",
        1: "right",
        2: "forward",
        3: "interact",
        4: "stay",
    }
    env.step = jax.jit(env.step)
    for t in range(20):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        # a1 = jnp.array(2)
        # a2 = jnp.array(2)
        a1 = jax.random.choice(
            rng1, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
        )
        a2 = jax.random.choice(
            rng2, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
        )
        obs, state, reward, done, info = env.step(
            rng, state, (a1 * action, a2 * action), params
        )
        print(t, int_action[a1.item()], int_action[a2.item()])
        print(reward[0], reward[1])

        img = env.render(state)
        img1 = env.render_agent_view(state, agent=0)
        img2 = env.render_agent_view(state, agent=1)

        pics.append(img)
        pics1.append(img1)
        pics2.append(img2)

    pics = [Image.fromarray(img) for img in pics]
    pics1 = [Image.fromarray(img) for img in pics1]
    pics2 = [Image.fromarray(img) for img in pics2]

    pics[0].save(
        "state.gif",
        format="gif",
        save_all=True,
        append_images=pics[1:],
        duration=300,
        loop=0,
    )

    pics1[0].save(
        "agent1.gif",
        format="gif",
        save_all=True,
        append_images=pics1[1:],
        duration=300,
        loop=0,
    )
    pics2[0].save(
        "agent2.gif",
        format="gif",
        save_all=True,
        append_images=pics2[1:],
        duration=300,
        loop=0,
    )
