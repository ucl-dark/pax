import math
from enum import IntEnum
from typing import Any, Optional, Tuple, Union, List

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
NUM_COINS = 6  # per type
NUM_COIN_TYPES = 2
NUM_OBJECTS = (
    2 + NUM_COIN_TYPES * NUM_COINS + 1
)  # red, blue, 2 red coin, 2 blue coin
NUM_AGENTS = 8

INTERACT_THRESHOLD = 0

# PLAYER3_COLOUR = (236.0, 64.0, 122.0) #kinda pink
# PLAYER4_COLOUR = (255.0, 235.0, 59.0) #yellow
# PLAYER5_COLOUR = (41.0, 182.0, 246.0) #baby blue
# PLAYER6_COLOUR = (171.0, 71.0, 188.0) #purple
# PLAYER7_COLOUR = (121.0, 85.0, 72.0) #brown
# PLAYER8_COLOUR = (255.0, 205.0, 210.0) #salmon

@chex.dataclass
class EnvState:
    # n players x 2
    agent_positions: List[jnp.ndarray]
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    # n players x 2
    agent_inventories: List[jnp.ndarray]
    # coin positions
    coin_coop: List[jnp.ndarray]
    coin_defect: List[jnp.ndarray]
    agent_freezes: List[int]


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    interact = 3
    stay = 4

# PLAYER3_COLOUR = (236.0, 64.0, 122.0) #kinda pink
# PLAYER4_COLOUR = (255.0, 235.0, 59.0) #yellow
# PLAYER5_COLOUR = (41.0, 182.0, 246.0) #baby blue
# PLAYER6_COLOUR = (171.0, 71.0, 188.0) #purple
# PLAYER7_COLOUR = (121.0, 85.0, 72.0) #brown
# PLAYER8_COLOUR = (255.0, 205.0, 210.0) #salmon

class Items(IntEnum):
    empty = 0
    agent1 = 1
    agent2 = 2
    agent3 = 3
    agent4 = 4
    agent5 = 5
    agent6 = 6
    agent7 = 7
    agent8 = 8
    cooperation_coin = 9
    defection_coin = 10
    wall = 11
    interact = 12
    #agents if  1 < val < item.WALL 

ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap`
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

COIN_SPAWNS = [
    [1, 1],
    [1, 2],
    [2, 1],
    [1, GRID_SIZE - 2],
    [2, GRID_SIZE - 2],
    [1, GRID_SIZE - 3],
    # [2, 2],
    # [2, GRID_SIZE - 3],
    [GRID_SIZE - 2, 2],
    [GRID_SIZE - 3, 1],
    [GRID_SIZE - 2, 1],
    [GRID_SIZE - 2, GRID_SIZE - 2],
    [GRID_SIZE - 2, GRID_SIZE - 3],
    [GRID_SIZE - 3, GRID_SIZE - 2],
    # [GRID_SIZE - 3, 2],
    # [GRID_SIZE - 3, GRID_SIZE - 3],
]

COIN_SPAWNS = jnp.array(
    COIN_SPAWNS,
    dtype=jnp.int8,
)

# Coins
RED_SPAWN = jnp.array(
    COIN_SPAWNS[::2, :],
    dtype=jnp.int8,
)
# Coins
BLUE_SPAWN = jnp.array(
    COIN_SPAWNS[1::2, :],
    dtype=jnp.int8,
)

#have them spawn in more places
AGENT_SPAWNS = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    # [1, 1],
    [1, 2],
    [0, GRID_SIZE - 1],
    [0, GRID_SIZE - 2],
    [0, GRID_SIZE - 3],
    [1, GRID_SIZE - 1],
    # [1, GRID_SIZE - 2],
    # [1, GRID_SIZE - 3],
    [GRID_SIZE - 1, 0],
    [GRID_SIZE - 1, 1],
    [GRID_SIZE - 1, 2],
    [GRID_SIZE - 2, 0],
    # [GRID_SIZE - 2, 1],
    # [GRID_SIZE - 2, 2],
    [GRID_SIZE - 1, GRID_SIZE - 1],
    [GRID_SIZE - 1, GRID_SIZE - 2],
    [GRID_SIZE - 1, GRID_SIZE - 3],
    [GRID_SIZE - 2, GRID_SIZE - 1],
    # [GRID_SIZE - 2, GRID_SIZE - 2],
    [GRID_SIZE - 2, GRID_SIZE - 3],
]

# spawn agents in opposite diagonals
# for n agents probably better just randomly
# rng choose expensive
AGENT_SPAWNS = jnp.array(
    [
        [(j, i), (GRID_SIZE - 1 - j, GRID_SIZE - 1 - i)]
        for (i, j) in AGENT_SPAWNS
    ],
    dtype=jnp.int8,
).reshape(-1, 2, 2)


PLAYER1_COLOUR = (255.0, 127.0, 14.0) #kinda orange
PLAYER2_COLOUR = (31.0, 119.0, 180.0) #kinda blue
PLAYER3_COLOUR = (236.0, 64.0, 122.0) #kinda pink
PLAYER4_COLOUR = (255.0, 235.0, 59.0) #yellow
PLAYER5_COLOUR = (41.0, 182.0, 246.0) #baby blue
PLAYER6_COLOUR = (171.0, 71.0, 188.0) #purple
PLAYER7_COLOUR = (121.0, 85.0, 72.0) #brown
PLAYER8_COLOUR = (255.0, 205.0, 210.0) #salmon
GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)


class InTheMatrix(environment.Environment):
    """
    JAX Compatible version of *inTheMatix environment.
    """

    # used for caching
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps: int,
        num_outer_steps: int,
        num_agents: int,
        fixed_coin_location: bool,
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
            grid = jnp.pad(
                state.grid,
                ((PADDING, PADDING), (PADDING, PADDING)),
                constant_values=Items.wall,
            )
            def mini_obs(agent_pos, frozen, grid, agent_idx):
                x, y = _get_obs_point(
                agent_pos[0], agent_pos[1], agent_pos[2]
                )
                grid1 = jax.lax.dynamic_slice(
                    grid,
                    start_indices=(x, y),
                    slice_sizes=(OBS_SIZE, OBS_SIZE),
                )
                # rotate
                grid1 = jnp.where(
                    agent_pos[2] == 1,
                    jnp.rot90(grid1, k=1, axes=(0, 1)),
                    grid1,
                )
                grid1 = jnp.where(
                    agent_pos[2] == 2,
                    jnp.rot90(grid1, k=2, axes=(0, 1)),
                    grid1,
                )
                grid1 = jnp.where(
                    agent_pos[2] == 3,
                    jnp.rot90(grid1, k=3, axes=(0, 1)),
                    grid1,
                )
                angle1 = -1 * jnp.ones_like(grid1, dtype=jnp.int8)
            #      i look where the blue agent is (with its own orientation)
            #     i correct its orientation relative to mine
            #     angle is 
                all_items = jnp.arange(1, num_agents+1)
                all_items = all_items[all_items != agent_idx]

                def check_angle(grid1, item, other_agent_pos):    
                    angle1 = jnp.where(
                        grid1 == item,
                        (other_agent_pos[2] - agent_pos[2]) % 4,
                        -1,
                    )
                    angle1 = jax.nn.one_hot(angle1, 4)
                    return angle1
                vmap_check_angle = vmap(check_angle, (None, 0, 0), 0)
                vmap_check_angle(grid1, all_items, state.agent_positions)

                # one-hot (drop first channel as its empty blocks)
                grid1 = jax.nn.one_hot(grid1 - 1, len(Items) - 1, dtype=jnp.int8)
                _grid1 = grid1.at[:, :, 0].set(grid1[:, :, agent_idx])
                _grid1 = _grid1.at[:, :, agent_idx].set(grid1[:, :, 0])
                obs1 = jnp.concatenate([_grid1, angle1], axis=-1)
                return obs1

            # has red/blue picked up something
            # can also be vmapped
            # state.inventories : [num_player x 2]
            # agent.pos : [num_player x 2]
            # vmap over first dim
            pickups = jnp.sum(state.agent_inventories, axis=1) > INTERACT_THRESHOLD
            # red_pickup = jnp.sum(state.red_inventory) > INTERACT_THRESHOLD
            # blue_pickup = jnp.sum(state.blue_inventory) > INTERACT_THRESHOLD
            # state.frozen = [0, 0, 0, 0, 1, 0, 0]
            # work out everyones inventory = 
            # state.inventory = [n_players x 2]
            # stuff_i_can_see = jnp.where(agent==grid, 1, 0)
            # inventories_to_show = stuff_i_can_see * state.frozen * state.inventory

            
            def agent2show(inventory, freeze):
                agent_to_show = jnp.where(
                    freeze >= 0, agent_inventory, 0
                )
                return agent_to_show

            vmap_agent2show = vmap(agent2show, (0, 0), (0))
            agents2show = vmap_agent2_show(state.agent_inventories, state.agent_freezes)
            # mask over all agents that are frozen
            return {
                "observations": obs,
                "inventory": jnp.array(
                    [
                        state.agent_inventories,
                        pickups,
                        agents2show,
                        # red_pickup,
                        # blue_pickup,
                        # blue_to_show[0],
                        # blue_to_show[1],
                    ],
                    dtype=jnp.int8,
                ),
            }

        def _get_reward(agent1_idx, agent2_idx, params: EnvParams) -> jnp.ndarray:
            inv1 = state.agent_inventories[agent1_idx] / state.agent_inventories[agent1_idx].sum()
            inv2 = state.agent_inventories[agent2_idx] / state.agent_inventories[agent2_idx].sum()
            r1 = inv1 @ params.payoff_matrix[0] @ inv2.T
            r2 = inv1 @ params.payoff_matrix[1] @ inv2.T
            return jnp.array([r1, r2])
        # pairwise??
        def _interact(
            state: EnvState, actions: jnp.array, params: EnvParams
        ) -> Tuple[bool, float, float, EnvState]:
            # if interact
            # a0, a1 = actions

            # red_zap = a0 == Actions.interact
            # blue_zap = a1 == Actions.interact
            interact_idx = jnp.int8(Items.interact)
            # zaps = actions == Actions.interact

            # remove old interacts
            state.grid = jnp.where(
                state.grid == interact_idx, jnp.int8(Items.empty), state.grid
            )
            def mini_interact(agent_pos, action, item):
                zap = action == Actions.interact
                # check 1 ahead
                target = jnp.clip(
                    agent_pos + STEP[agent_pos[2]], 0, GRID_SIZE - 1
                )
                interact = (
                    state.grid[target[0], target[1]] == item
                )
                target_ahead = jnp.clip(
                    agent_pos + 2 * STEP[agent_pos[2]], 0, GRID_SIZE - 1
                )
                interact_ahead = (
                    state.grid[target_ahead[0], target_ahead[1]]
                    == item
                )
                # check to your right  - clip can't be used here as it will wrap down
                target_right = (
                    agent_pos
                    + STEP[agent_pos[2]]
                    + STEP[(agent_pos[2] + 1) % 4]
                )
                oob = jnp.logical_or(
                    (target_right > GRID_SIZE - 1).any(),
                    (target_right < 0).any(),
                )
                target_right = jnp.where(oob, target, target_right)

                interact_right = (
                    state.grid[target_right[0], target_right[1]]
                    == item
                )
                target_left = (
                    agent_pos
                    + STEP[agent_pos[2]]
                    + STEP[(agent_pos[2] - 1) % 4]
                )
                oob = jnp.logical_or(
                    (target_left > GRID_SIZE - 1).any(),
                    (target_left < 0).any(),
                )
                target_left = jnp.where(oob, target, target_left)
                interact_left = (
                    state.grid[target_left[0], target_left[1]]
                    == item
                )
                interact = jnp.logical_or(
                    interact,
                    jnp.logical_or(
                        interact_ahead,
                        jnp.logical_or(interact_right, interact_left),
                    ),
                )
                # update grid with red zaps
                aux_grid = jnp.copy(state.grid)

                item = jnp.where(
                    state.grid[target[0], target[1]],
                    state.grid[target[0], target[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[target[0], target[1]].set(item)

                item = jnp.where(
                    state.grid[target_ahead[0], target_ahead[1]],
                    state.grid[target_ahead[0], target_ahead[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[
                    target_ahead[0], target_ahead[1]
                ].set(item)

                item = jnp.where(
                    state.grid[target_right[0], target_right[1]],
                    state.grid[target_right[0], target_right[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[
                    target_right[0], target_right[1]
                ].set(item)

                item = jnp.where(
                    state.grid[target_left[0], target_left[1]],
                    state.grid[target_left[0], target_left[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[target_left[0], target_left[1]].set(
                    item
                )
                # not 100% sure how to do this
                aux_grid = jnp.where(zap, aux_grid, state.grid)
                interact = zap * interact
                return aux_grid, interact

            #inputs: agent_pos, action, item
            #outputs: aux_grid, zap, interact
            # first vmap: vmap over different opponents
            # second vmap: vmap over all agents
            vmap_interact = vmap(vmap(mini_interact, (None, None, 0), (0, 0)), (0, 0, None), (0, 0))

            all_items = jnp.arange(1, num_agents+1)
            all_items = all_items[all_items != agent_idx]
            aux_grids, interacts = vmap_interact(state.agent_positions, actions, all_items)

            # rewards
            all_idx = jnp.arange(0, num_agents)
            rewards =  jnp.zeros((num_agents, ))
            vmap_get_rewards = vmap(vmap(_get_reward, (0, None, None), 0), (None, 0, None), 0)
            rewards = _get_reward(all_idx, all_idx, params)

            # single player red_zap * red_interact
            interact = jnp.sum(interacts)

            pickups = jnp.sum(state.agent_inventories, axis=1) > INTERACT_THRESHOLD

            interact = jnp.logical_and(
                interact, jnp.logical_and(red_pickup, blue_pickup)
            )
            rewards = 
            red_reward = jnp.where(
                interact, red_reward + _r_reward, red_reward
            )
            blue_reward = jnp.where(
                interact, blue_reward + _b_reward, blue_reward
            )
            return interact, red_reward, blue_reward, state

        def _interact(state: EnvState, actions: jnp.array, params: EnvParams, rng_key: jnp.ndarray) -> Tuple[bool, jnp.array, EnvState]:
            interact_idx = jnp.int8(Items.interact)

            # Remove old interacts from the grid
            state.grid = jnp.where(
                state.grid == interact_idx, jnp.int8(Items.empty), state.grid
            )

            # Get agent positions and orientations
            agent_positions = state.agent_positions[:, :2]
            # n x 3 array 
            agent_orientations = state.agent_positions[:, 2]

            # Calculate all possible agent pairs and their corresponding interact positions
            agent_pairs = jnp.array(list(itertools.combinations(range(num_agents), 2)))
            agent_pair_positions = agent_positions[agent_pairs]

            # Compute interact positions for each agent pair
            interact_positions = agent_pair_positions + jnp.expand_dims(STEP[agent_orientations[agent_pairs[:, 0]]], axis=1)

            # Check if the interact positions are within the grid boundaries
            valid_positions = jnp.all(jnp.logical_and(interact_positions >= 0, interact_positions < GRID_SIZE), axis=-1)

            # Calculate the interact action for each agent
            agent_interact_actions = actions == Actions.interact

            # Determine which agent pairs are interacting
            interacting_pairs = jnp.logical_and(agent_interact_actions[agent_pairs[:, 0]], agent_interact_actions[agent_pairs[:, 1]])

            # Filter valid and interacting pairs
            valid_interacting_pairs = jnp.logical_and(valid_positions, interacting_pairs)

            # Sort the agent pairs randomly using the provided rng_key
            random_key, subkey = random.split(rng_key)
            shuffled_indices = random.permutation(subkey, jnp.arange(valid_interacting_pairs.shape[0]))

            # Create a boolean array to track whether each agent is in an interaction
            agent_interaction_mask = jnp.zeros((num_agents,), dtype=bool)

            # Define a function to process each pair of agents and update the agent_interaction_mask
            def process_pair(agent_interaction_mask, pair_idx):
                agents = agent_pairs[pair_idx]
                is_interacting = valid_interacting_pairs[pair_idx]

                # Check if either agent in the pair is already part of another interaction
                interacting = jnp.logical_and(is_interacting, jnp.logical_not(agent_interaction_mask[agents[0]] | agent_interaction_mask[agents[1]]))

                # Update the agent_interaction_mask
                agent_interaction_mask = agent_interaction_mask.at[agents].set(interacting)

                return agent_interaction_mask

            # Process all pairs of agents using the reduce function
            agent_interaction_mask = reduce(process_pair, shuffled_indices, agent_interaction_mask)

            # Compute the selected interact positions and rewards
            selected_interactions = agent_interaction_mask[agent_pairs].reshape(-1, 2).all(axis=-1)
            selected_interact_positions = interact_positions[selected_interactions]
            selected_pairs = agent_pairs[selected_interactions]
            rewards = jnp.zeros((num_agents,))
            rewards = rewards.at[selected_pairs.ravel()].set(_get_reward(selected_pairs, params).ravel())

            # Update the grid with the selected interact positions
            state.grid = state.grid.at[selected_interact_positions[:, :, 0], selected_interact

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: jnp.array,
            params: EnvParams,
        ):

            """Step the environment."""

            # freeze check
            # action_0, action_1 = actions
            # action_0 = jnp.where(state.freeze > 0, Actions.stay, action_0)
            # action_1 = jnp.where(state.freeze > 0, Actions.stay, action_1)
            actions = jnp.where(state.agent_freezes > 0, Actions.stay, actions)

            def turning_and_moving(agent_pos, action):

                # turning red
                new_pos = jnp.int8(
                    (agent_pos + ROTATIONS[action])
                    % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4])
                )

                # get new positions
                # moving red 
                move = action == Actions.forward
                new_pos = jnp.where(
                    move, new_pos + STEP[agent_pos[2]], new_pos
                )
                new_agent_pos = jnp.clip(
                    new_pos,
                    a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                    a_max=jnp.array(
                        [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                    ),
                )

                # if you bounced back to ur original space, we change your move to stay (for collision logic)
                move = (new_pos[:2] != agent_pos[:2]).any()
                return new_agent_pos, move

            vmap_turning_and_moving = vmap(turning_and_moving, (0, 0), (0, 0))
            new_agent_positions = vmap_turning_and_moving(state.agent_positions, actions)


            # if collision, priority to whoever didn't move
            def check_collision(agent1_pos, agent2_pos):
                collision = jnp.all(agent1_pos[:2] == agent2_pos[:2])
                return collision

            vmap_check_collision = vmap(vmap(check_collision, (None, 0), 0), (0, None), 0)

            def update_position(new_agent_pos, old_agent_pos, move1, move2, collision):
                new_agent_pos = jnp.where(
                collision
                * move1
                * (1 - move2),  # red moved, blue didn't
                old_agent_pos,
                new_agent_pos,
                )
                return new_agent_pos


            vmap_update_position = vmap(vmap(update_position, (None, None, None, 0, 0), 0), (0, 0, 0, None, None), 0)

            num_samples = num_agents * (num_agents - 1) // 2
            key, subkey = random.split(key)
            upper_triangular = jax.random.choice(key, jnp.array([0, 1]), shape=(num_agents, num_agents))
            indices = jnp.triu_indices(num_agents, k=1)
            matrix = jnp.zeros((num_agents, num_agents))

            # Fill the matrix using the generated random samples and index array
            matrix = matrix.at[indices].set(upper_triangular)

            # Copy the upper triangular part to the lower triangular part to make it symmetric
            takes_square = matrix + matrix.T

            def update_positions_randomized(agent1_idx, agent2_idx)
                # if both moved, then randomise
                new_agent_pos = jnp.where(
                    collisions[agent1_idx, agent2_idx]
                    * moves[agent1_idx]
                    * moves[agent2_idx]
                    * (
                        1 - takes_square[agent1_idx, agent2_idx]
                    ),  # if both collide and red doesn't take square
                    state.agent_positions[agent1_idx],
                    new_agent_pos[agent1_idx],
                )
                return new_agent_pos

            def update_inventories(new_pos, inventory):
                coop_matches = = (
                state.grid[new_pos[0], new_pos[1]] == Items.cooperation_coin
                )
                defect_matches = (
                state.grid[new_pos[0], new_pos[1]] == Items.defection_coin
                )
                inventory = inventory + jnp.array(
                [coop_matches, defect_matches]
                )
                return inventory

            vmap_update_inventory = vmap(update_inventories, (0, 0), 0)
            state.agent_inventories = vmap_update_inventory(agent_new_positons, state.agent_inventories)

            def update_grid(agent_pos, new_agent_pos, item)
                # update grid
                state.grid = state.grid.at[
                    (agent_pos[0], agent_pos[1])
                ].set(jnp.int8(Items.empty))
                state.grid = state.grid.at[(new_agent_pos[0], new_agent_pos[1])].set(
                    jnp.int8(item)
                )

            vmap_update_grid = vmap(update_grid, (0, 0, 0))
            items = jnp.arange(1, num_agents+1)
            vmap_update_grid(state.agent_positions, agent_new_positons, items)

            state.agent_positions = agent_new_positons

            red_reward, blue_reward = 0, 0
            (
                interact,
                rewards
                state,
            ) = _interact(state, actions, params)
            red_reward += red_interact_reward
            blue_reward += blue_interact_reward

            # if we interacted, then we set freeze
            state.freeze = jnp.where(
                interact, params.freeze_penalty, state.freeze
            )

            # if we didn't interact, then we decrement freeze
            state.freeze = jnp.where(
                state.freeze > 0, state.freeze - 1, state.freeze
            )
            state_sft_re = _soft_reset_state(key, state)
            state = jax.tree_map(
                lambda x, y: jnp.where(state.freeze == 0, x, y),
                state_sft_re,
                state,
            )
            state_nxt = EnvState(
                red_pos=state.red_pos,
                blue_pos=state.blue_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                red_inventory=state.red_inventory,
                blue_inventory=state.blue_inventory,
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=jnp.where(
                    interact, params.freeze_penalty, state.freeze
                ),
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

        def _soft_reset_state(key: jnp.ndarray, state: EnvState) -> EnvState:
            """Reset the grid to original state and"""
            # Find the free spaces in the grid
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)

            # if coin location can change, then we need to reset the coins
            for i in range(NUM_COINS):
                grid = grid.at[
                    state.red_coins[i, 0], state.red_coins[i, 1]
                ].set(jnp.int8(Items.red_coin))

            for i in range(NUM_COINS):
                grid = grid.at[
                    state.blue_coins[i, 0], state.blue_coins[i, 1]
                ].set(jnp.int8(Items.blue_coin))

            agent_pos = jax.random.choice(
                key, AGENT_SPAWNS, shape=(), replace=False
            )

            player_dir = jax.random.randint(
                key, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [agent_pos[:2, 0], agent_pos[:2, 1], player_dir]
            ).T

            red_pos = player_pos[0, :]
            blue_pos = player_pos[1, :]

            grid = grid.at[red_pos[0], red_pos[1]].set(
                jnp.int8(Items.red_agent)
            )
            grid = grid.at[blue_pos[0], blue_pos[1]].set(
                jnp.int8(Items.blue_agent)
            )

            return EnvState(
                red_pos=red_pos,
                blue_pos=blue_pos,
                inner_t=state.inner_t,
                outer_t=state.outer_t,
                grid=grid,
                red_inventory=jnp.zeros(2),
                blue_inventory=jnp.zeros(2),
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=jnp.int16(-1),
            )

        def _reset_state(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            key, subkey = jax.random.split(key)

            # coin_pos = jax.random.choice(
            #     subkey, COIN_SPAWNS, shape=(NUM_COIN_TYPES*NUM_COINS,), replace=False
            # )

            agent_pos = jax.random.choice(
                subkey, AGENT_SPAWNS, shape=(), replace=False
            )
            player_dir = jax.random.randint(
                subkey, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [agent_pos[:2, 0], agent_pos[:2, 1], player_dir]
            ).T
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)
            grid = grid.at[player_pos[0, 0], player_pos[0, 1]].set(
                jnp.int8(Items.red_agent)
            )
            grid = grid.at[player_pos[1, 0], player_pos[1, 1]].set(
                jnp.int8(Items.blue_agent)
            )
            if fixed_coin_location:
                rand_idx = jax.random.randint(
                    subkey, shape=(), minval=0, maxval=1
                )
                red_coins = jnp.where(rand_idx, RED_SPAWN, BLUE_SPAWN)
                blue_coins = jnp.where(rand_idx, BLUE_SPAWN, RED_SPAWN)
            else:
                coin_spawn = jax.random.permutation(
                    subkey, COIN_SPAWNS, axis=0
                )
                red_coins = coin_spawn[:NUM_COINS, :]
                blue_coins = coin_spawn[NUM_COINS:, :]

            for i in range(NUM_COINS):
                grid = grid.at[red_coins[i, 0], red_coins[i, 1]].set(
                    jnp.int8(Items.red_coin)
                )

            for i in range(NUM_COINS):
                grid = grid.at[blue_coins[i, 0], blue_coins[i, 1]].set(
                    jnp.int8(Items.blue_coin)
                )

            return EnvState(
                red_pos=player_pos[0, :],
                blue_pos=player_pos[1, :],
                inner_t=0,
                outer_t=0,
                grid=grid,
                red_inventory=jnp.zeros(2),
                blue_inventory=jnp.zeros(2),
                red_coins=red_coins,
                blue_coins=blue_coins,
                freeze=jnp.int16(-1),
            )

        def reset(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            state = _reset_state(key, params)
            obs = _get_obs(state)
            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        # self.step = jax.jit(_step)
        self.step = _step
        self.reset = jax.jit(reset)
        self.get_obs_point = _get_obs_point
        self.get_reward = _get_reward

        # for debugging
        self.get_obs = _get_obs
        self.cnn = True

        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

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
                low=0,
                high=NUM_COINS,
                shape=NUM_COIN_TYPES + 4,
                dtype=jnp.uint8,
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
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

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
            agent_color = PLAYER1_COLOUR

        elif obj == Items.blue_agent:
            # Draw agent 2
            agent_color = PLAYER2_COLOUR
        elif obj == Items.red_coin:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (44.0, 160.0, 44.0)
            )
        elif obj == Items.blue_coin:
            # Draw the blue coin as DEFECT/ RED COIN
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
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

        grid = onp.array(obs[agent]["observation"][:, :, :-4])
        empty_space_channel = onp.zeros((OBS_SIZE, OBS_SIZE, 1))
        grid = onp.concatenate((empty_space_channel, grid), axis=-1)
        grid = onp.argmax(grid.reshape(-1, grid.shape[-1]), axis=1)
        grid = grid.reshape(OBS_SIZE, OBS_SIZE)

        angles = onp.array(obs[agent]["observation"][:, :, -4:])
        angles = onp.argmax(angles.reshape(-1, angles.shape[-1]), axis=1)
        angles = angles.reshape(OBS_SIZE, OBS_SIZE)

        # Compute the total grid size
        width_px = grid.shape[0] * tile_size
        height_px = grid.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)

        # agent direction
        pricipal_dir = 0
        red_hat = bool(state.red_inventory.sum() > INTERACT_THRESHOLD)
        blue_hat = bool(state.blue_inventory.sum() > INTERACT_THRESHOLD)
        if agent == 0:
            principal_hat = red_hat
            other_hat = blue_hat
        else:
            principal_hat = blue_hat
            other_hat = red_hat

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None

                principal_agent_here = cell == 1
                other_agent_here = cell == 2

                if principal_agent_here:
                    agent_dir = pricipal_dir
                    agent_hat = principal_hat

                elif other_agent_here:
                    agent_dir = angles[i, j]
                    agent_hat = other_hat

                else:
                    agent_dir = None
                    agent_hat = None

                tile_img = InTheMatrix.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        img = onp.rot90(img, 2, axes=(0, 1))
        inv = self.render_inventory(
            state.red_inventory if agent == 0 else state.blue_inventory,
            width_px,
        )
        return onp.concatenate((img, inv), axis=0)

    def render(
        self,
        state: EnvState,
        params: EnvParams,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(GRID))

        # Compute the total grid size
        width_px = GRID.shape[0] * tile_size
        height_px = GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        grid = onp.pad(
            grid, ((PADDING, PADDING), (PADDING, PADDING)), constant_values=5
        )

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
        if state.freeze > 0:
            # check which agent won
            r1, r2 = self.get_reward(state, params)

            if r1 == -r2:
                # zero sum game
                if r1 > r2:
                    # red won
                    img = onp.tile(
                        PLAYER1_COLOUR, (img.shape[0], img.shape[1], 1)
                    )
                elif r2 > r1:
                    # blue won
                    img = onp.tile(
                        PLAYER2_COLOUR, (img.shape[0], img.shape[1], 1)
                    )
                elif r1 == r2:
                    img[:, : width_px // 2, :] = onp.tile(
                        PLAYER2_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                    img[:, width_px // 2 :, :] = onp.tile(
                        PLAYER1_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
            else:
                # otherwise we got some cool general sum game
                welfare = r1 + r2
                if welfare > 5:
                    # cooperate
                    img[:, width_px // 2 :, :] = onp.tile(
                        GREEN_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                else:
                    img[:, width_px // 2 :, :] = onp.tile(
                        RED_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                if r1 > r2:
                    # red won
                    img[:, : width_px // 2, :] = onp.tile(
                        PLAYER1_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                elif r1 < r2:
                    # blue won
                    img[:, : width_px // 2, :] = onp.tile(
                        PLAYER2_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                elif r1 == r2:
                    img[height_px // 2 :, : width_px // 2, :] = onp.tile(
                        PLAYER1_COLOUR,
                        (img.shape[0] // 2, img.shape[1] // 2, 1),
                    )
                    img[: height_px // 2, : width_px // 2, :] = onp.tile(
                        PLAYER2_COLOUR,
                        (img.shape[0] // 2, img.shape[1] // 2, 1),
                    )

            img = img.astype(onp.uint8)
        else:
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
                        state.red_pos[2].item()
                        if red_agent_here
                        else agent_dir
                    )
                    agent_dir = (
                        state.blue_pos[2].item()
                        if blue_agent_here
                        else agent_dir
                    )
                    agent_hat = False
                    agent_hat = (
                        bool(state.red_inventory.sum() > INTERACT_THRESHOLD)
                        if red_agent_here
                        else agent_hat
                    )
                    agent_hat = (
                        bool(state.blue_inventory.sum() > INTERACT_THRESHOLD)
                        if blue_agent_here
                        else agent_hat
                    )

                    tile_img = InTheMatrix.render_tile(
                        cell,
                        agent_dir=agent_dir,
                        agent_hat=agent_hat,
                        highlight=highlight_mask[i, j],
                        tile_size=tile_size,
                    )

                    ymin = j * tile_size
                    ymax = (j + 1) * tile_size
                    xmin = i * tile_size
                    xmax = (i + 1) * tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img

        img = onp.rot90(
            img[
                (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
                (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        # Render the inventory
        red_inv = self.render_inventory(state.red_inventory, img.shape[1])
        blue_inv = self.render_inventory(state.blue_inventory, img.shape[1])

        time = self.render_time(state, img.shape[1])
        img = onp.concatenate((img, red_inv, blue_inv, time), axis=0)
        return img

    def render_inventory(self, inventory, width_px) -> onp.array:
        tile_height = 32
        height_px = NUM_COIN_TYPES * tile_height
        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // NUM_COINS
        for j in range(0, NUM_COIN_TYPES):
            num_coins = inventory[j]
            for i in range(int(num_coins)):
                cell = None
                if j == 0:
                    cell = 99
                elif j == 1:
                    cell = 100
                tile_img = InTheMatrix.render_tile(cell, tile_size=tile_height)
                ymin = j * tile_height
                ymax = (j + 1) * tile_height
                xmin = i * tile_width
                xmax = (i + 1) * tile_width
                img[ymin:ymax, xmin:xmax, :] = onp.resize(
                    tile_img, (tile_height, tile_width, 3)
                )
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img

if __name__ == "__main__":
    from PIL import Image

    action = 1
    render_agent_view = True
    num_outer_steps = 1
    num_inner_steps = 150

    rng = jax.random.PRNGKey(0)
    env = InTheMatrix(num_inner_steps, num_outer_steps, True)
    num_actions = env.action_space().n
    params = EnvParams(
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,
    )
    obs, old_state = env.reset(rng, params)
    pics = []
    pics1 = []
    pics2 = []

    img = env.render(old_state, params)
    img1 = env.render_agent_view(old_state, agent=0)
    img2 = env.render_agent_view(old_state, agent=1)
    pics.append(img)

    int_action = {
        0: "left",
        1: "right",
        2: "forward",
        3: "interact",
        4: "stay",
    }

    key_int = {"w": 2, "a": 0, "s": 4, "d": 1, " ": 4}
    env.step = jax.jit(env.step)

    for t in range(num_outer_steps * num_inner_steps):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        # a1 = jnp.array(2)
        # a2 = jnp.array(4)
        a1 = jax.random.choice(
            rng1, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
        )
        a2 = jax.random.choice(
            rng2, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
        )
        obs, state, reward, done, info = env.step(
            rng, old_state, (a1 * action, a2 * action), params
        )

        if (state.red_pos[:2] == state.blue_pos[:2]).all():
            import pdb

            # pdb.set_trace()
            print("collision")
            print(
                f"timestep: {t}, A1: {int_action[a1.item()]} A2:{int_action[a2.item()]}"
            )
            print(state.red_pos, state.blue_pos)

        img = env.render(state, params)
        pics.append(img)

        if render_agent_view:
            img1 = env.render_agent_view(state, agent=0)
            img2 = env.render_agent_view(state, agent=1)
            pics1.append(img1)
            pics2.append(img2)
        old_state = state
    print("Saving GIF")
    pics = [Image.fromarray(img) for img in pics]
    pics[0].save(
        "state.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=100,
        loop=0,
    )

    if render_agent_view:
        pics1 = [Image.fromarray(img) for img in pics1]
        pics2 = [Image.fromarray(img) for img in pics2]
        pics1[0].save(
            "agent1.gif",
            format="GIF",
            save_all=True,
            optimize=False,
            append_images=pics1[1:],
            duration=100,
            loop=0,
        )
        pics2[0].save(
            "agent2.gif",
            format="GIF",
            save_all=True,
            optimize=False,
            append_images=pics2[1:],
            duration=100,
            loop=0,
        )
