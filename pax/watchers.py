import enum
import pickle
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

import pax.agents.hyper.ppo as HyperPPO
import pax.agents.ppo.ppo as PPO
from pax.agents.naive_exact import NaiveExact
from pax.envs.iterated_matrix_game import EnvState, IteratedMatrixGame
from pax.envs.in_the_matrix import InTheMatrix

# five possible states
START = jnp.array([[0, 0, 0, 0, 1]])
CC = jnp.array([[1, 0, 0, 0, 0]])
CD = jnp.array([[0, 1, 0, 0, 0]])
DC = jnp.array([[0, 0, 1, 0, 0]])
DD = jnp.array([[0, 0, 0, 1, 0]])
STATE_NAMES = ["START", "CC", "CD", "DC", "DD"]
ALL_STATES = [START, CC, CD, DC, DD]

START_3P = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]])
CCC = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])
CCD = jnp.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])
CDC = jnp.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]])
CDD = jnp.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]])
DCC = jnp.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]])
DCD = jnp.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]])
DDC = jnp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]])
DDD = jnp.array([[0, 0, 0, 0, 0, 0, 0, 1, 0]])
STATE_NAMES_3P = [
    "START",
    "CCC",
    "CCD",
    "CDC",
    "CDD",
    "DCC",
    "DCD",
    "DDC",
    "DDD",
]
ALL_STATES_3P = [START, CCC, CCD, CDC, CDD, DCC, DCD, DDC, DDD]


class State(enum.IntEnum):
    CC = 0
    CD = 1
    DC = 2
    DD = 3
    START = 4


class State_3P(enum.IntEnum):
    CCC = 0
    CCD = 1
    CDC = 2
    CDD = 3
    DCC = 4
    DCD = 5
    DDC = 6
    DDD = 7
    START = 8


def policy_logger(agent, three_players=False) -> dict:
    weights = agent.actor_optimizer.target["Dense_0"][
        "kernel"
    ]  # [layer_name]['w']
    log_pi = nn.softmax(weights)
    if three_players:
        state = State_3P
    else:
        state = State
    probs = {
        "policy/" + str(s): p[0] for (s, p) in zip(state, log_pi)
    }  # probability of cooperating is p[0]
    return probs


def value_logger(agent, three_players=False) -> dict:
    weights = agent.critic_optimizer.target["Dense_0"]["kernel"]
    if three_players:
        state = State_3P
    else:
        state = State
    values = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(state, weights)
    }
    values.update(
        {f"value/{str(s)}.defect": p[1] for (s, p) in zip(state, weights)}
    )
    return values


def policy_logger_dqn(agent, three_players=False) -> None:
    # this assumes using a linear layer, so this logging won't work using MLP
    weights = agent._state.target_params["linear"]["w"]  # 5 x 2 matrix
    pi = nn.softmax(weights)
    pid = agent.player_id
    target_steps = agent.target_step_updates
    if three_players:
        state = State_3P
    else:
        state = State
    probs = {
        f"policy/player_{str(pid)}/{str(s)}.cooperate": p[0]
        for (s, p) in zip(state, pi)
    }
    probs.update(
        {
            f"policy/player_{str(pid)}/{str(s)}.defect": p[1]
            for (s, p) in zip(state, pi)
        }
    )
    probs.update({"policy/target_step_updates": target_steps})
    return probs


def value_logger_dqn(agent, three_players=False) -> dict:
    weights = agent._state.target_params["linear"]["w"]  # 5 x 2 matrix
    pid = agent.player_id
    target_steps = agent.target_step_updates
    if three_players:
        state = State_3P
    else:
        state = State
    values = {
        f"value/player_{str(pid)}/{str(s)}.cooperate": p[0]
        for (s, p) in zip(state, weights)
    }
    values.update(
        {
            f"value/player_{str(pid)}/{str(s)}.defect": p[1]
            for (s, p) in zip(state, weights)
        }
    )
    values.update({"value/target_step_updates": target_steps})
    return values


def policy_logger_ppo(agent: PPO, three_players=False) -> dict:
    weights = agent._state.params["categorical_value_head/~/linear"]["w"]
    pi = nn.softmax(weights)
    sgd_steps = agent._total_steps / agent._num_steps
    if three_players:
        state = State_3P
    else:
        state = State
    probs = {f"policy/{str(s)}.cooperate": p[0] for (s, p) in zip(state, pi)}
    probs.update({"policy/total_steps": sgd_steps})
    return probs


def value_logger_ppo(agent: PPO, three_players=False) -> dict:
    weights = agent._state.params["categorical_value_head/~/linear_1"][
        "w"
    ]  # 5 x 1 matrix
    sgd_steps = agent._total_steps / agent._num_steps
    if three_players:
        state = State_3P
    else:
        state = State
    probs = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(state, weights)
    }
    probs.update({"value/total_steps": sgd_steps})
    return probs


def policy_logger_ppo_with_memory(agent) -> dict:
    """Calculate probability of coopreation"""
    # n = 5
    # params = agent._state.params
    # hidden = agent._state.hidden
    # pid = agent.player_id

    # # episode = int(
    # #     agent._logger.metrics["total_steps"]
    # #     / (agent._num_steps * agent._num_envs)
    # # )
    # episode = int(agent._logger.metrics["total_steps"] / agent._num_steps)
    # cooperation_probs = {"episode": episode}

    # TODO: Figure out how to JIT the forward function
    # Works when the forward function is not jitted.
    # for state, state_name in zip(ALL_STATES, STATE_NAMES):
    #     (dist, _), hidden = agent.forward(params, state, hidden)
    #     cooperation_probs[f"policy/ppo_{pid}/{state_name}"] = float(
    #         dist.probs[0][0].mean()
    #         )
    return {}


def naive_pg_losses(agent) -> None:
    pid = agent.player_id
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    losses = {
        f"train/naive_learner{pid}/sgd_steps": sgd_steps,
        f"train/naive_learner{pid}/total": loss_total,
        f"train/naive_learner{pid}/policy": loss_policy,
        f"train/naive_learner{pid}/value": loss_value,
    }
    return losses


def logger_hyper(agent: HyperPPO) -> dict:
    episode = int(
        agent._logger.metrics["total_steps"]
        / (agent._num_steps * agent._num_envs)
    )
    cooperation_probs = {"episode": episode}
    return cooperation_probs


def losses_ppo(agent: PPO) -> dict:
    pid = agent.player_id
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    loss_entropy = agent._logger.metrics["loss_entropy"]
    entropy_cost = agent._logger.metrics["entropy_cost"]
    losses = {
        f"train/ppo_{pid}/sgd_steps": sgd_steps,
        f"train/ppo_{pid}/total": loss_total,
        f"train/ppo_{pid}/policy": loss_policy,
        f"train/ppo_{pid}/value": loss_value,
        f"train/ppo_{pid}/entropy": loss_entropy,
        f"train/ppo_{pid}/entropy_coefficient": entropy_cost,
    }
    return losses


def losses_naive(agent: NaiveExact) -> dict:
    pid = agent.player_id
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    num_episodes = agent._logger.metrics["num_episodes"]
    losses = {
        f"train/naive_learner_{pid}/sgd_steps": sgd_steps,
        f"train/naive_learner_{pid}/loss": loss_total,
        f"train/naive_learner_{pid}/num_episodes": num_episodes,
    }
    return losses


def logger_naive_exact(agent: NaiveExact, three_players=False) -> dict:
    params = agent._mem.hidden
    pid = agent.player_id
    params = params.mean(axis=0)
    cooperation_probs = {"episode": agent._logger.metrics["total_steps"]}
    if three_players:
        state_names = STATE_NAMES_3P
    else:
        state_names = STATE_NAMES
    for i, state_name in enumerate(state_names):
        cooperation_probs[
            f"policy/naive_learner_{pid}/avg/{state_name}"
        ] = float(params[i])
    return cooperation_probs


def policy_logger_naive(agent, three_players=False) -> None:
    weights = agent._state.params["categorical_value_head/~/linear"]["w"]
    pi = nn.softmax(weights)
    sgd_steps = agent._total_steps / agent._num_steps
    if three_players:
        state = State_3P
    else:
        state = State
    probs = {
        f"policy/{str(s)}/{agent.player_id}.cooperate": p[0]
        for (s, p) in zip(state, pi)
    }
    probs.update({"policy/total_steps": sgd_steps})
    return probs


class ESLog(object):
    def __init__(
        self, num_dims: int, num_generations: int, top_k: int, maximize: bool
    ):
        """Simple jittable logging tool for ES rollouts."""
        self.num_dims = num_dims
        self.num_generations = num_generations
        self.top_k = top_k
        self.maximize = maximize

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self) -> chex.ArrayTree:
        """Initialize the logger storage."""
        log = {
            "top_fitness": jnp.zeros(self.top_k)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_params": jnp.zeros((self.top_k, self.num_dims))
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_1": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_gen_fitness": jnp.zeros(self.top_k)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_gen_params": jnp.zeros((self.top_k, self.num_dims))
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_1": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_gen_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_gen_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "gen_counter": 0,
        }
        return log

    # @partial(jax.jit, static_argnums=(0,))
    def update(
        self, log: chex.ArrayTree, x: chex.Array, fitness: chex.Array
    ) -> chex.ArrayTree:
        """Update the logging storage with newest data."""
        # Check if there are solutions better than current archive
        def get_top_idx(maximize: bool, vals: jnp.ndarray) -> jnp.ndarray:
            top_idx = maximize * ((-1) * vals).argsort() + (
                (1 - maximize) * vals.argsort()
            )
            return top_idx

        vals = jnp.hstack([log["top_fitness"], fitness])
        params = jnp.vstack([log["top_params"], x])
        top_idx = get_top_idx(self.maximize, vals)
        log["top_fitness"] = vals[top_idx[: self.top_k]]
        log["top_params"] = params[top_idx[: self.top_k]]
        log["log_top_1"] = (
            log["log_top_1"].at[log["gen_counter"]].set(log["top_fitness"][0])
        )
        log["log_top_mean"] = (
            log["log_top_mean"]
            .at[log["gen_counter"]]
            .set(jnp.mean(log["top_fitness"]))
        )
        log["log_top_std"] = (
            log["log_top_std"]
            .at[log["gen_counter"]]
            .set(jnp.std(log["top_fitness"]))
        )
        top_idx = get_top_idx(self.maximize, fitness)
        log["top_gen_fitness"] = fitness[top_idx[: self.top_k]]
        log["top_gen_params"] = x[top_idx[: self.top_k]]
        log["log_gen_1"] = (
            log["log_gen_1"]
            .at[log["gen_counter"]]
            .set(
                self.maximize * jnp.max(fitness)
                + (1 - self.maximize) * jnp.min(fitness)
            )
        )
        log["log_top_gen_mean"] = (
            log["log_top_gen_mean"]
            .at[log["gen_counter"]]
            .set(jnp.mean(log["top_gen_fitness"]))
        )
        log["log_top_gen_std"] = (
            log["log_top_gen_std"]
            .at[log["gen_counter"]]
            .set(jnp.std(log["top_gen_fitness"]))
        )
        log["log_gen_mean"] = (
            log["log_gen_mean"].at[log["gen_counter"]].set(jnp.mean(fitness))
        )
        log["log_gen_std"] = (
            log["log_gen_std"].at[log["gen_counter"]].set(jnp.std(fitness))
        )
        log["gen_counter"] += 1
        return log

    def save(self, log: chex.ArrayTree, filename: str):
        """Save different parts of logger in .pkl file."""
        with open(filename, "wb") as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str):
        """Reload the pickle logger and return dictionary."""
        with open(filename, "rb") as handle:
            es_logger = pickle.load(handle)
        return es_logger

    def plot(
        self,
        log,
        title,
        ylims=None,
        fig=None,
        ax=None,
        no_legend=False,
    ):
        """Plot fitness trajectory from evo logger over generations."""
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        int_range = jnp.arange(1, log["gen_counter"] + 1)
        ax.plot(
            int_range, log["log_top_1"][: log["gen_counter"]], label="Top 1"
        )
        ax.plot(
            int_range,
            log["log_top_mean"][: log["gen_counter"]],
            label=f"Top-{self.top_k} Mean",
        )
        ax.plot(
            int_range, log["log_gen_1"][: log["gen_counter"]], label="Gen. 1"
        )
        ax.plot(
            int_range,
            log["log_gen_mean"][: log["gen_counter"]],
            label="Gen. Mean",
        )
        if ylims is not None:
            ax.set_ylim(ylims)
        if not no_legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Number of Generations")
        ax.set_ylabel("Fitness Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig, ax


def ipd_visitation(
    observations: jnp.ndarray, actions: jnp.ndarray, final_obs: jnp.ndarray
) -> dict:
    # obs [num_outer_steps, num_inner_steps, num_opps, num_envs, ...]
    # final_t [num_opps, num_envs, ...]
    num_timesteps = observations.shape[0] * observations.shape[1]
    # obs = [0, 1, 2, 3, 4], a = [0, 1]
    # combine = [0, .... 9]
    state_actions = 2 * jnp.argmax(observations, axis=-1) + actions
    state_actions = jnp.reshape(
        state_actions,
        (num_timesteps,) + state_actions.shape[2:],
    )
    # assume final step taken is cooperate
    final_obs = jax.lax.expand_dims(2 * jnp.argmax(final_obs, axis=-1), [0])
    state_actions = jnp.append(state_actions, final_obs, axis=0)
    hist = jnp.bincount(state_actions.flatten(), length=10)
    state_freq = hist.reshape((int(hist.shape[0] / 2), 2)).sum(axis=1)
    state_probs = state_freq / state_freq.sum()
    action_probs = jnp.nan_to_num(hist[::2] / state_freq)
    return {
        "state_visitation/CC": state_freq[0],
        "state_visitation/CD": state_freq[1],
        "state_visitation/DC": state_freq[2],
        "state_visitation/DD": state_freq[3],
        "state_visitation/START": state_freq[4],
        "state_probability/CC": state_probs[0],
        "state_probability/CD": state_probs[1],
        "state_probability/DC": state_probs[2],
        "state_probability/DD": state_probs[3],
        "state_probability/START": state_probs[4],
        "cooperation_probability/CC": action_probs[0],
        "cooperation_probability/CD": action_probs[1],
        "cooperation_probability/DC": action_probs[2],
        "cooperation_probability/DD": action_probs[3],
        "cooperation_probability/START": action_probs[4],
    }


def tensor_ipd_visitation(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    final_obs: jnp.ndarray,
    agent_idx=1,
) -> dict:
    # obs [num_outer_steps, num_inner_steps, num_opps, num_envs, ...]
    # final_t [num_opps, num_envs, ...]
    num_timesteps = observations.shape[0] * observations.shape[1]
    # obs = [0....8], a = [0, 1]
    # combine = [0, .... 17]
    state_actions = 2 * jnp.argmax(observations, axis=-1) + actions
    state_actions = jnp.reshape(
        state_actions,
        (num_timesteps,) + state_actions.shape[2:],
    )
    # assume final step taken is cooperate
    final_obs = jax.lax.expand_dims(2 * jnp.argmax(final_obs, axis=-1), [0])
    state_actions = jnp.append(state_actions, final_obs, axis=0)
    hist = jnp.bincount(state_actions.flatten(), length=18)
    state_freq = hist.reshape((int(hist.shape[0] / 2), 2)).sum(axis=1)
    state_probs = state_freq / state_freq.sum()
    action_probs = jnp.nan_to_num(hist[::2] / state_freq)
    print(action_probs.shape)
    return {
        "state_visitation/CCC": state_freq[0],
        "state_visitation/CCD": state_freq[1],
        "state_visitation/CDC": state_freq[2],
        "state_visitation/CDD": state_freq[3],
        "state_visitation/DCC": state_freq[4],
        "state_visitation/DCD": state_freq[5],
        "state_visitation/DDC": state_freq[6],
        "state_visitation/DDD": state_freq[7],
        "state_visitation/START": state_freq[8],
        "state_probability/CCC": state_probs[0],
        "state_probability/CCD": state_probs[1],
        "state_probability/CDC": state_probs[2],
        "state_probability/CDD": state_probs[3],
        "state_probability/DCC": state_probs[4],
        "state_probability/DCD": state_probs[5],
        "state_probability/DDC": state_probs[6],
        "state_probability/DDD": state_probs[7],
        "state_probability/START": state_probs[8],
        "cooperation_probability/1/CCC": action_probs[0],
        "cooperation_probability/1/CCD": action_probs[1],
        "cooperation_probability/1/CDC": action_probs[2],
        "cooperation_probability/1/CDD": action_probs[3],
        "cooperation_probability/1/DCC": action_probs[4],
        "cooperation_probability/1/DCD": action_probs[5],
        "cooperation_probability/1/DDC": action_probs[6],
        "cooperation_probability/1/DDD": action_probs[7],
        "cooperation_probability/1/START": action_probs[8],
    }


def tensor_ipd_coop_probs(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    final_obs: jnp.ndarray,
    agent_idx: int = 1,
) -> dict:
    num_timesteps = observations.shape[0] * observations.shape[1]
    # obs = [0....8], a = [0, 1]
    # combine = [0, .... 17]
    state_actions = 2 * jnp.argmax(observations, axis=-1) + actions
    state_actions = jnp.reshape(
        state_actions,
        (num_timesteps,) + state_actions.shape[2:],
    )
    final_obs = jax.lax.expand_dims(2 * jnp.argmax(final_obs, axis=-1), [0])
    state_actions = jnp.append(state_actions, final_obs, axis=0)
    hist = jnp.bincount(state_actions.flatten(), length=18)
    state_freq = hist.reshape((int(hist.shape[0] / 2), 2)).sum(axis=1)
    action_probs = jnp.nan_to_num(hist[::2] / state_freq)
    # THIS IS FROM AGENTS OWN PERSPECTIVE
    return {
        f"cooperation_probability/{agent_idx}/CCC": action_probs[0],
        f"cooperation_probability/{agent_idx}/CCD": action_probs[1],
        f"cooperation_probability/{agent_idx}/CDC": action_probs[2],
        f"cooperation_probability/{agent_idx}/CDD": action_probs[3],
        f"cooperation_probability/{agent_idx}/DCC": action_probs[4],
        f"cooperation_probability/{agent_idx}/DCD": action_probs[5],
        f"cooperation_probability/{agent_idx}/DDC": action_probs[6],
        f"cooperation_probability/{agent_idx}/DDD": action_probs[7],
        f"cooperation_probability/{agent_idx}/START": action_probs[8],
    }


def cg_visitation(state: NamedTuple) -> dict:
    # [num_opps, num_envs, num_outer_episodes]
    total_1 = state.red_coop + state.red_defect
    total_2 = state.blue_coop + state.blue_defect
    avg_prob_1 = jnp.sum(state.red_coop, axis=-1) / jnp.sum(total_1, axis=-1)
    avg_prob_2 = jnp.sum(state.blue_coop, axis=-1) / jnp.sum(total_2, axis=-1)
    final_prob_1 = state.red_coop[:, :, -1] / total_1[:, :, -1]
    final_prob_2 = state.blue_coop[:, :, -1] / total_2[:, :, -1]

    # [num_opps, num_envs, num_states]
    prob_coop_1 = jnp.sum(state.coop1, axis=(0, 1)) / jnp.sum(
        state.counter, axis=(0, 1)
    )
    prob_coop_2 = jnp.sum(state.coop2, axis=(0, 1)) / jnp.sum(
        state.counter, axis=(0, 1)
    )
    count = jnp.nanmean(state.counter, axis=(0, 1))
    return {
        "prob_coop/1": jnp.nanmean(avg_prob_1),  # [1]
        "prob_coop/2": jnp.nanmean(avg_prob_2),  # [1]
        "final_prob_coop/1": jnp.nanmean(final_prob_1),  # [1]
        "final_prob_coop/2": jnp.nanmean(final_prob_2),  # [1]
        "total_coins/1": total_1.sum(),  # int
        "total_coins/2": total_2.sum(),  # int
        "coins_per_episode/1": total_1.sum(axis=-1).mean(axis=(0, 1)),  # [1]
        "coins_per_episode/2": total_2.sum(axis=-1).mean(axis=(0, 1)),  # [1]
        "final_coin_total/1": total_1[:, :, -1].mean(axis=(0, 1)),  # [1]
        "final_coin_total/2": total_2[:, :, -1].mean(axis=(0, 1)),  # [1]
        "cooperation_probability/1/SS": prob_coop_1[0],
        "cooperation_probability/1/CC": prob_coop_1[1],
        "cooperation_probability/1/CD": prob_coop_1[2],
        "cooperation_probability/1/DC": prob_coop_1[3],
        "cooperation_probability/1/DD": prob_coop_1[4],
        "cooperation_probability/1/SC": prob_coop_1[5],
        "cooperation_probability/1/SD": prob_coop_1[6],
        "cooperation_probability/1/CS": prob_coop_1[7],
        "cooperation_probability/1/DS": prob_coop_1[8],
        "cooperation_probability/2/SS": prob_coop_2[0],
        "cooperation_probability/2/CC": prob_coop_2[1],
        "cooperation_probability/2/CD": prob_coop_2[2],
        "cooperation_probability/2/DC": prob_coop_2[3],
        "cooperation_probability/2/DD": prob_coop_2[4],
        "cooperation_probability/2/SC": prob_coop_2[5],
        "cooperation_probability/2/SD": prob_coop_2[6],
        "cooperation_probability/2/CS": prob_coop_2[7],
        "cooperation_probability/2/DS": prob_coop_2[8],
        "state_visitation/SS": count[0],
        "state_visitation/CC": count[1],
        "state_visitation/CD": count[2],
        "state_visitation/DC": count[3],
        "state_visitation/DD": count[4],
        "state_visitation/SC": count[5],
        "state_visitation/SD": count[6],
        "state_visitation/CS": count[7],
        "state_visitation/DS": count[8],
    }


def ipditm_stats(
    state: EnvState, traj1: NamedTuple, traj2: NamedTuple, num_envs: int
) -> dict:
    from pax.envs.in_the_matrix import Actions

    """Compute statistics for IPDITM."""
    interacts1 = (
        jnp.count_nonzero(traj1.actions == Actions.interact) / num_envs
    )
    interacts2 = (
        jnp.count_nonzero(traj2.actions == Actions.interact) / num_envs
    )

    soft_reset_mask = jnp.where(traj1.rewards != 0, 1, 0)
    num_soft_resets = jnp.count_nonzero(traj1.rewards) / num_envs

    num_sft_resets = jnp.maximum(1, num_soft_resets)
    coops1 = (
        soft_reset_mask * traj1.observations["inventory"][..., 0]
    ).sum() / (num_envs * num_sft_resets)
    defect1 = (
        soft_reset_mask * traj1.observations["inventory"][..., 1]
    ).sum() / (num_envs * num_sft_resets)
    coops2 = (
        soft_reset_mask * traj2.observations["inventory"][..., 0]
    ).sum() / (num_envs * num_sft_resets)
    defect2 = (
        soft_reset_mask * traj2.observations["inventory"][..., 1]
    ).sum() / (num_envs * num_sft_resets)

    rewards1 = traj1.rewards.sum() / num_envs
    rewards2 = traj2.rewards.sum() / num_envs
    f_rewards1 = traj1.rewards[-1, ...].sum() / num_envs
    f_rewards2 = traj2.rewards[-1, ...].sum() / num_envs

    return {
        "interactions/1": interacts1,
        "interactions/2": interacts2,
        "coop_coin/1": coops1,
        "coop_coin/2": coops2,
        "defect_coin/1": defect1,
        "defect_coin/2": defect2,
        "total_coin/1": coops1 + defect1,
        "total_coin/2": coops2 + defect2,
        "ratio/1": jnp.nan_to_num(coops1 / (coops1 + defect1), nan=0),
        "ratio/2": jnp.nan_to_num(coops2 / (coops2 + defect2), nan=0),
        "num_soft_resets": num_soft_resets,
        "train/total_reward/player_1": rewards1,
        "train/total_reward/player_2": rewards2,
        "train/final_reward/player1": f_rewards1,
        "train/final_reward/player2": f_rewards2,
    }
