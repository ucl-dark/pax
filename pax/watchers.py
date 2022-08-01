import jax.numpy as jnp
from flax import linen as nn

from pax.lola.lola import LOLA
import pax.hyper.ppo as HyperPPO
import pax.ppo.ppo as PPO
from pax.naive.naive import NaiveLearner
from pax.naive_exact import NaiveLearnerEx

from .env import State

# five possible states
START = jnp.array([[0, 0, 0, 0, 1]])
CC = jnp.array([[1, 0, 0, 0, 0]])
CD = jnp.array([[0, 1, 0, 0, 0]])
DC = jnp.array([[0, 0, 1, 0, 0]])
DD = jnp.array([[0, 0, 0, 1, 0]])
STATE_NAMES = ["START", "CC", "DC", "CD", "DD"]
ALL_STATES = [START, CC, DC, CD, DD]


# General policy logger
def policy_logger(agent) -> dict:
    # Categorical plotting
    weights = agent._state.params["categorical_value_head/~/linear"]["w"]
    pi = nn.softmax(weights)
    # ####### # #
    # # Bernoulli plotting
    # weights = agent._state.params["bernoulli_value_head/~/linear"]["w"]
    # pi = nn.sigmoid(weights)
    # sgd_steps = agent._total_steps / agent._num_steps
    sgd_steps = agent._logger.metrics["sgd_steps"]
    probs = {
        f"policy/{agent.player_id}/{str(s)}": p[0] for (s, p) in zip(State, pi)
    }
    probs.update({"policy/total_steps": sgd_steps})
    return probs


# General value logger
def value_logger(agent) -> dict:
    # Categorical plotting
    weights = agent._state.params["categorical_value_head/~/linear_1"]["w"]
    # Bernoulli plotting
    # weights = agent._state.params["bernoulli_value_head/~/linear_1"]["w"]
    # sgd_steps = agent._total_steps / agent._num_steps
    sgd_steps = agent._logger.metrics["sgd_steps"]
    probs = {
        f"value/{agent.player_id}/{str(s)}": p[0]
        for (s, p) in zip(State, weights)
    }
    probs.update({"value/total_steps": sgd_steps})
    return probs


# General loss logger
def losses_logger(agent) -> None:
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    losses = {
        "sgd_steps": sgd_steps,
        f"loss/{agent.player_id}/total": loss_total,
        f"loss/{agent.player_id}/policy": loss_policy,
        f"loss/{agent.player_id}/value": loss_value,
    }
    return losses


# Loss logger for PPO (has additional terms)
def losses_ppo(agent: PPO) -> dict:
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    loss_entropy = agent._logger.metrics["loss_entropy"]
    entropy_cost = agent._logger.metrics["entropy_cost"]
    losses = {
        "sgd_steps": sgd_steps,
        f"loss/{agent.player_id}/total": loss_total,
        f"loss/{agent.player_id}/policy": loss_policy,
        f"loss/{agent.player_id}/value": loss_value,
        f"loss/{agent.player_id}/entropy": loss_entropy,
        f"loss/{agent.player_id}/entropy_coefficient": entropy_cost,
    }
    return losses


# General policy logger for networks with more than on layer
def policy_logger_ppo_with_memory(agent) -> dict:
    """Calculate probability of cooperation"""
    params = agent._state.params
    hidden = agent._state.hidden
    episode = int(agent._logger.metrics["total_steps"] / agent._num_steps)
    cooperation_probs = {"episode": episode}
    for state, state_name in zip(ALL_STATES, STATE_NAMES):
        (dist, _), hidden = agent.forward(params, state, hidden)
        cooperation_probs[f"policy/{agent.player_id}/{state_name}"] = float(
            dist.probs[0][0]
        )
    return cooperation_probs


# TODO: Clean these up (possibly redundant)
def policy_logger_dqn(agent) -> None:
    # this assumes using a linear layer, so this logging won't work using MLP
    weights = agent._state.target_params["linear"]["w"]  # 5 x 2 matrix
    pi = nn.softmax(weights)
    pid = agent.player_id
    target_steps = agent.target_step_updates
    probs = {
        f"policy/player_{str(pid)}/{str(s)}.cooperate": p[0]
        for (s, p) in zip(State, pi)
    }
    probs.update(
        {
            f"policy/player_{str(pid)}/{str(s)}.defect": p[1]
            for (s, p) in zip(State, pi)
        }
    )
    probs.update({"policy/target_step_updates": target_steps})
    return probs


def value_logger_dqn(agent) -> dict:
    weights = agent._state.target_params["linear"]["w"]  # 5 x 2 matrix
    pid = agent.player_id
    target_steps = agent.target_step_updates
    values = {
        f"value/player_{str(pid)}/{str(s)}.cooperate": p[0]
        for (s, p) in zip(State, weights)
    }
    values.update(
        {
            f"value/player_{str(pid)}/{str(s)}.defect": p[1]
            for (s, p) in zip(State, weights)
        }
    )
    values.update({"value/target_step_updates": target_steps})
    return values


def logger_hyper(agent: HyperPPO) -> dict:
    episode = int(
        agent._logger.metrics["total_steps"]
        / (agent._num_steps * agent._num_envs)
    )
    cooperation_probs = {"episode": episode}
    return cooperation_probs


def losses_naive(agent: NaiveLearnerEx) -> dict:
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


def logger_naive(agent: NaiveLearnerEx) -> dict:
    params = agent._state.params
    pid = agent.player_id
    params = params.mean(axis=0)
    cooperation_probs = {"episode": agent._logger.metrics["total_steps"]}
    for i, state_name in enumerate(STATE_NAMES):
        cooperation_probs[
            f"policy/naive_learner_{pid}/avg/{state_name}"
        ] = float(params[i])
    return cooperation_probs
