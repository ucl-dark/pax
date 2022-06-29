from flax import linen as nn
from .env import State
import jax.numpy as jnp


# five possible states
START = jnp.array([[0, 0, 0, 0, 1]])
CC = jnp.array([[1, 0, 0, 0, 0]])
CD = jnp.array([[0, 1, 0, 0, 0]])
DC = jnp.array([[0, 0, 1, 0, 0]])
DD = jnp.array([[0, 0, 0, 1, 0]])
STATE_NAMES = ["START", "CC", "CD", "DC", "DD"]
ALL_STATES = [START, CC, CD, DC, DD]


def policy_logger(agent) -> None:
    weights = agent.actor_optimizer.target["Dense_0"][
        "kernel"
    ]  # [layer_name]['w']
    log_pi = nn.softmax(weights)
    probs = {
        "policy/" + str(s): p[0] for (s, p) in zip(State, log_pi)
    }  # probability of cooperating is p[0]
    return probs


def value_logger(agent) -> None:
    weights = agent.critic_optimizer.target["Dense_0"]["kernel"]
    values = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
    }
    values.update(
        {f"value/{str(s)}.defect": p[1] for (s, p) in zip(State, weights)}
    )
    return values


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


def value_logger_dqn(agent) -> None:
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


def policy_logger_ppo(agent) -> None:
    weights = agent._state.params["categorical_value_head/~/linear"]["w"]
    pi = nn.softmax(weights)
    sgd_steps = agent._total_steps / agent._num_steps
    probs = {f"policy/{str(s)}.cooperate": p[0] for (s, p) in zip(State, pi)}
    probs.update({"policy/total_steps": sgd_steps})
    return probs


def value_logger_ppo(agent) -> None:
    weights = agent._state.params["categorical_value_head/~/linear_1"][
        "w"
    ]  # 5 x 1 matrix
    sgd_steps = agent._total_steps / agent._num_steps
    probs = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
    }
    probs.update({"value/total_steps": sgd_steps})
    return probs


def policy_logger_ppo_with_memory(agent) -> None:
    """Calculate probability of coopreation"""
    # n = 5
    params = agent._state.params
    hidden = agent._state.hidden
    episode = int(
        agent._logger.metrics["total_steps"]
        / (agent._num_steps * agent._num_envs)
    )
    cooperation_probs = {"episode": episode}

    # Works when the forward function is not jitted.
    for state, state_name in zip(ALL_STATES, STATE_NAMES):
        (dist, _), hidden = agent.forward(params, state, hidden)
        cooperation_probs[state_name] = float(dist.probs[0][0])
    return cooperation_probs


def value_logger_ppo_with_memory(agent) -> None:
    weights = agent._state.params["categorical_value_head/~/linear_1"][
        "w"
    ]  # 5 x 1 matrix
    sgd_steps = agent._total_steps / agent._num_steps
    probs = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
    }
    probs.update({"value/total_steps": sgd_steps})
    return probs


def ppo_losses(agent) -> None:
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    loss_entropy = agent._logger.metrics["loss_entropy"]
    losses = {
        "sgd_steps": sgd_steps,
        "losses/total": loss_total,
        "losses/policy": loss_policy,
        "losses/value": loss_value,
        "losses/entropy": loss_entropy,
    }
    return losses
