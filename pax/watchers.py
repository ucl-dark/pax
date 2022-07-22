from flax import linen as nn

from pax.naive_learners import NaiveLearnerEx
from .env import State
import jax.numpy as jnp
import pax.hyper.ppo as HyperPPO
import pax.hyper.ppo_gru as HyperPPOMemory
import pax.ppo.ppo as PPO

# five possible states
START = jnp.array([[0, 0, 0, 0, 1]])
CC = jnp.array([[1, 0, 0, 0, 0]])
CD = jnp.array([[0, 1, 0, 0, 0]])
DC = jnp.array([[0, 0, 1, 0, 0]])
DD = jnp.array([[0, 0, 0, 1, 0]])
STATE_NAMES = ["START", "CC", "CD", "DC", "DD"]
ALL_STATES = [START, CC, CD, DC, DD]


def policy_logger(agent) -> dict:
    weights = agent.actor_optimizer.target["Dense_0"][
        "kernel"
    ]  # [layer_name]['w']
    log_pi = nn.softmax(weights)
    probs = {
        "policy/" + str(s): p[0] for (s, p) in zip(State, log_pi)
    }  # probability of cooperating is p[0]
    return probs


def value_logger(agent) -> dict:
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


def policy_logger_ppo(agent: PPO) -> dict:
    weights = agent._state.params["categorical_value_head/~/linear"]["w"]
    pi = nn.softmax(weights)
    sgd_steps = agent._total_steps / agent._num_steps
    probs = {f"policy/{str(s)}.cooperate": p[0] for (s, p) in zip(State, pi)}
    probs.update({"policy/total_steps": sgd_steps})
    return probs


def value_logger_ppo(agent: PPO) -> dict:
    weights = agent._state.params["categorical_value_head/~/linear_1"][
        "w"
    ]  # 5 x 1 matrix
    sgd_steps = agent._total_steps / agent._num_steps
    probs = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
    }
    probs.update({"value/total_steps": sgd_steps})
    return probs


def policy_logger_ppo_with_memory(agent) -> dict:
    """Calculate probability of coopreation"""
    # n = 5
    params = agent._state.params
    hidden = agent._state.hidden
    # episode = int(
    #     agent._logger.metrics["total_steps"]
    #     / (agent._num_steps * agent._num_envs)
    # )
    episode = int(agent._logger.metrics["total_steps"] / agent._num_steps)
    cooperation_probs = {"episode": episode}

    # TODO: Figure out how to JIT the forward function
    # Works when the forward function is not jitted.
    for state, state_name in zip(ALL_STATES, STATE_NAMES):
        (dist, _), hidden = agent.forward(params, state, hidden)
        cooperation_probs[f"policy/{state_name}"] = float(dist.probs[0][0])
    return cooperation_probs


def policy_logger_hyper_gru(agent: HyperPPOMemory) -> dict:
    """Calculate probability of coopreation"""
    # n = 5
    params = agent._state.params
    hidden = agent._state.hidden
    episode = int(
        agent._logger.metrics["total_steps"]
        / (agent._num_steps * agent._num_envs)
    )
    cooperation_probs = {"episode": episode}
    pid = agent.player_id

    # TODO: Figure out how to JIT the forward function
    # Works when the forward function is not jitted.
    (dist, _), hidden = agent.forward(params, 0.5 * jnp.ones((1, 10)), hidden)
    mu, var = dist.mean(), dist.variance()
    for i, state_name in enumerate(STATE_NAMES):
        cooperation_probs[f"policy/hyper_{pid}/mu/{state_name}"] = float(
            mu[0][i]
        )
        cooperation_probs[f"policy/hyper_{pid}/var/{state_name}"] = float(
            var[0][i]
        )
    return cooperation_probs


def naive_losses(agent) -> None:
    sgd_steps = agent._logger.metrics["sgd_steps"]
    loss_total = agent._logger.metrics["loss_total"]
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    losses = {
        "sgd_steps": sgd_steps,
        "train/total": loss_total,
        "train/policy": loss_policy,
        "train/value": loss_value,
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


def policy_logger_naive(agent) -> None:
    weights = agent._state.params["categorical_value_head/~/linear"]["w"]
    pi = nn.softmax(weights)
    sgd_steps = agent._total_steps / agent._num_steps
    probs = {
        f"policy/{str(s)}/{agent.player_id}.cooperate": p[0]
        for (s, p) in zip(State, pi)
    }
    probs.update({"policy/total_steps": sgd_steps})
    return probs
