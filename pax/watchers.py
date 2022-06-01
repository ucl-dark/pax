from flax import linen as nn
from .env import State


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
    probs = {f"policy/{str(s)}.cooperate": p[0] for (s, p) in zip(State, pi)}
    probs.update(
        {f"policy/{str(s)}.defect": p[1] for (s, p) in zip(State, pi)}
    )
    probs.update({"policy/target_update_steps": agent.num_target_updates})
    return probs


def value_logger_dqn(agent) -> None:
    weights = agent._state.target_params["linear"]["w"]  # 5 x 2 matrix
    values = {
        f"value/{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
    }
    values.update(
        {f"value/{str(s)}.defect": p[1] for (s, p) in zip(State, weights)}
    )
    values.update({"value/target_update_steps": agent.num_target_updates})
    return values
