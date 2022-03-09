from flax import linen as nn
from .env import State


def policy_logger(agent) -> None:
    weights = agent.actor_optimizer.target["Dense_0"]["kernel"]
    log_pi = nn.softmax(weights)
    probs = {"policy/" + str(s): p[0] for (s, p) in zip(State, log_pi)}
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
