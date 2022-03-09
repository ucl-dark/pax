from math import log2

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax import nn

import wandb
from pax.env import IteratedPrisonersDilemma, State
from pax.sac.buffers import ReplayBuffer
from pax.sac.agent import SAC
from pax.strategies import Altruistic, Defect, Human, Random, TitForTat

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer(
    "num_exploration_episodes",
    10000,
    "Number of env episodes to run training for",
)
flags.DEFINE_integer(
    "train_steps", 500, "Number of gradient steps to run training for."
)
flags.DEFINE_integer("batch_size", 256, "batch size for gradient update")
flags.DEFINE_integer(
    "max_episode_length", 100, "max number of iterations in Prisoners Dilemma"
)
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_float("discount_rate", 0.99, "discount rate")


def collect_data(agent, other_agents, seed):
    key_0 = jax.random.PRNGKey(seed)
    key_1 = jax.random.PRNGKey(seed + 1)

    NUM_ENVS = 2 ** int(log2(FLAGS.num_episodes))
    env = IteratedPrisonersDilemma(FLAGS.max_episode_length)
    buffer = ReplayBuffer(state_dim=len(State), action_dim=2)

    print("Iterated Prisoners Dilemma")
    print(f"Max time limit {FLAGS.max_episode_length}")
    print(f"Exploring for {FLAGS.num_episodes} episodes with {NUM_ENVS} envs")
    print("-----------------------")
    for other_agent in other_agents:
        for _ in range(FLAGS.num_episodes // NUM_ENVS + 1):
            rewards = []
            _state = _previous_state = State.START * jnp.ones((NUM_ENVS, 1))
            for _ in range(FLAGS.max_episode_length):
                _obs_0, _obs_1 = env._observation(_state, 1), env._observation(
                    _state, 2
                )
                _action_0, _ = agent.actor_step(key_0, _obs_0, False)
                _action_1, _ = other_agent.actor_step(key_1, _obs_1, False)

                _state, _reward, _not_done = env.step(
                    _state, _action_0, _action_1
                )
                rewards.append(_reward)

                # add experience to buffer
                buffer.add_batch(
                    env._observation(_previous_state, 1),
                    jax.nn.one_hot(_action_0[:, 0], 2),
                    _obs_0,
                    _reward[:, 0],
                    1 - _not_done,
                )

                _previous_state = _state
                _, key_0 = jax.random.split(key_0)
                _, key_1 = jax.random.split(key_1)

            rewards = jnp.array(rewards)
            print(
                f"Mean Episode Rewards: {float(rewards[:,:,0].mean()), float(rewards[:,:,1].mean())}"
            )
    return buffer


def main(_):
    expert = Defect()
    # train script is SAC with SAC
    agent_0 = SAC(
        state_dim=len(State),
        action_dim=2,
        discount=FLAGS.discount_rate,
        lr=FLAGS.learning_rate,
        seed=FLAGS.seed,
    )

    dataset = collect_data(
        expert,
        [Defect(), Altruistic(), TitForTat()],
        seed=0,
    )
    agent_0.pre_train(dataset, 1000, 3e-3, 32, jax.random.PRNGKey(FLAGS.seed))
    agent_0.save(f"models/experts/{type(expert).__name__}")

    # evaluate against population of agents
    # population = [agent_0, TitForTat(), Defect(), Altruistic(), Random()]
    # [
    #     evaluate(
    #         agent_0, opponent, FLAGS.seed, FLAGS.max_episode_length, 1, False
    #     )[:, :, 0]
    #     for opponent in population
    # ]
    # total_reward = jnp.mean(jnp.asarray(rewards))
    # print(policy_logger(agent_0))


def policy_logger(agent) -> None:
    weights = agent.actor_optimizer.target["Dense_0"]["kernel"]
    log_pi = nn.softmax(weights)
    probs = {"policy." + str(s): p[0] for (s, p) in zip(State, log_pi)}
    return probs


def value_logger(agent) -> None:
    weights = agent.critic_optimizer.target["Dense_0"]["kernel"]
    values = {
        f"value.{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
    }
    values.update(
        {f"value.{str(s)}.defect": p[1] for (s, p) in zip(State, weights)}
    )
    return values


if __name__ == "__main__":
    app.run(main)
