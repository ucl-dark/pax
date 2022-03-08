# from math import log2

# import jax
# import jax.numpy as jnp
# from absl import app, flags
# from flax import linen as nn

# import wandb
# from pax.game import IteratedPrisonersDilemma, State
# from pax.sac.buffers import ReplayBuffer
# from pax.sac.discrete_SAC import SAC
# from pax.strategies import Altruistic, Defect, Human, Random, TitForTat

# FLAGS = flags.FLAGS
# flags.DEFINE_integer("seed", 0, "Random seed.")
# flags.DEFINE_integer(
#     "num_episodes", 10000, "Number of env episodes to run training for"
# )

# flags.DEFINE_integer("num_envs", 100, "Number of parallel envs")
# flags.DEFINE_integer(
#     "train_steps", 500, "Number of gradient steps to run training for."
# )
# flags.DEFINE_integer("batch_size", 256, "batch size for gradient update")
# flags.DEFINE_integer(
#     "max_episode_length", 100, "max number of iterations in Prisoners Dilemma"
# )
# flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
# flags.DEFINE_float("discount_rate", 0.99, "discount rate")


# def train(agent_0, agent_1, seed, num_episodes, train_steps, batch_size):
#     key_0 = jax.random.PRNGKey(seed)
#     key_1 = jax.random.PRNGKey(seed + 1)
#     env = IteratedPrisonersDilemma(FLAGS.max_episode_length, FLAGS.num_envs)
#     buffer_0 = ReplayBuffer(state_dim=len(State), action_dim=2)
#     buffer_1 = ReplayBuffer(state_dim=len(State), action_dim=2)

#     print("Iterated Prisoners Dilemma")
#     print(f"Max time limit {FLAGS.max_episode_length}")
#     print(f"Exploring for {num_episodes} episodes with {FLAGS.num_envs} envs")
#     print(f"Training for {train_steps} episodes with batch size: {batch_size}")
#     print("-----------------------")

#     for i in range(train_steps):
#         print(f"Exploration step {i}")
#         for _ in range(num_episodes // (FLAGS.num_envs * train_steps) + 1):
#             rewards = []
#             _state = _previous_state = State.START * jnp.ones(
#                 (FLAGS.num_envs, 1)
#             )
#             for _ in range(FLAGS.max_episode_length):
#                 _obs_0, _obs_1 = env._observation(_state, 1), env._observation(
#                     _state, 2
#                 )
#                 _action_0, _pi_0 = agent_0.actor_step(key_0, _obs_0, False)
#                 _action_1, _pi_1 = agent_1.actor_step(key_1, _obs_1, False)

#                 _state, _reward, _not_done = env.step(
#                     _state, _action_0, _action_1
#                 )
#                 rewards.append(_reward)

#                 # add experience to buffer
#                 buffer_0.add_batch(
#                     env._observation(_previous_state, 1),
#                     jax.nn.one_hot(_action_0[:, 0], 2),
#                     _obs_0,
#                     _reward[:, 0],
#                     1 - _not_done,
#                 )

#                 buffer_1.add_batch(
#                     env._observation(_previous_state, 2),
#                     jax.nn.one_hot(_action_1[:, 0], 2),
#                     _obs_1,
#                     _reward[:, 1],
#                     1 - _not_done,
#                 )

#                 _previous_state = _state
#                 _, key_0 = jax.random.split(key_0)
#                 _, key_1 = jax.random.split(key_1)

#             rewards = jnp.array(rewards)
#             print(
#                 f"Mean Episode Rewards: {float(rewards[:,:,0].mean()), float(rewards[:,:,1].mean())}"
#             )

#             wandb.log(
#                 {
#                     "Episode Reward Player 1": float(rewards[:, :, 0].mean()),
#                     "Episode Reward Player 2": float(rewards[:, :, 1].mean()),
#                 }
#             )
#         # here (if training) we should do some gradient update
#         print(f"Train Step: {i}")
#         if i % 10:
#             wandb.log(policy_logger(agent_0))
#             wandb.log(value_logger(agent_0))

#         agent_0.train(buffer_0, batch_size)
#         agent_1.train(buffer_1, batch_size)

#     return agent_0, agent_1


# def main(_):
#     wandb.init(project="iterated_prisoners_dilemma")
#     wandb.config.update(flags.FLAGS)
#     experiment_name = wandb.run.name

#     # train script is SAC with SAC
#     agent_0 = SAC(
#         state_dim=len(State),
#         action_dim=2,
#         discount=FLAGS.discount_rate,
#         lr=FLAGS.learning_rate,
#         seed=FLAGS.seed,
#     )
#     agent_1 = SAC(
#         state_dim=len(State),
#         action_dim=2,
#         discount=FLAGS.discount_rate,
#         lr=FLAGS.learning_rate,
#         seed=FLAGS.seed + 1,
#     )

#     agent_0, agent_1 = train(
#         agent_0,
#         agent_1,
#         FLAGS.seed,
#         FLAGS.num_episodes,
#         FLAGS.train_steps,
#         FLAGS.batch_size,
#     )

#     agent_0.save(f"models/{experiment_name}")

#     # evaluate against population of agents
#     # population = [agent_0, TitForTat(), Defect(), Altruistic(), Random()]
#     # rewards = [
#     #     evaluate(
#     #         agent_0, opponent, FLAGS.seed, FLAGS.max_episode_length, 1, True
#     #     )[:, :, 0]
#     #     for opponent in population
#     # ]
#     total_reward = jnp.mean(jnp.asarray(rewards))
#     wandb.run.summary["Population Mean Reward"] = total_reward


# def policy_logger(agent) -> None:
#     weights = agent.actor_optimizer.target["Dense_0"]["kernel"]
#     log_pi = nn.softmax(weights)
#     probs = {"policy." + str(s): p[0] for (s, p) in zip(State, log_pi)}
#     return probs


# def value_logger(agent) -> None:
#     weights = agent.critic_optimizer.target["Dense_0"]["kernel"]
#     values = {
#         f"value.{str(s)}.cooperate": p[0] for (s, p) in zip(State, weights)
#     }
#     values.update(
#         {f"value.{str(s)}.defect": p[1] for (s, p) in zip(State, weights)}
#     )
#     return values


# if __name__ == "__main__":
#     app.run(main)
