from absl import app, flags
import wandb

from .env import IteratedPrisonersDilemma
from .independent_learners import IndependentLearners
from .runner import evaluate_loop, train_loop
from .sac.agent import SAC
from .watchers import policy_logger, value_logger

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("wandb_group", "-", "wandb group")
flags.DEFINE_integer(
    "num_eps", 100000, "num of env episodes to run training for"
)

flags.DEFINE_integer("num_envs", 20, "num of parallel envs")
flags.DEFINE_integer("eval_every", 100, "evaluate performance every")
flags.DEFINE_integer("batch_size", 256, "batch size for gradient update")
flags.DEFINE_integer(
    "eps_length", 100, "max number of iterations in Prisoners Dilemma"
)
flags.DEFINE_float("learning_rate", 3e-2, "learning rate")
flags.DEFINE_float("discount_rate", 0.99, "discount rate")


def global_setup():
    if str(FLAGS.wandb_group) == "-":
        print("Please provide wandb_group flag")
        exit()

    wandb.init(
        project="ipd",
        entity="ucl-dark",
        group=str(FLAGS.wandb_group),
        name=f"run-{FLAGS.seed}",
    )
    wandb.config.update(flags.FLAGS)

    return wandb.run.name


def env_setup():
    train_env = IteratedPrisonersDilemma(FLAGS.eps_length, FLAGS.num_envs)
    test_env = IteratedPrisonersDilemma(FLAGS.eps_length, 1)
    return train_env, test_env


def agent_setup():
    # TODO: make configurable
    agent_0 = SAC(
        state_dim=5,
        action_dim=2,
        discount=FLAGS.discount_rate,
        lr=FLAGS.learning_rate,
        seed=FLAGS.seed,
    )

    agent_1 = SAC(
        state_dim=5,
        action_dim=2,
        discount=FLAGS.discount_rate,
        lr=FLAGS.learning_rate,
        seed=FLAGS.seed,
    )

    return IndependentLearners([agent_0, agent_1])


def watcher_setup():
    # TODO: make configurable based on previous agents

    def sac_log(agent):
        policy_dict = policy_logger(agent)
        value_dict = value_logger(agent)
        policy_dict.update(value_dict)
        wandb.log(policy_dict)
        return

    def dumb_log(agent):
        return

    return [sac_log, dumb_log]


def main(_):
    _ = global_setup()
    train_env, test_env = env_setup()
    agents = agent_setup()
    watchers = watcher_setup()

    train_episodes = FLAGS.num_eps
    eval_every = FLAGS.eval_every
    assert not train_episodes % eval_every
    for _ in range(int(train_episodes // eval_every)):
        evaluate_loop(test_env, agents, 1, watchers)
        train_loop(train_env, agents, eval_every, watchers)


if __name__ == "__main__":
    app.run(main)
