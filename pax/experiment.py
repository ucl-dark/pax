import logging
import os

from pax.dqn.agent import default_agent
from pax.env import (
    SequentialMatrixGame,
    IteratedPrisonersDilemma,
    StagHunt,
    BattleOfTheSexes,
    Chicken,
)
from pax.centralized_learners import CentralizedLearners
from pax.independent_learners import IndependentLearners
from pax.ppo.ppo import make_agent
from pax.ppo.ppo_gru import make_gru_agent
from pax.lola.lola import make_lola
from pax.runner import Runner
from pax.sac.agent import SAC
from pax.strategies import (
    TitForTat,
    Defect,
    Altruistic,
    Random,
    Human,
    GrimTrigger,
)
from pax.utils import Section
from pax.watchers import (
    policy_logger,
    policy_logger_dqn,
    value_logger,
    value_logger_dqn,
    ppo_losses,
    policy_logger_ppo,
    value_logger_ppo,
    policy_logger_ppo_with_memory,
)

import hydra
import wandb


def global_setup(args):
    """Set up global variables."""
    os.makedirs(args.save_dir, exist_ok=True)
    if args.wandb.log:
        print("name", str(args.wandb.name))
        wandb.init(
            reinit=True,
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            config=vars(args),
        )


def env_setup(args, logger):
    """Set up env variables."""
    games = {
        "ipd": IteratedPrisonersDilemma,
        "stag": StagHunt,
        "sexes": BattleOfTheSexes,
        "chicken": Chicken,
    }
    if args.payoff is not None:
        assert (
            len(args.payoff) == 4
        ), f"Expected length 4 but got {len(args.payoff)}"
        assert (
            len(args.payoff[0]) == 2
        ), f"Expected length 2 but got {len(args.payoff[0])}"
        assert (
            len(args.payoff[1]) == 2
        ), f"Expected length 2 but got {len(args.payoff[1])}"
        assert (
            len(args.payoff[2]) == 2
        ), f"Expected length 2 but got {len(args.payoff[2])}"
        assert (
            len(args.payoff[3]) == 2
        ), f"Expected length 2 but got {len(args.payoff[3])}"
        train_env = SequentialMatrixGame(
            args.num_steps, args.num_envs, args.payoff
        )
        test_env = SequentialMatrixGame(args.num_steps, 1, args.payoff)
        logger.info(f"Game: Custom | payoff: {args.payoff}")
    else:
        assert args.game in games, f"{args.game} not in {games.keys()}"
        train_env = games[args.game](args.num_steps, args.num_envs)
        test_env = games[args.game](args.num_steps, 1)
        logger.info(f"Game: {args.game} | payoff: {train_env.payoff}")

    return train_env, test_env


def runner_setup():
    return Runner()


def agent_setup(args, logger):
    """Set up agent variables."""

    def get_SAC_agent(*args):
        sac_agent = SAC(
            state_dim=5,
            action_dim=2,
            discount=args.discount,
            lr=args.lr,
            seed=args.seed,
        )
        return sac_agent

    def get_DQN_agent(seed, player_id):
        # dummy environment to get observation and action spec
        dummy_env = SequentialMatrixGame(
            args.num_steps, args.num_envs, args.payoff
        )
        dqn_agent = default_agent(
            args,
            obs_spec=dummy_env.observation_spec(),
            action_spec=dummy_env.action_spec(),
            seed=seed,
            player_id=player_id,
        )
        return dqn_agent

    def get_PPO_agent(seed, player_id):
        # dummy environment to get observation and action spec
        dummy_env = SequentialMatrixGame(
            args.num_steps, args.num_envs, args.payoff
        )
        if args.ppo.with_memory:
            ppo_agent = make_gru_agent(
                args,
                obs_spec=(dummy_env.observation_spec().num_values,),
                action_spec=dummy_env.action_spec().num_values,
                seed=seed,
                player_id=player_id,
            )
        else:
            ppo_agent = make_agent(
                args,
                obs_spec=(dummy_env.observation_spec().num_values,),
                action_spec=dummy_env.action_spec().num_values,
                seed=seed,
                player_id=player_id,
            )
        return ppo_agent

    def get_LOLA_agent(seed, player_id):
        lola_agent = make_lola(seed)
        return lola_agent

    strategies = {
        "TitForTat": TitForTat,
        "Defect": Defect,
        "Altruistic": Altruistic,
        "Human": Human,
        "Random": Random,
        "Grim": GrimTrigger,
        # "ZDExtortion": ZDExtortion,
        "SAC": get_SAC_agent,
        "DQN": get_DQN_agent,
        "PPO": get_PPO_agent,
        "LOLA": get_LOLA_agent,
    }

    assert args.agent1 in strategies
    assert args.agent2 in strategies

    # TODO: Is there a better way to start agents with different seeds?
    num_agents = 2
    seeds = [seed for seed in range(args.seed, args.seed + num_agents)]
    # Create Player IDs by normalizing seeds to 1, 2 respectively
    pids = [
        seed % seed + i if seed != 0 else 1
        for seed, i in zip(seeds, range(1, num_agents + 1))
    ]
    agent_0 = strategies[args.agent1](seeds[0], pids[0])  # player 1
    agent_1 = strategies[args.agent2](seeds[1], pids[1])  # player 2

    logger.info(f"PPO with memory: {args.ppo.with_memory}")
    logger.info(f"Agent Pair: {args.agent1} | {args.agent2}")
    logger.info(f"Agent seeds: {seeds[0]} | {seeds[1]}")

    if args.centralized:
        return CentralizedLearners([agent_0, agent_1])

    return IndependentLearners([agent_0, agent_1])


def watcher_setup(args, logger):
    """Set up watcher variables."""

    def sac_log(agent, *args):
        policy_dict = policy_logger(agent)
        value_dict = value_logger(agent)
        policy_dict.update(value_dict)
        wandb.log(policy_dict)
        return

    def dqn_log(agent):
        policy_dict = policy_logger_dqn(agent)
        value_dict = value_logger_dqn(agent)
        policy_dict.update(value_dict)
        if args.wandb.log:
            wandb.log(policy_dict)
        return

    def ppo_log(agent):
        losses = ppo_losses(agent)
        if args.ppo.with_memory:
            policy = policy_logger_ppo_with_memory(agent)
        else:
            policy = policy_logger_ppo(agent)
            value = value_logger_ppo(agent)
            losses.update(value)
        losses.update(policy)
        if args.wandb.log:
            wandb.log(losses)
        return

    def dumb_log(agent, *args):
        return

    strategies = {
        "TitForTat": dumb_log,
        "Defect": dumb_log,
        "Altruistic": dumb_log,
        "Human": dumb_log,
        "Random": dumb_log,
        "Grim": dumb_log,
        # "ZDExtortion": dumb_log,
        "SAC": sac_log,
        "DQN": dqn_log,
        "PPO": ppo_log,
        "LOLA": dumb_log,
    }

    assert args.agent1 in strategies
    assert args.agent2 in strategies

    agent_0_log = strategies[args.agent1]
    agent_1_log = strategies[args.agent2]

    return [agent_0_log, agent_1_log]


@hydra.main(config_path="conf", config_name="config")
def main(args):
    """Set up main."""
    logger = logging.getLogger()
    with Section("Global setup", logger=logger):
        global_setup(args)

    with Section("Env setup", logger=logger):
        train_env, test_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent_pair = agent_setup(args, logger)

    with Section("Watcher setup", logger=logger):
        watchers = watcher_setup(args, logger)

    with Section("Runner setup", logger=logger):
        runner = runner_setup()

    # TODO: Do we need this?
    # assert not train_episodes % eval_every
    num_episodes = int(args.total_timesteps / (args.num_steps * args.num_envs))
    print(f"Number of training episodes = {num_episodes}")
    print()
    if not args.wandb.log:
        watchers = False
    for num_update in range(int(num_episodes // args.eval_every)):
        print(f"Update: {num_update}/{int(num_episodes // args.eval_every)}")
        print()
        runner.evaluate_loop(test_env, agent_pair, 1, watchers)
        runner.train_loop(train_env, agent_pair, args.eval_every, watchers)


if __name__ == "__main__":
    main()
