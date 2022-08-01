import time
from typing import NamedTuple
import copy

import jax
import jax.numpy as jnp


import wandb
from pax.env import IteratedPrisonersDilemma
from pax.independent_learners import IndependentLearners
from pax.strategies import Defect, TitForTat
from pax.utils import TrainingState, copy_state_and_network

# TODO: make these a copy of acme


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class Runner:
    """Holds the runner's state."""

    def __init__(self, args):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()
        self.args = args

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents

        def _env_rollout(carry, unused):
            t1, t2, a1_state, a2_state = carry
            a1, a1_state = agent1._policy(
                a1_state.params, t1.observation, a1_state
            )
            a2, a2_state = agent2._policy(
                a2_state.params, t2.observation, a2_state
            )
            tprime_1, tprime_2 = env.runner_step(
                [
                    a1,
                    a2,
                ]
            )
            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                a1_state.extras["log_probs"],
                a1_state.extras["values"],
                tprime_1.last() * jnp.zeros(env.num_envs),
                a1_state.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                a2_state.extras["log_probs"],
                a2_state.extras["values"],
                tprime_2.last() * jnp.zeros(env.num_envs),
                a2_state.hidden,
            )
            return (
                tprime_1,
                tprime_2,
                a1_state,
                a2_state,
            ), (traj1, traj2)

        def _env_rollout2(carry, unused):
            t1, t2, a1_state, a2_state = carry
            a1, a1_state = agent2._policy(
                a1_state.params, t1.observation, a1_state
            )
            a2, a2_state = agent1._policy(
                a2_state.params, t2.observation, a2_state
            )
            tprime_1, tprime_2 = env.runner_step(
                [
                    a1,
                    a2,
                ]
            )
            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                a1_state.extras["log_probs"],
                a1_state.extras["values"],
                tprime_1.last() * jnp.zeros(env.num_envs),
                a1_state.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                a2_state.extras["log_probs"],
                a2_state.extras["values"],
                tprime_2.last() * jnp.zeros(env.num_envs),
                a2_state.hidden,
            )
            return (tprime_1, tprime_2, a1_state, a2_state), (traj1, traj2)

        for _ in range(0, max(int(num_episodes / env.num_envs), 1)):
            ######################### LOLA #########################
            # start of unique lola code
            if self.args.agent1 == "LOLA" and self.args.agent2 == "LOLA":
                # copy state and haiku network
                (
                    agent1.other_state,
                    agent1.other_network,
                ) = copy_state_and_network(agent2)
                (
                    agent2.other_state,
                    agent2.other_network,
                ) = copy_state_and_network(agent1)
                # inner rollout
                for _ in range(self.args.lola.num_lookaheads):
                    agent1.in_lookahead(env, _env_rollout)
                    agent2.in_lookahead(env, _env_rollout2)

                # outer rollout
                agent1.out_lookahead(env, _env_rollout)
                agent2.out_lookahead(env, _env_rollout2)

            elif self.args.agent1 == "LOLA" and self.args.agent2 != "LOLA":
                # copy state and haiku network of agent 2
                (
                    agent1.other_state,
                    agent1.other_network,
                ) = copy_state_and_network(agent2)
                # inner rollout
                for _ in range(self.args.lola.num_lookaheads):
                    agent1.in_lookahead(env, _env_rollout)
                # outer rollout
                agent1.out_lookahead(env, _env_rollout)

            elif self.args.agent1 != "LOLA" and self.args.agent2 == "LOLA":
                # copy state and haiku network of agent 1
                (
                    agent2.other_state,
                    agent2.other_network,
                ) = copy_state_and_network(agent1)
                # inner rollout
                for _ in range(self.args.lola.num_lookaheads):
                    agent2.in_lookahead(env, _env_rollout)
                # outer rollout
                agent2.out_lookahead(env, _env_rollout)
            # end of unique lola code
            ######################### LOLA #########################

            t_init = env.reset()
            a1_state = agent1.reset_memory()
            a2_state = agent2.reset_memory()

            if self.args.agent2 == "NaiveLearnerEx":
                # unique naive-learner code
                a2_state = agent2.make_initial_state(t_init[1])

            # rollout episode
            vals, trajectories = jax.lax.scan(
                _env_rollout,
                (*t_init, a1_state, a2_state),
                None,
                length=env.episode_length,
            )

            self.train_episodes += 1
            rewards_0 = trajectories[0].rewards.mean()
            rewards_1 = trajectories[1].rewards.mean()

            # update agent / add final trajectory
            final_t1 = vals[0]._replace(step_type=2)
            a1_state = vals[2]
            a1_state = agent1.update(trajectories[0], final_t1, a1_state)

            final_t2 = vals[1]._replace(step_type=2)
            a2_state = vals[3]
            a2_state = agent2.update(trajectories[1], final_t2, a2_state)

            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
                f"| Joint reward: {(rewards_0.mean() + rewards_1.mean())*0.5}"
            )

            # print(
            #     f"Total Episode Reward: {mean_r_0, mean_r_1} | Joint reward: {(mean_r_0 + mean_r_1)*0.5}"
            # )

            # logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "episodes": self.train_episodes,
                        "train/episode_reward/player_1": float(
                            rewards_0.mean()
                        ),
                        "train/episode_reward/player_2": float(
                            rewards_1.mean()
                        ),
                        "train/episode_reward/joint": (
                            rewards_0.mean() + rewards_1.mean()
                        )
                        * 0.5,
                    },
                )
        print()

        # TODO: Why do we need this if we already update the state in agent.update?
        # update agents
        agents.agents[0]._state = a1_state
        agents.agents[1]._state = a2_state
        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        agent1, agent2 = agents.agents
        agent1.reset_memory()
        agent2.reset_memory()

        for _ in range(num_episodes // env.num_envs):
            rewards_0, rewards_1 = [], []
            timesteps = env.reset()
            while not timesteps[0].last():
                actions = agents.select_action(timesteps)
                timesteps = env.step(actions)
                r_0, r_1 = timesteps[0].reward, timesteps[1].reward
                rewards_0.append(timesteps[0].reward)
                rewards_1.append(timesteps[1].reward)

                # update evaluation steps
                self.eval_steps += 1

                if watchers:
                    agents.log(watchers)
                    env_info = {
                        "eval/evaluation_steps": self.eval_steps,
                        "eval/step_reward/player_1": float(
                            jnp.array(r_0).mean()
                        ),
                        "eval/step_reward/player_2": float(
                            jnp.array(r_1).mean()
                        ),
                    }
                    wandb.log(env_info)

            # end of episode stats
            self.eval_episodes += 1
            rewards_0 = jnp.array(rewards_0)
            rewards_1 = jnp.array(rewards_1)

            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )
            if watchers:
                wandb.log(
                    {
                        "train/episodes": self.train_episodes,
                        "eval/episodes": self.eval_episodes,
                        "eval/episode_reward/player_1": float(
                            rewards_0.mean()
                        ),
                        "eval/episode_reward/player_2": float(
                            rewards_1.mean()
                        ),
                    }
                )
        agents.eval(False)
        print()
        return agents


if __name__ == "__main__":
    agents = IndependentLearners([Defect(), TitForTat()])
    # TODO: accept the arguments from config file instead of hard-coding
    # Default to prisoner's dilemma
    env = IteratedPrisonersDilemma(50, 5)
    wandb.init(mode="online")

    def log_print(agent) -> dict:
        print(agent)
        return None
