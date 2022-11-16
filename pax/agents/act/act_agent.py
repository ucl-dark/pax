from typing import Any, Dict, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from pax.agents.agent import AgentInterface
from pax import utils
from pax.utils import Logger, MemoryState, TrainingState, get_advantages
from pax.agents.act.networks import make_act_network

class ActAgent(AgentInterface):
    def __init__(
        self,
        network: NamedTuple,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: Tuple,
        num_envs: int = 4,
        entropy_coeff_start: float = 0.1,
        player_id: int = 0,
    ):
        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            """Agent policy to select actions and calculate agent specific information"""
            values = network.apply(state.params, observation)
            mem.extras["values"] = values
            mem = mem._replace(extras=mem.extras)
            return values, state, mem

        def make_initial_state(key: Any, hidden: jnp.ndarray) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=obs_spec)

            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros((num_envs, 2)),
                    "log_probs": jnp.zeros(num_envs),
                },
            )
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(random_key, jnp.zeros(1))

        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
        }

        # Initialize functions
        self._policy = policy
        self.player_id = player_id

        # Other useful hyperparameters
        self._num_envs = num_envs  # number of environments

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros((num_envs, 2)),
                "log_probs": jnp.zeros(num_envs),
            },
        )
        return memory

def make_act_agent(
    args,
    obs_spec,
    action_spec,
    seed: int,
    player_id: int,
    tabular=False,
):
    """Make PPO agent"""
    if args.runner == "act":
        network = make_act_network(action_spec)
    else:
        raise NotImplementedError

    # Optimizer
    batch_size = int(args.num_envs * args.num_steps)
    transition_steps = (
        args.total_timesteps
        / batch_size
        * args.ppo.num_epochs
        * args.ppo.num_minibatches
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.ppo.max_gradient_norm),
        optax.scale_by_adam(eps=args.ppo.adam_epsilon),
        optax.scale(-args.ppo.learning_rate),
    )

    random_key = jax.random.PRNGKey(seed=seed)

    agent = ActAgent(
        network=network,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        num_envs=args.num_envs,
        entropy_coeff_start=args.ppo.entropy_coeff_start,
        player_id=player_id,
    )
    return agent