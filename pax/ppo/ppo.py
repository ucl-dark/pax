from typing import Any, Callable, NamedTuple, Sequence
import hydra
import jax.numpy as jnp
from pax.ppo.networks import critic_network, actor_network
import haiku as hk
import optax

# import wandb
# import acme
# import dm_env
# import distrax
# import os
# import jax
# import optax
# import rlax


class TrainingState(NamedTuple):
    """Holds the agent's training state."""

    critic_params: hk.Params
    actor_params: hk.Params
    opt_state: Any
    step: int


# TODO: Complete PPO implementation
class PPO(hk.Module):
    """A simple PPO agent using JAX"""

    # TODO: complete init function
    def __init__(
        self,
        obs_spec,
        action_spec,
        critic_network: Callable[[jnp.ndarray], jnp.ndarray],
        actor_network: Callable[[jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        rng: hk.PRNGSequence,
        player_id: int,
        args: dict,
    ):
        # Transform the (impure) network into a pure function.
        critic_network = hk.without_apply_rng(hk.transform(critic_network))
        actor_network = hk.without_apply_rng(hk.transform(actor_network))

        # Initialize the networks and optimizer.
        # TODO: this will change when using a DM_spec. Hard coded to 4 for now
        # dummy_observation = jnp.zeros((1, obs_spec.num_values), jnp.float32)
        dummy_observation = jnp.zeros((1, 4), jnp.float32)
        initial_critic_params = critic_network.init(
            next(rng), dummy_observation
        )
        initial_actor_params = actor_network.init(next(rng), dummy_observation)
        # TODO: Determine is this will require two separate optimizers
        # I think they do because critic_params and actor_params have a diff number of params
        initial_opt_state = optimizer.init(initial_critic_params)

        self._state = TrainingState(
            critic_params=initial_critic_params,
            actor_params=initial_actor_params,
            opt_state=initial_opt_state,
            step=0,
        )

        # TODO: complete loss function
        def loss():
            # Start here, then get replay buffer to do what you want
            pass

        # TODO: complete sgd step
        # @jax.jit
        def sgd_step():
            pass

    # TODO: complete select_action()
    def select_action(self):
        # then get select action to work
        pass

    # TODO: complete update()
    def update(self):
        pass


def make_agent(args, obs_spec, action_spec, seed: int, player_id: int):

    return PPO(
        obs_spec=obs_spec,
        action_spec=action_spec,
        critic_network=critic_network,
        actor_network=actor_network,
        optimizer=optax.adam(learning_rate=args.learning_rate, eps=args.eps),
        rng=hk.PRNGSequence(seed),
        player_id=player_id,
        args=args,
    )


if __name__ == "__main__":
    pass
