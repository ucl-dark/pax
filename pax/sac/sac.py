# copied from https://github.com/henry-prior/jax-rl/blob/master/jax_rl/SAC.py

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import optim
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence
import dm_env
import wandb
from pax.sac.buffers import ReplayBuffer
from pax.sac.models import (
    apply_categorical_policy_model,
    apply_constant_model,
    apply_discrete_double_critic_model,
    build_categorical_policy_model,
    build_constant_model,
    build_discrete_double_critic_model,
)
from pax.sac.saving import load_model, save_model
from pax.sac.utils import copy_params, double_mse


def cross_entropy_loss_fn(log_p: jnp.ndarray, target: jnp.ndarray):
    return -1 * (log_p * target).sum(-1)


def actor_loss_fn(
    log_alpha: jnp.ndarray, log_p: jnp.ndarray, min_q: jnp.ndarray
):
    return (
        (jnp.exp(log_p) * (jnp.exp(log_alpha) * log_p - min_q)).sum(-1).mean()
    )


def alpha_loss_fn(
    log_alpha: jnp.ndarray, target_entropy: float, log_p: jnp.ndarray
):
    return (
        -1
        * (
            log_alpha * (target_entropy + (log_p * jnp.exp(log_p)).sum(-1))
        ).mean()
    )


@partial(jax.jit, static_argnums=(6, 7))
def get_td_target(
    rng: PRNGSequence,
    state: jnp.ndarray,
    action: jnp.ndarray,
    next_state: jnp.ndarray,
    reward: jnp.ndarray,
    not_done: jnp.ndarray,
    discount: float,
    action_dim: int,
    actor_params: FrozenDict,
    critic_target_params: FrozenDict,
    log_alpha_params: FrozenDict,
) -> jnp.ndarray:
    _, next_log_p = apply_categorical_policy_model(
        actor_params, action_dim, next_state, rng, False
    )

    t_q1, t_q2 = apply_discrete_double_critic_model(
        critic_target_params, action_dim, next_state, False
    )
    next_q = jnp.minimum(t_q1, t_q2)
    target_Q = (
        next_q
        - jnp.exp(apply_constant_model(log_alpha_params, 0.0, False))
        * next_log_p
    )
    target_Q = (jnp.exp(next_log_p) * target_Q).sum(axis=-1, keepdims=True)
    target_Q = reward + not_done * discount * target_Q
    return target_Q


@partial(jax.jit, static_argnums=4)
def critic_step(
    optimizer: optim.Optimizer,
    state: jnp.ndarray,
    action: jnp.ndarray,
    target_Q: jnp.ndarray,
    action_dim: int,
) -> optim.Optimizer:
    action = jax.lax.stop_gradient(action)

    def loss_fn(critic_params):
        current_Q1, current_Q2 = apply_discrete_double_critic_model(
            critic_params, action_dim, state, False
        )
        # collect action taken
        q1 = (current_Q1 * action).sum(axis=-1)
        q2 = (current_Q2 * action).sum(axis=-1)
        critic_loss = double_mse(q1, q2, target_Q)
        return jnp.mean(critic_loss)

    critic_loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    return critic_loss, optimizer.apply_gradient(grad)


@partial(jax.jit, static_argnums=(5, 6))
def actor_step(
    rng: PRNGSequence,
    optimizer: optim.Optimizer,
    critic_params: FrozenDict,
    state: jnp.ndarray,
    log_alpha_params: FrozenDict,
    action_dim: int,
    evaluation: bool,
) -> Tuple[jnp.ndarray, optim.Optimizer, jnp.ndarray]:
    def loss_fn(actor_params):
        _, log_p = apply_categorical_policy_model(
            actor_params,
            action_dim,
            state,
            rng,
            evaluation,
        )
        q1, q2 = jax.lax.stop_gradient(
            apply_discrete_double_critic_model(
                critic_params, action_dim, state, False
            )
        )
        min_q = jnp.minimum(q1, q2)
        partial_loss_fn = jax.vmap(
            partial(
                actor_loss_fn,
                jax.lax.stop_gradient(
                    apply_constant_model(log_alpha_params, 0.0, False)
                ),
            ),
        )
        actor_loss = partial_loss_fn(log_p, min_q)
        return jnp.mean(actor_loss), log_p

    (actor_loss, log_p), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        optimizer.target
    )
    return actor_loss, optimizer.apply_gradient(grad), log_p


@partial(jax.jit, static_argnums=2)
def alpha_step(
    optimizer: optim.Optimizer, log_p: jnp.ndarray, target_entropy: float
) -> Tuple[jnp.ndarray, optim.Optimizer]:
    def loss_fn(log_alpha_params):
        partial_loss_fn = jax.vmap(
            partial(
                alpha_loss_fn,
                apply_constant_model(log_alpha_params, 0.0, False),
                target_entropy,
            )
        )
        return jnp.mean(partial_loss_fn(jax.lax.stop_gradient(log_p)))

    value, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    return value, optimizer.apply_gradient(grad)


class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_freq: int = 2,
        lr: float = 3e-4,
        entropy_tune: bool = True,
        seed: int = 0,
    ):
        self.rng = PRNGSequence(seed)

        input_dim = (1, state_dim)
        actor_params = build_categorical_policy_model(
            input_dim, action_dim, next(self.rng)
        )
        actor_optimizer = optim.Adam(learning_rate=lr).create(actor_params)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_params = build_discrete_double_critic_model(
            input_dim, action_dim, init_rng
        )
        self.critic_target_params = build_discrete_double_critic_model(
            input_dim, action_dim, init_rng
        )
        critic_optimizer = optim.Adam(learning_rate=lr).create(critic_params)
        self.critic_optimizer = jax.device_put(critic_optimizer)

        self.entropy_tune = entropy_tune
        log_alpha_params = build_constant_model(0.0, next(self.rng))

        log_alpha_optimizer = optim.Adam(learning_rate=lr).create(
            log_alpha_params
        )
        self.log_alpha_optimizer = jax.device_put(log_alpha_optimizer)
        self.target_entropy = 0.8 * (-np.log(1 / action_dim))

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.total_it = 0

        self._replay = ReplayBuffer(state_dim, action_dim)
        self.eval = False
        self._total_steps = 0
        self._sgd_period = 50

    @property
    def target_params(self):
        return (
            self.discount,
            self.action_dim,
            self.actor_optimizer.target,
            self.critic_target_params,
            self.log_alpha_optimizer.target,
        )

    def actor_step(
        self, rng: PRNGSequence, state: jnp.ndarray, evaluation: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action, log_pi = apply_categorical_policy_model(
            self.actor_optimizer.target,
            self.action_dim,
            state,
            rng,
            evaluation,
        )

        return jnp.expand_dims(jnp.argmax(action, axis=-1), axis=-1), log_pi

    def select_action(self, timestep: dm_env.TimeStep) -> jnp.ndarray:
        return self.actor_step(
            next(self.rng), timestep.observation, self.eval
        )[0]

    def update(self, old_timestep, action, new_timestep, *args) -> None:
        batch_size = 32
        batched_done = jnp.ones_like(action) * new_timestep.last
        self._replay.add_batch(
            state=old_timestep.observation,
            action=action,
            next_state=new_timestep.observation,
            reward=new_timestep.reward,
            done=batched_done,  # this is not batched currently
        )
        self._total_steps += 1
        if self._total_steps % self._sgd_period != 0:
            return

        if self._replay.size < batch_size:
            return

        self.train(self._replay, batch_size, True)

    def train(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 100,
        logging: bool = True,
    ):
        self.total_it += 1

        buffer_out = replay_buffer.sample(next(self.rng), batch_size)
        target_Q = jax.lax.stop_gradient(
            get_td_target(next(self.rng), *buffer_out, *self.target_params)
        )

        state, action, *_ = buffer_out
        critic_loss, self.critic_optimizer = critic_step(
            self.critic_optimizer, state, action, target_Q, self.action_dim
        )

        if logging:
            wandb.log({"Critic Loss": float(critic_loss)})

        if self.total_it % self.policy_freq == 0:

            actor_loss, self.actor_optimizer, log_p = actor_step(
                next(self.rng),
                self.actor_optimizer,
                self.critic_optimizer.target,
                state,
                self.log_alpha_optimizer.target,
                self.action_dim,
                False,
            )
            if logging:
                wandb.log({"Actor Loss": float(actor_loss)})

            if self.entropy_tune:
                log_alpha_loss, self.log_alpha_optimizer = alpha_step(
                    self.log_alpha_optimizer, log_p, self.target_entropy
                )

                if logging:
                    wandb.log(
                        {
                            "Alpha": float(
                                jnp.exp(
                                    self.log_alpha_optimizer.target["value"]
                                )
                            )
                        }
                    )
                    wandb.log({"Alpha Loss": float(log_alpha_loss)})

            self.critic_target_params = copy_params(
                self.critic_target_params,
                self.critic_optimizer.target,
                self.tau,
            )

    def save(self, filename):
        save_model(filename + "_critic", self.critic_optimizer)
        save_model(filename + "_actor", self.actor_optimizer)

    def load(self, filename):
        self.critic_optimizer = load_model(
            filename + "_critic", self.critic_optimizer
        )
        self.critic_optimizer = jax.device_put(self.critic_optimizer)
        self.critic_target_params = self.critic_optimizer.target.copy({})

        self.actor_optimizer = load_model(
            filename + "_actor", self.actor_optimizer
        )
        self.actor_optimizer = jax.device_put(self.actor_optimizer)

    def pre_train(
        self,
        dataset: ReplayBuffer,
        num_iterations: int,
        lr: float,
        batch_size: int,
        rng: jax.random.PRNGKey,
    ):
        _, action_dim, actor_params, _, _ = self.target_params
        bc_optimizer = jax.device_put(
            optim.Adam(learning_rate=lr).create(actor_params)
        )

        print("Starting Pre-Training")
        for _ in range(num_iterations):
            _, rng = jax.random.split(rng)
            state, action, _, _, _ = dataset.sample(rng, batch_size)

            def loss_fn(params, rng, state, target):
                _, log_p = apply_categorical_policy_model(
                    params, action_dim, state, rng, False
                )
                partial_loss_fn = jax.vmap(cross_entropy_loss_fn)
                return jnp.mean(partial_loss_fn(log_p, target))

            loss, grad = jax.value_and_grad(loss_fn)(
                bc_optimizer.target, rng, state, action
            )
            bc_optimizer = bc_optimizer.apply_gradient(grad)

        # this is the most sketch part
        actor_optimizer = optim.Adam(learning_rate=lr).create(
            bc_optimizer.target
        )
        self.actor_optimizer = jax.device_put(actor_optimizer)
