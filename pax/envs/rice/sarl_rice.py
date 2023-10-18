from typing import Optional, Tuple

import chex
import jax
import jax.debug
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from pax.envs.rice.rice import Rice, EnvState, EnvParams

"""
Wrapper to turn Rice-N into a single-agent environment.
"""


class SarlRice(environment.Environment):
    env_id: str = "SarlRice-N"

    def __init__(
        self,
        config_folder: str,
        fixed_mitigation_rate: int = None,
        episode_length: int = 20,
    ):
        super().__init__()
        self.rice = Rice(config_folder, episode_length=episode_length)
        self.fixed_mitigation_rate = fixed_mitigation_rate

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            action: chex.Array,
            params: EnvParams,
        ):
            actions = jnp.asarray(jnp.split(action, self.rice.num_players))
            # To facilitate learning and since we want to optimize a social planner
            # we disable defection actions
            actions = actions.at[:, self.rice.export_action_index].set(1.0)
            actions = actions.at[
                :,
                self.rice.tariffs_action_index : self.rice.tariffs_action_index
                + self.rice.num_players,
            ].set(0.0)

            if self.fixed_mitigation_rate is not None:
                actions = actions.at[
                    :, self.rice.mitigation_rate_action_index
                ].set(self.fixed_mitigation_rate)

            # actions = actions.at[:, self.rice.desired_imports_action_index:
            #                         self.rice.desired_imports_action_index + self.rice.num_players].set(0.0)
            obs, state, rewards, done, info = self.rice.step(
                key, state, tuple(actions), params
            )

            return (
                self._generate_observation(state),
                state,
                jnp.asarray(rewards).sum(),
                done,
                info,
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
            _, state = self.rice.reset(key, params)
            return self._generate_observation(state), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    def _generate_observation(self, state: EnvState):
        return jnp.concatenate(
            [
                # Public features
                jnp.asarray([state.inner_t]),
                state.global_temperature,
                state.global_carbon_mass,
                jnp.asarray([state.global_exogenous_emissions]),
                jnp.asarray([state.global_land_emissions]),
                state.labor_all,
                state.capital_all,
                state.gross_output_all,
                state.consumption_all,
                state.investment_all,
                state.balance_all,
                state.production_factor_all,
                state.intensity_all,
                state.mitigation_cost_all,
                state.damages_all,
                state.abatement_cost_all,
                state.production_all,
                state.utility_all,
                state.social_welfare_all,
            ]
        )

    @property
    def name(self) -> str:
        return self.env_id

    @property
    def num_actions(self) -> int:
        return self.rice.num_actions * self.rice.num_players

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(self.num_actions,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        _, state = self.reset(jax.random.PRNGKey(0), params)
        obs = self._generate_observation(state)
        return spaces.Box(
            low=0, high=float("inf"), shape=obs.shape, dtype=jnp.float32
        )
