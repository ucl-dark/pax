import os
from typing import Optional, Tuple

import chex
import jax
import jax.debug
import jax.numpy as jnp
import yaml
from gymnax.environments import environment, spaces
from jax import Array

from pax.envs.rice.rice import EnvState, EnvParams, get_consumption, get_armington_agg, get_utility, get_social_welfare, \
    get_global_temperature, load_rice_params, get_exogenous_emissions, get_mitigation_cost, get_carbon_intensity, \
    get_abatement_cost, get_damages, get_production, get_gross_output, get_investment, zero_diag, \
    get_max_potential_exports, get_land_emissions, get_aux_m, get_global_carbon_mass, get_capital_depreciation, \
    get_capital, get_labor, get_production_factor, get_carbon_price
from pax.utils import float_precision

eps = 1e-5

"""
Based off the MARL environment from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4189735
which in turn is an adaptation of the RICE IAM.
"""


class Rice(environment.Environment):
    env_id: str = "Rice-v1"

    def __init__(self, num_inner_steps: int, config_folder: str, has_mediator=False):
        super().__init__()

        # TODO refactor all the constants to use env_params
        # 1. Load env params in the experiment.py#env_setup
        # 2. type env params as a chex dataclass
        # 3. change the references in the code to env params
        params, num_regions = load_rice_params(config_folder)
        self.has_mediator = has_mediator
        self.num_players = num_regions
        self.num_actors = self.num_players + 1 if self.has_mediator else self.num_players
        self.rice_constant = params["_RICE_GLOBAL_CONSTANT"]
        self.dice_constant = params["_DICE_CONSTANT"]
        self.region_constants = params["_REGIONS"]
        self.region_params = params["_REGION_PARAMS"]

        self.savings_action_n = 1
        self.mitigation_rate_action_n = 1
        # Each region sets max allowed export from own region
        self.export_action_n = 1
        # Each region sets import bids (max desired imports from other countries)
        # TODO Find an "automatic" model for trade imports
        # Reason: without protocols such as clubs there is no incentive for countries to impose a tariff
        self.import_actions_n = self.num_players
        # Each region sets import tariffs imposed on other countries
        self.tariff_actions_n = self.num_players

        self.actions_n = (
                self.savings_action_n
                + self.mitigation_rate_action_n
                + self.export_action_n
                + self.import_actions_n
                + self.tariff_actions_n
        )

        # Determine the index of each action to slice them in the step function
        self.savings_action_index = 0
        self.mitigation_rate_action_index = self.savings_action_index + self.savings_action_n
        self.export_action_index = self.mitigation_rate_action_index + self.mitigation_rate_action_n
        self.tariffs_action_index = self.export_action_index + self.export_action_n
        self.desired_imports_action_index = self.tariffs_action_index + self.tariff_actions_n
        self.join_club_action_index = self.desired_imports_action_index + self.import_actions_n

        # Parameters for armington aggregation utility
        self.sub_rate = jnp.asarray(0.5, dtype=float_precision)
        self.dom_pref = jnp.asarray(0.5, dtype=float_precision)
        self.for_pref = jnp.asarray([0.5 / (self.num_players - 1)] * self.num_players, dtype=float_precision)

        def _step(
                key: chex.PRNGKey,
                state: EnvState,
                actions: Tuple[float, ...],
                params: EnvParams,
        ):
            t = state.inner_t + 1  # Rice equations expect to start at t=1
            key, _ = jax.random.split(key, 2)
            done = t >= num_inner_steps

            t_at = state.global_temperature[0]
            global_exogenous_emissions = get_exogenous_emissions(
                self.dice_constant["xf_0"], self.dice_constant["xf_1"], self.dice_constant["xt_f"], t
            )

            actions = jnp.asarray(actions).astype(float_precision).squeeze()
            actions = jnp.clip(actions, a_min=0, a_max=1)

            if self.has_mediator:
                region_actions = actions[1:]
            else:
                region_actions = actions

            club_membership_all = region_actions[:, self.join_club_action_index]
            mitigation_cost_all = get_mitigation_cost(
                self.rice_constant["xp_b"],
                self.rice_constant["xtheta_2"],
                self.rice_constant["xdelta_pb"],
                state.intensity_all,
                t
            )
            intensity_all = get_carbon_intensity(
                state.intensity_all,
                self.rice_constant["xg_sigma"],
                self.rice_constant["xdelta_sigma"],
                self.dice_constant["xDelta"],
                t
            )
            # Get the maximum carbon price of non-members from the last timestep
            club_price = jnp.max(state.carbon_price_all * (1 - state.club_membership_all))
            club_mitigation_rates = get_club_mitigation_rates(
                club_price,
                intensity_all,
                self.rice_constant["xtheta_2"],
                mitigation_cost_all,
                state.damages_all
            )
            mitigation_rate_all = jnp.where(
                club_membership_all == 1,
                club_mitigation_rates,
                region_actions[:, self.mitigation_rate_action_index]
            )

            abatement_cost_all = get_abatement_cost(
                mitigation_rate_all, mitigation_cost_all,
                self.rice_constant["xtheta_2"]
            )
            damages_all = get_damages(t_at, self.region_params["xa_1"], self.region_params["xa_2"],
                                      self.region_params["xa_3"])
            production_all = get_production(
                state.production_factor_all,
                state.capital_all,
                state.labor_all,
                self.region_params["xgamma"],
            )
            gross_output_all = get_gross_output(damages_all, abatement_cost_all, production_all)
            balance_all = state.balance_all * (1 + self.rice_constant["r"])
            investment_all = get_investment(region_actions[:, self.savings_action_index], gross_output_all)

            # Trade
            desired_imports = region_actions[:,
                              self.desired_imports_action_index: self.desired_imports_action_index + self.import_actions_n]
            # Countries cannot import from themselves
            desired_imports = zero_diag(desired_imports)
            total_desired_imports = desired_imports.sum(axis=1)
            clipped_desired_imports = jnp.clip(total_desired_imports, 0, gross_output_all)
            desired_imports = desired_imports * (
                                                    # Transpose to apply the scaling row and not column-wise
                                                        clipped_desired_imports / (total_desired_imports + eps))[:,
                                                jnp.newaxis]
            init_capital_multiplier = 10.0
            debt_ratio = init_capital_multiplier * balance_all / self.region_params["xK_0"]
            debt_ratio = jnp.clip(debt_ratio, -1.0, 0.0)
            scaled_imports = desired_imports * (1 + debt_ratio)

            max_potential_exports = get_max_potential_exports(
                region_actions[:, self.export_action_index], gross_output_all, investment_all)
            # Summing along columns yields the total number of exports requested by other regions
            total_desired_exports = jnp.sum(scaled_imports, axis=0)
            clipped_desired_exports = jnp.clip(total_desired_exports, 0, max_potential_exports)
            scaled_imports = scaled_imports * clipped_desired_exports / (total_desired_exports + eps)

            prev_tariffs = state.future_tariff
            tariffed_imports = scaled_imports * (1 - prev_tariffs)
            # calculate tariffed imports, tariff revenue and budget balance
            # In the paper this goes to a "special reserve fund", i.e. it's not used
            tariff_revenue_all = jnp.sum(scaled_imports * prev_tariffs, axis=0)

            total_exports = scaled_imports.sum(axis=0)
            balance_all = balance_all + self.dice_constant["xDelta"] * (
                    total_exports - scaled_imports.sum(axis=1))

            c_dom = get_consumption(gross_output_all, investment_all, total_exports)
            consumption_all = get_armington_agg(c_dom, tariffed_imports, self.sub_rate, self.dom_pref, self.for_pref)
            utility_all = get_utility(state.labor_all, consumption_all, self.rice_constant["xalpha"])
            social_welfare_all = get_social_welfare(utility_all, self.rice_constant["xrho"],
                                                    self.dice_constant["xDelta"], t)

            # Update ecology
            m_at = state.global_carbon_mass[0]
            global_temperature = get_global_temperature(
                self.dice_constant["xPhi_T"],
                state.global_temperature,
                self.dice_constant["xB_T"],
                self.dice_constant["xF_2x"],
                m_at,
                self.dice_constant["xM_AT_1750"],
                global_exogenous_emissions,
            )

            global_land_emissions = get_land_emissions(
                self.dice_constant["xE_L0"], self.dice_constant["xdelta_EL"], t, self.num_players
            )

            aux_m_all = get_aux_m(
                state.intensity_all,
                mitigation_rate_all,
                production_all,
                global_land_emissions
            )

            global_carbon_mass = get_global_carbon_mass(
                self.dice_constant["xPhi_M"],
                state.global_carbon_mass,
                self.dice_constant["xB_M"],
                jnp.sum(aux_m_all),
            )

            capital_depreciation = get_capital_depreciation(
                self.rice_constant["xdelta_K"], self.dice_constant["xDelta"]
            )
            capital_all = get_capital(
                capital_depreciation, state.capital_all,
                self.dice_constant["xDelta"],
                investment_all
            )
            labor_all = get_labor(state.labor_all, self.region_params["xL_a"], self.region_params["xl_g"])
            production_factor_all = get_production_factor(
                state.production_factor_all,
                self.region_params["xg_A"],
                self.region_params["xdelta_A"],
                self.dice_constant["xDelta"],
                t,
            )
            carbon_price_all = get_carbon_price(mitigation_cost_all, intensity_all,
                                                mitigation_rate_all,
                                                self.rice_constant["xtheta_2"],
                                                damages_all)

            next_state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t,
                global_temperature=global_temperature,
                global_carbon_mass=global_carbon_mass,
                global_exogenous_emissions=global_exogenous_emissions,
                global_land_emissions=global_land_emissions,

                labor_all=labor_all,
                capital_all=capital_all,
                production_factor_all=production_factor_all,
                intensity_all=intensity_all,
                balance_all=balance_all,

                future_tariff=region_actions[:,
                              self.tariffs_action_index: self.tariffs_action_index + self.num_players],

                gross_output_all=gross_output_all,
                investment_all=investment_all,
                production_all=production_all,
                utility_all=utility_all,
                social_welfare_all=social_welfare_all,
                capital_depreciation_all=jnp.asarray([capital_depreciation]),
                mitigation_cost_all=mitigation_cost_all,
                consumption_all=consumption_all,
                damages_all=damages_all,
                abatement_cost_all=abatement_cost_all,

                tariff_revenue_all=tariff_revenue_all,
                carbon_price_all=carbon_price_all,
                club_membership_all=jnp.zeros(self.num_players, dtype=jnp.int8),
            )

            reset_obs, reset_state = _reset(key, params)
            reset_state = reset_state.replace(outer_t=state.outer_t + 1)

            obs = []
            if self.has_mediator:
                obs.append(self._generate_mediator_observation(actions, next_state))

            for i in range(self.num_players):
                obs.append(self._generate_observation(i, actions, next_state))

            obs = jax.tree_map(lambda x, y: jnp.where(done, x, y), reset_obs, tuple(obs))

            state = jax.tree_map(
                lambda x, y: jnp.where(done, x, y),
                reset_state,
                next_state,
            )

            if self.has_mediator:
                rewards = jnp.insert(state.utility_all, 0, state.utility_all.sum())
            else:
                rewards = state.utility_all

            return (
                tuple(obs),
                state,
                tuple(rewards),
                done,
                {},
            )

        def _reset(
                key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = self._get_initial_state()
            actions = jnp.asarray([jax.random.uniform(key, (self.num_actions,)) for _ in range(self.num_actors)])
            obs = []
            if self.has_mediator:
                obs.append(self._generate_mediator_observation(actions, state))
            for i in range(self.num_players):
                obs.append(self._generate_observation(i, actions, state))
            return tuple(obs), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    def _get_initial_state(self) -> EnvState:
        return EnvState(
            inner_t=jnp.zeros((), dtype=jnp.int16),
            outer_t=jnp.zeros((), dtype=jnp.int16),
            global_temperature=jnp.array([self.rice_constant["xT_AT_0"], self.rice_constant["xT_LO_0"]]),
            global_carbon_mass=jnp.array(
                [self.dice_constant["xM_AT_0"], self.dice_constant["xM_UP_0"], self.dice_constant["xM_LO_0"]]),
            global_exogenous_emissions=jnp.zeros((), dtype=float_precision),
            global_land_emissions=jnp.zeros((), dtype=float_precision),
            labor_all=self.region_params["xL_0"],
            capital_all=self.region_params["xK_0"],
            production_factor_all=self.region_params["xA_0"],
            intensity_all=self.region_params["xsigma_0"],

            balance_all=jnp.zeros(self.num_players, dtype=float_precision),
            future_tariff=jnp.zeros((self.num_players, self.num_players), dtype=float_precision),

            gross_output_all=jnp.zeros(self.num_players, dtype=float_precision),
            investment_all=jnp.zeros(self.num_players, dtype=float_precision),
            production_all=jnp.zeros(self.num_players, dtype=float_precision),
            utility_all=jnp.zeros(self.num_players, dtype=float_precision),
            social_welfare_all=jnp.zeros(self.num_players, dtype=float_precision),
            capital_depreciation_all=jnp.zeros(0, dtype=float_precision),
            mitigation_cost_all=jnp.zeros(self.num_players, dtype=float_precision),
            consumption_all=jnp.zeros(self.num_players, dtype=float_precision),
            damages_all=jnp.zeros(self.num_players, dtype=float_precision),
            abatement_cost_all=jnp.zeros(self.num_players, dtype=float_precision),

            tariff_revenue_all=jnp.zeros(self.num_players, dtype=float_precision),
            carbon_price_all=jnp.zeros(self.num_players, dtype=float_precision),
            club_membership_all=jnp.zeros(self.num_players, dtype=jnp.int8),
        )

    def _generate_observation(self, index: int, actions: chex.ArrayDevice, state: EnvState) -> Array:
        return jnp.concatenate([
            # Public features
            jnp.asarray([index]),
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
            state.tariff_revenue_all,
            state.carbon_price_all,
            state.club_membership_all,
            # Private features
            jnp.asarray([state.damages_all[index]]),
            jnp.asarray([state.abatement_cost_all[index]]),
            jnp.asarray([state.production_all[index]]),
            # All agent actions
            actions.ravel()
        ], dtype=float_precision)

    def _generate_mediator_observation(self, actions: chex.ArrayDevice, state: EnvState) -> Array:
        return jnp.concatenate([
            jnp.zeros(1),
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
            state.tariff_revenue_all,
            state.carbon_price_all,
            state.club_membership_all,
            # Maintain same dimensionality as for other players
            jnp.zeros(3),
            # All agent actions
            actions.ravel()
        ], dtype=float_precision)

    @property
    def name(self) -> str:
        return self.env_id

    @property
    def num_actions(self) -> int:
        return self.actions_n

    def action_space(
            self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(self.actions_n,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        init_state = self._get_initial_state()
        obs = self._generate_observation(0, jnp.zeros(self.num_actions * self.num_actors), init_state)
        return spaces.Box(low=0, high=float('inf'), shape=obs.shape, dtype=float_precision)


def get_club_mitigation_rates(coalition_price, intensity, theta_2, mitigation_cost, damages):
    return pow(coalition_price * intensity / (theta_2 * mitigation_cost * damages), 1 / (theta_2 - 1))
