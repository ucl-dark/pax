import os
from typing import Optional, Tuple

import chex
import jax
import jax.debug
import jax.numpy as jnp
import yaml
from gymnax.environments import environment, spaces
from jax import Array


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int

    # Ecological
    global_temperature: chex.ArrayDevice
    global_carbon_mass: chex.ArrayDevice
    global_exogenous_emissions: float
    global_land_emissions: float

    # Economic
    labor_all: chex.ArrayDevice
    capital_all: chex.ArrayDevice
    production_factor_all: chex.ArrayDevice
    intensity_all: chex.ArrayDevice
    balance_all: chex.ArrayDevice

    # Tariffs are applied to the next time step
    future_tariff: chex.ArrayDevice

    # The following values are intermediary values
    # that we only track in the state for easier evaluation and logging
    gross_output_all: chex.ArrayDevice
    investment_all: chex.ArrayDevice
    production_all: chex.ArrayDevice
    utility_all: chex.ArrayDevice
    social_welfare_all: chex.ArrayDevice
    capital_depreciation_all: chex.ArrayDevice
    mitigation_cost_all: chex.ArrayDevice
    consumption_all: chex.ArrayDevice
    damages_all: chex.ArrayDevice
    abatement_cost_all: chex.ArrayDevice


@chex.dataclass
class EnvParams:
    pass


"""
Based off the MARL environment from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4189735
which in turn is an adaptation of the RICE IAM
"""


class Rice(environment.Environment):
    env_id: str = "Rice-v1"

    def __init__(self, num_inner_steps: int, config_folder: str, mediator=False):
        super().__init__()

        # TODO refactor all the constants to use env_params
        # 1. Load env params in the experiment.py#env_setup
        # 2. type env params as a chex dataclass
        # 3. change the references in the code to env params
        params, num_regions = load_rice_params(config_folder)
        self.mediator = mediator
        self.num_players = num_regions
        self.num_actors = self.num_players + 1 if self.mediator else self.num_players
        self.rice_constant = params["_RICE_GLOBAL_CONSTANT"]
        self.dice_constant = params["_DICE_CONSTANT"]
        self.region_constants = params["_REGIONS"]

        self.savings_action_n = 1
        self.mitigation_rate_action_n = 1
        # Each region sets max allowed export from own region
        self.export_action_n = 1
        # Each region sets import bids (max desired imports from other countries)
        # TODO Find an "automatic" model for trade imports
        # Reason: it's counter-intuitive that regions would both set import tariffs and imports
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

        # Parameters for armington aggregation utility
        self.sub_rate = 0.5
        self.dom_pref = 0.5
        self.for_pref = jnp.asarray([0.5 / (self.num_players - 1)] * self.num_players)

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

            actions = jnp.asarray(actions).squeeze()
            actions = jnp.clip(actions, a_min=0, a_max=1)

            # Intermediary variables
            scaled_imports = jnp.zeros((self.num_players, self.num_players))
            gross_output_all = []
            investment_all = []
            production_all = []
            utility_all = []
            social_welfare_all = []
            capital_depreciation_all = []
            mitigation_cost_all = []
            consumption_all = []
            damages_all = []
            abatement_cost_all = []

            # Next state variables
            next_labor_all = []
            next_production_factor_all = []
            next_balance_all = []

            if self.mediator:
                region_actions = actions[1:]
            else:
                region_actions = actions

            for i in range(self.num_players):
                savings = region_actions[i, self.savings_action_index]
                mitigation_rate = region_actions[i, self.mitigation_rate_action_index]

                intensity = state.intensity_all[i]
                production_factor = state.production_factor_all[i]
                capital = state.capital_all[i]
                labor = state.labor_all[i]
                gov_balance = state.balance_all[i]

                region_const = self.region_constants[i]

                mitigation_cost = get_mitigation_cost(
                    self.rice_constant["xp_b"],
                    self.rice_constant["xtheta_2"],
                    self.rice_constant["xdelta_pb"],
                    intensity,
                    t
                )
                mitigation_cost_all.append(mitigation_cost)
                damages = get_damages(t_at, region_const["xa_1"], region_const["xa_2"], region_const["xa_3"])
                damages_all.append(damages)
                abatement_cost = get_abatement_cost(
                    mitigation_rate, mitigation_cost, self.rice_constant["xtheta_2"]
                )
                abatement_cost_all.append(abatement_cost)
                production = get_production(
                    production_factor,
                    capital,
                    labor,
                    region_const["xgamma"],
                )
                production_all.append(production)

                gross_output = get_gross_output(damages, abatement_cost, production)
                gross_output_all.append(gross_output)
                gov_balance = gov_balance * (1 + self.rice_constant["r"])
                next_balance_all.append(gov_balance)
                investment = get_investment(savings, gross_output)
                investment_all.append(investment)

                # Trade
                desired_imports = region_actions[i,
                                  self.desired_imports_action_index: self.desired_imports_action_index + self.import_actions_n]
                # Countries cannot import from themselves
                desired_imports = desired_imports.at[i].set(0)
                total_desired_imports = desired_imports.sum()
                clipped_desired_imports = jnp.clip(total_desired_imports, 0, gross_output)
                desired_imports = jnp.where(total_desired_imports != 0,
                                            desired_imports * clipped_desired_imports / total_desired_imports,
                                            jnp.zeros_like(desired_imports))

                # Scale imports based on gov balance
                init_capital_multiplier = 10.0
                debt_ratio = init_capital_multiplier * gov_balance / region_const["xK_0"]
                debt_ratio = jnp.clip(debt_ratio, -1.0, 0.0)
                desired_imports *= 1 + debt_ratio
                scaled_imports = scaled_imports.at[i].set(desired_imports)

            gross_output_all = jnp.asarray(gross_output_all)
            investment_all = jnp.asarray(investment_all)
            max_potential_exports = get_max_potential_exports(
                region_actions[:, self.export_action_index], gross_output_all, investment_all)
            # Summing along columns yields the total number of exports requested by other regions
            total_desired_exports = jnp.sum(scaled_imports, axis=0)
            clipped_desired_exports = jnp.clip(total_desired_exports, 0, max_potential_exports)
            scaled_imports = jnp.where(total_desired_exports != 0, scaled_imports * clipped_desired_exports / total_desired_exports,
                                    jnp.zeros((self.num_players, self.num_players)))


            tariffed_imports = jnp.zeros((self.num_players, self.num_players))
            prev_tariffs = state.future_tariff
            # Third loop to calculate tariffs and welfare
            for i in range(self.num_players):
                gross_output = gross_output_all[i]
                investment = investment_all[i]
                labor = state.labor_all[i]

                # calculate tariffed imports, tariff revenue and budget balance
                tariffed_imports = tariffed_imports.at[i].set(scaled_imports[i] * (1 - state.future_tariff[i]))

                # In the paper this goes to a "special reserve fund", i.e. it's not used
                tariff_revenue = jnp.sum(
                    scaled_imports[i, :] * prev_tariffs[i, :]
                )
                # Aggregate consumption from domestic and foreign goods
                # domestic consumption
                c_dom = get_consumption(gross_output, investment, exports=scaled_imports[:, i])

                consumption = get_armington_agg(
                    c_dom=c_dom,
                    c_for=tariffed_imports[i, :],
                    sub_rate=self.sub_rate,
                    dom_pref=self.dom_pref,
                    for_pref=self.for_pref,
                )
                consumption_all.append(consumption)

                utility = get_utility(labor, consumption, self.rice_constant["xalpha"])
                utility_all.append(utility)
                social_welfare_all.append(get_social_welfare(
                    utility, self.rice_constant["xrho"], self.dice_constant["xDelta"], t
                ))

                # Update government balance
                next_balance_all[i] = (next_balance_all[i]
                                       + self.dice_constant["xDelta"] * (
                                               jnp.sum(scaled_imports[:, i])
                                               - jnp.sum(scaled_imports[i, :])
                                       ))

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
            production_all = jnp.asarray(production_all)
            aux_m_all = get_aux_m(
                state.intensity_all,
                region_actions[:, self.mitigation_rate_action_index],
                jnp.asarray(production_all),
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
            next_capital_all = get_capital(
                capital_depreciation, state.capital_all,
                self.dice_constant["xDelta"],
                investment_all
            )
            next_intensity_all = get_carbon_intensity(
                state.intensity_all,
                self.rice_constant["xg_sigma"],
                self.rice_constant["xdelta_sigma"],
                self.dice_constant["xDelta"],
                t
            )
            for i in range(self.num_players):
                region_const = self.region_constants[i]
                capital_depreciation_all.append(capital_depreciation)
                next_labor_all.append(
                    get_labor(state.labor_all[i], region_const["xL_a"], region_const["xl_g"]))
                next_production_factor_all.append(get_production_factor(
                    state.production_factor_all[i],
                    region_const["xg_A"],
                    region_const["xdelta_A"],
                    self.dice_constant["xDelta"],
                    t,
                ))

            next_state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t,
                global_temperature=global_temperature,
                global_carbon_mass=global_carbon_mass,
                global_exogenous_emissions=global_exogenous_emissions,
                global_land_emissions=global_land_emissions,

                labor_all=jnp.asarray(next_labor_all),
                capital_all=next_capital_all,
                production_factor_all=jnp.asarray(next_production_factor_all),
                intensity_all=next_intensity_all,
                balance_all=jnp.asarray(next_balance_all),

                future_tariff=region_actions[:, self.tariffs_action_index: self.tariffs_action_index + self.num_players],

                gross_output_all=gross_output_all,
                investment_all=investment_all,
                production_all=production_all,
                utility_all=jnp.asarray(utility_all),
                social_welfare_all=jnp.asarray(social_welfare_all),
                capital_depreciation_all=jnp.asarray([capital_depreciation]),
                mitigation_cost_all=jnp.asarray(mitigation_cost_all),
                consumption_all=jnp.asarray(consumption_all),
                damages_all=jnp.asarray(damages_all),
                abatement_cost_all=jnp.asarray(abatement_cost_all),
            )

            reset_obs, reset_state = _reset(key, params)
            reset_state = reset_state.replace(outer_t=state.outer_t + 1)

            obs = []
            if self.mediator:
                obs.append(self._generate_mediator_observation(actions, next_state))

            for i in range(self.num_players):
                obs.append(self._generate_observation(i, actions, next_state))


            obs = jax.tree_map(lambda x, y: jnp.where(done, x, y), reset_obs, tuple(obs))

            state = jax.tree_map(
                lambda x, y: jnp.where(done, x, y),
                reset_state,
                next_state,
            )

            if self.mediator:
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
            if self.mediator:
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
            global_exogenous_emissions=jnp.zeros((), dtype=jnp.float32),
            global_land_emissions=jnp.zeros((), dtype=jnp.float32),
            labor_all=jnp.array(
                [self.region_constants[region_id]["xL_0"] for region_id in range(self.num_players)]),
            capital_all=jnp.array(
                [self.region_constants[region_id]["xK_0"] for region_id in range(self.num_players)]),
            production_factor_all=jnp.array(
                [self.region_constants[region_id]["xA_0"] for region_id in range(self.num_players)]),
            intensity_all=jnp.array(
                [self.region_constants[region_id]["xsigma_0"] for region_id in range(self.num_players)]),

            balance_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            future_tariff=jnp.zeros((self.num_players, self.num_players), dtype=jnp.float32),

            gross_output_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            investment_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            production_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            utility_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            social_welfare_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            capital_depreciation_all=jnp.zeros(0, dtype=jnp.float32),
            mitigation_cost_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            consumption_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            damages_all=jnp.zeros(self.num_players, dtype=jnp.float32),
            abatement_cost_all=jnp.zeros(self.num_players, dtype=jnp.float32),
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
            # Private features
            jnp.asarray([state.production_factor_all[index]]),
            jnp.asarray([state.intensity_all[index]]),
            jnp.asarray([state.mitigation_cost_all[index]]),
            jnp.asarray([state.damages_all[index]]),
            jnp.asarray([state.abatement_cost_all[index]]),
            jnp.asarray([state.production_all[index]]),
            jnp.asarray([state.utility_all[index]]),
            jnp.asarray([state.social_welfare_all[index]]),
            # All agent actions
            actions.ravel()
        ])

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
            # Maintain same dimensionality as for other players
            jnp.zeros(8),
            # All agent actions
            actions.ravel()
        ])

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
        return spaces.Box(low=0, high=float('inf'), shape=obs.shape, dtype=jnp.float32)


def load_rice_params(config_dir=None):
    """Helper function to read yaml data and set environment configs."""
    assert config_dir is not None
    base_params = load_yaml_data(os.path.join(config_dir, "default.yml"))
    file_list = sorted(os.listdir(config_dir))  #
    yaml_files = []
    for file in file_list:
        if file[-4:] == ".yml" and file != "default.yml":
            yaml_files.append(file)

    region_params = []
    for file in yaml_files:
        region_params.append(load_yaml_data(os.path.join(config_dir, file)))

    # Overwrite rice params)
    base_params["_REGIONS"] = []
    for idx, param in enumerate(region_params):
        region_to_append = param["_RICE_CONSTANT"]
        for k in base_params["_RICE_CONSTANT_DEFAULT"].keys():
            if k not in region_to_append.keys():
                region_to_append[k] = base_params["_RICE_CONSTANT_DEFAULT"][k]
        base_params["_REGIONS"].append(region_to_append)

    return base_params, len(region_params)


def get_exogenous_emissions(f_0, f_1, t_f, timestep):
    return f_0 + jnp.min(jnp.array([f_1 - f_0, (f_1 - f_0) / t_f * (timestep - 1)]))


def get_land_emissions(e_l0, delta_el, timestep, num_regions):
    return e_l0 * pow(1 - delta_el, timestep - 1) / num_regions


def get_mitigation_cost(p_b, theta_2, delta_pb, intensity, timestep):
    return p_b / (1000 * theta_2) * pow(1 - delta_pb, timestep - 1) * intensity


def get_damages(t_at, a_1, a_2, a_3):
    return 1 / (1 + a_1 * t_at + a_2 * pow(t_at, a_3))


def get_abatement_cost(mitigation_rate, mitigation_cost, theta_2):
    return mitigation_cost * pow(mitigation_rate, theta_2)


def get_production(production_factor, capital, labor, gamma):
    """Obtain the amount of goods produced."""
    return production_factor * pow(capital, gamma) * pow(labor / 1000, 1 - gamma)


def get_gross_output(damages, abatement_cost, production):
    return damages * (1 - abatement_cost) * production


def get_investment(savings, gross_output):
    return savings * gross_output


def get_consumption(gross_output, investment, exports):
    total_exports = jnp.sum(exports)
    return jnp.max(jnp.asarray([0.0, gross_output - investment - total_exports]))


def get_max_potential_exports(x_max, gross_output, investment):
    return jnp.min(jnp.array([x_max * gross_output, gross_output - investment]), axis=0)


def get_capital_depreciation(x_delta_k, x_delta):
    return pow(1 - x_delta_k, x_delta)


# Returns shape 2
def get_global_temperature(
        phi_t, temperature, b_t, f_2x, m_at, m_at_1750, exogenous_emissions
):
    return jnp.dot(phi_t, temperature) + jnp.dot(
        b_t, f_2x * jnp.log(m_at / m_at_1750) / jnp.log(2) + exogenous_emissions
    )


def get_aux_m(intensity, mitigation_rate, production, land_emissions):
    """Auxiliary variable to denote carbon mass levels."""
    return intensity * (1 - mitigation_rate) * production + land_emissions


def get_global_carbon_mass(phi_m, carbon_mass, b_m, aux_m):
    return jnp.dot(phi_m, carbon_mass) + jnp.dot(b_m, aux_m)


def get_capital(capital_depreciation, capital, delta, investment):
    return capital_depreciation * capital + delta * investment


def get_labor(labor, l_a, l_g):
    return labor * pow((1 + l_a) / (1 + labor), l_g)


def get_production_factor(production_factor, g_a, delta_a, delta, timestep):
    return production_factor * (
            jnp.exp(0.0033) + g_a * jnp.exp(-delta_a * delta * (timestep - 1))
    )


def get_carbon_intensity(intensity, g_sigma, delta_sigma, delta, timestep):
    return intensity * jnp.exp(
        -g_sigma * pow(1 - delta_sigma, delta * (timestep - 1)) * delta
    )


_SMALL_NUM = 1e-0


def get_utility(labor, consumption, alpha):
    return (
            (labor / 1000.0)
            * (pow(consumption / (labor / 1000.0) + _SMALL_NUM, 1 - alpha) - 1)
            / (1 - alpha)
    )


def get_social_welfare(utility, rho, delta, timestep):
    return utility / pow(1 + rho, delta * timestep)


def get_armington_agg(
        c_dom,
        c_for,  # np.array
        sub_rate=0.5,  # in (0,1)
        dom_pref=0.5,  # in [0,1]
        for_pref=None,  # np.array
):
    """
    Armington aggregate from Lessmann, 2009.
    Consumption goods from different regions act as imperfect substitutes.
    As such, consumption of domestic and foreign goods are scaled according to
    relative preferences, as well as a substitution rate, which are modeled
    by a CES functional form.
    Inputs :
        `C_dom`     : A scalar representing domestic consumption. The value of
                    C_dom is what is left over from initial production after
                    investment and exports are deducted.
        `C_for`     : An array reprensenting foreign consumption. Each element
                    is the consumption imported from a given country.
        `sub_rate`  : A substitution parameter in (0,1). The elasticity of
                    substitution is 1 / (1 - sub_rate).
        `dom_pref`  : A scalar in [0,1] representing the relative preference for
                    domestic consumption over foreign consumption.
        `for_pref`  : An array of the same size as `C_for`. Each element is the
                    relative preference for foreign goods from that country.
    """

    c_dom_pref = dom_pref * (c_dom ** sub_rate)
    c_for_pref = jnp.sum(for_pref * pow(c_for, sub_rate))

    c_agg = (c_dom_pref + c_for_pref) ** (1 / sub_rate)  # CES function
    return c_agg


def load_yaml_data(yaml_file: str):
    """Helper function to read yaml configuration data."""
    with open(yaml_file, "r", encoding="utf-8") as file_ptr:
        file_data = file_ptr.read()
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return rec_array_conversion(data)


def rec_array_conversion(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = jnp.asarray(value)
            elif isinstance(value, dict):
                data[key] = rec_array_conversion(value)
    elif isinstance(data, list):
        data = jnp.asarray(data)
    return data
