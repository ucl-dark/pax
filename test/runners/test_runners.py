import os

import pytest
from hydra import compose, initialize_config_dir

from pax.experiment import main

shared_overrides = [
    "++wandb.mode=disabled",
    "++num_iters=1",
    "++popsize=2",
    "++num_outer_steps=1",
    "++num_inner_steps=8",  # required for ppo minibatch size
    "++num_devices=1",
    "++num_envs=1",
    "++num_epochs=1",
]


@pytest.fixture(scope="module", autouse=True)
def setup_hydra():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../pax/conf"
    )
    initialize_config_dir(config_dir=path)


def _test_runner(overrides):
    cfg = compose(
        config_name="config.yaml", overrides=shared_overrides + overrides
    )
    main(cfg)


def test_runner_evo_nroles_runs():
    _test_runner(["+experiment/rice=shaper_v_ppo"])


def test_runner_evo_runs():
    _test_runner(["+experiment/cg=mfos"])


def test_runner_sarl_runs():
    _test_runner(["+experiment/sarl=cartpole"])


def test_runner_eval_runs():
    _test_runner(
        [
            "+experiment/c_rice=eval_mediator_gs_ppo",
            "++model_path=test/runners/files/eval_mediator/generation_1499",
            # Eval requires a full episode to be played
            "++num_inner_steps=20",
        ]
    )


def test_runner_marl_runs():
    _test_runner(["+experiment/cg=tabular"])


def test_runner_weight_sharing():
    _test_runner(["+experiment/rice=weight_sharing"])


def test_runner_evo_multishaper():
    _test_runner(
        ["+experiment/multiplayer_ipd=3pl_2shap_ipd", "++num_inner_steps=10"]
    )


def test_runner_marl_nplayer():
    _test_runner(
        ["+experiment/multiplayer_ipd=lola_vs_ppo_ipd", "++num_inner_steps=10"]
    )


def test_runner_evo_hardstop():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_hardstop"])


def test_runner_evo_mixed_rl():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_lr"])


def test_runner_evo_mixed_payoff():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_payoff"])


def test_runner_evo_mixed_ipd_payoff():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_ipd_payoff"])


def test_runner_evo_mixed_payoff_gen():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_payoff_gen"])


def test_runner_evo_mixed_payoff_input():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_payoff_input"])


def test_runner_evo_mixed_payoff_input():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_payoff_input"])


def test_runner_evo_scanned():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_scanned"])


def test_runner_evo_mixed_payoff_only_opp():
    _test_runner(["+experiment/ipd=shaper_att_v_tabular", "++runner=evo_mixed_payoff_only_opp"])