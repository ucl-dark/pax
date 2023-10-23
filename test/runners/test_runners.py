import os

import pytest
from hydra import compose, initialize_config_dir

from pax.experiment import main

shared_overrides = [
    "++wandb.mode=disabled",
    "++num_iters=1",
    "++popsize=2",
    "++num_outer_steps=1",
    "++num_inner_steps=4",  # required for ppo
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


def test_runner_evo_runs():
    _test_runner(["+experiment/rice=shaper_v_ppo"])


def test_runner_sarl_runs():
    _test_runner(["+experiment/sarl=cartpole"])


def test_runner_eval_runs():
    _test_runner(
        [
            "+experiment/c_rice=eval_mediator_gs_ppo",
            "++model_path=test/runners/files/eval_mediator/generation_1499",
            "++num_inner_steps=20",
        ]
    )


def test_runner_marl_runs():
    _test_runner(["+experiment/imp=ppo_v_all_heads"])


def test_runner_weight_sharing():
    _test_runner(["+experiment/rice=weight_sharing"])
