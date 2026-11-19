"""Subspace-Net -- Baseline models trainer (DA-MUSIC, DeepCNN) -- FAST VERSION

Purpose
-------
Trains and saves the two additional deep-learning baselines needed for the
full comparison figures (generate_comparison_figures.py): DA-MUSIC and
DeepCNN. SubspaceNet and SubspaceNetAttn are trained separately via
main.py and main_attn.py.

This is a LIGHTWEIGHT version: samples_size and epochs are reduced compared
to main.py/main_attn.py, since DA-MUSIC/DeepCNN are only comparison
baselines here (not the method being proposed). Expect roughly 1-2h total
instead of ~10-14h with the full-size settings. If you want the original
full-size settings back, set samples_size=100000 and .set_epochs(80) below.

Run this once (after main.py / main_attn.py) with SAVE_MODEL=True, using
the SAME BASE_PARAMS scenario as main.py / main_attn.py /
generate_comparison_figures.py, so the saved checkpoints can be found and
loaded later.
"""
raise SystemExit("v1 script, superseded by train_v2.py / evaluate_v2.py (notes/03_rerun_plan.md). To rerun v1: git checkout v1-snapshot")
# Imports
import sys
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data_handler import create_dataset, load_datasets
from src.models import ModelGenerator
from src.system_model import SystemModelParams
from src.training import TrainingParams, get_simulation_filename, train, simulation_summary
from src.utils import set_unified_seed

warnings.simplefilter("ignore")
plt.close("all")

if __name__ == "__main__":
    # Initialize paths (same layout as main.py)
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "uniform_bias_spacing"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)
    (saving_path / "final_models").mkdir(parents=True, exist_ok=True)

    commands = {
        "CREATE_DATA": True,  # DA-MUSIC and DeepCNN each need their own tensor
        # format (raw X for DA-MUSIC, covariance tensor for DeepCNN) --
        # NOT the SubspaceNet tau-lag tensor, so this data must be
        # generated separately the first time you run this script.
        "LOAD_DATA": False,  # Set True (and CREATE_DATA False) on subsequent
        # runs, once the DA-MUSIC/DeepCNN datasets already exist on disk.
        "SAVE_MODEL": True,
    }

    # MUST MATCH BASE_PARAMS in main.py / main_attn.py / generate_comparison_figures.py
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", 8)
        .set_parameter("M", 3)
        .set_parameter("T", 200)
        .set_parameter("snr", 10)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0.05)
        .set_parameter("sv_noise_var", 0)
    )

    # --- REDUCED for speed (was 100000 / 80 epochs in the full-size script) ---
    samples_size = 20000
    train_test_ratio = 0.05
    n_epochs = 25

    set_unified_seed()

    for model_type in ["DA-MUSIC", "DeepCNN"]:
        print(f"\n========== Training {model_type} ==========")
        model_config = (
            ModelGenerator()
            .set_model_type(model_type)
            .set_diff_method("esprit")  # unused by these models, kept for filename consistency
            .set_tau(8)  # unused by these models, kept for filename consistency
            .set_model(system_model_params)
        )
        simulation_filename = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )

        if commands["CREATE_DATA"]:
            train_dataset, _, _ = create_dataset(
                system_model_params=system_model_params,
                samples_size=samples_size,
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="train",
            )
            # Also save a matching test set (not used by this script directly,
            # but kept on disk in case you want to load it later).
            create_dataset(
                system_model_params=system_model_params,
                samples_size=int(train_test_ratio * samples_size),
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="test",
            )
        elif commands["LOAD_DATA"]:
            train_dataset, test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,  # each model needs its own tensor format
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
                is_training=True,
            )

        simulation_parameters = (
            TrainingParams()
            .set_batch_size(2048)
            .set_epochs(n_epochs)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=0.00001, weight_decay=1e-9)
            .set_training_dataset(train_dataset)
            .set_schedular(step_size=n_epochs, gamma=0.2)
            .set_criterion()
        )
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
            print(f"Saved: {saving_path / 'final_models' / simulation_filename}")

    print("done")

