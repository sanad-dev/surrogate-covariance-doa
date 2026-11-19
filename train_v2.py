"""Single training entry point for the v2 rerun (notes/03_rerun_plan.md, CC5).

Every model trains on the same shared observations (CC3), generated once under
DATA_SEED. The run's own --seed only sets weight initialization, dropout and
batch order (R2); the train/validation split is fixed.

Examples
--------
    python train_v2.py --model SubspaceNetIter --toeplitz 0.1 --seed 0 --lr 1e-4
    python train_v2.py --model DeepCNN --seed 0 --lr 1e-4
Outputs go to data/weights/v2/<run name>/ (checkpoint, curves.csv, config.json).
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch

from src.data_handler import create_observations, model_features
from src.models import ModelGenerator
from src.system_model import SystemModelParams
from src.training import TrainingParams, train
from src.utils import set_unified_seed

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "datasets" / "v2"
WEIGHTS_DIR = ROOT / "data" / "weights" / "v2"
DATA_SEED = 2026
TAU = 8
# Training distribution (plan §2): all SNR values in true dB
TRAIN_PARAMS = dict(
    N=8, M=3, T=200, snr=(-10, 20), signal_type="NarrowBand",
    signal_nature="non-coherent", eta=0, bias=0.05, sv_noise_var=0,
)
# X9 (reviewer 1, comment 1): sensor positions in half wavelengths. nested = two-level nested array,
# inner ULA of 4 sensors at unit spacing, outer ULA of 4 sensors at spacing 5
ARRAYS = {"ula": None, "nested": [0, 1, 2, 3, 4, 9, 14, 19]}
MODELS = ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter", "DA-MUSIC", "DeepCNN")
# X7 module ablations (reviewer 1, comment 3): constructor overrides, see src/models.py
VARIANTS = {
    "SubspaceNetAttn": {
        "nolayers": dict(num_layers=0),                             # no Transformer layers
        "meanpool": dict(pool="mean"),                              # mean instead of attention pooling
        "nopos": dict(use_pos=False),                               # no positional embedding
        "mlp": dict(num_layers=0, pool="mean", use_pos=False),      # embedding -> mean -> head
        # capacity control: 40,400 parameters with --tau 1 (SubspaceNet: 41,761)
        "mlp48": dict(num_layers=0, pool="mean", use_pos=False, d_model=48),
        # X6 (reviewer 2, M2): correction added to the sample covariance instead of a full covariance
        "res": dict(output="res_psd"),                              # positive-definite correction
        "resherm": dict(output="res_herm"),                         # Hermitian correction, any sign
        # residual-correction design of Dkhan et al. in this far-field setting: use with --diff-method root_music
        "dkhan": dict(output="res_psd", pool="mean"),
    },
    "SubspaceNetIter": {
        "L0": dict(n_iterations=0),                                 # no refinement loop
        "L1": dict(n_iterations=1),
        "meanctx": dict(context="mean"),                            # refinement without cross-attention
    },
}


def model_generator(model, tau=TAU, diff_method="esprit", variant=""):
    gen = ModelGenerator().set_model_type(model).set_diff_method(diff_method).set_tau(tau)
    gen.variant = VARIANTS[model][variant] if variant else {}
    return gen.set_model(system_params())


def system_params(**overrides) -> SystemModelParams:
    smp = SystemModelParams()
    for name, value in {**TRAIN_PARAMS, **overrides}.items():
        smp.set_parameter(name, value)
    return smp


def shared_observations(samples: int, sir_range=None, coherent_fraction=None):
    """Loads the shared training observations, generating them once if missing.
    sir_range: unequal source powers, see create_observations (X10, reviewer 2, M6a).
    coherent_fraction: share of coherent samples in training (X12, reviewer 2, M5b)."""
    p = TRAIN_PARAMS
    path = DATA_DIR / (
        f"train_obs_v2_N{p['N']}_M{p['M']}_T{p['T']}_snr{p['snr'][0]}to{p['snr'][1]}"
        f"_bias{p['bias']}_{p['signal_nature']}_n{samples}_seed{DATA_SEED}"
        + ("_nonuniform" if p.get("array_positions") else "")
        + (f"_sir{sir_range[0]:g}to{sir_range[1]:g}" if sir_range else "")
        + (f"_coh{coherent_fraction:g}" if coherent_fraction is not None else "") + ".pt"
    )
    if path.exists():
        return torch.load(path, weights_only=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_unified_seed(DATA_SEED)
    observations, _ = create_observations(system_params(), samples, sir_range=sir_range,
                                          coherent_fraction=coherent_fraction)
    torch.save(observations, path)
    return observations


def src_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--toeplitz", type=float, default=0.0, help="Toeplitz regularization weight (SubspaceNet family)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--samples", type=int, default=100000, help="size of the shared data set (train + validation)")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--tag", default="", help="suffix for pilot or benchmark runs")
    # Revision controls: the run name does not carry them, so always pair them with a --tag
    ap.add_argument("--tau", type=int, default=TAU, help="number of autocorrelation lags fed to the model")
    ap.add_argument("--diff-method", default="esprit", choices=("esprit", "root_music"))
    ap.add_argument("--variant", default="", help="module ablation, a key of VARIANTS[model]")
    ap.add_argument("--sir-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    help="train with one source stronger by -SIR dB, SIR ~ U[LOW, HIGH] dB per sample")
    ap.add_argument("--coherent-fraction", type=float, help="train with this share of coherent samples")
    ap.add_argument("--N", type=int, default=TRAIN_PARAMS["N"], help="number of sensors")
    ap.add_argument("--M", type=int, default=TRAIN_PARAMS["M"], help="number of sources")
    ap.add_argument("--array", default="ula", choices=tuple(ARRAYS), help="array geometry")
    args = ap.parse_args()
    default = (args.tau == TAU and args.diff_method == "esprit" and not args.variant and not args.sir_range
               and args.coherent_fraction is None
               and (args.N, args.M, args.array) == (TRAIN_PARAMS["N"], TRAIN_PARAMS["M"], "ula"))
    assert args.tag or default, "non-default data or model option (--tau, --variant, --sir-range, ...) needs --tag"
    # X8, X9: the training condition itself changes (data set, model size, saved train_params)
    TRAIN_PARAMS.update(N=args.N, M=args.M)
    if args.array != "ula":
        TRAIN_PARAMS["array_positions"] = ARRAYS[args.array]

    observations = shared_observations(args.samples, args.sir_range, args.coherent_fraction)
    # Run seed before the model exists, so it controls initialization (R2)
    set_unified_seed(args.seed)
    model_config = model_generator(args.model, args.tau, args.diff_method, args.variant)
    dataset = model_features(
        observations, args.model, tau=args.tau, grid_labels=args.model.startswith("DeepCNN")
    )
    params = (
        TrainingParams()
        .set_batch_size(args.batch_size)
        .set_epochs(args.epochs)
        .set_model(model=model_config)
        .set_optimizer(optimizer="Adam", learning_rate=args.lr, weight_decay=1e-9)
        .set_training_dataset(dataset)
        .set_schedular(step_size=args.epochs, gamma=1.0)  # constant lr; early stopping decides the end
        .set_criterion()
    )
    if args.toeplitz > 0:
        params.set_toeplitz_regularization(weight=args.toeplitz)
    params.patience = args.patience

    name = f"{args.model}_tz{args.toeplitz:g}_seed{args.seed}_lr{args.lr:g}{args.tag}"
    run_dir = WEIGHTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model, loss_train, loss_valid = train(
        training_parameters=params, model_name=f"{name}.pt", plot_curves=False,
        saving_path=run_dir, seed=args.seed,
    )
    with open(run_dir / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss"])
        w.writerows([e + 1, a, b] for e, (a, b) in enumerate(zip(loss_train, loss_valid)))
    config = dict(
        vars(args), name=name, train_params={k: v for k, v in TRAIN_PARAMS.items()},
        data_seed=DATA_SEED, src_sha256=src_hash(),  # tau and diff_method come from vars(args)
        epochs_run=len(loss_train), best_valid_loss=min(loss_valid),
        linalg_errors=params.linalg_errors, wall_time_s=round(time.time() - start, 1),
        n_parameters=sum(p.numel() for p in model.parameters()),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
