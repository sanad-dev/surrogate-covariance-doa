"""X8, X9 (reviewer 1, comment 1): other numbers of sources, array sizes and a non-uniform array.

Each condition has its own test data at SNR = 10 dB, with non-coherent (5000 samples) and coherent
(2000 samples) sources. Reports the RMSPE of the classical estimators on the sample covariance, of
the runs trained for that condition (tag x8-* / x9-*), and, for M != 3, of the submitted models
(trained with M = 3) used with the new M. The base condition (N = 8, M = 3, ULA) is included as a
reference on the same kind of test data.

    python x8_x9_eval.py
Writes results_v2/X8_X9.csv and results_v2/per_sample/X8_X9_<condition>_<nature>.npz.
"""
import csv
import json

import numpy as np
import torch

import evaluate_v2 as ev
from eval_tagged import paired
from src.data_handler import create_observations, model_features
from src.evaluation import add_random_predictions
from src.methods import MUSIC, Esprit, RootMUSIC
from src.models import ModelGenerator
from src.utils import set_unified_seed
from train_v2 import ARRAYS, VARIANTS, system_params

CONDITIONS = {  # tag of the runs trained for the condition: overrides of the training condition
    "": {}, "x8-M2": dict(M=2), "x8-M4": dict(M=4), "x8-M5": dict(M=5),
    "x9-N12": dict(N=12), "x9-N16": dict(N=16), "x9-nested": dict(array_positions=ARRAYS["nested"]),
}
MODELS = ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter", "Dense-48")


def load(pattern, cond):
    """{label: (cfg, model)} for the run folders matching pattern, built for the condition's N."""
    runs = {}
    for d in sorted(ev.WEIGHTS_DIR.glob(pattern)):
        cfg = json.loads((d / "config.json").read_text())
        gen = ModelGenerator().set_model_type(cfg["model"]).set_diff_method("esprit").set_tau(cfg.get("tau", 8))
        gen.variant = VARIANTS[cfg["model"]][cfg["variant"]] if cfg.get("variant") else {}
        model = gen.set_model(system_params(**cond)).model
        model.load_state_dict(torch.load(d / f"{cfg['name']}.pt", map_location=ev.device))
        name = "Dense-48" if cfg.get("variant") == "mlp48" else cfg["model"]
        runs[f"{name}#s{cfg['seed']}"] = (cfg, model.to(ev.device).eval())
    return runs


def submitted(cond):
    """The models of the base condition (M = 3, N = 8, ULA): submitted runs and Dense-48."""
    runs = load("*_lr0.001abl1-mlp48", cond)
    for m in MODELS[:3]:
        runs.update(load(f"{m}_tz0_seed?_lr0.001", cond))
    return runs


def main():
    rng = np.random.default_rng(0)
    rows = []
    for i, (tag, cond) in enumerate(CONDITIONS.items()):
        name = tag or "base"
        groups = {"own": {**load(f"*_lr0.001{tag}", cond), **load(f"*_lr0.001{tag}-mlp48", cond)} if tag else submitted(cond)}
        if "M" in cond:  # the models trained with M = 3, asked for another number of sources
            groups["M=3"] = submitted({})
        assert all(len(g) == 12 for g in groups.values()), {k: sorted(g) for k, g in groups.items()}
        for j, (nature, n) in enumerate((("non-coherent", 5000), ("coherent", 2000))):
            smp = system_params(**{**cond, "snr": 10, "signal_nature": nature})
            set_unified_seed(30_000 + 10 * i + j)
            obs, samples_model = create_observations(smp, n)
            true = [np.rad2deg(d.numpy()) for _, d in obs]
            err = lambda preds: np.rad2deg([np.sqrt(np.mean(ev.source_errors(p, t) ** 2)) for p, t in zip(preds, true)])
            per_sample = {}
            # ESPRIT and Root-MUSIC need a uniform array: only MUSIC is reported for the nested array
            classical = {"MUSIC": MUSIC} if "array_positions" in cond else {"ESPRIT": Esprit, "R-MUSIC": RootMUSIC, "MUSIC": MUSIC}
            for algo, cls in classical.items():
                method = cls(samples_model)
                per_sample[algo] = err([add_random_predictions(smp.M, method.narrowband(X=X.numpy(), mode="sample")[0], algo)
                                        for X, _ in obs])
                rows.append(dict(condition=name, nature=nature, method=algo, trained="", n_seeds="",
                                 mean_deg=per_sample[algo].mean(), seed_std="", diff_subspacenet="", lo="", hi="",
                                 n_parameters=""))
            for trained, runs in groups.items():
                for label, (cfg, model) in runs.items():
                    model.M = smp.M
                    x = torch.stack([f for f, _ in model_features(obs, cfg["model"], tau=cfg.get("tau", 8))])
                    per_sample[f"{label}|{trained}"] = err(list(np.rad2deg(ev.batched(model, x)[0].real.numpy())))
                seeds = lambda m: [v for k, v in per_sample.items() if k.startswith(f"{m}#") and k.endswith(f"|{trained}")]
                base = np.mean(seeds("SubspaceNet"), axis=0)
                for m in MODELS:
                    s = seeds(m)
                    d = paired(np.mean(s, axis=0) - base, rng) if m != "SubspaceNet" else [float("nan")] * 4
                    cfg = next(c for k, (c, _) in runs.items() if k.startswith(f"{m}#"))
                    rows.append(dict(condition=name, nature=nature, method=m, trained=trained, n_seeds=len(s),
                                     mean_deg=np.mean(s), seed_std=np.std([x.mean() for x in s]),
                                     diff_subspacenet=d[0], lo=d[1], hi=d[2], n_parameters=cfg.get("n_parameters", "")))
            np.savez(ev.OUT / "per_sample" / f"X8_X9_{name}_{nature}.npz", **per_sample)
            for r in rows[-len(classical) - 4 * len(groups):]:
                print(f"{name:10s} {nature:13s} {r['method']:16s} {r['trained']:4s} {r['mean_deg']:7.3f}", flush=True)
    with open(ev.OUT / "X8_X9.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
