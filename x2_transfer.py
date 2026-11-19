"""X2 (reviewer 2, M3a and M7b): does a surrogate covariance transfer to an estimator it was not trained through?

On the test observations of E10 and E12, the covariance Rz produced by each model is passed to ESPRIT,
Root-MUSIC and MUSIC (same calls as the "+R-MUSIC" / "+MUSIC" entries of evaluate_v2, which cover
seed 0 of the submitted runs only). Reports the RMSPE and the share of samples where the estimator
returns fewer than M directions, for the submitted SubspaceNet / Attn / Iter runs (all seeds) and for
the tagged runs given by --runs.

    python x2_transfer.py --runs "*rootmusic" "*abl1-mlp48"
Writes results_v2/X2_transfer.csv.
"""
import argparse
import csv

import numpy as np
import torch

import eval_tagged
import evaluate_v2 as ev
from src.data_handler import create_observations, model_features
from src.evaluation import add_random_predictions
from src.methods import MUSIC, Esprit, RootMUSIC
from src.utils import set_unified_seed
from train_v2 import TAU


def root_radius(Rz, M):
    """Per sample, the smallest modulus among the M Root-MUSIC roots that are retained (the M roots
    inside the unit circle that are closest to it). MUSIC needs these roots near the circle: a root
    far inside gives no sharp peak in the spectrum."""
    N = Rz.shape[-1]
    w, V = torch.linalg.eig(Rz)
    Un = torch.gather(V, 2, torch.argsort(w.abs(), dim=-1, descending=True)[:, M:].unsqueeze(1).expand(-1, N, -1))
    F = Un @ Un.conj().transpose(1, 2)
    c = torch.stack([torch.diagonal(F, k, dim1=1, dim2=2).sum(-1) for k in range(-N + 1, N)], dim=1)
    r = np.abs(np.array([np.roots(ci) for ci in c.numpy()]))
    r = np.sort(np.where(r < 1, r, 0), axis=1)[:, ::-1][:, :M]
    return r.min(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[], help="glob patterns of tagged run folders in data/weights/v2")
    args = ap.parse_args()
    # (label, estimator used in training, tau, cfg, model)
    runs = [(k, "esprit", TAU, cfg, m) for k, (cfg, m) in ev.load_runs().items()
            if k.split("#")[0] in ev.SUBNET]
    for pattern in args.runs:  # one pattern at a time: eval_tagged.load accepts a single tau per call
        runs += [(k, cfg.get("diff_method", "esprit"), cfg.get("tau", TAU), cfg, m)
                 for k, (cfg, m) in eval_tagged.load([pattern]).items()]
    saved = {(r["experiment"], r["method"]): r for name in ev.TABLES
             for r in csv.DictReader(open(ev.OUT / f"{name}.csv"))}
    rows = []
    for name, overrides in ev.TABLES.items():
        smp = ev.system_params(**{**ev.BASE, **overrides})
        set_unified_seed(10_000 * (int(name[1:]) + 1))  # seed of evaluate_v2.run_experiment
        obs, samples_model = create_observations(smp, 5000)
        true = [np.rad2deg(d.numpy()) for _, d in obs]
        algos = {"ESPRIT": Esprit(samples_model), "R-MUSIC": RootMUSIC(samples_model), "MUSIC": MUSIC(samples_model)}
        rmspe = lambda preds: float(np.rad2deg(np.mean(
            [np.sqrt(np.mean(ev.source_errors(p, t) ** 2)) for p, t in zip(preds, true)])))
        radius = root_radius(torch.stack([X @ X.conj().T / X.shape[1] for X, _ in obs]), smp.M)
        rows.append(dict(experiment=name, model="sample covariance", trained_through="", seed="", estimator="",
                         mean_deg=float("nan"), fail_rate=float("nan"),
                         root_radius_median=float(np.median(radius)), root_radius_q10=float(np.quantile(radius, 0.1))))
        print(f"{name} sample covariance: retained-root radius median {rows[-1]['root_radius_median']:.3f}, q10 {rows[-1]['root_radius_q10']:.3f}", flush=True)
        for label, through, tau, cfg, model in runs:
            model.M = smp.M
            x = torch.stack([f for f, _ in model_features(obs, cfg["model"], tau=tau)])
            out, Rz = ev.batched(model, x)
            res = {f"native ({through})": (rmspe(list(np.rad2deg(out.real.numpy()))), 0.0)}
            radius = root_radius(Rz, smp.M)
            for algo in algos:
                p = [algos[algo].narrowband(X=R.numpy(), mode="covariance")[0] for R in Rz]
                res[algo] = (rmspe([add_random_predictions(smp.M, q, algo) for q in p]),
                             float(np.mean([len(q) < smp.M for q in p])))
            if through == "esprit" and cfg["seed"] == 0 and "/" not in label:
                # Same test data and same calls as the submitted evaluation (missed peaks are filled
                # at random, so only the miss rate and the miss-free Root-MUSIC error are compared)
                ref = saved[(name, f"{label}+MUSIC")], saved[(name, f"{label}+R-MUSIC")]
                # GPU round-off may flip a borderline peak: allow 2 samples out of 5000
                assert abs(res["MUSIC"][1] - float(ref[0]["fail_rate"])) < 5e-4, f"{name} {label}: MUSIC miss rate"
                assert abs(res["R-MUSIC"][0] - float(ref[1]["mean_deg"])) < 1e-3, f"{name} {label}: Root-MUSIC error"
            for est, (mean, fail) in res.items():
                rows.append(dict(experiment=name, model=label.split("#")[0], trained_through=through,
                                 seed=cfg["seed"], estimator=est, mean_deg=mean, fail_rate=fail,
                                 root_radius_median=float(np.median(radius)), root_radius_q10=float(np.quantile(radius, 0.1))))
                print(f"{name} {label:26s} trained through {through:10s} {est:22s} {mean:7.3f} deg  miss {100 * fail:5.2f} %  radius {np.median(radius):.3f}",
                      flush=True)
    with open(ev.OUT / "X2_transfer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
