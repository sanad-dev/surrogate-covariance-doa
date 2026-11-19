"""v2 evaluation suite (notes/03_rerun_plan.md, CC7): sweeps E1-E9, tables E10-E12, complexity E13.

Every method sees the same test observations at each point (fixed seed per point).
Outputs in data/simulations/results_v2/:
    per_sample/<exp>.npz  per-sample RMSPE [rad] for every method and point
    <exp>.csv             mean, bootstrap 95% CI, MSPE [dB], failure rate, paired
                          Wilcoxon p-values (vs SubspaceNet seed 0 and vs ESPRIT)

    python evaluate_v2.py                          # everything
    python evaluate_v2.py --only E1 E10 --samples 200   # quick subset
"""
import argparse
import csv
import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import wilcoxon

from src.data_handler import create_observations, model_features
from src.evaluation import add_random_predictions
from src.methods import MUSIC, Esprit, RootMUSIC
from src.models import ModelGenerator
from src.utils import device, get_k_peaks, set_unified_seed
from train_v2 import TAU, TRAIN_PARAMS, WEIGHTS_DIR, system_params

OUT = Path(__file__).resolve().parent / "data" / "simulations" / "results_v2"
BASE = {**TRAIN_PARAMS, "snr": 10}  # evaluation base point, true dB
SNRS = [-10, -5, 0, 5, 10, 15, 20]
SWEEPS = {  # name: (swept parameter, values, overrides of BASE)
    "E1": ("snr", SNRS, dict(T=200)),
    "E2": ("snr", SNRS, dict(T=2)),
    "E3": ("snr", SNRS, dict(T=200, signal_nature="coherent")),
    "E4": ("snr", SNRS, dict(T=20, signal_nature="coherent")),
    "E5": ("snr", SNRS, dict(T=2, signal_nature="coherent")),
    "E6": ("T", [2, 5, 10, 20, 50, 100, 200], dict(M=2, signal_nature="coherent")),
    "E7": ("bias", [0, 0.02, 0.04, 0.06, 0.08, 0.10], {}),
    "E8": ("sv_noise_var", [0, 0.05, 0.1, 0.25, 0.5, 0.75], dict(bias=0)),
    "E9": ("sir", [0, -5, -10, -15, -20, -30], {}),  # source 3 is the interferer
}
TABLES = {"E10": {}, "E12": dict(bias=0)}  # E11 = the ablation runs inside E10
# Configurations shown in the sweeps (proposed methods without Toeplitz, D15/D16);
# the Toeplitz ablations (+T) appear in the tables only
SWEEP_CONFIGS = {"SubspaceNetIter", "SubspaceNetAttn", "SubspaceNet", "DA-MUSIC"}
# DeepCNN failed the P4 gate at every lr tried (about 22-23 deg, near random guessing):
# dropped by decision D13, see notes/03_rerun_plan.md section 9
DROPPED_MODELS = {"DeepCNN"}
SUBNET = ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter")


def load_runs():
    """Loads every finished v2 run (pilot/benchmark runs, which carry a tag, are skipped)."""
    runs = {}
    for cfg_path in sorted(WEIGHTS_DIR.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("tag") or cfg["model"] in DROPPED_MODELS:
            continue
        model = (
            ModelGenerator().set_model_type(cfg["model"]).set_diff_method("esprit")
            .set_tau(TAU).set_model(system_params()).model
        )
        model.load_state_dict(torch.load(cfg_path.parent / f"{cfg['name']}.pt", map_location=device))
        label = cfg["model"] + ("+T" if cfg["toeplitz"] > 0 else "") + f"#s{cfg['seed']}"
        runs[label] = (cfg, model.to(device).eval())
    return runs


def source_errors(pred_deg, true_deg):
    """Wrapped per-source errors [rad] under the permutation of minimal RMSPE (eq. 16)."""
    true = np.asarray(true_deg, dtype=float)
    best = None
    for p in permutations(np.asarray(pred_deg, dtype=float)):
        e = ((np.array(p) - true) * np.pi / 180 + np.pi / 2) % np.pi - np.pi / 2
        if best is None or np.sum(e**2) < np.sum(best**2):
            best = e
    return best


def batched(model, x, chunk=256):
    with torch.no_grad():
        outs = [model(x[i : i + chunk].to(device)) for i in range(0, len(x), chunk)]
    if isinstance(outs[0], tuple):  # SubspaceNet family: (doa, all, roots, Rz)
        return torch.cat([o[0] for o in outs]).cpu(), torch.cat([o[3] for o in outs]).cpu()
    return torch.cat(outs).cpu(), None


def evaluate_point(runs, overrides, n, seed, sweep_only, augment):
    """Returns {method: (predictions in deg (list), failure flags)} and the true DoAs."""
    smp = system_params(**{**BASE, **overrides})
    set_unified_seed(seed)
    obs, samples_model = create_observations(smp, n)
    true_deg = [np.rad2deg(d.numpy()) for _, d in obs]
    M, coherent = smp.M, smp.signal_nature == "coherent"
    preds = {}
    algos = {"ESPRIT": Esprit(samples_model), "R-MUSIC": RootMUSIC(samples_model), "MUSIC": MUSIC(samples_model)}

    def subspace(algo, X, mode):
        p = algos[algo].narrowband(X=X, mode=mode)[0]
        return add_random_predictions(M, p, algo), len(p) < M

    for label, (cfg, model) in runs.items():
        kind = cfg["model"]
        if sweep_only and label.split("#")[0] not in SWEEP_CONFIGS:
            continue
        if kind == "DA-MUSIC" and (model.M != M or model.T != smp.T):
            continue  # architecture fixes M and T
        if kind in SUBNET:
            model.M = M
        x = torch.stack([f for f, _ in model_features(obs, kind, tau=cfg.get("tau", TAU))])  # revision runs record their tau
        out, Rz = batched(model, x)
        if kind == "DeepCNN":
            rows = [r.numpy() for r in out]
            found = [len(scipy.signal.find_peaks(r, prominence=0.05, height=0.01)[0]) for r in rows]
            preds[label] = ([get_k_peaks(361, M, torch.tensor(r)).numpy() for r in rows], [f < M for f in found])
            continue
        preds[label] = (list(np.rad2deg(out.real.numpy())), [False] * n)  # ESPRIT/DA-MUSIC output
        if kind in SUBNET and augment and cfg["seed"] == 0:
            for algo in ("R-MUSIC", "MUSIC"):
                res = [subspace(algo, R.numpy(), "covariance") for R in Rz]
                preds[f"{label}+{algo}"] = ([r[0] for r in res], [r[1] for r in res])
    classical = [(a, a, "sample") for a in algos]
    if coherent:
        classical += [("SPS-ESPRIT", "ESPRIT", "spatial_smoothing"), ("SPS-R-MUSIC", "R-MUSIC", "spatial_smoothing")]
    for name, algo, mode in classical:
        res = [subspace(algo, X.numpy(), mode) for X, _ in obs]
        preds[name] = ([r[0] for r in res], [r[1] for r in res])
    return preds, true_deg


def summarize(err, refs, rng):
    deg = np.rad2deg(err)
    boot = [deg[rng.integers(0, len(deg), len(deg))].mean() for _ in range(10000)]
    row = dict(mean_deg=deg.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
               mspe_db=10 * np.log10(np.mean(err**2)))
    for key, ref in refs.items():
        try:
            row[f"p_vs_{key}"] = wilcoxon(err, ref).pvalue if ref is not None else np.nan
        except ValueError:  # identical samples (e.g. the reference itself)
            row[f"p_vs_{key}"] = np.nan
    return row


def run_experiment(name, runs, points, n, sweep_only, augment):
    """points: list of (value, overrides). Writes <name>.csv and per_sample/<name>.npz."""
    rng = np.random.default_rng(0)
    rows, per_sample = [], {}
    for i, (value, overrides) in enumerate(points):
        print(f"--- {name}: {value}")
        preds, true_deg = evaluate_point(runs, overrides, n, seed=10_000 * (int(name[1:]) + 1) + i,
                                         sweep_only=sweep_only, augment=augment)
        errs, target = {}, {}
        for method, (p, fails) in preds.items():
            e = [source_errors(pp, t) for pp, t in zip(p, true_deg)]
            errs[method] = np.array([np.sqrt(np.mean(x**2)) for x in e])
            if name == "E9":  # targets are sources 1-2, source 3 is the interferer
                target[method] = np.array([np.sqrt(np.mean(x[:2] ** 2)) for x in e])
            per_sample.setdefault(method, []).append(errs[method])
        refs = {"SubspaceNet_s0": errs.get("SubspaceNet#s0"), "ESPRIT": errs["ESPRIT"]}
        for method, err in errs.items():
            row = dict(experiment=name, value=value, method=method, n=n,
                       fail_rate=float(np.mean(preds[method][1])), **summarize(err, refs, rng))
            if name == "E9":
                t = np.rad2deg(target[method])
                row["target_mean_deg"] = t.mean()
            rows.append(row)
        rows += seed_aggregates(rows, name, value)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_sample").mkdir(exist_ok=True)
    np.savez(OUT / "per_sample" / f"{name}.npz", **{m: np.array(v, dtype=object) for m, v in per_sample.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))  # columns in first-seen order
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUT / f'{name}.csv'}")


def seed_aggregates(rows, name, value):
    """Mean and std across training seeds of the per-seed mean RMSPE, per configuration."""
    groups = {}
    for r in rows:
        if r["value"] == value and "#s" in r["method"] and "seeds" not in r["method"]:
            config, rest = r["method"].split("#s")
            suffix = rest[1:] if "+" in rest else ""  # e.g. "+MUSIC"
            groups.setdefault(config + suffix, []).append(r["mean_deg"])
    return [dict(experiment=name, value=value, method=f"{c} ({len(v)} seeds)",
                 mean_deg=np.mean(v), seed_std_deg=np.std(v)) for c, v in groups.items() if len(v) > 1]


def complexity(runs):
    """E13: parameters and median inference time per sample (batch 1)."""
    smp = system_params(**BASE)
    set_unified_seed(1)
    obs, samples_model = create_observations(smp, 1)
    X = obs[0][0]
    rows = []

    def timed(fn, reps=1000):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return 1000 * float(np.median(times))

    for label, (cfg, model) in runs.items():
        if cfg["seed"] != 0:
            continue
        x = model_features(obs, cfg["model"], tau=cfg.get("tau", TAU))[0][0].unsqueeze(0)
        row = dict(method=label, n_parameters=sum(p.numel() for p in model.parameters()))
        for dev in ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",):
            m, xd = model.to(dev), x.to(dev)
            with torch.no_grad():
                row[f"ms_{dev}"] = timed(lambda: m(xd))
        model.to(device)
        rows.append(row)
    for name, algo in (("ESPRIT", Esprit), ("R-MUSIC", RootMUSIC), ("MUSIC", MUSIC)):
        a = algo(samples_model)
        rows.append(dict(method=name, n_parameters=0, ms_cpu=timed(lambda: a.narrowband(X=X.numpy()))))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "E13.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_parameters", "ms_cuda", "ms_cpu"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="experiments to run, e.g. E1 E10 (default: all)")
    ap.add_argument("--samples", type=int, default=None, help="override test samples per point")
    args = ap.parse_args()
    runs = load_runs()
    print(f"Loaded runs: {list(runs)}")
    todo = args.only or [*SWEEPS, *TABLES, "E13"]
    for name in todo:
        if name in SWEEPS:
            param, values, overrides = SWEEPS[name]
            if param == "sir":
                points = [(v, {**overrides, "source_power_db": [0, 0, -v]}) for v in values]
            else:
                points = [(v, {**overrides, param: v}) for v in values]
            run_experiment(name, runs, points, args.samples or 2000, sweep_only=True, augment=True)
        elif name in TABLES:
            run_experiment(name, runs, [("table", TABLES[name])], args.samples or 5000,
                           sweep_only=False, augment=True)
        elif name == "E13":
            complexity(runs)


if __name__ == "__main__":
    main()
