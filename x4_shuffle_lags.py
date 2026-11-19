"""X4 control (reviewer 2, M1a): do the submitted 8-lag models use the lags tau >= 1?

On the test observations of E10, E3 and E4, the lag matrices tau = 1..7 of every sample are
replaced by those of another test sample (same condition, random permutation); lag 0 is kept.
A model that relies on the higher lags must get worse; one that reads lag 0 only is unchanged.

    python x4_shuffle_lags.py
Writes results_v2/X4_shuffle.csv (seed-averaged errors in degrees, paired bootstrap CI).
"""
import csv

import numpy as np
import torch

import evaluate_v2 as ev
from src.data_handler import create_observations, model_features
from src.utils import set_unified_seed

MODELS = ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter")


def errors(model, x, true_deg):
    pred = np.rad2deg(ev.batched(model, x)[0].real.numpy())
    return np.rad2deg([np.sqrt(np.mean(ev.source_errors(p, t) ** 2)) for p, t in zip(pred, true_deg)])


def main():
    runs = {k: v for k, v in ev.load_runs().items() if k.split("#")[0] in MODELS}
    rng = np.random.default_rng(0)
    rows = []
    for name in ("E10", "E3", "E4"):
        if name in ev.SWEEPS:
            param, values, overrides = ev.SWEEPS[name]
            points, n = [(v, {**overrides, param: v}) for v in values], 2000
        else:
            points, n = [("table", ev.TABLES[name])], 5000
        saved = np.load(ev.OUT / "per_sample" / f"{name}.npz", allow_pickle=True)
        for i, (value, ov) in enumerate(points):
            smp = ev.system_params(**{**ev.BASE, **ov})
            set_unified_seed(10_000 * (int(name[1:]) + 1) + i)  # seed of evaluate_v2.run_experiment
            obs, _ = create_observations(smp, n)
            true_deg = [np.rad2deg(d.numpy()) for _, d in obs]
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(i))
            intact, shuffled = {m: [] for m in MODELS}, {m: [] for m in MODELS}
            for label, (cfg, model) in runs.items():
                model.M = smp.M
                x = torch.stack([f for f, _ in model_features(obs, cfg["model"], tau=ev.TAU)])
                assert x.shape[1] == ev.TAU, f"lag axis is not axis 1: {tuple(x.shape)}"
                e = errors(model, x, true_deg)
                # Intact inputs must give back the submitted errors, or the test data differ
                ref = np.rad2deg(saved[label][i].astype(float))
                assert abs(e.mean() - ref.mean()) < 0.01, f"{name} {value} {label}: {e.mean():.4f} vs {ref.mean():.4f}"
                x[:, 1:] = x[perm][:, 1:]
                intact[cfg["model"]].append(e)
                shuffled[cfg["model"]].append(errors(model, x, true_deg))
            for m in MODELS:
                a, b = np.mean(intact[m], axis=0), np.mean(shuffled[m], axis=0)
                d = b - a
                boot = [d[rng.integers(0, n, n)].mean() for _ in range(10000)]
                rows.append(dict(experiment=name, value=value, model=m, intact_deg=a.mean(), shuffled_deg=b.mean(),
                                 diff=d.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
                r = rows[-1]
                print(f"{name} {str(value):>5} {m:16s} intact {r['intact_deg']:6.3f}  lags 1-7 shuffled {r['shuffled_deg']:6.3f}"
                      f"  diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", flush=True)
    with open(ev.OUT / "X4_shuffle.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
