"""Evaluates revision runs that carry a --tag (evaluate_v2.py skips them) on the test data of
the submitted experiments, and compares them with the submitted runs sample by sample.

X4 (reviewer 2, M1a), models trained on the lag-0 covariance only:
    python eval_tagged.py --out tau1 --runs "*tau1" --only E10 E12 E3 E4
Option 2 of the revision, single-lag dense encoder on every experiment, and its capacity control:
    python eval_tagged.py --out dense --runs "*abl1-mlp" --only E9 E1 E2 E3 E4 E5 E6 E7 E8 E10 E12
    python eval_tagged.py --out cap --runs "*abl1-mlp48" --only E10 E12 E3 E4
X3 (reviewer 2, M3b), each model at its best validation learning rate:
    python eval_tagged.py --out lrbest --only E10 E12 --runs "SubspaceNet_*lr0.003lrsweep" \
        "SubspaceNetAttn_*lr0.0001lrsweep" "SubspaceNetIter_*lr0.0001lrsweep"

Writes results_v2/X_<out>.csv and results_v2/per_sample/X_<out>_<exp>.npz. Differences are
seed-averaged per-sample errors (new - submitted), with a paired bootstrap CI and a paired
Wilcoxon p-value, against the submitted runs of the same model and of SubspaceNet.
"""
import argparse
import csv
import json

import numpy as np
import torch
from scipy.stats import wilcoxon

import evaluate_v2 as ev
from train_v2 import model_generator


def load(patterns):
    runs = {}
    for pattern in patterns:
        for d in sorted(ev.WEIGHTS_DIR.glob(pattern)):
            cfg = json.loads((d / "config.json").read_text())
            variant = cfg.get("variant", "")
            label = f"{cfg['model']}{'/' + variant if variant else ''}#s{cfg['seed']}"
            assert label not in runs, f"two runs for {label}: narrow --runs"
            model = model_generator(cfg["model"], cfg.get("tau", 8), cfg.get("diff_method", "esprit"), variant).model
            model.load_state_dict(torch.load(d / f"{cfg['name']}.pt", map_location=ev.device))
            runs[label] = (cfg, model.to(ev.device).eval())
    taus = {cfg.get("tau", 8) for cfg, _ in runs.values()}
    assert len(taus) == 1, f"runs with different tau in one call: {taus}"
    ev.TAU = taus.pop()  # evaluate_point builds the model inputs with this global
    return runs


def paired(d, rng):
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), wilcoxon(d).pvalue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="glob patterns of run folders in data/weights/v2")
    ap.add_argument("--out", required=True, help="name of the output files")
    ap.add_argument("--only", nargs="+", default=["E10", "E12"])
    args = ap.parse_args()
    runs = load(args.runs)
    models = sorted({k.split("#")[0] for k in runs})
    print(f"tau = {ev.TAU}; runs: {list(runs)}")
    rng = np.random.default_rng(0)
    rows = []
    for name in args.only:
        if name in ev.SWEEPS:
            param, values, overrides = ev.SWEEPS[name]
            # E9: same override as evaluate_v2.main (source 3 is the interferer)
            points = [(v, {**overrides, **({"source_power_db": [0, 0, -v]} if param == "sir" else {param: v})})
                      for v in values]
            n = 2000
        else:
            points, n = [("table", ev.TABLES[name])], 5000
        old = np.load(ev.OUT / "per_sample" / f"{name}.npz", allow_pickle=True)
        deg = lambda label, i: np.rad2deg(old[label][i].astype(float))
        per_sample = {}
        for i, (value, ov) in enumerate(points):
            preds, true = ev.evaluate_point(runs, ov, n, seed=10_000 * (int(name[1:]) + 1) + i,
                                            sweep_only=False, augment=False)
            err = {k: np.rad2deg([np.sqrt(np.mean(ev.source_errors(p, t) ** 2)) for p, t in zip(pp, true)])
                   for k, (pp, _) in preds.items() if k in runs or k == "ESPRIT"}
            # Same test observations as the submitted evaluation, or the paired tests mean nothing
            assert np.allclose(err["ESPRIT"], deg("ESPRIT", i), atol=1e-6), f"{name} {value}: test data differ"
            for k in runs:
                per_sample.setdefault(k, []).append(np.deg2rad(err[k]))
            # E9 is reported on the targets (sources 1-2; source 3 is the interferer), as in evaluate_v2
            tgt = {k: np.rad2deg(np.mean([np.sqrt(np.mean(ev.source_errors(p, t)[:2] ** 2))
                                          for p, t in zip(preds[k][0], true)])) for k in runs} if name == "E9" else {}
            sub_base = np.mean([deg(f"SubspaceNet#s{s}", i) for s in range(3)], axis=0)
            for m in models:
                seeds = [err[k] for k in runs if k.split("#")[0] == m]
                new = np.mean(seeds, axis=0)
                sub = np.mean([deg(f"{m.split('/')[0]}#s{s}", i) for s in range(3)], axis=0)  # submitted full model
                d_same, d_base = paired(new - sub, rng), paired(new - sub_base, rng)
                rows.append(dict(
                    experiment=name, value=value, model=m, n_seeds=len(seeds), esprit_deg=err["ESPRIT"].mean(),
                    fail_rate=np.mean([np.mean(preds[k][1]) for k in runs if k.split("#")[0] == m]),
                    new_target_deg=np.mean([tgt[k] for k in runs if k.split("#")[0] == m]) if tgt else float("nan"),
                    new_deg=new.mean(), new_seed_std=np.std([s.mean() for s in seeds]), submitted_deg=sub.mean(),
                    diff_same=d_same[0], same_lo=d_same[1], same_hi=d_same[2], p_same=d_same[3],
                    submitted_subspacenet_deg=sub_base.mean(),
                    diff_base=d_base[0], base_lo=d_base[1], base_hi=d_base[2], p_base=d_base[3]))
                r = rows[-1]
                print(f"{name} {str(value):>5} {m:16s} new {r['new_deg']:6.3f} (std {r['new_seed_std']:.3f})  submitted {r['submitted_deg']:6.3f}"
                      f"  diff {r['diff_same']:+.3f} [{r['same_lo']:+.3f}, {r['same_hi']:+.3f}] p={r['p_same']:.2g}"
                      f"  | vs submitted SubspaceNet {r['submitted_subspacenet_deg']:6.3f}: {r['diff_base']:+.3f}"
                      f" [{r['base_lo']:+.3f}, {r['base_hi']:+.3f}] p={r['p_base']:.2g}")
        np.savez(ev.OUT / "per_sample" / f"X_{args.out}_{name}.npz", **{k: np.array(v) for k, v in per_sample.items()})
    with open(ev.OUT / f"X_{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
