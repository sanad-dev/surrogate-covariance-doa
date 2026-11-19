"""X13 (reviewer 2, comment M4): multiple-comparison control and seed-pair tests.

1. Benjamini-Hochberg over every paired Wilcoxon p-value of the study: those stored in
   results_v2/E*.csv (each method against SubspaceNet seed 0 and against ESPRIT) plus the
   seed-pair and seed-averaged tests added here.
2. E10 / E12: each seed of each configuration against each SubspaceNet seed, one
   seed-averaged paired test per configuration, and a paired bootstrap CI of the mean
   difference (the statistic the overlapping per-method CIs cannot show).

3. SubspaceNet-FC (the single-lag dense encoder of X15, evaluated by eval_tagged.py): the same
   E10 / E12 tests, and at every point of E1-E10 and E12 one seed-averaged paired test against
   SubspaceNet, Attn, Iter and ESPRIT. These tests join the Benjamini-Hochberg family.

    python stats_x13.py
Writes results_v2/X13_fdr.csv, results_v2/X13_seed_pairs.csv and results_v2/X13_fc.csv.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import false_discovery_control, wilcoxon

OUT = Path(__file__).resolve().parent / "data" / "simulations" / "results_v2"
CONFIGS = ["SubspaceNetIter", "SubspaceNetIter+T", "SubspaceNetAttn", "SubspaceNetAttn+T", "SubspaceNet+T"]
FC = "SubspaceNet-FC"
SEEDS = (0, 1, 2)
Q = 0.05


def errors(exp):
    """Returns label -> per-sample errors in degrees, [points, samples], for one experiment."""
    z = np.load(OUT / "per_sample" / f"{exp}.npz", allow_pickle=True)
    # SubspaceNet-FC: E10, E12, E3, E4 are in X_cap_*, the other experiments in X_cap_all_*
    fc = np.load(next(f for f in (OUT / "per_sample" / f"X_cap{a}_{exp}.npz" for a in ("", "_all")) if f.exists()))

    def err(label):
        if label.startswith(FC):
            return np.rad2deg(fc[label.replace(FC, "SubspaceNetAttn/mlp48")])
        return np.rad2deg(np.stack(z[label]).astype(float))
    return err


def csv_tests():
    """Every p-value already stored by evaluate_v2.py: (experiment, value, method, reference, p)."""
    tests = []
    for path in sorted(OUT.glob("E*.csv")):
        for r in csv.DictReader(open(path)):
            for ref in ("SubspaceNet_s0", "ESPRIT"):
                p = r.get(f"p_vs_{ref}", "")
                if p not in ("", "nan"):
                    tests.append((r["experiment"], r["value"], r["method"], ref, float(p)))
    return tests


def seed_tests():
    """E10/E12 seed-pair and seed-averaged tests against SubspaceNet. Errors in degrees."""
    pair_rows, tests = [], []
    rng, rng_fc = np.random.default_rng(0), np.random.default_rng(2)
    for exp in ("E10", "E12"):
        err = lambda label, e=errors(exp): e(label)[0]
        base = {s: err(f"SubspaceNet#s{s}") for s in SEEDS}
        base_mean = np.mean(list(base.values()), axis=0)
        for cfg in [*CONFIGS, FC]:
            runs = {s: err(f"{cfg}#s{s}") for s in SEEDS}
            for i in SEEDS:
                for j in SEEDS:
                    p = wilcoxon(runs[i], base[j]).pvalue
                    pair_rows.append(dict(experiment=exp, config=cfg, seed=i, ref_seed=j,
                                          diff_deg=runs[i].mean() - base[j].mean(), p=p))
                    if j != 0 or cfg == FC:  # reference seed 0 is already in the E*.csv family, except for FC
                        tests.append((exp, "table", f"{cfg}#s{i}", f"SubspaceNet_s{j}", p))
            # One test per configuration on the per-sample errors averaged over the seeds
            d = np.mean(list(runs.values()), axis=0) - base_mean
            r = rng_fc if cfg == FC else rng  # own stream, so the intervals of the other configurations do not move
            boot = [d[r.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            p = wilcoxon(d).pvalue
            pair_rows.append(dict(experiment=exp, config=cfg, seed="mean", ref_seed="mean", diff_deg=d.mean(),
                                  ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), p=p))
            tests.append((exp, "table", f"{cfg} (seed-averaged)", "SubspaceNet (seed-averaged)", p))
    return pair_rows, tests


def fc_tests():
    """SubspaceNet-FC against each reference at every point, on errors averaged over the seeds."""
    rows, tests = [], []
    rng = np.random.default_rng(1)
    for exp in [f"E{k}" for k in (*range(1, 11), 12)]:
        err = errors(exp)
        mean = lambda cfg: np.mean([err(f"{cfg}#s{s}") for s in SEEDS], axis=0)
        values = [r["value"] for r in csv.DictReader(open(OUT / f"{exp}.csv")) if r["method"] == "ESPRIT"]
        fc = mean(FC)
        assert fc.shape[0] == len(values), f"{exp}: {fc.shape[0]} points for {len(values)} values"
        for ref in ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter", "ESPRIT"):
            other = err("ESPRIT") if ref == "ESPRIT" else mean(ref)
            for i, value in enumerate(values):
                d = fc[i] - other[i]
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
                p = wilcoxon(d).pvalue
                rows.append(dict(experiment=exp, value=value, reference=ref, fc_deg=fc[i].mean(), ref_deg=other[i].mean(),
                                 diff_deg=d.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), p=p))
                if not (ref == "SubspaceNet" and exp in ("E10", "E12")):  # that test is already in seed_tests
                    tests.append((exp, value, f"{FC} (seed-averaged)", ref if ref == "ESPRIT" else f"{ref} (seed-averaged)", p))
    return rows, tests


def main():
    pair_rows, new_tests = seed_tests()
    fc_rows, fc_new = fc_tests()
    tests = csv_tests() + new_tests + fc_new
    p = np.array([t[4] for t in tests])
    p_bh = false_discovery_control(p, method="bh")
    adjusted = {t[:4]: q for t, q in zip(tests, p_bh)}
    with open(OUT / "X13_fdr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "value", "method", "reference", "p_raw", "p_bh"])
        w.writerows([*t, q] for t, q in zip(tests, p_bh))
    for r in pair_rows:
        key = ((r["experiment"], "table", f"{r['config']} (seed-averaged)", "SubspaceNet (seed-averaged)")
               if r["seed"] == "mean" else
               (r["experiment"], "table", f"{r['config']}#s{r['seed']}", f"SubspaceNet_s{r['ref_seed']}"))
        r["p_bh"] = adjusted[key]
    with open(OUT / "X13_seed_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "config", "seed", "ref_seed", "diff_deg", "ci_lo", "ci_hi", "p", "p_bh"])
        w.writeheader()
        w.writerows(pair_rows)

    for r in fc_rows:
        ref = r["reference"] if r["reference"] == "ESPRIT" else f"{r['reference']} (seed-averaged)"
        value = "table" if ref == "SubspaceNet (seed-averaged)" and r["experiment"] in ("E10", "E12") else r["value"]
        r["p_bh"] = adjusted[(r["experiment"], value, f"{FC} (seed-averaged)", ref)]
    with open(OUT / "X13_fc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fc_rows[0]))
        w.writeheader()
        w.writerows(fc_rows)

    # A raw p-value below Q that BH no longer accepts is what the correction changes
    lost = [(t, q) for t, q in zip(tests, p_bh) if t[4] < Q <= q]
    print(f"{len(tests)} paired tests; significant at {Q}: {int((p < Q).sum())} raw, {int((p_bh < Q).sum())} after BH")
    print(f"tests that lose significance: {len(lost)}")
    for t, q in lost:
        print(f"   {t[0]} {t[1]:>6} {t[2]:32s} vs {t[3]:16s} p={t[4]:.3g} -> {q:.3g}")
    print("\nSeed-averaged paired tests (configuration - SubspaceNet), degrees:")
    for r in pair_rows:
        if r["seed"] == "mean":
            print(f"   {r['experiment']} {r['config']:20s} diff={r['diff_deg']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p']:.3g}  p_bh={r['p_bh']:.3g}")
    print("\nSeed pairs significant after BH (out of 9):")
    for exp in ("E10", "E12"):
        for cfg in [*CONFIGS, FC]:
            rows = [r for r in pair_rows if r["experiment"] == exp and r["config"] == cfg and r["seed"] != "mean"]
            better = sum(r["p_bh"] < Q and r["diff_deg"] < 0 for r in rows)
            worse = sum(r["p_bh"] < Q and r["diff_deg"] > 0 for r in rows)
            print(f"   {exp} {cfg:20s} better than SubspaceNet: {better}/9, worse: {worse}/9")
    print(f"\n{FC} against each reference, points where it is significantly better / worse after BH:")
    for exp in dict.fromkeys(r["experiment"] for r in fc_rows):
        parts = []
        for ref in ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter", "ESPRIT"):
            rows = [r for r in fc_rows if r["experiment"] == exp and r["reference"] == ref]
            parts.append(f"{ref} {sum(r['p_bh'] < Q and r['diff_deg'] < 0 for r in rows)}/"
                         f"{sum(r['p_bh'] < Q and r['diff_deg'] > 0 for r in rows)} of {len(rows)}")
        print(f"   {exp:4s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
