"""SubspaceNet-FC (the X15 runs) evaluated with the code of the submitted evaluation, so that
paper_assets.py reads it like any other method.

Writes results_v2/fc/E*.csv, with the columns of results_v2/E*.csv (seed rows, 3-seed aggregates,
transfer of the seed-0 covariance to Root-MUSIC and MUSIC, E9 target error), on the same test
observations, and results_v2/fc/E13.csv (parameters and latency of every seed-0 model, timed in
one session). SubspaceNet seed 0 is evaluated again because the p-values refer to it.

    python fc_eformat.py
"""
import numpy as np
import pandas as pd

import eval_tagged
import evaluate_v2 as ev

FC = "SubspaceNet-FC"
FC_RUNS = "SubspaceNetAttn_tz0_seed?_lr0.001abl1-mlp48"


def main():
    submitted = ev.load_runs()
    fc = {k.replace("SubspaceNetAttn/mlp48", FC): v for k, v in eval_tagged.load([FC_RUNS]).items()}
    assert len(fc) == 3, list(fc)
    ev.TAU = 8  # eval_tagged.load set it to the single lag of FC; each run now uses the tau of its config
    results = ev.OUT
    ev.OUT = results / "fc"
    runs = {"SubspaceNet#s0": submitted["SubspaceNet#s0"], **fc}
    for name, (param, values, overrides) in ev.SWEEPS.items():
        points = [(v, {**overrides, **({"source_power_db": [0, 0, -v]} if param == "sir" else {param: v})})
                  for v in values]
        ev.run_experiment(name, runs, points, 2000, sweep_only=False, augment=True)
    for name, overrides in ev.TABLES.items():
        ev.run_experiment(name, runs, [("table", overrides)], 5000, sweep_only=False, augment=True)
    ev.complexity({**{k: v for k, v in submitted.items() if "+T" not in k}, **fc})

    # Same test data and same models as the files already on disk, or the merged tables mean nothing
    for name in [*ev.SWEEPS, *ev.TABLES]:
        new, old = pd.read_csv(ev.OUT / f"{name}.csv"), pd.read_csv(results / f"{name}.csv")
        for method in ("ESPRIT", "SubspaceNet#s0"):
            a, b = (df[df.method == method].mean_deg.to_numpy() for df in (new, old))
            assert np.allclose(a, b, atol=1e-6), f"{name} {method}: differs from the submitted evaluation"
    print("ESPRIT and SubspaceNet seed 0 match the submitted evaluation in every experiment")


if __name__ == "__main__":
    main()
