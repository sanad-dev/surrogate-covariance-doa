"""X11 (reviewer 2, comments M6b and M7a): what the covariance passed to ESPRIT looks like
under a strong interferer.

For every SIR of E9, on the E9 test observations, and for the sample covariance and each
learned surrogate covariance:
    lam1..lam8      eigenvalues, largest first, as shares of the trace
    gap34_db        10 log10(lam3 / lam4), the signal/noise eigenvalue gap for M = 3
    frac_weak       share of the energy of the nominal steering vector of a weak target
                    that lies in the signal subspace (M dominant eigenvectors), mean of the
                    two weak targets
    frac_interferer same for the strong source
    weak_lost       share of weak targets with less than half their energy in that subspace

The nominal steering vector is used because it is the one ESPRIT assumes. With the spacing
error of the base point it differs from the true one, which lowers the fractions of the
sample covariance by the same amount at every SIR: read its trend, not its level.

    python x11_spectrum.py
Writes results_v2/X11_spectrum.csv.
"""
import csv

import numpy as np
import torch

import evaluate_v2 as ev
import eval_tagged
from src.data_handler import create_observations, model_features
from src.utils import set_unified_seed

N_SAMPLES = 2000
MODELS = ("SubspaceNet", "SubspaceNetAttn", "SubspaceNetIter", "SubspaceNetAttn/mlp48")
FC_RUNS = "SubspaceNetAttn_tz0_seed?_lr0.001abl1-mlp48"  # SubspaceNet-FC: single-lag dense encoder (X15)
COLS = [f"lam{k}" for k in range(1, 9)] + ["gap34_db", "frac_weak", "frac_interferer", "weak_lost"]


def diagnose(R, doa, M):
    """R: [n, N, N] complex covariances, doa: [n, M] true angles in rad (last = interferer)."""
    lam, U = np.linalg.eigh(R)                     # ascending
    lam, U = lam[:, ::-1], U[:, :, ::-1]
    share = lam / lam.sum(axis=1, keepdims=True)
    N = R.shape[1]
    a = np.exp(-1j * np.pi * np.arange(N)[None, :, None] * np.sin(doa)[:, None, :])  # [n, N, M]
    proj = np.einsum("nik,nim->nkm", U[:, :, :M].conj(), a)                          # Us^H a
    frac = (np.abs(proj) ** 2).sum(axis=1) / N                                        # [n, M]
    weak = frac[:, : M - 1]
    return [*share.mean(axis=0), float(np.mean(10 * np.log10(lam[:, M - 1] / lam[:, M]))),
            weak.mean(), frac[:, M - 1].mean(), (weak < 0.5).mean()]


def main():
    runs = {k: v for k, v in ev.load_runs().items() if k.split("#")[0] in MODELS}
    runs.update(eval_tagged.load([FC_RUNS]))
    _, values, overrides = ev.SWEEPS["E9"]
    rows = []
    for i, v in enumerate(values):
        smp = ev.system_params(**{**ev.BASE, **overrides, "source_power_db": [0, 0, -v]})
        set_unified_seed(10_000 * 10 + i)  # the E9 seed of evaluate_v2.run_experiment
        obs, _ = create_observations(smp, N_SAMPLES)
        doa = np.stack([d.numpy() for _, d in obs])
        X = torch.stack([x for x, _ in obs]).numpy()
        covs = {("sample covariance", ""): np.einsum("nit,njt->nij", X, X.conj()) / X.shape[2]}
        for label, (cfg, model) in runs.items():
            model.M = smp.M
            x = torch.stack([f for f, _ in model_features(obs, cfg["model"], tau=cfg.get("tau", 8))])
            covs[(label.split("#")[0], cfg["seed"])] = ev.batched(model, x)[1].numpy()
        for (method, seed), R in covs.items():
            rows.append(dict(sir=v, method=method, seed=seed, **dict(zip(COLS, diagnose(R, doa, smp.M)))))
    with open(ev.OUT / "X11_spectrum.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sir", "method", "seed", *COLS])
        w.writeheader()
        w.writerows(rows)

    print(f"{'SIR':>4} {'method':22s} lam1  lam2  lam3  lam4-8 gap34[dB] frac_weak (seed std) frac_interf weak_lost")
    for v in values:
        for method in ("sample covariance", *MODELS):
            r = [x for x in rows if x["sir"] == v and x["method"] == method]
            m = lambda c: np.mean([x[c] for x in r])
            noise = np.mean([sum(x[f"lam{k}"] for k in range(4, 9)) for x in r])
            print(f"{v:4d} {method:22s} {m('lam1'):.3f} {m('lam2'):.3f} {m('lam3'):.3f} {noise:.3f}  {m('gap34_db'):7.2f}"
                  f"     {m('frac_weak'):.3f} ({np.std([x['frac_weak'] for x in r]):.3f})       {m('frac_interferer'):.3f}"
                  f"      {m('weak_lost'):.3f}")


if __name__ == "__main__":
    main()
