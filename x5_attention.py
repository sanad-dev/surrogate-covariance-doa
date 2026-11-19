"""X5 (reviewer 2, M1b and M1c): attention weights of the submitted Attn and Iter runs.

Attn: weight each lag receives in the encoder self-attention (per layer, mean over heads and
query tokens) and in the attention pooling. Iter: cross-attention over the lags at each
refinement step. Four conditions at SNR 10 dB: non-coherent / coherent x T = 200 / T = 2.

    python x5_attention.py

Writes results_v2/X5_attention.csv: mean weight per lag over samples and seeds, its spread
between samples (does the weighting depend on the input?) and between seeds.
"""
import csv

import numpy as np
import torch

import evaluate_v2 as ev
from src.data_handler import create_observations, model_features
from src.utils import set_unified_seed
from train_v2 import system_params

CONDITIONS = {f"{nature}, T = {T}": dict(signal_nature=nature, T=T)
              for nature in ("non-coherent", "coherent") for T in (200, 2)}
N_SAMPLES = 2000


def record(module, store, key):
    """Makes an nn.MultiheadAttention keep its weights, mean over heads: [batch, queries, lags]."""
    forward = module.forward

    def wrapped(*args, **kwargs):
        kwargs.update(need_weights=True, average_attn_weights=True)
        out, weights = forward(*args, **kwargs)
        store.setdefault(key, []).append(weights.detach().cpu())
        return out, weights

    module.forward = wrapped


def attention(model, kind, x):
    """Returns {stage: weights [samples, lags]}; encoder weights are averaged over the query tokens."""
    store = {}
    if kind == "SubspaceNetAttn":
        hooked = [layer.self_attn for layer in model.transformer.layers] + [model.pool_attn]
        for i, layer in enumerate(model.transformer.layers):
            record(layer.self_attn, store, f"encoder layer {i + 1}")
        record(model.pool_attn, store, "pooling")
    else:
        hooked = [model.cross_attn]
        record(model.cross_attn, store, "step")
    # Gradients stay enabled: under no_grad the encoder takes a fused path that never calls self_attn
    for i in range(0, len(x), 256):
        model(x[i:i + 256].to(ev.device))
    for module in hooked:
        del module.forward  # back to the class method
    if kind == "SubspaceNetIter":  # one call per refinement step, in order
        n = model.n_iterations
        store = {f"step {k + 1}": store["step"][k::n] for k in range(n)}
    out = {stage: torch.cat(w).mean(dim=1).numpy() for stage, w in store.items()}
    expected = len(model.transformer.layers) + 1 if kind == "SubspaceNetAttn" else model.n_iterations
    assert len(out) == expected, f"{kind}: stages recorded {list(out)}"
    assert all(w.shape == (len(x), model.tau) and np.allclose(w.sum(1), 1, atol=1e-4) for w in out.values())
    return out


def main():
    runs = {k: v for k, v in ev.load_runs().items() if k.split("#")[0] in ("SubspaceNetAttn", "SubspaceNetIter")}
    rows = []
    for i, (condition, overrides) in enumerate(CONDITIONS.items()):
        set_unified_seed(50_000 + i)
        obs, _ = create_observations(system_params(**{**ev.BASE, **overrides}), N_SAMPLES)
        per_seed = {}  # (model, stage): [seeds, samples, lags]
        for label, (cfg, model) in runs.items():
            x = torch.stack([f for f, _ in model_features(obs, cfg["model"], tau=ev.TAU)])
            for stage, w in attention(model, cfg["model"], x).items():
                per_seed.setdefault((cfg["model"], stage), []).append(w)
        for (kind, stage), w in per_seed.items():
            w = np.array(w)
            mean, sample_std, seed_std = w.mean((0, 1)), w.std(1).mean(0), w.mean(1).std(0)
            rows += [dict(condition=condition, model=kind, stage=stage, n_seeds=len(w), lag=lag, weight=mean[lag],
                          sample_std=sample_std[lag], seed_std=seed_std[lag]) for lag in range(w.shape[2])]
            print(f"{condition:22s} {kind:16s} {stage:16s} " + " ".join(f"{v:.3f}" for v in mean)
                  + f" | sample std {sample_std.max():.3f} seed std {seed_std.max():.3f}")
    with open(ev.OUT / "X5_attention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
