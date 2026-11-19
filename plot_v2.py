"""Phase 8: v2 sweep figures E1-E9 from data/simulations/results_v2/E*.csv.

Values are plotted exactly as stored in the CSVs (nothing recomputed or rounded):
    learned models (3 seeds)  line = "<model> (3 seeds)" mean_deg, band = +/- seed_std_deg
    DA-MUSIC, classical       line = mean_deg, error bars = bootstrap 95% CI (ci_lo, ci_hi)
    E9 panel (b)              target_mean_deg, seed-0 runs only (no 3-seed target aggregate in the CSV)
Plug-and-play rows (+R-MUSIC / +MUSIC) and the Toeplitz ablations are tables, not plotted here.

    python plot_v2.py   # writes data/simulations/results_v2/plots/E*.pdf and .png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluate_v2 import SWEEPS

RES = Path(__file__).resolve().parent / "data" / "simulations" / "results_v2"
OUT = RES / "plots"
# Palette validated with the dataviz validator (light surface, CVD-safe); markers carry identity too
LEARNED = {  # model: (label, colour, marker)
    # SubspaceNet-FC rows exist only in the revision assets (paper_assets.py); its colour passes the
    # validator against every other learned series (all pairs, light surface)
    "SubspaceNet-FC": ("SubspaceNet-FC (proposed)", "#b8336a", "P"),
    "SubspaceNetIter": ("Iter (lag variant)", "#2a78d6", "o"),
    "SubspaceNetAttn": ("Attn (lag variant)", "#eb6834", "s"),
    "SubspaceNet": ("SubspaceNet", "#1baf7a", "^"),
    "DA-MUSIC": ("DA-MUSIC", "#6250d6", "D"),
}
CLASSICAL = {  # method: (marker, linestyle), all in one neutral ink
    "ESPRIT": ("x", "-"),
    "R-MUSIC": ("+", "--"),
    "MUSIC": ("1", ":"),
    "SPS-ESPRIT": ("v", "-."),
    "SPS-R-MUSIC": ("<", (0, (5, 1, 1, 1))),
}
GRAY = "#52514e"
RANDOM_GUESS = {3: 32.17, 2: 37.58}  # measured in the audit (notes/02_code_audit.md)
XLABEL = {"snr": "SNR [dB]", "T": "Snapshots T", "bias": "Spacing error bound b [λ]",
          "sv_noise_var": "Steering-vector noise variance σ²$_{sv}$", "sir": "SIR [dB]"}


def title(name, param, overrides):
    o = {"M": 3, "T": 200, "signal_nature": "non-coherent", "snr": 10, "bias": 0.05, **overrides}
    parts = [f"M={o['M']}", o["signal_nature"]]
    parts += [f"T={o['T']}"] if param != "T" else []
    parts += [f"SNR {o['snr']} dB"] if param != "snr" else []
    parts += [(f"b≤{o['bias']}λ" if o["bias"] else "b=0")] if param != "bias" else []
    return f"{name}: " + ", ".join(parts)


def draw(ax, df, param, m, target=False):
    col = "target_mean_deg" if target else "mean_deg"
    for model, (label, colour, marker) in LEARNED.items():
        agg = model != "DA-MUSIC" and not target
        method = f"{model} (3 seeds)" if agg else f"{model}#s0"
        s = df[df.method == method].sort_values("value")
        if s.empty:
            continue
        assert len(s) == df.value.nunique(), (method, len(s))
        x, y = s.value.to_numpy(), s[col].to_numpy()
        # Iter drawn with a larger marker so it stays visible where it overlaps Attn
        ax.plot(x, y, color=colour, marker=marker, ms={"SubspaceNetIter": 8, "SubspaceNet-FC": 7}.get(model, 5),
                mec="white", mew=0.8, lw=2, label=label, zorder=4 if model == "SubspaceNet-FC" else 3)
        if agg:
            sd = s.seed_std_deg.to_numpy()
            ax.fill_between(x, y - sd, y + sd, color=colour, alpha=0.2, lw=0, zorder=2)
        elif not target:
            ax.errorbar(x, y, yerr=[y - s.ci_lo, s.ci_hi - y], fmt="none", ecolor=colour, capsize=2, lw=1)
    for method, (marker, ls) in CLASSICAL.items():
        s = df[df.method == method].sort_values("value")
        if s.empty:
            continue
        x, y = s.value.to_numpy(), s[col].to_numpy()
        ax.plot(x, y, color=GRAY, marker=marker, ms=7, lw=1.2, ls=ls, label=method, zorder=3)
        if not target:
            ax.errorbar(x, y, yerr=[y - s.ci_lo, s.ci_hi - y], fmt="none", ecolor=GRAY, capsize=2, lw=0.8)
    if not target:  # the measured random-guess level is for all M sources, not the 2 targets
        ax.axhline(RANDOM_GUESS[m], color=GRAY, lw=0.8, ls=(0, (1, 3)), label=f"random guess ({RANDOM_GUESS[m]}°)")
    ax.set_xlabel(XLABEL[param])
    if param == "T":
        ax.set_xscale("log")
        ax.set_xticks(SWEEPS["E6"][1], [str(t) for t in SWEEPS["E6"][1]])
    if param == "sir":
        ax.invert_xaxis()  # stronger interferer to the right
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})
    for name, (param, _, overrides) in SWEEPS.items():
        df = pd.read_csv(RES / f"{name}.csv")
        m = overrides.get("M", 3)
        panels = 2 if name == "E9" else 1
        fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 3.4), squeeze=False)
        draw(axes[0, 0], df, param, m)
        axes[0, 0].set_ylabel("Mean RMSPE [°]")
        if name == "E9":
            axes[0, 0].set_title("(a) all 3 sources")
            draw(axes[0, 1], df, param, m, target=True)
            axes[0, 1].set_title("(b) targets only (sources 1-2), learned: seed 0")
        handles, labels = [], []
        for ax in axes[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.suptitle(title(name, param, overrides), fontsize=9)
        ncol = 2 * panels
        rows = -(-len(labels) // ncol)
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)
        fig.set_figheight(3.4 + 0.18 * rows)
        fig.tight_layout(rect=(0, 0.045 * rows + 0.02, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"{name}: {len(labels)} series -> {OUT / name}.pdf/.png")


if __name__ == "__main__":
    main()
