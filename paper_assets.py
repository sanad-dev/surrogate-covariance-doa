"""Paper assets for the revised manuscript: Figures 2-6 and Tables 1-6, with SubspaceNet-FC.
Figure 1 (Fig1.jpg) is the pipeline image supplied by the authors and is not generated here.

Every number is read from data/simulations/results_v2/E*.csv (submitted evaluation) and from
results_v2/fc/E*.csv (SubspaceNet-FC, written by fc_eformat.py); nothing is recomputed.
The assets of the submitted version stay in results_v2/paper/.
Tables print values to 3 decimals (display only; the CSVs keep full precision).
Figures are 180 mm wide (Frontiers two-column width), 300 dpi.

    python paper_assets.py   # writes data/simulations/results_v2/paper_rev/
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluate_v2 import SWEEPS
from plot_v2 import RES, draw, title
from train_v2 import DATA_SEED, TAU, TRAIN_PARAMS

OUT = RES / "paper_rev"
FC = "SubspaceNet-FC"
MM = 1 / 25.4
WIDTH = 180 * MM  # Frontiers two-column width
CONFIGS = [(FC, "SubspaceNet-FC (proposed)"), ("SubspaceNetIter", "Iter (lag variant)"),
           ("SubspaceNetAttn", "Attn (lag variant)"), ("SubspaceNet", "SubspaceNet")]
CLASSICAL = ["ESPRIT", "R-MUSIC", "MUSIC"]


def csv(name):
    """Submitted results plus the SubspaceNet-FC rows of fc_eformat.py."""
    fc = pd.read_csv(RES / "fc" / f"{name}.csv")
    return pd.concat([pd.read_csv(RES / f"{name}.csv"), fc[fc.method.str.startswith(FC)]], ignore_index=True)


def row(df, method):
    r = df[df.method == method]
    assert len(r) == 1, (method, len(r))
    return r.iloc[0]


def f3(x):
    return f"{x:.3f}"


def pval(p):
    return "--" if pd.isna(p) else ("$<10^{-4}$" if p < 1e-4 else f"{p:.3g}")


# ---------------------------------------------------------------- figures
def legend_below(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)
    return -(-len(labels) // ncol)


def composite(name, panels, target_panel=False):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 75 * MM))
    for k, (ax, exp) in enumerate(zip(axes, panels)):
        param, _, overrides = SWEEPS[exp]
        target = target_panel and k == 1
        draw(ax, csv(exp), param, overrides.get("M", 3), target=target)
        letter = "AB"[k]
        sub = "targets only (sources 1-2), learned: seed 0" if target else title(exp, param, overrides)
        if target_panel and k == 0:
            sub = "all 3 sources; " + title(exp, param, overrides)
        ax.set_title(f"({letter}) {sub}", fontsize=7)
    axes[0].set_ylabel("Mean RMSPE [°]")
    rows = legend_below(fig, axes)
    fig.set_figheight(75 * MM + rows * 4.2 * MM)
    fig.tight_layout(rect=(0, rows * 0.045 + 0.01, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def tabular(cols, header, body, caption, label, note=None):
    """Rows given as a string (e.g. MIDRULE) are written as is; `note` is printed below the table."""
    lines = [r"\begin{table}[h!]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in body]
    lines += [r"\hline", r"\end{tabular}"]
    if note:
        lines += [rf"\par\smallskip\parbox{{0.95\textwidth}}{{\footnotesize {note}}}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


MIDRULE = r"\midrule"
SHOW = {"R-MUSIC": "Root-MUSIC"}  # display names of CSV method keys


def table1():
    t = TRAIN_PARAMS
    sweep_lines = {"E1": "SNR, non-coherent, $T=200$", "E2": "SNR, non-coherent, $T=2$",
                   "E3": "SNR, coherent, $T=200$", "E4": "SNR, coherent, $T=20$", "E5": "SNR, coherent, $T=2$",
                   "E6": "$T$, coherent, $M=2$", "E7": "spacing error $b$", "E8": r"steering-vector noise $\sigma^2_{sv}$ ($b=0$)",
                   "E9": "SIR, one strong interferer"}
    body = [["Array", f"ULA, $N={t['N']}$, $d=\\lambda/2$"],
            ["Training sources", f"$M={t['M']}$, {t['signal_nature']}, equal powers"],
            ["Training snapshots / SNR", f"$T={t['T']}$; SNR $\\sim\\mathcal{{U}}[{t['snr'][0]},{t['snr'][1]}]$ dB (true SNR)"],
            ["Spacing error", f"$b\\sim\\mathcal{{U}}(\\pm{t['bias']})\\lambda$, one per array realization"],
            ["Training data", f"100,000 samples (90k / 10k), data seed {DATA_SEED}"],
            ["Autocorrelation lags", f"$\\tau=0,\\dots,{TAU - 1}$; SubspaceNet-FC: lag 0 only"],
            ["Optimizer", "Adam, lr $10^{-3}$, batch 2048, early stopping (patience 15)"],
            ["Seeds", "0, 1, 2"],
            ["Test samples", "2,000 per sweep point; 5,000 for E10 and E12"]]
    body += [MIDRULE, r"\textit{Experiment} & \textit{Swept parameter: values} \\", MIDRULE]
    body += [[e, f"{sweep_lines[e]}: ${', '.join(str(v) for v in SWEEPS[e][1])}$"] for e in SWEEPS]  # math mode: true minus signs
    body += [["E10 / E12", "base point ($b\\le0.05\\lambda$) / calibrated array ($b=0$), SNR 10 dB, $T=200$"],
             ["E11", "Toeplitz-regularizer ablation (inside E10)"], ["E13", "parameters and latency (batch 1)"]]
    return tabular("ll", ["Item", "Setting"], body, "Simulation, training and evaluation settings.", "tab:settings")


def seed_table(exp, caption, label):
    df = csv(exp)
    body = []
    for model, name in CONFIGS:
        for suffix, tag in (("", ""), ("+T", " + Toeplitz")):
            if model == FC and suffix:
                continue  # no Toeplitz run of SubspaceNet-FC
            agg = row(df, f"{model}{suffix} (3 seeds)")
            seeds = [row(df, f"{model}{suffix}#s{s}") for s in range(3)]
            body.append([name + tag, f"{f3(agg.mean_deg)} ({f3(agg.seed_std_deg)})",
                         " / ".join(f3(s.mean_deg) for s in seeds), " / ".join(pval(s.p_vs_SubspaceNet_s0) for s in seeds)])
    for m in ["DA-MUSIC#s0", *CLASSICAL]:
        r = row(df, m)
        if m == "DA-MUSIC#s0":
            body.append(MIDRULE)
        body.append([SHOW.get(m, m).replace("#s0", " (seed 0)"), f"{f3(r.mean_deg)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]", "--",
                     pval(r.p_vs_SubspaceNet_s0)])
    header = ["Method", r"RMSPE [$^\circ$]", r"Seeds 0 / 1 / 2 [$^\circ$]", "$p$ vs.\\ SubspaceNet (seed 0)"]
    note = (r"Learned models: mean (standard deviation) over three training seeds; other methods: 95\% bootstrap "
            r"confidence interval. 5,000 test samples. $p$: two-sided Wilcoxon signed-rank test on the per-sample errors.")
    return tabular("llll", header, body, caption, label, note)


def table4():
    df = csv("E10")
    body = []
    for model, name in CONFIGS:
        cells = [name]
        for est in ("", "+R-MUSIC", "+MUSIC"):
            r = row(df, f"{model}#s0{est}")
            cells.append(f3(r.mean_deg) + (f" ({100 * r.fail_rate:.1f}\\%)" if r.fail_rate > 0 else ""))
        body.append(cells)
    body += [MIDRULE, ["Sample covariance"] + [f3(row(df, m).mean_deg) for m in CLASSICAL]]
    header = ["Covariance", "ESPRIT", "Root-MUSIC", "MUSIC"]
    return tabular("llll", header, body, "Transfer of the learned covariances to other estimators (E10, seed 0).", "tab:pnp",
                   r"Mean RMSPE [$^\circ$]. In parentheses: share of samples in which MUSIC misses a peak, when nonzero. "
                   r"All learned models were trained through ESPRIT.")


def table5():
    df = pd.read_csv(RES / "fc" / "E13.csv")  # every row timed in one session: latency depends on the machine state
    names = dict(CONFIGS, **SHOW)
    body = []
    for m in [f"{c}#s0" for c, _ in CONFIGS] + ["DA-MUSIC#s0", *CLASSICAL]:
        r = row(df, m)
        if m == CLASSICAL[0]:
            body.append(MIDRULE)
        key = m.replace("#s0", "")
        body.append([names.get(key, key), f"{int(r.n_parameters):,}" if r.n_parameters else "--",
                     "--" if pd.isna(r.ms_cuda) else f3(r.ms_cuda), f3(r.ms_cpu)])
    return tabular("lrrr", ["Method", "Parameters", "GPU [ms]", "CPU [ms]"], body,
                   "Model size and per-sample latency (E13).", "tab:complexity",
                   "Latency at batch size 1. The classical methods were timed on CPU only.")


def table6():
    body = [["TransMUSIC (Ji et al.)", "snapshots", "noise subspace $\\to$ MUSIC + NN peak finder", "non-coherent, 1-bit ADC"],
            ["Dkhan et al. (2026)", "autocorrelation lags", "residual correction $dR$ $\\to$ Root-MUSIC", "THz near field, coherent"],
            ["SubspaceNet / NF-SubspaceNet", "autocorrelation lags (CNN)", "surrogate covariance $\\to$ (Root-)MUSIC / ESPRIT", "far / near field"],
            ["Zohar \\& Shlezinger (2026)", "as SubspaceNet", "surrogate covariance $\\to$ ESPRIT + uncertainty", "coherent, few snapshots"],
            ["Barthelme \\& Utschick; Jiang et al.; Wu et al.", "sample covariance(s)", "supervised covariance reconstruction $\\to$ MUSIC", "subarrays, coupling, sparse arrays"],
            ["Thattukolla (2026); Yeganegi et al. (2024)", "covariance / one-bit data", "unrolled sparse recovery (LISTA)", "coupling, one-bit"],
            ["\\textbf{SubspaceNet-FC (this work)}", "sample covariance (lag 0), fully connected", "full surrogate $K^HK+\\epsilon I$ $\\to$ differentiable ESPRIT", "far field, spacing error, coherent, few snapshots"],
            ["Attn / Iter (this work, lag variants)", "autocorrelation lags", "as SubspaceNet-FC", "as SubspaceNet-FC"]]
    cols = r">{\raggedright}p{3.9cm}>{\raggedright}p{2.6cm}>{\raggedright}p{4.6cm}>{\raggedright\arraybackslash}p{3.6cm}"
    return tabular(cols, ["Work", "Attention / input over", "Output $\\to$ estimator", "Setting"], body,
                   "Comparison with the closest related designs.", "tab:positioning")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6.5})
    for name, panels in {"Fig2": ("E1", "E2"), "Fig3": ("E3", "E4"), "Fig4": ("E5", "E6"), "Fig5": ("E7", "E8")}.items():
        composite(name, panels)
    composite("Fig6", ("E9", "E9"), target_panel=True)
    tables = {"Table1": table1(),
              "Table2": seed_table("E10", "Base point (E10) and Toeplitz ablation (E11).", "tab:e10"),
              "Table3": seed_table("E12", r"Calibrated array, $b=0$ (E12).", "tab:e12"),
              "Table4": table4(), "Table5": table5(), "Table6": table6()}
    for name, tex in tables.items():
        (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
