"""Three candidate pipeline diagrams (replacements for Figure 1) with blocks, tensor shapes and
hyperparameters taken from src/models.py (SubspaceNetAttn, SubspaceNetIter, SubspaceNet), train_v2.py
and evaluate_v2.py. Nothing here is measured; every number is a configuration value.

figure1() is Figure 1 of the revision: the same data flow with SubspaceNet-FC as the named encoder
(variant mlp48 of SubspaceNetAttn, one lag) and Attn / Iter as lag variants.

    python pipeline_figures.py   # writes data/simulations/results_v2/paper/pipeline_variants/
                                 # and data/simulations/results_v2/paper_rev/Fig1.{pdf,png,tif}
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from plot_v2 import RES

OUT = RES / "paper" / "pipeline_variants"
MM = 1 / 25.4
W = 180 * MM
INK, GREY = "#2b2b2b", "#6b6a66"
C = {"io": ("#f2f1ee", "#52514e"), "cnn": ("#e8f6f0", "#1baf7a"), "attn": ("#fdeee7", "#eb6834"),
     "iter": ("#e8f0fb", "#2a78d6"), "est": ("#f3f0fb", "#6250d6"), "loss": ("#fff6e0", "#c98500"),
     "fc": ("#f9eaf0", "#b8336a")}


def box(ax, x, y, w, h, title, body="", kind="io", ts=6.4, bs=5.6, lw=0.9):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=ts, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.012 - ts * 0.0072, body, ha="center", va="top", fontsize=bs, color=INK, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts, weight="bold", color=INK, linespacing=1.2)


def arrow(ax, p, q, label="", dashed=False, color=GREY, lw=0.9, rad=0.0, ls=5.0, off=(0, 0.012)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                 ls=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", va="bottom",
                fontsize=ls, color=GREY, style="italic")


def canvas(h_mm):
    fig, ax = plt.subplots(figsize=(W, h_mm * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    return fig, ax


ATTN_STEPS = [
    ("Lag tokenizer", "Linear 128→128\nLayerNorm, GELU\n+ learned lag embedding"),
    ("Transformer encoder", "2 layers, 4 heads\nFFN 512, dropout 0.2\nself-attention over lags"),
    ("Attention pooling", "learned query (1×128)\n4-head attention\n8 tokens → 1 vector"),
    ("Decoder", "Linear 128→512, GELU\ndropout 0.2\nLinear 512→128 → K"),
]
ITER_STEPS = [
    ("Lag tokenizer", "Linear 128→128\nLayerNorm, GELU\n(no lag embedding)"),
    ("Initial state", "s₀ = Linear 128→128\n(mean of the 8 tokens)\nstate size 2N² = 128"),
    ("Refinement × 5 (shared)", "q = Linear(s)  →  4-head\ncross-attention on tokens\ns ← GRUCell(context, s)"),
    ("Decoder", "Linear 128→512, GELU\ndropout 0.1\nLinear 512→128 → K"),
]


def variant_a():
    """Top: end-to-end data flow with tensor shapes. Bottom: inside the two proposed encoders."""
    fig, ax = canvas(105)
    y, h = 0.73, 0.19
    top = [(0.000, 0.115, "Snapshots", "X ∈ ℂ⁸ˣᵀ\nN = 8 ULA\nT = 200 (train)", "io"),
           (0.135, 0.140, "Lag autocorr.", "R̂[τ], τ = 0…7\nRe/Im stacked\n8 × (16 × 8)", "io"),
           (0.295, 0.150, "Learned encoder", "SubspaceNet (CNN), or\nAttn / Iter (proposed)\ndetails below", "attn"),
           (0.465, 0.120, "Surrogate", "K ∈ ℂ⁸ˣ⁸\nR̃ = KᴴK + I\n(ε = 1)", "est"),
           (0.605, 0.125, "ESPRIT", "differentiable, M = 3\ndominant eigenvectors\nLS shift invariance", "est"),
           (0.750, 0.090, "DoA", "θ̂ (3 angles)", "io"),
           (0.860, 0.140, "Loss (training)", "periodic, permutation-\ninvariant RMSPE", "loss")]
    for x, w, t, b, k in top:
        box(ax, x, y, w, h, t, b, k, ts=6.2)
    for (x0, w0, *_), (x1, *__) in zip(top[:-1], top[1:]):
        arrow(ax, (x0 + w0, y + h / 2), (x1, y + h / 2))
    arrow(ax, (0.93, y - 0.035), (0.37, y - 0.035), dashed=True, color="#c98500")
    ax.text(0.65, y - 0.075, "training: gradients flow back through ESPRIT and both eigendecompositions",
            ha="center", fontsize=5.4, color=GREY, style="italic")
    arrow(ax, (0.93, y), (0.93, y - 0.035), dashed=True, color="#c98500")
    ax.text(0.5, 0.965, "Data flow (one sample)", ha="center", fontsize=7, weight="bold", color=INK)
    for name, steps, kind, yb in (("Attn (single pass, 612,224 parameters)", ATTN_STEPS, "attn", 0.345),
                                  ("Iter (iterative refinement, 346,624 parameters)", ITER_STEPS, "iter", 0.03)):
        ax.text(0.0, yb + 0.20, name, ha="left", fontsize=6.6, weight="bold", color=C[kind][1])
        xs, w, gap = 0.0, 0.225, 0.033
        for i, (t, b) in enumerate(steps):
            box(ax, xs + i * (w + gap), yb, w, 0.18, t, b, kind)
            if i:
                arrow(ax, (xs + i * (w + gap) - gap, yb + 0.09), (xs + i * (w + gap), yb + 0.09))
    return fig


FC_STEPS = [  # the mlp48 variant of SubspaceNetAttn with tau = 1: 40,400 parameters
    ("Input", "lag-0 sample covariance\nRe/Im stacked (16 × 8)\nflattened to 128 values"),
    ("Embedding", "Linear 128→48\nLayerNorm, GELU"),
    ("Hidden layer", "Linear 48→192\nGELU, dropout 0.2"),
    ("Output layer", "Linear 192→128\nreshaped to K ∈ ℂ⁸ˣ⁸"),
]


def panel(ax, x, y, w, h, kind, title, sub):
    """Rounded frame of one encoder, with its name in bold and its size below."""
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.7, alpha=0.55))
    ax.text(x + 0.012, y + h - 0.018, title, ha="left", va="top", fontsize=7.2, weight="bold", color=ec)
    ax.text(x + 0.012, y + h - 0.048, sub, ha="left", va="top", fontsize=5.8, color=INK)


DARK = {"io": "#52514e", "fc": "#9c2a59", "attn": "#b84a17", "iter": "#1f5fb0", "est": "#4b3bb0",
        "loss": "#9a6600"}  # title strips; white text on each keeps a contrast ratio above 4.5


def hbox(ax, x, y, w, h, title, body, kind, ts=6.8, bs=6.2, strip=0.042):
    """Block with a coloured title strip (white bold title) and the body text below it."""
    fc, ec = C[kind]
    frame = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01", fc=fc, ec=ec, lw=1.1)
    ax.add_patch(frame)
    band = plt.Rectangle((x - 0.01, y + h - strip), w + 0.02, strip + 0.01, fc=DARK[kind], ec="none")
    ax.add_patch(band)
    band.set_clip_path(frame)
    ax.text(x + w / 2, y + h - strip / 2, title, ha="center", va="center", fontsize=ts, weight="bold", color="white")
    ax.text(x + w / 2, y + (h - strip) / 2, body, ha="center", va="center", fontsize=bs, color=INK, linespacing=1.3)


def figure1():
    """Figure 1, high-legibility version of figure1_rows: same content, larger type, coloured title
    strips, a header line above each encoder row. Sizes from src/models.py."""
    fig, ax = canvas(128)
    ax.set_xlim(-0.004, 1.004)
    gold = "#c98500"
    A = dict(lw=1.2, mutation_scale=9)

    def arr(p, q, color=GREY):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=color, **A))

    def line(xs, ys, color=GREY, dashed=False):
        ax.plot(xs, ys, color=color, lw=1.2, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="round")

    # input
    hbox(ax, 0.004, 0.565, 0.12, 0.17, "Snapshots", "X ∈ ℂ⁸ˣᵀ\nN = 8 ULA\nT = 200 (train)", "io")
    hbox(ax, 0.136, 0.565, 0.12, 0.17, "Autocorrelation", "R̂[τ]\nRe/Im stacked\n16 × 8 per lag", "io", ts=6.4)
    arr((0.124, 0.65), (0.136, 0.65))
    # encoder rows
    ax.add_patch(FancyBboxPatch((0.282, 0.305), 0.714, 0.68, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="none", ec="#6250d6", lw=1.0, ls=(0, (2, 2))))
    ax.text(0.639, 0.985, "Learned encoder (one of the three)", ha="center", va="center", fontsize=7.4,
            weight="bold", color="#4b3bb0", bbox=dict(fc="white", ec="none", pad=1.5))
    rows = [
        (0.775, "fc", "SubspaceNet-FC", "proposed  ·  input τ = 0  ·  40,400 parameters", [
            ("Input", "lag-0 sample\ncovariance, Re/Im\n16 × 8 → 128 values"),
            ("Embedding", "Linear 128→48\nLayerNorm, GELU"),
            ("Hidden layer", "Linear 48→192\nGELU, dropout 0.2"),
            ("Output layer", "Linear 192→128\nreshaped to K")]),
        (0.565, "attn", "Attn", "lag variant  ·  input τ = 0…7  ·  612,224 parameters", [
            ("Lag tokenizer", "8 tokens:\nLinear 128→128\nLayerNorm, GELU\n+ learned lag embed."),
            ("Transformer", "2 layers, 4 heads\nFFN 512, dropout 0.2\nself-attention over lags"),
            ("Attention pooling", "learned query (1×128)\n4-head attention\n8 tokens → 1 vector"),
            ("Decoder", "Linear 128→512, GELU\ndropout 0.2\nLinear 512→128 → K")]),
        (0.355, "iter", "Iter", "lag variant  ·  input τ = 0…7  ·  346,624 parameters", [
            ("Lag tokenizer", "8 tokens:\nLinear 128→128\nLayerNorm, GELU\n(no lag embedding)"),
            ("Initial state", "s₀ = Linear 128→128\n(mean of the 8 tokens)\nstate size 2N² = 128"),
            ("Refinement ×5", "shared weights\nq = Linear(s), 4-head\ncross-attention\ns ← GRUCell(c, s)"),
            ("Decoder", "Linear 128→512, GELU\ndropout 0.1\nLinear 512→128 → K")])]
    rh, bw, gap, xs0, bus_in, bus_out = 0.165, 0.152, 0.014, 0.300, 0.27, 0.982
    x0s = [xs0 + i * (bw + gap) for i in range(4)]
    fig.canvas.draw()
    for y, kind, name, sub, steps in rows:
        ym = y + rh / 2
        t = ax.text(x0s[0], y + rh + 0.016, name, ha="left", va="center", fontsize=7.6, weight="bold", color=DARK[kind])
        bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        ax.text(bb.x1 + 0.01, y + rh + 0.016, sub, ha="left", va="center", fontsize=6.4, color=INK)
        arr((bus_in, ym), (x0s[0], ym))
        for i, (tt, b) in enumerate(steps):
            hbox(ax, x0s[i], y, bw, rh, tt, b, kind)
            if i:
                arr((x0s[i - 1] + bw, ym), (x0s[i], ym))
        line([x0s[-1] + bw, bus_out], [ym, ym])
    ym = [y + rh / 2 for y, *_ in rows]
    line([0.256, bus_in], [0.65, 0.65])
    line([bus_in, bus_in], [ym[0], ym[-1]])
    # subspace chain
    yb, hb = 0.07, 0.165
    chain = [(0.03, 0.15, "Surrogate matrix", "K ∈ ℂ⁸ˣ⁸\n(network output)", "est"),
             (0.2, 0.18, "Surrogate covariance", "R̃ = KᴴK + εI\nε = 1", "est"),
             (0.402, 0.215, "ESPRIT (differentiable)", "M = 3 dominant eigenvectors\n→ least-squares shift\ninvariance, Eq. (5)", "est"),
             (0.638, 0.105, "DoA", "θ̂\n(3 angles)", "io"),
             (0.765, 0.195, "Training loss", "periodic, permutation-\ninvariant RMSPE", "loss")]
    for x, w, tt, b, k in chain:
        hbox(ax, x, yb, w, hb, tt, b, k, ts=6.5)
    for (x0, w0, *_), (x1, *__) in zip(chain[:-1], chain[1:]):
        arr((x0 + w0, yb + hb / 2), (x1, yb + hb / 2))
    line([bus_out, bus_out, 0.105], [ym[0], 0.27, 0.27])
    arr((0.105, 0.27), (0.105, yb + hb))
    # training path back to the encoder weights
    line([0.865, 0.865, 0.012, 0.012], [yb, 0.024, 0.024, 0.33], color=gold, dashed=True)
    arr((0.012, 0.33), (0.282, 0.33), color=gold)
    ax.text(0.43, 0.03, "training: gradients flow back through ESPRIT and both eigendecompositions into the encoder",
            ha="center", va="bottom", fontsize=6.2, color="#7a5300", style="italic")
    return fig


def figure1_rows():
    """Figure 1 of 17 Nov 2026, laid out after the authors' sketch revision/figure11.jpeg: input on the
    left, the three encoders as parallel rows, ESPRIT chain at the bottom. Sizes from src/models.py."""
    fig, ax = canvas(108)
    ax.set_xlim(-0.004, 1.004)
    gold = "#c98500"
    # input
    box(ax, 0.005, 0.585, 0.118, 0.16, "Snapshots", "X ∈ ℂ⁸ˣᵀ\nN = 8 ULA\nT = 200 (train)", "io", ts=6.0, bs=5.4)
    box(ax, 0.138, 0.585, 0.118, 0.16, "Autocorrelation", "R̂[τ]\nRe/Im stacked\n16 × 8 per lag", "io", ts=6.0, bs=5.4)
    arrow(ax, (0.123, 0.665), (0.138, 0.665))
    # the three encoders, one row each
    ax.add_patch(FancyBboxPatch((0.285, 0.31), 0.71, 0.68, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="none", ec="#6250d6", lw=0.8, ls=(0, (2, 2))))
    ax.text(0.64, 0.982, "Learned encoder (one of three)", ha="center", va="center", fontsize=6.6,
            weight="bold", color="#6250d6", bbox=dict(fc="white", ec="none", pad=1.5))
    rows = [
        (0.78, "fc", "SubspaceNet-FC", "τ = 0", "proposed\n40,400 parameters", [
            ("Input", "lag-0 sample\ncovariance, Re/Im\nflattened to 128"),
            ("Embedding", "Linear 128→48\nLayerNorm, GELU"),
            ("Hidden layer", "Linear 48→192\nGELU, dropout 0.2"),
            ("Output layer", "Linear 192→128\nreshaped to K")]),
        (0.585, "attn", "Attn", "τ = 0…7", "lag variant\n612,224 parameters", [
            ("Lag tokenizer", "8 tokens: Linear\n128→128, LayerNorm,\nGELU + lag embedding"),
            ("Transformer", "2 layers, 4 heads\nFFN 512, dropout 0.2\nself-attention"),
            ("Attention pooling", "learned query\n4-head attention\n8 tokens → 1 vector"),
            ("Decoder", "Linear 128→512, GELU\ndropout 0.2\nLinear 512→128 → K")]),
        (0.39, "iter", "Iter", "τ = 0…7", "lag variant\n346,624 parameters", [
            ("Lag tokenizer", "8 tokens: Linear\n128→128, LayerNorm,\nGELU (no lag emb.)"),
            ("Initial state", "s₀ = Linear 128→128\n(mean of tokens)\nstate size 2N² = 128"),
            ("Refinement ×5", "shared weights\nq = Linear(s), 4-head\ncross-attention\ns ← GRUCell(c, s)"),
            ("Decoder", "Linear 128→512, GELU\ndropout 0.1\nLinear 512→128 → K")])]
    rh, bw, x0s, bus_in, bus_out = 0.16, 0.125, (0.418, 0.555, 0.692, 0.829), 0.272, 0.978
    for y, kind, name, sub, size, steps in rows:
        ym = y + rh / 2
        ax.text(0.292, ym + 0.05, name, ha="left", va="center", fontsize=5.8, weight="bold", color=C[kind][1])
        ax.text(0.292, ym + 0.018, sub, ha="left", va="center", fontsize=5.2, color=INK)
        ax.text(0.292, ym - 0.012, size, ha="left", va="top", fontsize=5.2, color=GREY, linespacing=1.3)
        arrow(ax, (bus_in, ym), (x0s[0], ym))
        for i, (t, b) in enumerate(steps):
            box(ax, x0s[i], y, bw, rh, t, b, kind, ts=5.8, bs=5.2)
            if i:
                arrow(ax, (x0s[i - 1] + bw, ym), (x0s[i], ym))
        ax.plot([x0s[-1] + bw, bus_out], [ym, ym], color=GREY, lw=0.9)
    ym = [y + rh / 2 for y, *_ in rows]
    ax.plot([0.256, bus_in], [0.665, 0.665], color=GREY, lw=0.9)
    ax.plot([bus_in, bus_in], [ym[0], ym[-1]], color=GREY, lw=0.9)
    # from the encoder output K to the subspace chain
    yb, hb = 0.07, 0.15
    chain = [(0.05, 0.13, "Surrogate matrix", "K ∈ ℂ⁸ˣ⁸\n(network output)", "est"),
             (0.21, 0.15, "Surrogate covariance", "R̃ = KᴴK + εI\nε = 1", "est"),
             (0.39, 0.22, "ESPRIT (differentiable)", "M = 3 dominant eigenvectors\n→ least-squares shift\ninvariance, Eq. (5)", "est"),
             (0.64, 0.10, "DoA", "θ̂ (3 angles)", "io"),
             (0.77, 0.18, "Training loss", "periodic, permutation-\ninvariant RMSPE", "loss")]
    for x, w, t, b, k in chain:
        box(ax, x, yb, w, hb, t, b, k, ts=6.0, bs=5.4)
    for (x0, w0, *_), (x1, *__) in zip(chain[:-1], chain[1:]):
        arrow(ax, (x0 + w0, yb + hb / 2), (x1, yb + hb / 2))
    ax.plot([bus_out, bus_out, 0.115], [ym[0], 0.265, 0.265], color=GREY, lw=0.9)
    arrow(ax, (0.115, 0.265), (0.115, yb + hb))
    # training: the gradient reaches the encoder weights
    ax.plot([0.86, 0.86, 0.012, 0.012], [yb, 0.025, 0.025, 0.35], color=gold, lw=0.9, ls=(0, (3, 2)))
    arrow(ax, (0.012, 0.35), (0.285, 0.35), color=gold)
    ax.text(0.44, 0.031, "training: gradients flow back through ESPRIT and both eigendecompositions into the encoder",
            ha="center", va="bottom", fontsize=5.4, color=GREY, style="italic")
    return fig


def figure1_columns():
    """Figure 1 of 17 Nov 2026 (morning), laid out after revision/figure1.jpg: data flow on
    top, then the three encoders side by side. Every block and size is taken from src/models.py."""
    fig, ax = canvas(118)
    ax.set_xlim(-0.006, 1.006)
    # top: data flow
    ax.add_patch(FancyBboxPatch((0.0, 0.695), 1.0, 0.295, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc="#f7f6f3", ec="#9a9893", lw=0.6))
    ax.text(0.012, 0.975, "Data flow (one sample)", ha="left", va="top", fontsize=7.2, weight="bold", color=INK)
    y, h = 0.765, 0.15
    top = [(0.012, 0.108, "Snapshots", "X ∈ ℂ⁸ˣᵀ\nN = 8 ULA\nT = 200 (train)", "io"),
           (0.135, 0.150, "Autocorrelation", "R̂[τ], Re/Im stacked\nSubspaceNet-FC: τ = 0\nAttn, Iter: τ = 0…7", "io"),
           (0.300, 0.158, "Learned encoder", "SubspaceNet-FC (proposed)\nor Attn / Iter\n(lag variants)", "fc"),
           (0.473, 0.105, "Surrogate", "K ∈ ℂ⁸ˣ⁸\nR̃ = KᴴK + I\n(ε = 1)", "est"),
           (0.593, 0.135, "ESPRIT", "differentiable, M = 3\ndominant eigenvectors\nLS shift invariance", "est"),
           (0.743, 0.080, "DoA", "θ̂ (3 angles)", "io"),
           (0.838, 0.150, "Training loss", "periodic, permutation-\ninvariant RMSPE", "loss")]
    for x, w, t, b, k in top:
        box(ax, x, y, w, h, t, b, k, ts=6.4)
    for (x0, w0, *_), (x1, *__) in zip(top[:-1], top[1:]):
        arrow(ax, (x0 + w0, y + h / 2), (x1, y + h / 2))
    ax.plot([0.913, 0.913, 0.379], [y + h, y + h + 0.03, y + h + 0.03], color="#c98500", lw=0.9, ls=(0, (3, 2)))
    arrow(ax, (0.379, y + h + 0.03), (0.379, y + h), color="#c98500")
    ax.text(0.988, 0.708, "training: gradients flow back through ESPRIT and both eigendecompositions",
            ha="right", fontsize=5.4, color=GREY, style="italic")

    # bottom: the three encoders, one column each
    bh, gap = 0.128, 0.024
    rows = [0.477 - i * (bh + gap) for i in range(4)]  # block bottoms, top to bottom
    for x0, kind, title, sub, steps in (
            (0.0, "fc", "SubspaceNet-FC", "proposed, 40,400 parameters", FC_STEPS),
            (0.336, "attn", "Attn", "lag variant, 612,224 parameters", ATTN_STEPS),
            (0.672, "iter", "Iter", "lag variant, 346,624 parameters", ITER_STEPS)):
        panel(ax, x0, 0.005, 0.328, 0.677, kind, title, sub)
        for i, (t, b) in enumerate(steps):
            box(ax, x0 + 0.014, rows[i], 0.30, bh, t, b, kind)
            if i:
                arrow(ax, (x0 + 0.164, rows[i - 1]), (x0 + 0.164, rows[i] + bh))
    return fig


def figure1_previous():
    """Previous version of Figure 1 (16 Nov 2026), kept for comparison."""
    fig, ax = canvas(124)
    ax.set_xlim(-0.012, 1.012)  # the boxes at x = 0 and x = 1 keep their outer border
    y, h = 0.79, 0.155
    top = [(0.000, 0.115, "Snapshots", "X ∈ ℂ⁸ˣᵀ\nN = 8 ULA\nT = 200 (train)", "io"),
           (0.132, 0.150, "Autocorrelation", "R̂[τ], Re/Im stacked\nFC: τ = 0 only\nlag variants: τ = 0…7", "io"),
           (0.299, 0.168, "Learned encoder", "SubspaceNet-FC (proposed)\nAttn / Iter (lag variants)\ndetails below", "fc"),
           (0.484, 0.105, "Surrogate", "K ∈ ℂ⁸ˣ⁸\nR̃ = KᴴK + I\n(ε = 1)", "est"),
           (0.606, 0.140, "ESPRIT", "differentiable, M = 3\ndominant eigenvectors\nLS shift invariance", "est"),
           (0.763, 0.077, "DoA", "θ̂ (3 angles)", "io"),
           (0.857, 0.143, "Loss (training)", "periodic, permutation-\ninvariant RMSPE", "loss")]
    for x, w, t, b, k in top:
        box(ax, x, y, w, h, t, b, k, ts=6.2)
    for (x0, w0, *_), (x1, *__) in zip(top[:-1], top[1:]):
        arrow(ax, (x0 + w0, y + h / 2), (x1, y + h / 2))
    ax.plot([0.93, 0.93, 0.383], [y, y - 0.03, y - 0.03], color="#c98500", lw=0.9, ls=(0, (3, 2)))
    arrow(ax, (0.383, y - 0.03), (0.383, y), color="#c98500")
    ax.text(0.656, y - 0.063, "training: gradients flow back through ESPRIT and both eigendecompositions",
            ha="center", fontsize=5.4, color=GREY, style="italic")
    ax.text(0.5, 0.972, "Data flow (one sample)", ha="center", fontsize=7, weight="bold", color=INK)
    for name, steps, kind, yb in (("SubspaceNet-FC (proposed, 40,400 parameters)", FC_STEPS, "fc", 0.49),
                                  ("Attn (lag variant, single pass, 612,224 parameters)", ATTN_STEPS, "attn", 0.265),
                                  ("Iter (lag variant, iterative refinement, 346,624 parameters)", ITER_STEPS, "iter", 0.04)):
        ax.text(0.0, yb + 0.17, name, ha="left", fontsize=6.6, weight="bold", color=C[kind][1])
        w, gap = 0.225, 0.033
        for i, (t, b) in enumerate(steps):
            box(ax, i * (w + gap), yb, w, 0.155, t, b, kind)
            if i:
                arrow(ax, (i * (w + gap) - gap, yb + 0.0775), (i * (w + gap), yb + 0.0775))
    return fig


def variant_b():
    """Three encoders side by side between a shared input and a shared output."""
    fig, ax = canvas(130)
    box(ax, 0.20, 0.885, 0.60, 0.095, "Input (shared)",
        "X ∈ ℂ⁸ˣᵀ → lag autocorrelations R̂[τ], τ = 0…7 → tensor 8 × 16 × 8 (Re/Im stacked)", "io")
    cols = [("SubspaceNet (baseline)\n41,761 parameters", "cnn",
             [("Encoder", "Conv 2×2: 8→16, 32→32, 64→64\nAntiReLU after each\n(doubles the channels)"),
              ("Decoder", "Deconv 2×2: 128→32,\n64→16, 32→1\ndropout 0.2"),
              ("Output", "2N × N map → K")]),
            ("Attn (proposed)\n612,224 parameters", "attn",
             [("Tokenizer", "8 tokens: Linear 128→128,\nLayerNorm, GELU\n+ learned lag embedding"),
              ("Transformer ×2", "4 heads, FFN 512\ndropout 0.2"),
              ("Pool + decoder", "learned-query attention\nLinear 128→512→128\ndropout 0.2 → K")]),
            ("Iter (proposed)\n346,624 parameters", "iter",
             [("Tokenizer", "8 tokens: Linear 128→128,\nLayerNorm, GELU\n(no lag embedding)"),
              ("Refine ×5 (weight-tied)", "s₀ = Linear(mean token)\nq = Linear(s); 4-head\ncross-attn; GRUCell(128)"),
              ("Decoder", "Linear 128→512→128\ndropout 0.1 → K")])]
    w, xs = 0.30, (0.01, 0.35, 0.69)
    for x, (title, kind, blocks) in zip(xs, cols):
        ax.text(x + w / 2, 0.80, title, ha="center", va="center", fontsize=6.8, weight="bold", color=C[kind][1])
        ys = (0.575, 0.375, 0.175)
        for yy, (t, b) in zip(ys, blocks):
            box(ax, x, yy, w, 0.17, t, b, kind)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            arrow(ax, (x + w / 2, y0), (x + w / 2, y1 + 0.17))
        arrow(ax, (0.5, 0.885), (x + w / 2, 0.83), rad=0.0)
        arrow(ax, (x + w / 2, 0.175), (0.5, 0.115))
    box(ax, 0.20, 0.015, 0.60, 0.095, "Output (shared)",
        "R̃ = KᴴK + I → differentiable ESPRIT (M = 3) → θ̂ ;   training loss: RMSPE (+0.1·Toeplitz in the ablation only)", "est")
    return fig


def variant_c():
    """Workflow view: data generation, model, training loop and evaluation, with hyperparameters."""
    fig, ax = canvas(68)
    stages = [
        (0.000, 0.185, "1  Data generation", "100,000 samples, seed 2026\nM = 3 non-coherent, T = 200\nSNR ~ U[−10, 20] dB\nspacing error b ~ U(±0.05)λ\nsplit 90k / 10k", "io"),
        (0.205, 0.165, "2  Features", "lag autocorrelations\nτ = 0…7 → 8 × 16 × 8", "io"),
        (0.390, 0.215, "3  Encoder (one of)", "Attn: tokens + lag emb. →\n2×Transformer (4 h, FFN 512)\n→ query pooling → MLP\nIter: tokens → s₀ →\n5× [cross-attn → GRUCell]\n→ MLP   (both d = 128)", "attn"),
        (0.625, 0.165, "4  Subspace step", "R̃ = KᴴK + I\ndifferentiable ESPRIT\nM = 3 → θ̂", "est"),
        (0.810, 0.190, "5  Optimization", "RMSPE loss (sum over batch)\nAdam, lr 10⁻³, wd 10⁻⁹\nbatch 2048\nearly stop: patience 15,\n≤ 100 epochs; seeds 0–2", "loss"),
    ]
    y, h = 0.47, 0.50
    for x, w, t, b, k in stages:
        box(ax, x, y, w, h, t, b, k, bs=5.5)
    for (x0, w0, *_), (x1, *__) in zip(stages[:-1], stages[1:]):
        arrow(ax, (x0 + w0, y + h / 2), (x1, y + h / 2))
    arrow(ax, (0.905, y), (0.905, y - 0.05), dashed=True, color="#c98500")
    arrow(ax, (0.905, y - 0.05), (0.497, y - 0.05), dashed=True, color="#c98500")
    arrow(ax, (0.497, y - 0.05), (0.497, y), dashed=True, color="#c98500")
    ax.text(0.58, y - 0.105, "backpropagation through ESPRIT", ha="center", fontsize=5.2, color=GREY, style="italic")
    box(ax, 0.0, 0.02, 1.0, 0.24, "6  Evaluation (fixed test seeds, identical observations for every method)",
        "E1–E9 sweeps: SNR, snapshots, coherence, spacing error, steering-vector noise, interferer (2,000 samples per point)   ·   "
        "E10 / E12 tables (5,000 samples)\nmetrics: mean RMSPE, 95% bootstrap CI, seed std, paired Wilcoxon tests   ·   "
        "baselines: SubspaceNet, DA-MUSIC, ESPRIT, Root-MUSIC, MUSIC, SPS", "io", bs=5.6)
    arrow(ax, (0.7075, y), (0.7075, 0.26))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("VarA", variant_a), ("VarB", variant_b), ("VarC", variant_c)):
        fig = fn()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300)
        plt.close(fig)
    print("written:", sorted(p.name for p in OUT.iterdir()))
    rev = RES / "paper_rev"  # the revised figures of paper_assets.py are here
    fig = figure1()
    for ext in ("pdf", "png"):
        fig.savefig(rev / f"Fig1.{ext}", dpi=300)
    Image.open(rev / "Fig1.png").convert("RGB").save(rev / "Fig1.tif", compression="tiff_lzw", dpi=(300, 300))
    print("written:", sorted(p.name for p in rev.glob("Fig1.*")))


if __name__ == "__main__":
    main()
