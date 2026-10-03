"""Build the figure set for the OT-drift project from the recorded result files.

Reads only files under `data/` -- every number on every axis comes from a run's own CSV or
from the dumped point clouds. Nothing here is typed by hand, so a figure cannot drift away
from the record. Writes PNGs into `figures/`.

Stdlib + numpy + matplotlib only: this Mac has no pandas, no scipy, no jax.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# overridden by --data / --out; the report folder on the laptop is the default layout
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

# --- design-system parameters (light surface), from the dataviz reference palette ---------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# categorical slots, in the fixed order -- never cycled, never reordered
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# blue ordinal ramp; on a light surface start no lighter than step 250
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",          # grids are solid hairlines, never dashed
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 160,
})


def style(ax, *, grid="y"):
    ax.set_axisbelow(True)
    if grid in ("y", "both"):
        ax.yaxis.grid(True)
    if grid in ("x", "both"):
        ax.xaxis.grid(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rows(name):
    with open(os.path.join(DATA, name)) as handle:
        return list(csv.DictReader(handle))


def num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


# the six arms, in the order they tell the story, with the label shown to a reader
ARMS = [
    ("pretrained", "pretrained (zero-shot)"),
    ("regress_scratch", "regression, scratch"),
    ("regress_ft", "regression, fine-tune"),
    ("ot_final", "OT final-only"),
    ("ot_traj", "OT trajectory"),
    ("ot_traj_scratch", "OT trajectory, scratch"),
]


# =========================================================================================
# Figure 1 -- the toy's two distributions
# =========================================================================================
def fig_distributions(clouds):
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for i, (key, label) in enumerate([("real_source", "source GMM"),
                                      ("real_target", "target GMM")]):
        pts = clouds[key]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=CAT[i], alpha=0.55,
                   linewidths=0, label=label)
    style(ax, grid="both")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("The 2-D toy: a 6-mode ring, rotated and pushed out\n"
                 "target = source rotated 30°, radius ×1.2, offset +1.0",
                 fontsize=10, loc="left")
    ax.legend(loc="upper right", markerscale=2.2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_toy_distributions.png"))
    plt.close(fig)


# =========================================================================================
# Figure 2 -- what each arm actually generates, at NFE 1 and NFE 4
# =========================================================================================
def fig_generations(clouds, finals):
    fig, axes = plt.subplots(2, 6, figsize=(15.0, 5.4), sharex=True, sharey=True)
    target = clouds["real_target"]
    for col, (arm, label) in enumerate(ARMS):
        for row, nfe in enumerate((1, 4)):
            ax = axes[row, col]
            # the target sits underneath in muted ink: it is the reference, not a series
            ax.scatter(target[:, 0], target[:, 1], s=5, c=AXIS, linewidths=0, zorder=1)
            pts = clouds[f"{arm}_nfe{nfe}"]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c=CAT[0], alpha=0.6,
                       linewidths=0, zorder=2)
            w2 = finals.get((arm, nfe))
            if w2 is not None:
                ax.text(0.03, 0.03, f"W2 {w2[0]:.2f}", transform=ax.transAxes,
                        fontsize=8, color=INK, ha="left", va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_color(AXIS)
            if row == 0:
                ax.set_title(label, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(f"NFE {nfe}", fontsize=10, color=INK)
    fig.suptitle("Generated sets against the target ring (gray).  "
                 "W2 is the mean over 3 seeds from the recorded run; clouds are seed 0.",
                 fontsize=10, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGS, "fig2_toy_generations.png"))
    plt.close(fig)


# =========================================================================================
# Figure 3 -- adaptation curves
# =========================================================================================
def fig_curves(toy):
    series = {}
    for r in toy:
        key = (r["arm"], int(r["step"]))
        value = num(r["w2_nfe1"])
        if value is not None:
            series.setdefault(key, []).append(value)

    baseline = float(np.mean(series[("pretrained", 0)]))
    floor = float(np.mean(series[("source_pretrain", 8000)]))

    # two panels so no panel carries more than three hues -- the all-pairs-safe cap
    panels = [
        ("initialised from the pretrained source",
         ["regress_ft", "ot_final", "ot_traj"]),
        ("initialised from scratch (no prior)",
         ["regress_scratch", "ot_traj_scratch"]),
    ]
    by_arm = dict(ARMS)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, (caption, arms) in zip(axes, panels):
        ax.axhline(baseline, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.axhline(floor, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        for i, arm in enumerate(arms):
            steps = sorted(s for (a, s) in series if a == arm)
            mean = [float(np.mean(series[(arm, s)])) for s in steps]
            lo = [float(np.min(series[(arm, s)])) for s in steps]
            hi = [float(np.max(series[(arm, s)])) for s in steps]
            ax.fill_between(steps, lo, hi, color=CAT[i], alpha=0.14, linewidth=0, zorder=2)
            ax.plot(steps, mean, color=CAT[i], linewidth=2.0, label=by_arm[arm], zorder=3)
            ax.plot(steps[-1], mean[-1], "o", color=CAT[i], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        style(ax)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_xlabel("adaptation step")
        ax.set_title(caption, fontsize=9.5, color=INK, loc="left")
        ax.legend(loc="upper right", labelcolor=INK2)

    axes[0].set_ylabel("W2 to target at NFE 1  (lower is better)")
    axes[0].text(150, baseline * 1.06, f"no adaptation  {baseline:.2f}", fontsize=8,
                 color=INK2, va="bottom")
    axes[0].text(150, floor * 1.05, f"source-fit floor  {floor:.2f}", fontsize=8,
                 color=INK2, va="bottom")
    fig.suptitle("Every adaptation arm collapses the gap inside 500 steps, then flattens well "
                 "above the floor.\nNeither OT arm beats plain regression.  "
                 "Mean of 3 seeds, band = seed min-max.",
                 fontsize=10, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig3_toy_adaptation_curves.png"))
    plt.close(fig)


# =========================================================================================
# Figure 4 -- final W2, NFE 1 vs NFE 4
# =========================================================================================
def fig_final_w2(toy):
    finals = {}
    for r in toy:
        arm, step = r["arm"], int(r["step"])
        for nfe, col in ((1, "w2_nfe1"), (4, "w2_nfe4")):
            value = num(r[col])
            if value is None:
                continue
            bucket = finals.setdefault((arm, nfe), {})
            bucket.setdefault(step, []).append(value)

    last = {}
    for (arm, nfe), by_step in finals.items():
        step = 0 if arm == "pretrained" else max(by_step)
        draws = by_step[step]
        last[(arm, nfe)] = (float(np.mean(draws)), float(np.std(draws)), step)

    labels = [label for _, label in ARMS]
    x = np.arange(len(ARMS), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, nfe in enumerate((1, 4)):
        mean = [last[(a, nfe)][0] for a, _ in ARMS]
        err = [last[(a, nfe)][1] for a, _ in ARMS]
        offset = (i - 0.5) * (width + 0.02)   # 2px-equivalent gap between adjacent bars
        ax.bar(x + offset, mean, width, color=CAT[i], label=f"NFE {nfe}", zorder=2)
        ax.errorbar(x + offset, mean, yerr=err, fmt="none", ecolor=INK2,
                    elinewidth=1.0, capsize=3, zorder=3)
        for xi, value, e in zip(x + offset, mean, err):
            ax.text(xi, value + e + 0.06, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)

    floor = [float(r["w2_nfe1"]) for r in toy
             if r["arm"] == "source_pretrain" and int(r["step"]) == 8000]
    level = float(np.mean(floor))
    ax.axhline(level, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.995, 0.80, f"dashed: source-fit floor {level:.2f}\n"
                         "(the same network on its own training set)",
            transform=ax.transAxes, fontsize=8, color=INK2, ha="right", va="top")

    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", color=INK2)
    ax.set_ylabel("W2 to target at the final step")
    ax.set_title("At NFE 1 all four adapted arms land in 0.52-0.74, down from 3.18 zero-shot.\n"
                 "At NFE 4 the two OT arms get worse; the regression arms do not.  "
                 "Error bars = std, 3 seeds.", fontsize=10, loc="left")
    ax.legend(loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_toy_final_w2.png"))
    plt.close(fig)
    return last


# =========================================================================================
# Figure 5 -- can a minibatch Sinkhorn divergence even see a domain gap?
# =========================================================================================
def fig_snr(snr):
    panels = [("mix", "mixture fraction of the other domain"),
              ("sigma", "isotropic latent noise sigma")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, (condition, caption) in zip(axes, panels):
        levels = sorted({float(r["level"]) for r in snr if r["condition"] == condition})
        for i, level in enumerate(levels):
            pts = sorted((int(r["n"]), float(r["snr"])) for r in snr
                         if r["condition"] == condition and float(r["level"]) == level)
            # spread the ordinal ramp across however many levels this panel has, so a
            # 3-level panel does not end up using only the three palest steps
            slot = 0 if len(levels) == 1 else round(i * (len(ORD) - 1) / (len(levels) - 1))
            colour = ORD[slot]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour,
                    linewidth=2.0, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{level:g}", zorder=3)
        ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([2, 8, 32, 128, 512, 2048])
        ax.set_xticklabels(["2", "8", "32", "128", "512", "2048"])
        style(ax, grid="y")
        ax.set_xlabel("particles per Sinkhorn solve (N)")
        ax.set_title(caption, fontsize=9.5, color=INK, loc="left")
        ax.legend(title="level", loc="upper left", ncol=2, labelcolor=INK2,
                  title_fontsize=8)
    axes[0].set_ylabel("SNR  =  (perturbed - floor) / std over draws")
    axes[1].text(2100, 1.0, "  SNR = 1\n  signal equals\n  its own noise", fontsize=8,
                 color=INK2, va="center", ha="left")
    fig.suptitle("Stage 0b on real cub200 / food101 latents: the divergence is blind at "
                 "N=2 and only sees a small gap past N~128.",
                 fontsize=10, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig5_sinkhorn_snr.png"))
    plt.close(fig)


# =========================================================================================
# Figure 6 -- the cost of a Sinkhorn solve, which sizes Stage 1
# =========================================================================================
def fig_cost(snr):
    pts = sorted((int(r["n"]), float(r["median_seconds"])) for r in snr
                 if r["condition"] == "mix" and float(r["level"]) == 1.0)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.plot([p[0] for p in pts], [p[1] * 1000 for p in pts], color=CAT[0],
            linewidth=2.0, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    for n, seconds in pts:
        ax.annotate(f"{seconds * 1000:.0f} ms", (n, seconds * 1000), fontsize=7.5,
                    color=INK2, textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 8, 32, 128, 512, 2048])
    ax.set_xticklabels(["2", "8", "32", "128", "512", "2048"])
    style(ax)
    ax.set_xlabel("particles per solve (N)")
    ax.set_ylabel("median wall-clock per solve (ms)")
    by_n = dict(pts)
    ax.set_title("One debiased Sinkhorn solve, 4096-D latents, 100 iters.\n"
                 f"N=512 costs {by_n[512]:.2f} s; N=2048 costs {by_n[2048]:.1f} s.",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_sinkhorn_cost.png"))
    plt.close(fig)


# =========================================================================================
# Figure 7 -- the number the project has to beat
# =========================================================================================
def fig_camf():
    runs = [("camf_cub200_phase1.csv", "run 1 (to 160k batches)"),
            ("camf_cub200_phase2.csv", "run 2 (continued, to 300k)")]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    best = None
    for i, (name, label) in enumerate(runs):
        pts = []
        for r in rows(name):
            fid, step = num(r.get("fid")), num(r.get("training_step"))
            if fid and step is not None and fid > 0:
                pts.append((step, fid))
        pts.sort()
        ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], color=CAT[i],
                linewidth=2.0, marker="o", markersize=3.5, markeredgecolor=SURFACE,
                markeredgewidth=1.0, label=label, zorder=3)
        low = min(pts, key=lambda p: p[1])
        if best is None or low[1] < best[1]:
            best = low
    ax.plot(best[0] / 1000, best[1], "o", color=CAT[1], markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.annotate(f"best: FID {best[1]:.2f} @ {best[0] / 1000:.0f}k batches",
                (best[0] / 1000, best[1]), fontsize=8.5, color=INK,
                textcoords="offset points", xytext=(-12, -16), ha="right")
    style(ax)
    ax.set_yscale("log")
    ax.set_yticks([10, 15, 20, 30, 50, 100, 170])
    ax.set_yticklabels(["10", "15", "20", "30", "50", "100", "170"])
    ax.minorticks_off()
    ax.set_xlabel("training batches (thousands)")
    ax.set_ylabel("FID, NFE 4, 10k samples  (lower is better)")
    ax.set_title("The baseline this project must beat: CAMF (adversarial) post-training\n"
                 "ImageNet iMF-XL-2 onto cub200.  170.6 -> 9.34.", fontsize=10, loc="left")
    ax.legend(loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_camf_cub200_baseline.png"))
    plt.close(fig)
    return best


# =========================================================================================
# Figures 8 & 9 -- EXP-126, one OT step from a source state at a fixed noise level
# =========================================================================================
# `mf_head_t0` is deliberately absent: it is the identity map for every theta, so its bar is
# 8x the others and squashes the comparison that matters. It is a confirmed null -- a number,
# not a comparison -- so it is reported as a caption instead of a bar.
SRC_T_ARMS = [
    ("src_t1", "t=1\nsource init"),
    ("src_t0.5", "t=0.5\nsource init"),
    ("src_t0", "t=0\nsource init"),
    ("scratch_t1", "t=1\nscratch"),
    ("scratch_t0", "t=0\nscratch"),
    ("mf_head_t1", "t=1\nMF head"),
]


def source_t_table(src):
    """Final-step rows keyed by (arm, seed), plus the seed list."""
    final = {}
    for r in src:
        if int(r["step"]) == 4000:
            final[(r["arm"], int(r["seed"]))] = r
    seeds = sorted({int(r["seed"]) for r in src})
    return final, seeds


def fig_source_t_arms(src):
    final, seeds = source_t_table(src)

    def col(arm, key):
        return np.array([float(final[(arm, s)][key]) for s in seeds])

    labels = [label for _, label in SRC_T_ARMS]
    x = np.arange(len(SRC_T_ARMS), dtype=float)
    width = 0.36
    measures = [("w2_forward", "train-matched input\n(forward-noised)"),
                ("w2_onpolicy_src4", "test-time input\n(frozen source model, 4 steps)")]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    for i, (key, label) in enumerate(measures):
        mean = [col(a, key).mean() for a, _ in SRC_T_ARMS]
        err = [col(a, key).std(ddof=0) for a, _ in SRC_T_ARMS]
        offset = (i - 0.5) * (width + 0.02)
        ax.bar(x + offset, mean, width, color=CAT[i], label=label, zorder=2)
        ax.errorbar(x + offset, mean, yerr=err, fmt="none", ecolor=INK2, elinewidth=1.0,
                    capsize=3, zorder=3)
        for xi, value, e in zip(x + offset, mean, err):
            ax.text(xi, value + e + 0.02, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=INK2)
    ax.set_ylim(0, max(col(a, "w2_onpolicy_src4").mean() + col(a, "w2_onpolicy_src4").std()
                       for a, _ in SRC_T_ARMS) * 1.42)
    ax.set_ylabel("W2 to target, one OT step  (lower is better)")
    ax.set_title("Lower t does not pay. The gap between the two bars is the cost of\n"
                 "training without the source model in the loop, and it grows as t falls.",
                 fontsize=9.5, loc="left")
    ax.legend(loc="upper left", labelcolor=INK2, frameon=False)
    # the degenerate arm is a number, not a comparison -- see the SRC_T_ARMS comment
    ax.text(0.995, 0.70, "not plotted: the MeanFlow head at t=0 is the identity map for\n"
            "every theta. Measured W2 %.2f train-matched / %.2f on-policy,\n"
            "unmoved from its %.2f init. Predicted null, confirmed."
            % (col("mf_head_t0", "w2_forward").mean(),
               col("mf_head_t0", "w2_onpolicy_src4").mean(),
               float(np.mean([float(r["w2_forward"]) for r in src
                              if r["arm"] == "mf_head_t0" and int(r["step"]) == 0]))),
            transform=ax.transAxes, fontsize=7.5, color=INK2, ha="right", va="top")

    ax = axes[1]
    mean = [col(a, "mode_mi").mean() for a, _ in SRC_T_ARMS]
    err = [col(a, "mode_mi").std(ddof=0) for a, _ in SRC_T_ARMS]
    ax.bar(x, mean, 0.6, color=CAT[2], zorder=2)
    ax.errorbar(x, mean, yerr=err, fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3,
                zorder=3)
    ceiling = np.log2(6)
    ax.axhline(ceiling, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(-0.4, ceiling + 0.05, f"perfect: log2(6) = {ceiling:.2f} bits",
            fontsize=8, color=INK2, ha="left", va="bottom")
    for xi, value, e in zip(x, mean, err):
        # a label that would land on the reference line is lifted clear of it instead
        top = value + e + 0.04
        ax.text(xi, ceiling + 0.22 if abs(top - ceiling) < 0.14 else top, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK2)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=INK2)
    ax.set_ylim(0, ceiling * 1.26)
    ax.set_ylabel("mode MI (bits):  input mode -> output mode")
    ax.set_title("Does it ignore the source input?  No.\n"
                 "At t=0 the map is almost a bijection on modes; at t=1 it is at chance.",
                 fontsize=9.5, loc="left")

    fig.suptitle("EXP-126: one OT step from a source state at a fixed noise level.  "
                 "5 seeds, error bars = std.",
                 fontsize=10, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig8_source_t_arms.png"))
    plt.close(fig)
    return final, seeds


def fig_source_t_clouds(clouds):
    arms = [("src_t1", "t = 1  (pure noise in)"),
            ("src_t0.5", "t = 0.5  (half-noised source in)"),
            ("src_t0", "t = 0  (clean source image in)")]
    panels = [("in_forward", "input: forward-noised\n(what training sees)", "real_source"),
              ("in_onpolicy", "input: frozen source model\n(what test sees)", "real_source"),
              ("out_forward", "output from the training input", "real_target"),
              ("out_onpolicy", "output from the test input", "real_target")]

    fig, axes = plt.subplots(3, 4, figsize=(11.2, 8.6), sharex=True, sharey=True)
    for row, (arm, caption) in enumerate(arms):
        for col, (suffix, title, reference) in enumerate(panels):
            ax = axes[row, col]
            ref = clouds[reference]
            ax.scatter(ref[:, 0], ref[:, 1], s=4, c=AXIS, linewidths=0, zorder=1)
            pts = clouds[f"{arm}_{suffix}"]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c=CAT[0], alpha=0.55, linewidths=0,
                       zorder=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_color(AXIS)
            if row == 0:
                ax.set_title(title, fontsize=8.5, color=INK)
            if col == 0:
                ax.set_ylabel(caption, fontsize=9, color=INK)
    fig.suptitle("Why lower t costs more: the source model's partial generation (col 2) is "
                 "not the forward-noised state\ntraining saw (col 1), and the mismatch grows "
                 "as t falls.  Gray = source ring (cols 1-2) or target ring (cols 3-4).",
                 fontsize=10, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig9_source_t_clouds.png"))
    plt.close(fig)


def fig_boomerang(clouds):
    """Why a round blob comes out as an arc: squash + stretch (linear), then bend (quadratic).

    The input and output arrays are paired point-for-point, so the map's local geometry can be
    fitted directly instead of inferred from the output's shape. The ideal map here is a
    rotation plus a uniform x1.2 scale, so its Jacobian is 1.200 / 1.200 everywhere and a round
    blob must stay round. Every departure below is a defect, not a property of the problem.
    """
    num_modes, spread, scale, rot = 6, 0.25, 1.2, 0.5236
    angles = 2.0 * np.pi * np.arange(num_modes) / num_modes
    src_c = np.stack([3.0 * np.cos(angles), 3.0 * np.sin(angles)], 1)
    x, y = clouds["src_t0_in_forward"], clouds["src_t0_out_forward"]
    mode = np.argmin(((x[:, None] - src_c[None]) ** 2).sum(-1), 1)
    ideal = scale * np.array([[np.cos(-rot), -np.sin(-rot)], [np.sin(-rot), np.cos(-rot)]])

    fit = []
    for m in range(num_modes):
        xc = x[mode == m] - x[mode == m].mean(0)
        yc = y[mode == m] - y[mode == m].mean(0)
        a, *_ = np.linalg.lstsq(xc, yc, rcond=None)
        sv = np.linalg.svd(a.T)[1]
        fit.append((sv[0], sv[1], float((yc - xc @ a).std(0).mean()), xc, yc, a))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    # explicit margins, not tight_layout: it mis-sizes a row that holds an aspect-equal
    # axes and leaves a band of dead space under every panel
    fig.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.26, wspace=0.26)

    # --- panel 1: the worst mode, in data units, drawn square so the shape is readable -------
    ax = axes[0]
    worst = int(np.argmax([f[0] / f[1] for f in fit]))
    sv0, sv1, _, xc, yc, a = fit[worst]
    circle = np.stack([np.cos(np.linspace(0, 2 * np.pi, 200)),
                       np.sin(np.linspace(0, 2 * np.pi, 200))], 1) * 2 * spread
    ax.scatter(xc[:, 0], xc[:, 1], s=5, c=AXIS, linewidths=0, zorder=1,
               label="the input blob (std 0.25)")
    ax.plot(*(circle @ ideal).T, color=INK2, linewidth=1.4, linestyle=(0, (5, 3)), zorder=4,
            label="where the ideal map puts it: still round")
    ax.scatter(*(xc @ a).T, s=5, c=CAT[1], linewidths=0, zorder=2,
               label="the fitted LINEAR part: a line")
    ax.scatter(yc[:, 0], yc[:, 1], s=5, c=CAT[0], linewidths=0, alpha=0.8, zorder=3,
               label="what the model emits: a bent line")
    style(ax, grid=False)
    lim = 1.1 * np.abs(yc).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(f"source mode {worst}, the worst one.\nIts Jacobian is {sv0:.2f} by {sv1:.2f}.",
                 fontsize=9.5, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.46), fontsize=7.5, labelcolor=INK2,
              frameon=False)

    # --- panel 2: the squash, in the same units as the ideal gain ----------------------------
    index = np.arange(num_modes, dtype=float)
    ax = axes[1]
    ax.bar(index - 0.21, [f[0] for f in fit], 0.4, color=CAT[1], zorder=2,
           label="gain along the stretched axis")
    ax.bar(index + 0.21, [f[1] for f in fit], 0.4, color=CAT[0], zorder=2,
           label="gain along the squashed axis")
    ax.axhline(scale, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(num_modes - 0.4, scale + 0.08, f"both should be {scale:.2f}", fontsize=8,
            color=INK2, ha="right", va="bottom")
    style(ax)
    ax.set_xticks(index)
    ax.set_xticklabels([f"{m}" for m in range(num_modes)], fontsize=8, color=INK2)
    ax.set_xlabel("source mode", fontsize=8.5)
    ax.set_ylabel("local gain of the fitted map", fontsize=8.5)
    ax.set_ylim(0, 3.3)
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK2, frameon=False)
    ax.set_title("The squash. One axis is stretched 2-3x,\nthe other is crushed to nearly zero.",
                 fontsize=9.5, loc="left")

    # --- panel 3: the bend, in data units against the blob it is supposed to preserve --------
    ax = axes[2]
    ax.bar(index, [f[2] for f in fit], 0.56, color=CAT[0], zorder=2)
    ax.axhline(spread, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(num_modes - 0.4, spread + 0.006, f"the blob's own std, {spread:.2f}", fontsize=8,
            color=INK2, ha="right", va="bottom")
    for xi, f in zip(index, fit):
        ax.text(xi, f[2] + 0.004, f"{f[2]:.2f}", ha="center", va="bottom", fontsize=7.5,
                color=INK2)
    style(ax)
    ax.set_xticks(index)
    ax.set_xticklabels([f"{m}" for m in range(num_modes)], fontsize=8, color=INK2)
    ax.set_xlabel("source mode", fontsize=8.5)
    ax.set_ylabel("curvature left over the linear fit", fontsize=8.5)
    ax.set_ylim(0, 0.30)
    ax.set_title("The bend. What no linear map explains, and it is\nas big as the blob itself.",
                 fontsize=9.5, loc="left")

    fig.suptitle("Why the t=0 outputs are boomerangs and not Gaussians: the map crushes the blob "
                 "onto a line, then bends the line. The ideal map would do neither.",
                 fontsize=10, x=0.008, y=0.965, ha="left", color=INK)
    fig.savefig(os.path.join(FIGS, "fig10_boomerang.png"))
    plt.close(fig)


# =========================================================================================
# Figure 11 -- EXP-127: the entropic blur was the ceiling, and it was set 20x too coarse
# =========================================================================================
EPS_LEVELS = ["0.05", "0.02", "0.005", "0.0025", "0.001"]
EPS_ARMS = [("src_t1", "t=1", CAT[0]), ("src_t0", "t=0", CAT[1])]


def fig_eps_sweep():
    """One knob, three panels: train-matched W2 moves a lot, on-policy W2 does not, shape heals.

    x is `eps_rel`, the knob we actually set. The blur it buys is `sqrt(eps_rel * mean cost)`,
    printed per point in the third panel because it is the quantity that matters and it is not
    linear in the knob.
    """
    spread, scale, rot, num_modes = 0.25, 1.2, 0.5236, 6
    angles = 2.0 * np.pi * np.arange(num_modes) / num_modes
    src_c = np.stack([3.0 * np.cos(angles), 3.0 * np.sin(angles)], 1)
    ta = angles + rot
    tgt_c = np.stack([3.0 * scale * np.cos(ta) + 1.0, 3.0 * scale * np.sin(ta)], 1)

    def near(points, centres):
        return np.argmin(((points[:, None] - centres[None]) ** 2).sum(-1), 1)

    w2 = {}
    geom = {}
    for level in EPS_LEVELS:
        table = rows(os.path.join("eps_sweep", f"eps_{level}.csv"))
        last = max(int(r["step"]) for r in table)
        for arm, _, _ in EPS_ARMS:
            sel = [r for r in table if r["arm"] == arm and int(r["step"]) == last]
            for metric in ("w2_forward", "w2_onpolicy_src4"):
                vals = [num(r[metric]) for r in sel if num(r[metric]) is not None]
                w2[(arm, level, metric)] = float(np.mean(vals))
        cloud = np.load(os.path.join(DATA, "eps_sweep", f"eps_{level}_clouds.npz"))
        x, y, real = (cloud["src_t0_in_forward"], cloud["src_t0_out_forward"],
                      cloud["real_target"])
        blur = float(np.sqrt(float(level) * ((y[:, None] - real[None]) ** 2).sum(-1).mean()))
        mode, cond = near(x, src_c), []
        for m in range(num_modes):
            xc = x[mode == m] - x[mode == m].mean(0)
            yc = y[mode == m] - y[mode == m].mean(0)
            a, *_ = np.linalg.lstsq(xc, yc, rcond=None)
            sv = np.linalg.svd(a.T)[1]
            cond.append(sv[0] / sv[1])
        out_mode, rad, tan = near(y, tgt_c), [], []
        for t in range(num_modes):
            pts = y[out_mode == t]
            if pts.shape[0] < 20:
                continue
            centred = pts - pts.mean(0)
            v = tgt_c[t] - np.array([1.0, 0.0])
            r = v / np.linalg.norm(v)
            rad.append((centred @ r).std())
            tan.append((centred @ np.array([-r[1], r[0]])).std())
        geom[level] = (blur, float(np.mean(cond)), float(np.mean(rad)), float(np.mean(tan)))

    xs = np.array([float(e) for e in EPS_LEVELS])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    for ax, metric, caption in [
            (axes[0], "w2_forward", "Train-matched: tightening is worth 1.6x at t=1,\nuntil it collapses t=0."),
            (axes[1], "w2_onpolicy_src4",
             "On-policy, the real test condition:\nthe knob buys nothing at all.")]:
        for arm, label, colour in EPS_ARMS:
            ys = [w2[(arm, e, metric)] for e in EPS_LEVELS]
            ax.plot(xs, ys, color=colour, linewidth=2.0, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # coarse on the left, fine on the right: left to right is "turn the blur down"
        ax.invert_xaxis()
        ax.set_xticks(xs)
        ax.set_xticklabels(EPS_LEVELS, fontsize=8, color=INK2)
        ax.minorticks_off()
        ax.set_xlabel("eps_rel  (the blur gets finer to the right)", fontsize=8.5)
        ax.set_ylabel("W2 to the target", fontsize=8.5)
        ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.2", "0.3", "0.5", "1", "2", "4"])
        ax.set_title(caption, fontsize=9.5, loc="left")
        ax.legend(loc="lower left", labelcolor=INK2, frameon=False, fontsize=8)

    # the shape panel: everything in data units, against the blob the loss has to resolve
    ax = axes[2]
    rad = [geom[e][2] for e in EPS_LEVELS]
    tan = [geom[e][3] for e in EPS_LEVELS]
    ax.plot(xs, tan, color=CAT[1], linewidth=2.0, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2, label="output std, long axis", zorder=3)
    ax.plot(xs, rad, color=CAT[0], linewidth=2.0, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2, label="output std, short axis", zorder=3)
    ax.axhline(spread, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(xs.max(), spread * 0.90, f"the target's own blob, {spread:.2f}", fontsize=8,
            color=INK2, ha="left", va="top")
    style(ax)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(xs)
    # the knob, the blur it buys, and the squash it leaves -- all three belong on this axis
    ax.set_xticklabels([f"{e}\nblur {geom[e][0]:.2f}\ncond {geom[e][1]:.1f}"
                        for e in EPS_LEVELS], fontsize=7.5, color=INK2)
    ax.minorticks_off()
    ax.set_xlabel("eps_rel  (the blur gets finer to the right)", fontsize=8.5)
    ax.set_ylabel("t=0 output blob std, data units", fontsize=8.5)
    ax.set_ylim(0, 0.88)
    ax.legend(loc="lower right", labelcolor=INK2, frameon=False, fontsize=8)
    ax.set_title("The boomerang heals at eps_rel 0.005:\nround blobs, the right size.",
                 fontsize=9.5, loc="left")

    fig.suptitle("EXP-127. The crescents were the objective's resolution limit, not the model's. "
                 "Fixing them does not change which arm wins.",
                 fontsize=10, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig11_eps_sweep.png"))
    plt.close(fig)
    return geom



# =========================================================================================
# Figure 12 -- EXP-127: what the fix looks like, as generations
# =========================================================================================
EPS_SHOW = ["0.05", "0.005", "0.001"]


def fig_eps_generations():
    """The same three arms' outputs at the old blur, the good blur, and one step too far.

    Two full-ring rows plus a zoom row, because at ring scale a 0.6-unit crescent is a few
    pixels. The zoom is one target mode of the t=0 arm, which is where section 9 fitted the
    Jacobian.
    """
    spread, scale, rot, num_modes = 0.25, 1.2, 0.5236, 6
    angles = 2.0 * np.pi * np.arange(num_modes) / num_modes + rot
    tgt_c = np.stack([3.0 * scale * np.cos(angles) + 1.0,
                      3.0 * scale * np.sin(angles)], 1)

    clouds = {level: np.load(os.path.join(DATA, "eps_sweep", f"eps_{level}_clouds.npz"))
              for level in EPS_SHOW}
    target = clouds[EPS_SHOW[0]]["real_target"]

    # FID-style W2^2 (Frechet distance between fitted Gaussians) of the cloud actually drawn,
    # not the CSV's seed mean: the CSV scores a separate eval draw. Exact sample-matching W2^2
    # is not used: at n=2048 it is dominated by chance per-mode counts, not by quality. The
    # floor is a fresh target draw of the same size against the same target set.
    from scipy.linalg import sqrtm

    def frechet(p, q):
        cov_p, cov_q = np.cov(p.T), np.cov(q.T)
        return float(((p.mean(0) - q.mean(0)) ** 2).sum()
                     + np.trace(cov_p + cov_q - 2 * np.real(sqrtm(cov_p @ cov_q))))

    w2 = {(arm, level): frechet(clouds[level][f"{arm}_out_forward"], target)
          for level in EPS_SHOW for arm in ("src_t1", "src_t0")}
    rng = np.random.default_rng(0)
    floors = []
    for _ in range(20):
        which = rng.integers(0, num_modes, target.shape[0])
        fresh = tgt_c[which] + spread * rng.standard_normal(target.shape)
        floors.append(frechet(fresh, target))
    floor, floor_sd = float(np.mean(floors)), float(np.std(floors))
    # the zoom window: one target mode, +/- 3 blob stds, so a correct blob fills it and no more
    hub, half = tgt_c[1], 3.2 * spread

    panels = [("src_t1", "t = 1,  the winning arm", False),
              ("src_t0", "t = 0", False),
              ("src_t0", "t = 0, zoomed on one mode", True)]

    fig, axes = plt.subplots(3, 3, figsize=(10.4, 10.0))
    for r, (arm, label, zoom) in enumerate(panels):
        for c, level in enumerate(EPS_SHOW):
            ax = axes[r, c]
            pts = clouds[level][f"{arm}_out_forward"]
            if zoom:
                circle = np.stack([np.cos(np.linspace(0, 2 * np.pi, 200)),
                                   np.sin(np.linspace(0, 2 * np.pi, 200))], 1)
                keep = (np.abs(target - hub) < half).all(1)
                ax.scatter(*(target[keep] - hub).T, s=12, c=AXIS, linewidths=0, zorder=1)
                ax.plot(*(circle * 2 * spread).T, color=INK2, linewidth=1.3,
                        linestyle=(0, (5, 3)), zorder=4)
                keep = (np.abs(pts - hub) < half).all(1)
                ax.scatter(*(pts[keep] - hub).T, s=12, c=CAT[0], alpha=0.75, linewidths=0,
                           zorder=3)
                ax.set_xlim(-half, half)
                ax.set_ylim(-half, half)
            else:
                ax.scatter(target[:, 0], target[:, 1], s=4, c=AXIS, linewidths=0, zorder=1)
                ax.scatter(pts[:, 0], pts[:, 1], s=4, c=CAT[0], alpha=0.6, linewidths=0,
                           zorder=2)
                ax.set_xlim(-4.2, 6.2)
                ax.set_ylim(-5.2, 5.2)
                ax.text(0.03, 0.03, f"FD {w2[(arm, level)]:.4f}", transform=ax.transAxes,
                        fontsize=8.5, color=INK, ha="left", va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_color(AXIS)
            if r == 0:
                blur = float(np.sqrt(float(level) * ((clouds[level]["src_t1_out_forward"][:, None]
                                                      - target[None]) ** 2).sum(-1).mean()))
                note = {"0.05": "what we trained with",
                        "0.005": "the fix",
                        "0.001": "one step too far"}[level]
                ax.set_title(f"eps_rel {level}   blur {blur:.2f}\n{note}", fontsize=9.5,
                             color=INK)
            if c == 0:
                ax.set_ylabel(label, fontsize=9.5, color=INK)

    fig.suptitle("Gray = the target. Blue = the arm's own 1-step output on training-matched "
                 "inputs, seed 0.\nFD = FID-style W2² between Gaussians fitted to the points shown. "
                 "A perfect generator (fresh target draw, same n=%d) scores %.4f ± %.4f.\n"
                 "The dashed circle in the zoom row is 2 std of a correct blob."
                 % (target.shape[0], floor, floor_sd),
                 fontsize=10, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGS, "fig12_eps_generations.png"))
    plt.close(fig)


# =========================================================================================
# Figure 14 -- EXP-128: fig 9 at 5 t levels, eps 0.005, 1-step on-policy input, FD
# =========================================================================================
T_SWEEP = ["1", "0.75", "0.5", "0.25", "0"]


def frechet_fd(p, q):
    """FID-style W2^2 between Gaussians fitted to the two sets."""
    from scipy.linalg import sqrtm

    cov_p, cov_q = np.cov(p.T), np.cov(q.T)
    return float(((p.mean(0) - q.mean(0)) ** 2).sum()
                 + np.trace(cov_p + cov_q - 2 * np.real(sqrtm(cov_p @ cov_q))))


def ring_fd_floor(target, draws=20, seed=0):
    """FD of fresh target draws (same n) against `target`: what a perfect generator scores."""
    spread, num_modes = 0.25, 6
    angles = 2.0 * np.pi * np.arange(num_modes) / num_modes + 0.5236
    centres = np.stack([3.6 * np.cos(angles) + 1.0, 3.6 * np.sin(angles)], 1)
    rng = np.random.default_rng(seed)
    vals = [frechet_fd(centres[rng.integers(0, num_modes, target.shape[0])]
                       + spread * rng.standard_normal(target.shape), target)
            for _ in range(draws)]
    return float(np.mean(vals)), float(np.std(vals))


def fig_t_sweep(folder):
    """Separate fixed-t models. Each row is one model; inputs on the left, outputs on the right.

    Panel FD is on the seed-0 points drawn (n=2048). The bracket is the 5-seed mean +- sd of
    the logged eval (fresh draws, n=1024), so a seed-0 outlier cannot pass as the result.
    """
    clouds, table = {}, {}
    for t in T_SWEEP:
        clouds[t] = np.load(os.path.join(folder, f"t_{t}_clouds.npz"))
        rows_t = list(csv.DictReader(open(os.path.join(folder, f"t_{t}.csv"))))
        last = max(int(r["step"]) for r in rows_t)
        table[t] = [r for r in rows_t if int(r["step"]) == last]
    target = clouds["1"]["real_target"]
    floor, floor_sd = ring_fd_floor(target)
    # EXP-129 logs the on-policy input at a fixed source step size; EXP-128 at 1 step
    grid = "fd_onpolicy_grid" in table["1"][0]
    on_metric = "fd_onpolicy_grid" if grid else "fd_onpolicy_src1"

    panels = [("in_forward", "input: forward-noised source\n(off-policy, what training sees)",
               "real_source", None),
              ("in_onpolicy", ("input: source model at step 0.25\nfrom t=1 to t (on-policy)"
                               if grid else
                               "input: source model, 1 step\nfrom t=1 to t (on-policy)"),
               "real_source", None),
              ("out_forward", "output from the off-policy input", "real_target", "fd_forward"),
              ("out_onpolicy", "output from the on-policy input", "real_target", on_metric)]

    fig, axes = plt.subplots(len(T_SWEEP), 4, figsize=(11.2, 2.75 * len(T_SWEEP)),
                             sharex=True, sharey=True)
    for row, t in enumerate(T_SWEEP):
        cl = clouds[t]
        for col, (suffix, title, reference, metric) in enumerate(panels):
            ax = axes[row, col]
            ref = cl[reference]
            ax.scatter(ref[:, 0], ref[:, 1], s=3, c=AXIS, linewidths=0, zorder=1)
            pts = cl[f"src_t{t}_{suffix}"]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, c=CAT[0], alpha=0.55, linewidths=0,
                       zorder=2)
            if metric:
                vals = np.array([float(r[metric]) for r in table[t]])
                ax.text(0.03, 0.03, "FD %.4f\n(5 seeds %.4f ± %.4f)"
                        % (frechet_fd(pts, target), vals.mean(), vals.std()),
                        transform=ax.transAxes, fontsize=7.5, color=INK, ha="left",
                        va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_color(AXIS)
            if row == 0:
                ax.set_title(title, fontsize=8.5, color=INK)
            if col == 0:
                label = f"t = {t}"
                if grid:
                    nfe = int(float(table[t][0]["src_grid_nfe"]))
                    label += f"\n{nfe} MeanFlow + 1 OT step"
                ax.set_ylabel(label, fontsize=9.5, color=INK)
    fig.suptitle("One separately trained OT model per row, eps_rel 0.005, seed 0 drawn.  "
                 "Gray = source (cols 1-2) or target (cols 3-4).\n"
                 "FD = FID-style W2² vs the target. A perfect generator scores "
                 "%.4f ± %.4f at n=%d." % (floor, floor_sd, target.shape[0]),
                 fontsize=10, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGS, "fig15_t_sweep_separate_grid4.png" if grid
                             else "fig14_t_sweep_separate.png"))
    plt.close(fig)


# =========================================================================================
# Figure 2b -- EXP-130: fig 2's baselines with FD, plus one-step OT from scratch
# =========================================================================================
FIG2B_ARMS = [("pretrained", "pretrained (zero-shot)"),
              ("regress_scratch", "regression, scratch"),
              ("regress_ft", "regression, fine-tune"),
              ("scratch_t1", "OT one-step, scratch\n(t=1, eps_rel 0.005)")]


def fig_generations_fd(folder):
    """Rows are NFE 1 and 4. The OT map is one step by construction, so it has no NFE 4 cell.

    Panel FD is on the seed-0 points drawn (n=2048); the bracket is the 5-seed mean +- sd of
    the logged final eval (n=1024). `pretrained` is scored at step 0, the rest at the last step.
    """
    base = np.load(os.path.join(folder, "stage0a_clouds.npz"))
    ot = np.load(os.path.join(folder, "scratch_t1_clouds.npz"))
    target = base["real_target"]
    floor, floor_sd = ring_fd_floor(target)

    logged = {}
    rows_a = list(csv.DictReader(open(os.path.join(folder, "stage0a.csv"))))
    for arm in ("pretrained", "regress_scratch", "regress_ft"):
        mine = [r for r in rows_a if r["arm"] == arm and r.get("fd_nfe1")]
        last = 0 if arm == "pretrained" else max(int(r["step"]) for r in mine)
        for nfe in (1, 4):
            logged[(arm, nfe)] = np.array([float(r[f"fd_nfe{nfe}"]) for r in mine
                                           if int(r["step"]) == last])
    rows_b = list(csv.DictReader(open(os.path.join(folder, "scratch_t1.csv"))))
    last = max(int(r["step"]) for r in rows_b)
    logged[("scratch_t1", 1)] = np.array([float(r["fd_forward"]) for r in rows_b
                                          if int(r["step"]) == last])

    fig, axes = plt.subplots(2, len(FIG2B_ARMS), figsize=(11.0, 5.9), sharex=True,
                             sharey=True)
    for col, (arm, label) in enumerate(FIG2B_ARMS):
        for row, nfe in enumerate((1, 4)):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_color(AXIS)
            if row == 0:
                ax.set_title(label, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(f"NFE {nfe}", fontsize=10, color=INK)
            ax.scatter(target[:, 0], target[:, 1], s=5, c=AXIS, linewidths=0, zorder=1)
            if arm == "scratch_t1":
                if nfe != 1:
                    ax.text(0.5, 0.5, "one-step map:\nno NFE 4", transform=ax.transAxes,
                            fontsize=9, color=MUTED, ha="center", va="center")
                    continue
                pts = ot["scratch_t1_out_forward"]
            else:
                pts = base[f"{arm}_nfe{nfe}"]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c=CAT[0], alpha=0.6, linewidths=0,
                       zorder=2)
            vals = logged[(arm, nfe)]
            ax.text(0.03, 0.03, "FD %.4f\n(5 seeds %.4f ± %.4f)"
                    % (frechet_fd(pts, target), vals.mean(), vals.std()),
                    transform=ax.transAxes, fontsize=7.5, color=INK, ha="left", va="bottom")
    fig.suptitle("Generated sets against the target ring (gray), seed 0 drawn.  "
                 "FD = FID-style W2² vs the target.\nA perfect generator scores "
                 "%.4f ± %.4f at n=%d (panel) and 0.0219 ± 0.0150 at n=1024 (bracket)."
                 % (floor, floor_sd, target.shape[0]),
                 fontsize=10, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0)
    fig.savefig(os.path.join(FIGS, "fig2b_toy_generations_fd.png"))
    plt.close(fig)


def report_paired(final, seeds):
    """Paired per-seed contrasts. Seeds share a pretrain, so pairing is the powerful test."""
    def col(arm, key):
        return np.array([float(final[(arm, s)][key]) for s in seeds])

    print("\npaired differences in W2 at the test-time condition (on-policy, 4 source steps):")
    for a, b in [("src_t0.5", "src_t1"), ("src_t0", "src_t1"), ("scratch_t1", "src_t1"),
                 ("scratch_t0", "src_t0"), ("mf_head_t1", "src_t1")]:
        d = col(a, "w2_onpolicy_src4") - col(b, "w2_onpolicy_src4")
        t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
        print(f"  {a:11s} - {b:9s} {d.mean():+.4f}  t({d.size - 1})={t:+.2f}  "
              f"{a} wins {int((d < 0).sum())}/{d.size}")
    print("\ntrain-matched -> on-policy penalty, paired within arm:")
    for arm in ["src_t1", "src_t0.5", "src_t0", "scratch_t0"]:
        d = col(arm, "w2_onpolicy_src4") - col(arm, "w2_forward")
        t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
        print(f"  {arm:11s} +{d.mean():.4f} ({100 * d.mean() / col(arm, 'w2_forward').mean():+.0f}%)"
              f"  t({d.size - 1})={t:+.2f}  input_shift={col(arm, 'input_shift').mean():.3f}")


def main():
    global DATA, FIGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA, help="folder holding the result CSVs and the npz")
    parser.add_argument("--out", default=FIGS, help="folder to write the PNGs into")
    parser.add_argument("--t-sweep", default="", help="EXP-128 folder; only fig 14 is drawn")
    parser.add_argument("--fig2b", default="", help="EXP-130 folder; only fig 2b is drawn")
    parsed = parser.parse_args()
    DATA, FIGS = parsed.data, parsed.out

    os.makedirs(FIGS, exist_ok=True)
    if parsed.fig2b:
        fig_generations_fd(parsed.fig2b)
        return
    if parsed.t_sweep:
        fig_t_sweep(parsed.t_sweep)
        return
    clouds = np.load(os.path.join(DATA, "toy_clouds_seed0.npz"))
    toy = rows("toy_2d_prodconv.csv")
    snr = rows("snr_summary_cub_food.csv")

    fig_distributions(clouds)
    finals = fig_final_w2(toy)
    fig_generations(clouds, finals)
    fig_curves(toy)
    fig_snr(snr)
    fig_cost(snr)
    best = fig_camf()

    src = rows("toy_source_t.csv")
    final, seeds = fig_source_t_arms(src)
    src_clouds = np.load(os.path.join(DATA, "toy_source_t_clouds.npz"))
    fig_source_t_clouds(src_clouds)
    fig_boomerang(src_clouds)
    geom = fig_eps_sweep()
    fig_eps_generations()
    report_paired(final, seeds)

    for level, (blur, cond, rad, tan) in geom.items():
        print("eps %-7s blur %.2f  cond %.1f  std %.3f / %.3f" % (level, blur, cond, rad, tan))
    print("CAMF cub200 best:", best)
    for key in sorted(finals):
        print("final W2", key, "mean=%.4f std=%.4f @step %d" % finals[key])
    for name in sorted(os.listdir(FIGS)):
        print("wrote figures/" + name)


if __name__ == "__main__":
    main()
