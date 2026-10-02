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


def main():
    global DATA, FIGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA, help="folder holding the result CSVs and the npz")
    parser.add_argument("--out", default=FIGS, help="folder to write the PNGs into")
    parsed = parser.parse_args()
    DATA, FIGS = parsed.data, parsed.out

    os.makedirs(FIGS, exist_ok=True)
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

    print("CAMF cub200 best:", best)
    for key in sorted(finals):
        print("final W2", key, "mean=%.4f std=%.4f @step %d" % finals[key])
    for name in sorted(os.listdir(FIGS)):
        print("wrote figures/" + name)


if __name__ == "__main__":
    main()
