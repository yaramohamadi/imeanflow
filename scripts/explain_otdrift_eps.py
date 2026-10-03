"""What `eps_rel` actually is, drawn rather than described.

The OT loss has to decide which of OUR points corresponds to which REAL target point. A hard
one-to-one matching is not differentiable-friendly, so the plan is softened: every one of our
points is matched to MANY real points, with weights. `eps` is the knob that sets how many.

`eps` multiplies an entropy term against a squared-distance cost, so it has units of squared
distance and `sqrt(eps)` is a LENGTH -- the radius inside which the loss cannot tell two real
points apart. `eps_rel` is how we pick it: `eps = eps_rel * mean(cost)`.

This script solves the real plan (same log-domain Sinkhorn as `otdrift.py`, uniform marginals,
N=512 as in training) and draws, for one of our points, which real points it is matched to.

Writes `figures/fig13_what_eps_is.png`.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, FIGS = os.path.join(HERE, "data"), os.path.join(HERE, "figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 160,
})

PARTICLES = 512        # what the toy trains with
LEVELS = [0.05, 0.005, 0.001]
NOTES = {0.05: "what we trained with", 0.005: "the fix", 0.001: "one step too far"}


def sinkhorn_plan(cost, epsilon, iterations=400):
    """log-domain Sinkhorn with uniform marginals; returns the plan, which sums to 1."""
    n, m = cost.shape
    f, g = np.zeros(n), np.zeros(m)
    log_a, log_b = -np.log(n), -np.log(m)
    for _ in range(iterations):
        M = (f[:, None] + g[None] - cost) / epsilon + log_a + log_b
        g = g + epsilon * (log_b - (np.log(np.exp(M - M.max(0)).sum(0)) + M.max(0)))
        M = (f[:, None] + g[None] - cost) / epsilon + log_a + log_b
        f = f + epsilon * (log_a - (np.log(np.exp(M - M.max(1, keepdims=True)).sum(1)).ravel()
                                    + M.max(1)))
    M = (f[:, None] + g[None] - cost) / epsilon + log_a + log_b
    return np.exp(M)


def main():
    cloud = np.load(os.path.join(DATA, "eps_sweep", "eps_0.05_clouds.npz"))
    ours = cloud["src_t0_out_forward"][:PARTICLES]
    real = cloud["real_target"][:PARTICLES]
    cost = ((ours[:, None] - real[None]) ** 2).sum(-1)
    mean_cost = float(cost.mean())

    # one of our points, picked once and used in every panel so the panels are comparable
    pick = int(np.argmin(np.linalg.norm(ours - ours.mean(0), axis=1)
                         - np.linalg.norm(ours - ours.mean(0), axis=1)))  # = 0, kept explicit
    ramp = LinearSegmentedColormap.from_list("ord", [GRID] + ORD)
    circle = np.stack([np.cos(np.linspace(0, 2 * np.pi, 200)),
                       np.sin(np.linspace(0, 2 * np.pi, 200))], 1)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.0))
    for ax, level in zip(axes, LEVELS):
        epsilon = level * mean_cost
        plan = sinkhorn_plan(cost, epsilon)
        row = plan[pick] / plan[pick].sum()
        partners = 1.0 / (row ** 2).sum()      # effective number of matched real points

        order = np.argsort(row)                # faint points first, bright on top
        ax.scatter(real[order, 0], real[order, 1], s=46, c=row[order] / row.max(),
                   cmap=ramp, vmin=0, vmax=1, linewidths=0, zorder=2)
        ax.plot(*(circle * np.sqrt(epsilon) + ours[pick]).T, color=CAT[1], linewidth=1.6,
                linestyle=(0, (5, 3)), zorder=4)
        ax.plot(*(circle * 0.25 + ours[pick]).T, color=INK2, linewidth=1.4, zorder=5)
        ax.plot(ours[pick, 0], ours[pick, 1], "x", color=INK, markersize=11,
                markeredgewidth=2.2, zorder=6)
        # zoomed to the picked point: the figure is about two radii, so make them legible
        window = 1.45
        ax.set_xlim(ours[pick, 0] - window, ours[pick, 0] + window)
        ax.set_ylim(ours[pick, 1] - window, ours[pick, 1] + window)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for side in ax.spines.values():
            side.set_color(AXIS)
        ax.set_title(f"eps_rel {level}  ->  blur {np.sqrt(epsilon):.2f}   ({NOTES[level]})",
                     fontsize=9.5, loc="left")
        ax.text(0.03, 0.04, f"this one point is matched to\n{partners:.0f} of the "
                f"{PARTICLES} real points", transform=ax.transAxes, fontsize=8.5, color=INK,
                ha="left", va="bottom")

    axes[0].plot([], [], "x", color=INK, markersize=9, markeredgewidth=2,
                 label="one point WE generated")
    axes[0].plot([], [], color=CAT[1], linewidth=1.6, linestyle=(0, (5, 3)),
                 label="the blur radius it buys, sqrt(eps)")
    axes[0].plot([], [], color=INK2, linewidth=1.4, label="the blob we need it to resolve, 0.25")
    axes[0].scatter([], [], s=26, c=ORD[3], label="real target points, darker = more of the\n"
                    "match weight this one point was given")
    # one horizontal legend under the whole row: a vertical one under panel 1 leaves the
    # bottom-right third of the canvas empty
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, labelcolor=INK2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005), handletextpad=0.6, columnspacing=2.2)

    fig.suptitle("What eps_rel is: how many real points each of OUR points gets matched to.\n"
                 "eps = eps_rel x mean(cost). mean(cost) is set by the whole ring (26.0), not by "
                 "the blob, so eps_rel 0.05 buys a blur 4.5x too wide.",
                 fontsize=10, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    fig.savefig(os.path.join(FIGS, "fig13_what_eps_is.png"))
    plt.close(fig)
    print("wrote figures/fig13_what_eps_is.png")
    print(f"mean cost over {PARTICLES} particles: {mean_cost:.2f}")


if __name__ == "__main__":
    main()
