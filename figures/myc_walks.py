"""Figure: sampled random walks around the MYC locus (chr8, K562).

A: candidate elements (tick height = ATAC), the MYC TSS and the CRISPR-tested elements for MYC.
B: the first steps of one cached walk from the MYC promoter as arcs (height grows with the span; color = step order).
C: for three anchors, the share of their 80 cached walks (10 resamples x 8) that visit each node.
Usage: python figures/myc_walks.py [--data-dir data/processed] [--out figures/myc_walks]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from models import DEFAULT_WALKS

# Reference palette (light): categorical slot 2 (orange) for CRISPR positives, blue ordinal ramp 250->650 for step
# order, chart chrome and ink.
POSITIVE = "#eb6834"
STEP_RAMP = LinearSegmentedColormap.from_list("steps", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"])
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WINDOW = (127.45e6, 130.05e6)
ARC_STEPS = 16


def style(ax, left=False):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"] + ([] if left else ["left"]):
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=3)
    ax.set_xlim(WINDOW[0] / 1e6, WINDOW[1] / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("figures/myc_walks"))
    args = parser.parse_args()
    nodes = pd.read_parquet(args.data_dir / "nodes.parquet", columns=["chrom", "center", "log_atac"])
    crispr = pd.read_parquet(args.data_dir / "crispr.parquet")
    pairs = crispr[(crispr.elem_node >= 0) & (crispr.gene_node >= 0)]
    anchors = np.unique(np.r_[pairs.elem_node, pairs.gene_node])
    with np.load(args.data_dir / DEFAULT_WALKS) as cache:
        walks, steps = cache["arr_0"], cache["arr_1"]                 # nodes and sorted directed edge ids per step
    edges = pd.read_parquet(args.data_dir / "edges.parquet")
    src, dst = np.r_[edges.u, edges.v], np.r_[edges.v, edges.u]
    edge_type = np.r_[edges.type, edges.type][np.lexsort((dst, src))]  # same ordering as the walk cache
    center = nodes.center.to_numpy() / 1e6
    myc = crispr[crispr.gene == "MYC"]
    tss, promoter = myc.tss.iloc[0] / 1e6, int(myc.gene_node.iloc[0])
    tested = myc[myc.elem_node >= 0]
    positive_nodes = sorted(set(tested.loc[tested.label == 1, "elem_node"]))
    near, far = positive_nodes[0], max(positive_nodes, key=lambda e: abs(center[e] - tss))
    walks_of = lambda node: walks[:, np.searchsorted(anchors, node)].reshape(-1, walks.shape[-1])   # 80 x 65
    steps_of = lambda node: steps[:, np.searchsorted(anchors, node)].reshape(-1, steps.shape[-1])   # 80 x 64

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "svg.fonttype": "none"})
    fig = plt.figure(figsize=(10, 9.6), facecolor=SURFACE)
    grid = fig.add_gridspec(5, 1, height_ratios=[0.8, 2.1, 1, 1, 1], hspace=0.75,
                            left=0.08, right=0.97, top=0.85, bottom=0.07)

    # A: locus track.
    ax = fig.add_subplot(grid[0])
    style(ax)
    on = ((nodes.chrom == "chr8") & (nodes.center >= WINDOW[0]) & (nodes.center <= WINDOW[1])).to_numpy()
    ax.vlines(center[on], 0, nodes.log_atac.to_numpy()[on], color=AXIS, lw=0.6)
    negative, positive = tested[tested.label == 0], tested[tested.label == 1]
    ax.scatter(negative.center / 1e6, np.full(len(negative), -1), s=22, facecolor=SURFACE, edgecolor=MUTED, lw=1, zorder=3)
    ax.scatter(positive.center / 1e6, np.full(len(positive), -1), s=36, color=POSITIVE, edgecolor=SURFACE, lw=1.5, zorder=4)
    ax.plot([tss], [5.4], marker="v", color=INK, ms=7)
    ax.text(tss + 0.025, 5.4, "MYC", ha="left", va="center", fontsize=10, color=INK, fontweight="semibold")
    ax.text(129.64, 5.4, "distal enhancer cluster", ha="center", va="center", fontsize=9, color=INK_2)
    ax.set_ylim(-1.8, 6.4)
    ax.set_yticks([])
    ax.set_title("A   Locus: candidate elements (tick height = ATAC) and CRISPR-tested elements for MYC",
                 loc="left", fontsize=10.5, color=INK, fontweight="semibold", pad=8)

    # B: arcs for the first steps of one walk from the MYC promoter (the one visiting the most validated enhancers).
    ax = fig.add_subplot(grid[1], sharex=fig.axes[0])
    style(ax)
    ax.spines["bottom"].set_position(("data", 0))
    paths = walks_of(promoter)
    chosen = int(np.argmax([np.isin(p[:ARC_STEPS + 1], positive_nodes).sum() for p in paths]))
    path, path_edges = paths[chosen][:ARC_STEPS + 1], steps_of(promoter)[chosen][:ARC_STEPS]
    t = np.linspace(0, 1, 60)
    for step, (a, b) in enumerate(zip(path[:-1], path[1:])):
        span = abs(center[b] - center[a])
        height = max(span / 2.0, 0.03)                               # chain steps get a small visible arc
        ax.plot(center[a] + (center[b] - center[a]) * t, height * np.sin(np.pi * t),
                color=STEP_RAMP(step / (ARC_STEPS - 1)), lw=2, solid_capstyle="round", zorder=2)
    visited = np.isin(path, positive_nodes)
    ax.scatter(center[path], np.zeros(len(path)), s=12, color=INK_2, zorder=3)
    ax.scatter(center[path[visited]], np.zeros(visited.sum()), s=48, color=POSITIVE, edgecolor=SURFACE, lw=1.5, zorder=4)
    ax.scatter([center[promoter]], [0], s=60, marker="v", color=INK, edgecolor=SURFACE, lw=1, zorder=5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([])
    hic = int((edge_type[path_edges] == 1).sum())
    ax.set_title(f"B   One walk from the MYC promoter, first {ARC_STEPS} steps ({hic} along Hi-C edges, "
                 f"{ARC_STEPS - hic} along the chromosome)", loc="left", fontsize=10.5, color=INK,
                 fontweight="semibold", pad=8)
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=STEP_RAMP, norm=plt.Normalize(1, ARC_STEPS)),
                       cax=ax.inset_axes([0.83, 0.93, 0.15, 0.04]), orientation="horizontal")
    bar.set_label("Step", fontsize=8.5, color=INK_2, labelpad=1)
    bar.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK_2)
    bar.outline.set_visible(False)
    bar.set_ticks([1, 8, ARC_STEPS])

    # C: visit share per node for three anchors.
    panels = [(promoter, "from the MYC promoter"),
              (near, f"from the validated enhancer at +{(center[near] - tss) * 1e3:.0f} kb"),
              (far, f"from the validated enhancer at +{center[far] - tss:.2f} Mb")]
    for row, (anchor, label) in enumerate(panels):
        ax = fig.add_subplot(grid[2 + row], sharex=fig.axes[0])
        style(ax, left=True)
        paths = walks_of(anchor)
        ids, counts = np.unique(np.concatenate([np.unique(p) for p in paths]), return_counts=True)
        share = counts / len(paths)
        keep = (center[ids] >= WINDOW[0] / 1e6) & (center[ids] <= WINDOW[1] / 1e6)
        ids, share = ids[keep], share[keep]
        other = ~np.isin(ids, positive_nodes + [promoter, anchor])
        ax.vlines(center[ids[other]], 0, share[other], color=MUTED, lw=0.8)
        hit = np.isin(ids, positive_nodes) & (ids != anchor)
        ax.vlines(center[ids[hit]], 0, share[hit], color=POSITIVE, lw=2)
        if anchor != promoter:
            p = share[ids == promoter]
            ax.vlines([center[promoter]], 0, p, color=INK, lw=2)
            ax.text(center[promoter] + 0.025, p[0], f"MYC promoter: {p[0]:.0%} of walks", fontsize=8.5,
                    color=INK, va="top")
        ax.plot([center[anchor]], [-0.12], marker="^", color=INK_2, ms=6, clip_on=False)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "50%", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        title = "C   Share of 80 walks visiting each node, " if row == 0 else "     … "
        ax.set_title(title + label, loc="left", fontsize=10.5 if row == 0 else 10, color=INK,
                     fontweight="semibold" if row == 0 else "normal", pad=5)
    ax.set_xlabel("chr8 position (Mb, hg38)", fontsize=9, color=INK_2)

    fig.text(0.08, 0.955, "Random walks on the Hi-C graph link MYC to its distal enhancers", fontsize=14, color=INK,
             fontweight="semibold")
    fig.text(0.08, 0.932, "K562, 64-step node2vec walks over chain and Hi-C edges. Validated enhancers sit 160–230 kb "
             "and ~1.9 Mb downstream of MYC.", fontsize=9.5, color=INK_2)
    legend = [Line2D([], [], ls="", marker="o", ms=7, color=POSITIVE, markeredgecolor=SURFACE, label="CRISPR positive for MYC"),
              Line2D([], [], ls="", marker="o", ms=6, markerfacecolor=SURFACE, markeredgecolor=MUTED, label="CRISPR negative for MYC"),
              Line2D([], [], ls="", marker="v", ms=7, color=INK, label="MYC promoter / TSS"),
              Line2D([], [], color=MUTED, lw=1, label="Other visited nodes"),
              Line2D([], [], ls="", marker="^", ms=6, color=INK_2, label="Walk start")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.075, 0.922), ncol=5, frameon=False, fontsize=8.5,
               labelcolor=INK_2, handletextpad=0.4, columnspacing=1.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        fig.savefig(args.out.with_suffix(suffix), dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}.png/.svg")


if __name__ == "__main__":
    main()
