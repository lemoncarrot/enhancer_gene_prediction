"""Build sparser / more local Hi-C graph variants and their walk caches from data/processed.

Each variant changes only the Hi-C edges and the chain-edge walk weight; nodes, benchmark and Hi-C contacts are shared
(symlinked). Graph and walk code are executed from main.ipynb, so the sampler is exactly the notebook's.
Usage: python graph_variants.py [--data-dir data/processed] [--out-dir data/variants]
Train on a variant with: train.py --data-dir data/variants/<name> --walks <walks file listed in its stats.json>
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = [None, 5]            # keep each node's k strongest Hi-C edges (by O/E); an edge survives if either end keeps it
MAX_LENGTH = [None, 1_000_000]   # drop Hi-C edges longer than this (bp)
BETA = [5.0, 20.0]           # walk weight of chain edges (Hi-C edges are weighted by O/E)
DEFAULT_WALK = "WALK = dict(L=64, p=4.0, q=1.0, beta=5.0)"


def variant_name(k, cap, beta):
    return f"k{k or 'all'}_cap{cap // 1000 if cap else 'none'}kb_b{beta:g}".replace("capnonekb", "capnone")


def sparsify(edges, k, cap):
    chain, hic = edges[edges.type == 0], edges[edges.type == 1]
    if cap is not None:
        hic = hic[hic.dist.abs() <= cap]
    if k is not None:
        ends = pd.DataFrame({"node": np.r_[hic.u, hic.v], "edge": np.r_[hic.index, hic.index], "oe": np.r_[hic.oe, hic.oe]})
        ends = ends.sort_values(["node", "oe"], ascending=[True, False], kind="stable")
        keep = ends[ends.groupby("node").cumcount() < k].edge.unique()
        hic = hic.loc[np.sort(keep)]
    return pd.concat([chain, hic], ignore_index=True)   # original row order within each type


def walk_stats(directory, walk_file, edges, nodes):
    center = nodes.center.to_numpy()
    src, dst = np.r_[edges.u, edges.v], np.r_[edges.v, edges.u]
    types = np.r_[edges.type, edges.type][np.lexsort((dst, src))]
    with np.load(directory / walk_file) as cache:
        W, E = cache["arr_0"], cache["arr_1"]
    offset = np.abs(center[W] - center[W[..., :1]])
    hic = edges[edges.type == 1]
    degree = np.bincount(np.r_[hic.u, hic.v], minlength=len(nodes))
    return dict(walks=walk_file, hic_edges=int(len(hic)), hic_degree_median=float(np.median(degree)),
                hic_degree_max=int(degree.max()), hic_edge_length_median_kb=float(hic.dist.abs().median() / 1e3),
                hic_step_fraction=float((types[E] == 1).mean()),
                reach_median_kb=float(np.median(offset.max(-1)) / 1e3),
                reach_p25_p75_kb=[float(x) for x in np.percentile(offset.max(-1), [25, 75]) / 1e3],
                final_step_median_kb=float(np.median(offset[..., -1]) / 1e3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/variants"))
    args = parser.parse_args()
    os.cpu_count = lambda: len(os.sched_getaffinity(0))   # size the notebook's walk pool to this job's CPUs
    cells = json.loads(Path("main.ipynb").read_text())["cells"]
    walk_cell = "".join(cells[28]["source"])
    if walk_cell.count(DEFAULT_WALK) != 1:
        raise ValueError("main.ipynb walk settings changed; update DEFAULT_WALK")
    # Real __main__ namespace: the walk cell sends its worker function to a process pool by name.
    namespace = vars(sys.modules["__main__"])
    exec(compile("".join(cells[1]["source"]), "main.ipynb[1]", "exec"), namespace)
    nodes = pd.read_parquet(args.data_dir / "nodes.parquet", columns=["chrom", "center", "is_promoter"])
    crispr = pd.read_parquet(args.data_dir / "crispr.parquet")
    full = pd.read_parquet(args.data_dir / "edges.parquet")
    summary = {}
    for k, cap, beta in itertools.product(TOP_K, MAX_LENGTH, BETA):
        name = variant_name(k, cap, beta)
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        for shared in ["nodes.parquet", "crispr.parquet", "hic"]:
            link = out / shared
            if not link.exists():
                link.symlink_to((args.data_dir / shared).resolve())
        edges = sparsify(full, k, cap)
        edges.to_parquet(out / "edges.parquet")
        namespace.update(nodes=nodes, crispr=crispr, edges=edges, PROC=out)
        print(f"--- {name}: {int((edges.type == 1).sum()):,} Hi-C edges", flush=True)
        exec(compile("".join(cells[25]["source"]), "main.ipynb[25]", "exec"), namespace)   # Graph, random_walk, g
        cell = walk_cell.replace(DEFAULT_WALK, f"WALK = dict(L=64, p=4.0, q=1.0, beta={beta})")
        exec(compile(cell, "main.ipynb[28]", "exec"), namespace)
        stats = dict(top_k=k, max_length=cap, beta=beta, **walk_stats(out, namespace["walk_name"], edges, nodes))
        (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        summary[name] = stats
    print(pd.DataFrame(summary).T.drop(columns=["walks", "reach_p25_p75_kb"]).to_string())


if __name__ == "__main__":
    main()
