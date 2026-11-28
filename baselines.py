"""Non-learned baselines on the CRISPR benchmark: genomic distance, Hi-C contact and edges, and an ABC-style score.

Usage: python baselines.py [--data-dir data/processed] [--output results/baselines.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from evaluate import metrics

HIC_RES = 5_000   # bin size of the cached Hi-C contacts


def scores(directory):
    pairs = pd.read_parquet(directory / "crispr.parquet").reset_index(drop=True)
    edges = pd.read_parquet(directory / "edges.parquet")
    hic = edges[edges.type == 1]
    key = lambda a, b: np.minimum(a, b).astype(np.int64) * 10**7 + np.maximum(a, b)   # undirected node pair
    oe = pd.Series(hic.oe.values, index=key(hic.u.values, hic.v.values))
    pair_oe = oe.reindex(key(pairs.elem_node.values, pairs.gene_node.values)).to_numpy(copy=True)
    covered = ((pairs.elem_node >= 0) & (pairs.gene_node >= 0)).to_numpy()
    pair_oe[~covered] = np.nan                                  # uncovered pairs cannot have an edge
    distance = -pairs.distanceToTSS.abs()
    pairs["distance"] = distance
    pairs["hic_edge"] = (~np.isnan(pair_oe)).astype(float)      # 1 if the graph has a Hi-C edge enhancer–promoter
    pairs["hic_edge_oe"] = np.nan_to_num(pair_oe)               # that edge's O/E, 0 if absent
    # Edge first, distance as tie-break within each group (rank-based so scales don't matter).
    pairs["hic_edge_then_distance"] = pairs.hic_edge * len(pairs) + distance.rank()
    # Raw contact between the element's and the TSS's 5 kb bins, straight from the cached Hi-C records.
    # Defined for every pair (covered or not); bin pairs with no record have zero contact.
    contacts = pd.read_parquet(directory / "hic").set_index(["chrom", "bin1", "bin2"])[["observed", "oe"]]
    a, b = pairs.center // HIC_RES * HIC_RES, pairs.tss // HIC_RES * HIC_RES
    found = contacts.reindex(pd.MultiIndex.from_arrays([pairs.chrom, np.minimum(a, b), np.maximum(a, b)]))
    pairs["hic_contact"] = found.observed.fillna(0).to_numpy()      # SCALE-normalized observed contact
    pairs["hic_contact_oe"] = found.oe.fillna(0).to_numpy()         # same, divided by expected at that distance (no threshold)
    return pairs, covered, ["distance", "hic_contact", "hic_contact_oe",
                            "hic_edge", "hic_edge_oe", "hic_edge_then_distance"]


def abc_score(pairs, nodes, hic):
    """ABC-style share: activity x contact of the element, over the same sum for every node on the gene's chromosome
    (contacts are cached within 5 Mb). Activity is ATAC reads per million only; no H3K27ac, no close-range power-law fix."""
    out = np.zeros(len(pairs))
    activity = np.expm1(nodes.log_atac.to_numpy())
    node_bin = (nodes.center // HIC_RES).to_numpy()
    for chrom, group in pairs.groupby("chrom"):
        h = hic[hic.chrom == chrom]
        if h.empty:
            continue                                                    # no recorded contacts: score stays 0
        on = (nodes.chrom == chrom).to_numpy()
        size = int(max(h.bin2.max() // HIC_RES, group.tss.max() // HIC_RES, group.center.max() // HIC_RES, node_bin[on].max())) + 1
        i, j = (h.bin1 // HIC_RES).to_numpy(), (h.bin2 // HIC_RES).to_numpy()
        off = i != j                                                    # symmetrize without doubling the diagonal
        contact = coo_matrix((np.r_[h.observed, h.observed[off]], (np.r_[i, j[off]], np.r_[j, i[off]])), shape=(size, size)).tocsr()
        tss_bins = (group.tss // HIC_RES).to_numpy()
        unique, inverse = np.unique(tss_bins, return_inverse=True)
        rows = contact[unique][:, node_bin[on]]                         # contact from each TSS bin to every node on the chromosome
        denominator = np.asarray(rows @ activity[on])[inverse]
        covered = (group.elem_node >= 0).to_numpy()
        numerator = np.zeros(len(group))
        numerator[covered] = activity[group.elem_node.to_numpy()[covered]] * group.hic_contact.to_numpy()[covered]
        out[pairs.index.get_indexer(group.index)] = numerator / np.maximum(denominator, 1e-12)
    return out


def abc_baseline(directory):
    """ABC-style score for every benchmark row (crispr.parquet order); 0 when the element has no node."""
    pairs, _, _ = scores(directory)
    nodes = pd.read_parquet(directory / "nodes.parquet", columns=["chrom", "center", "log_atac"])
    return pairs, abc_score(pairs, nodes, pd.read_parquet(directory / "hic"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines.csv"))
    args = parser.parse_args()
    pairs, covered, models = scores(args.data_dir)
    _, pairs["abc"] = abc_baseline(args.data_dir)
    models.append("abc")
    rows = []
    for scope, data in [("All benchmark pairs", pairs), ("Covered pairs only", pairs[covered])]:
        for dataset, group in [("Pooled", data), *data.groupby("Dataset")]:
            rows.extend(metrics(group, models, scope=scope, dataset=dataset))
            rows[-len(models):] = [dict(row, hic_edge_rate=group.hic_edge.mean(),
                                        hic_edge_rate_positives=group.hic_edge[group.label == 1].mean())
                                   for row in rows[-len(models):]]
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
