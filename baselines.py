"""Non-learned baselines on the CRISPR benchmark: genomic distance and Hi-C edge presence.

Usage: python baselines.py [--data-dir data/processed] [--output results/baselines.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines.csv"))
    args = parser.parse_args()
    pairs, covered, models = scores(args.data_dir)
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
