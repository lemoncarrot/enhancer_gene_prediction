"""Pairwise control: can distance, raw Hi-C contact and ATAC alone match the walk encoders?

Uses the same chromosome-held-out folds as train.py. No graph walks, no sequence embeddings.
Usage: python pairwise_control.py [--data-dir data/processed] [--output-dir results]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import abc_score, scores
from evaluate import metrics
from models import chromosome_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    pairs, covered, _ = scores(args.data_dir)
    nodes = pd.read_parquet(args.data_dir / "nodes.parquet", columns=["chrom", "center", "log_atac"])
    atac = lambda column: np.where(pairs[column] >= 0, nodes.log_atac.to_numpy()[pairs[column].clip(lower=0)], 0.0)
    pairs["log_distance"] = np.log1p(pairs.distanceToTSS.abs() / 1000)
    pairs["log_contact"] = np.log1p(pairs.hic_contact)
    pairs["atac_elem"], pairs["atac_gene"] = atac("elem_node"), atac("gene_node")   # 0 when the end has no node (no peak)
    pairs["elem_covered"] = (pairs.elem_node >= 0).astype(float)
    pairs["abc"] = abc_score(pairs, nodes, pd.read_parquet(args.data_dir / "hic"))
    pairs["fold"] = pairs.chrom.map(chromosome_folds(pairs, args.folds))

    base = ["log_distance", "log_contact"]
    full = base + ["atac_elem", "atac_gene", "elem_covered"]
    logistic = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    learned = {"lr_distance_contact": (logistic, base),
               "lr_distance_contact_atac": (logistic, full),
               "gbm_distance_contact_atac": (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0), full)}
    for name, (make, features) in learned.items():
        pairs[name] = np.nan
        for k in range(args.folds):
            train, test = pairs.fold != k, pairs.fold == k
            model = make().fit(pairs.loc[train, features], pairs.loc[train, "label"])
            pairs.loc[test, name] = model.decision_function(pairs.loc[test, features])
    models = ["distance", "hic_contact", "abc", *learned]
    rows = []
    for scope, data in [("All benchmark pairs", pairs), ("Covered pairs only", pairs[covered])]:
        for dataset, group in [("Pooled", data), *data.groupby("Dataset")]:
            rows.extend(metrics(group, models, scope=scope, dataset=dataset))
    folds = pd.DataFrame([row for fold, group in pairs.groupby("fold") for row in metrics(group, models, fold=fold)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "pairwise_control.csv", index=False)
    folds.to_csv(args.output_dir / "pairwise_control_folds.csv", index=False)
    print(result.round(3).to_string(index=False))
    print(folds.pivot(index="fold", columns="model", values="AUPRC")[models].round(3))


if __name__ == "__main__":
    main()
