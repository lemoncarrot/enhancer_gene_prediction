"""Compare out-of-fold logits and an optional equal-weight probability ensemble."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve

MODELS = ["mamba", "gnn", "mamba_distance", "gnn_distance", "combined_distance"]


def load_predictions(paths):
    base, scores = None, []
    for path in paths:
        frame = pd.read_parquet(path)
        # Legacy files retain the original benchmark index rather than row_id.
        if "row_id" not in frame:
            if not frame.index.is_unique or not np.issubdtype(frame.index.dtype, np.integer):
                raise ValueError(f"{path}: legacy file needs a unique integer benchmark index")
            frame = frame.reset_index(names="row_id")
        if frame.row_id.isna().any() or frame.row_id.duplicated().any():
            raise ValueError(f"{path}: missing or duplicate row_id")
        frame = frame.set_index("row_id").sort_index()
        required = ["chrom", "label", "distanceToTSS", "elem_node", "gene_node", "Dataset", "fold"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{path}: missing benchmark metadata or fold assignments")
        columns = [name for name in MODELS if name in frame]
        if not columns:
            raise ValueError(f"{path}: no supported model logits")
        metadata = frame.drop(columns=[c for c in [*MODELS, "distance"] if c in frame])
        if base is None:
            base = metadata.copy()
        else:
            try:
                pd.testing.assert_frame_equal(base.drop(columns=scores).sort_index(axis=1),
                                              metadata.sort_index(axis=1), check_dtype=False)
            except AssertionError as error:
                raise ValueError(f"{path}: benchmark rows, labels, or folds do not match") from error
        for name in columns:
            if not np.isfinite(frame[name]).all():
                raise ValueError(f"{path}: nonfinite {name} logits")
            if name in scores:
                if not np.array_equal(base[name], frame[name]):
                    raise ValueError(f"{path}: conflicting predictions for {name}")
            else:
                base[name] = frame[name]
                scores.append(name)
    if not base.label.isin([0, 1]).all() or not np.isfinite(base.distanceToTSS).all():
        raise ValueError("Invalid benchmark labels/distances")
    base["distance"] = -base.distanceToTSS.abs()
    return base, ["distance", *scores]


def metrics(frame, models, **group):
    rows = []
    for model in models:
        pr, rc, _ = precision_recall_curve(frame.label, frame[model])
        rows.append(dict(**group, model=model, n=len(frame), positives=int(frame.label.sum()),
                         AUPRC=average_precision_score(frame.label, frame[model]),
                         precision_at_70_recall=float(pr[rc >= .7].max())))
    return rows


def evaluate(paths, output, ensemble=None):
    frame, models = load_predictions(paths)
    if ensemble:
        if len(set(ensemble)) != 2 or any(name not in models or name == "distance" for name in ensemble):
            raise ValueError("Ensemble needs two distinct learned models present in the inputs")
        frame["ensemble"] = (expit(frame[ensemble[0]]) + expit(frame[ensemble[1]])) / 2
        models.append("ensemble")
    covered = frame[(frame.elem_node >= 0) & (frame.gene_node >= 0)]
    rows = []
    for scope, data in [("All benchmark pairs", frame), ("Covered pairs only", covered)]:
        for dataset, group in [("Pooled", data), *data.groupby("Dataset")]:
            if len(group):
                rows.extend(metrics(group, models, scope=scope, dataset=dataset))
    results = pd.DataFrame(rows)
    folds = pd.DataFrame([row for fold, group in frame.groupby("fold")
                          for row in metrics(group, models, fold=fold)])
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "results.csv", index=False)
    folds.to_csv(output / "fold-results.csv", index=False)
    frame.reset_index().to_parquet(output / "predictions.parquet", index=False)
    (output / "evaluation.json").write_text(json.dumps(
        dict(inputs=[str(Path(p).resolve()) for p in paths], ensemble=ensemble,
             ensemble_rule="equal mean of sigmoid logits" if ensemble else None), indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    args = parser.parse_args()
    if any(args.output_dir.resolve() == p.parent.resolve() for p in args.predictions):
        parser.error("Use a separate evaluation directory to preserve input predictions")
    result = evaluate(args.predictions, args.output_dir, args.ensemble)
    print(result[result.dataset == "Pooled"].to_string(index=False))


if __name__ == "__main__":
    main()
