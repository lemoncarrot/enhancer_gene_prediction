# Results

## Pair-product and distance ablations (November 21, 2026)

These are two matched A100 runs on the `enbao` branch. The first used the default prediction head, which concatenates the enhancer embedding, promoter embedding, their elementwise product, and (when enabled) pair distance. It ran at commit `e23d1d7`. The second omitted the elementwise product with `--no-pair-product` and ran at commit `f009632`. The no-product head therefore has fewer input features; the graph, walks, node encoder, and distance calculation are unchanged.

Both runs used the same cached input file hashes, chromosome-held-out fold assignments, and settings: five folds, seed 0, embedding width 128, two encoder layers, 30 epochs, batch size 64, AdamW learning rate 0.001, and weight decay 0.01. The four learned variants in each run were GNN and Mamba, each with and without `--distance`. The GPU was an NVIDIA A100-SXM4-40GB. The Mamba run used the compiled CUDA backend.

### Pooled benchmark results

All-pair evaluation contains 10,356 enhancer–gene pairs, including 471 positives. Of these, 8,199 pairs (418 positives) have both nodes in the graph. Uncovered pairs receive the fixed learned-model score of `-1e9`, as in the original evaluation. The distance-only baseline has AUPRC **0.438** and precision at 70% recall **0.316** in both runs.

| Learned model | AUPRC with product | AUPRC without product | Precision at 70% recall with product | Precision at 70% recall without product |
| --- | ---: | ---: | ---: | ---: |
| GNN | 0.250 | 0.238 | 0.122 | 0.110 |
| GNN + distance | 0.516 | 0.505 | 0.278 | 0.331 |
| Mamba | 0.241 | 0.241 | 0.104 | 0.108 |
| Mamba + distance | **0.523** | 0.497 | **0.359** | 0.291 |

Adding explicit distance substantially improved both architectures in either head design. Omitting the product reduced pooled AUPRC by 0.011 for GNN + distance and 0.025 for Mamba + distance; Mamba without distance was nearly unchanged. Precision at 70% recall moved in both directions, so the product is not uniformly better by every metric. For Mamba + distance, the no-product AUPRC rose slightly on Gasperini2019 (0.575 to 0.582) but fell on Nasser2021 (0.506 to 0.407) and Schraivogel2020 (0.080 to 0.035; 23 positives). These are single-seed comparisons without validation-set tuning or uncertainty intervals; they do not establish that either head is optimal.

### Metric files

- [With-product results](results/with_pair_product.csv) and [fold results](results/with_pair_product_folds.csv)
- [Without-product results](results/without_pair_product.csv) and [fold results](results/without_pair_product_folds.csv)
- [Row-aligned comparison](results/pair_product_comparison.csv), including per-dataset and graph-covered metrics

The CSVs contain the unrounded values. Large prediction files and model checkpoints are kept outside Git in the run archives.

### Reproduce the no-product runs

Starting from the same cached `data/processed` inputs, run each variant in its own empty directory:

```bash
python train.py --model gnn --no-pair-product --batch 64 --run-dir runs/no_pair_product/gnn
python train.py --model gnn --distance --no-pair-product --batch 64 --run-dir runs/no_pair_product/gnn_distance
python train.py --model mamba --no-pair-product --batch 64 --run-dir runs/no_pair_product/mamba
python train.py --model mamba --distance --no-pair-product --batch 64 --run-dir runs/no_pair_product/mamba_distance
python evaluate.py runs/no_pair_product/{gnn,gnn_distance,mamba,mamba_distance}/predictions.parquet \
  --output-dir runs/no_pair_product/evaluation
```

The matching with-product run uses the same commands without `--no-pair-product` and with different run directories. See the [training instructions](README.md#train-and-resume) for setup and resume rules.
