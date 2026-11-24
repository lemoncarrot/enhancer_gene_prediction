# Experiments

One entry per experiment: question, status, how to run, result. Detailed write-ups of the A100 runs are in [RESULTS.md](RESULTS.md); unrounded metrics are under `results/`.

All numbers are pooled AUPRC on the K562 CRISPR benchmark with 5 chromosome-held-out folds and seed 0 unless stated. "All" is all 10,356 pairs (471 positives); "covered" is the 8,199 pairs (418 positives) whose element and gene both map to graph nodes. Learned walk models rank uncovered pairs last, so **covered is the fair scope for comparing against anything that scores every pair.**

| # | Experiment | Status | Headline |
| --- | --- | --- | --- |
| 1 | Mamba vs GNN, with and without distance | Done | Encoders alone 0.24–0.25; with distance 0.52 (all), 0.58 (covered) |
| 2 | Pair-product head ablation | Done | Removing product costs 0.01–0.03 |
| 3 | Walk length and count | Done | Mamba mildly prefers longer walks; 4 walks ≈ 8 walks |
| 4 | Non-learned baselines: distance, Hi-C edge, raw contact | Done | Raw contact 0.484 beats distance 0.438; thresholded edge 0.081 |
| 5 | Pairwise control without walks or sequence | Done | Boosted trees on distance + contact + ATAC: 0.549 (all), 0.562 (covered) |
| 6 | Dense attention instead of Mamba | **Not run** | — |
| 7 | Do walks add anything on top of pairwise features? | **Not run** | — |

## Done

### 1. Mamba vs GNN, with and without distance

Same sampled walks, same node/edge embeddings and head; only the encoder differs. A100, November 21, 2026, commit `e23d1d7`.

| Model | All | Covered |
| --- | ---: | ---: |
| Distance only | 0.438 | 0.465 |
| GNN | 0.250 | 0.276 |
| Mamba | 0.241 | 0.266 |
| GNN + distance | 0.516 | 0.576 |
| Mamba + distance | 0.523 | 0.583 |

Both combined models beat distance on 4 of 5 folds and lose on fold 0. On Schraivogel2020 (23 positives) every learned model collapses to 0.04–0.08 against 0.49 for distance. Mamba and GNN are not distinguishable at one seed. Note the GNN has two layers and reads out at the anchor, so it only sees nodes within two hops of the anchor in the walk subgraph, while Mamba sees the full walk.

Files: `results/with_pair_product.csv`, `results/with_pair_product_folds.csv`.

### 2. Pair-product head ablation

`--no-pair-product` drops the elementwise enhancer–promoter product from the head. Pooled AUPRC (all) changes by −0.011 for GNN + distance and −0.025 for Mamba + distance. Files: `results/without_pair_product*.csv`, `results/pair_product_comparison.csv`.

### 3. Walk length and count

With distance and the default head. Shorter caches are prefixes of the 64-step, 8-walk cache ([derive_walk_ablation.py](derive_walk_ablation.py)).

| Steps / walks | GNN + distance | Mamba + distance | Mamba fold time (s) |
| --- | ---: | ---: | ---: |
| 64 / 8 | 0.516 | 0.523 | 825 |
| 32 / 8 | 0.475 | 0.513 | 673 |
| 16 / 8 | 0.510 | 0.490 | 602 |
| 64 / 4 | 0.477 | 0.525 | 417 |

File: `results/walk_ablation.csv`.

### 4. Non-learned baselines

`python baselines.py` (CPU, seconds). Run locally November 23, 2026.

| Score | All | Covered | Precision at 70% recall (all) |
| --- | ---: | ---: | ---: |
| Distance | 0.438 | 0.465 | 0.316 |
| Raw Hi-C contact between element bin and TSS bin | 0.484 | 0.514 | 0.432 |
| Same, divided by expected (no threshold) | 0.091 | 0.110 | 0.060 |
| Graph has a Hi-C edge (yes/no) | 0.081 | 0.089 | 0.045 |

The graph's Hi-C edges require observed/expected ≥ 3, which almost never holds at short range: under 20 kb, 51% of pairs are positive but only 2% have an edge. Raw contact beats distance inside every distance band. **The graph is built from the weak version of the contact signal.** File: `results/baselines.csv`.

### 5. Pairwise control

`python pairwise_control.py` (CPU, under a minute). Same chromosome folds as `train.py`. Features: log distance, log raw contact, ATAC at element and promoter, element-covered flag. Run locally November 23, 2026.

| Model | All | Covered | Precision at 70% recall (all) |
| --- | ---: | ---: | ---: |
| ABC-style score, no training (ATAC × contact over its sum around the gene) | 0.491 | 0.550 | 0.411 |
| Logistic: distance + contact | 0.480 | 0.512 | 0.426 |
| Logistic: distance + contact + ATAC | 0.520 | 0.526 | 0.452 |
| Boosted trees: distance + contact + ATAC | 0.549 | 0.562 | 0.488 |
| *Mamba + distance (experiment 1)* | *0.523* | *0.583* | *0.359* |

Five pairwise numbers match the walk models (within ~0.02 on covered pairs, ahead on all pairs and on 5 of 5 folds against Mamba + distance). The ABC-style denominator adds 0.036 over raw contact on covered pairs, which is evidence that neighbourhood context carries some signal. Boosted-tree settings were fixed once and not tuned. Files: `results/pairwise_control.csv`, `results/pairwise_control_folds.csv`.

## Not run

Both need a GPU and the cached inputs described in the [README](README.md#cached-inputs) (`nodes.parquet` **with `seq_emb`**, `edges.parquet`, `crispr.parquet`, the walk cache). `data/` is not in Git.

### 6. Dense attention instead of Mamba

**Question.** Is the selective state-space mixer doing anything a plain transformer over the same walk tokens would not?

**Setup.** `--model attention` replaces the two bidirectional Mamba blocks with two pre-norm self-attention blocks (4 heads, feed-forward 4×, no dropout) plus a learned step-position embedding. Everything else is identical. 198k parameters per block against 233k for a bidirectional Mamba block. Needs no CUDA extras.

```bash
python train.py --model attention --distance --batch 64 --run-dir runs/attention-distance
python train.py --model attention --batch 64 --run-dir runs/attention
python evaluate.py runs/attention-distance/predictions.parquet runs/attention/predictions.parquet --output-dir runs/attention-evaluation
```

**Compare with** experiment 1: Mamba 0.241 / Mamba + distance 0.523 (all), 0.266 / 0.583 (covered).

**Record** `runs/attention-evaluation/results.csv` and `fold-results.csv` as `results/attention.csv` and `results/attention_folds.csv`, and fill in the table at the top.

### 7. Do walks add anything on top of pairwise features?

**Question.** Experiment 5 showed pairwise features alone match the walk models. The walk models were never given raw contact, so the open question is whether walk embeddings improve a model that already has distance, raw contact and ATAC. If yes, the local-network idea holds as a context correction on top of pairwise signal. If no, the earlier gain over distance was a blurred reconstruction of contact.

**Setup.** `--pair-features` appends three scalars to the prediction head next to distance: log raw Hi-C contact between the element's and TSS's 5 kb bins, and log ATAC at the element and promoter nodes. `--model pairwise` is the matched control: the same head, optimizer and epochs with no encoder at all, so the only difference from the walk runs is the walk embeddings.

**Extra input.** `--pair-features` reads the per-chromosome Hi-C contact cache at `<data-dir>/hic/*.parquet` (about 95 MB, written by the Hi-C section of `main.ipynb`). Check it exists on the GPU machine before queueing; if it is missing, copy it from a machine that has it or rerun that notebook section. The run manifest hashes these files.

**Runs.** Seed 0 is required. Seeds 1 and 2 are strongly recommended because the expected differences are around 0.02, which is the size of single-seed noise in earlier experiments.

```bash
for seed in 0 1 2; do
  python train.py --model pairwise  --distance --pair-features --batch 64 --seed $seed --run-dir runs/pairfeat/pairwise-s$seed
  python train.py --model gnn       --distance --pair-features --batch 64 --seed $seed --run-dir runs/pairfeat/gnn-s$seed
  python train.py --model mamba     --distance --pair-features --batch 64 --seed $seed --run-dir runs/pairfeat/mamba-s$seed
  python train.py --model attention --distance --pair-features --batch 64 --seed $seed --run-dir runs/pairfeat/attention-s$seed
  python evaluate.py runs/pairfeat/{pairwise,gnn,mamba,attention}-s$seed/predictions.parquet --output-dir runs/pairfeat/evaluation-s$seed
done
```

Expected time on an A100 per seed, from experiment 3: pairwise seconds, GNN about 2 minutes, Mamba about 14 minutes; attention is untimed on GPU. Each run is independent and can be queued separately (`sbatch chimera.sbatch <same arguments>`); if interrupted, repeat the same command with `--resume`.

**Read the result on covered pairs** (`scope == "Covered pairs only"`, `dataset == "Pooled"`), and per fold in `fold-results.csv`. Columns are `pairwise_distance_pairfeat`, `gnn_distance_pairfeat`, `mamba_distance_pairfeat`, `attention_distance_pairfeat`.

- **Walks help** if a walk model beats `pairwise_distance_pairfeat` by more than the spread across seeds, on most folds.
- **Walks do not help** if the walk models are within seed noise of the pairwise control or below it.
- Sanity check: `pairwise_distance_pairfeat` should land near the logistic and boosted-tree numbers from experiment 5 on covered pairs (0.53–0.56). If it is far below, the head is not using the scalars well and the comparison is not informative yet.
- Also report Schraivogel2020 and fold 0, where the walk models previously collapsed while pairwise features did not.

**Record** each seed's `results.csv` and `fold-results.csv` as `results/pair_features_s<seed>.csv` and `results/pair_features_folds_s<seed>.csv`, add a mean ± spread table here, and update the table at the top.

## Ideas not yet implemented

- **Raw contact as graph edge weight.** Experiment 4 says the thresholded observed/expected edges discard the useful part of Hi-C.
- **Pair-conditioned, causal Mamba.** Start the walk at the enhancer, drop the reverse pass, and read out at the token where the walk reaches the target promoter. The existing cache already reaches the tested gene in 72% of covered pairs with 8 walks and 98% with 80. This is what would make per-step interpretation (CTCF and gene crossings) possible. Needs CTCF as a node feature and "elements skipped" on Hi-C steps.
- **Larger sequence encoders** than HyenaDNA tiny (for example Evo 2).
- **ATAC ablations.** ATAC is 1 of about 257 node inputs and shares a LayerNorm with the sequence embedding; no run isolates its contribution.
- **Deeper GNN or pooled readout,** so the GNN sees the whole walk subgraph and not only two hops.
