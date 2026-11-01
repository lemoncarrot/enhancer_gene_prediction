# Enhancer–gene prediction

Compare bidirectional Mamba and GINE graph encoders on the same sampled regulatory neighborhoods, with optional enhancer–gene distance. The combined model trains both encoders and one classifier jointly. A prediction ensemble instead averages two separately trained models' probabilities.

## Setup

Cached-data training needs Python 3.10–3.12 and the small dependency set below. Use a Linux NVIDIA GPU for the original Mamba implementation; GNN and small smoke tests also run on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
# For Mamba/combined on Linux: first provide a CUDA toolkit matching torch.version.cuda.
python -m pip install packaging ninja wheel
python -m pip install -r requirements-cuda.txt --no-build-isolation
python -c 'import torch; from mamba_ssm import Mamba; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The minimal CPU environment was tested on macOS with Python 3.12. The compiled-kernel pins come from the original A100 environment (Python 3.10, Torch 2.7.0, Mamba 2.2.6.post3, causal-conv1d 1.7.0). A fresh GPU installation of this refactor has **not** been verified. The original server used PyG 2.8.1 and PyArrow 25.0.1; the minimal environment uses PyG 2.8.0 and PyArrow 20.0.0. Model parity checks passed in that environment; bitwise retraining reproducibility across environments/devices is not promised.

`environment.yml` is the broader, legacy **data preparation** environment, not a training lockfile. For rebuilding data, use `download_data.sh`, then the preparation, graph/walk, and embedding sections of `main.ipynb`. The notebook's training/evaluation cells now call the same Python modules as the CLI. The notebook is no longer parsed or patched by experiment scripts.

## Cached inputs

Pass a directory containing these four files with `--data-dir` (default `data/processed`):

- `nodes.parquet`, including `node_id`, `center`, `log_atac`, and cached `seq_emb`.
- `edges.parquet`, including `u`, `v`, `type`, and `oe` in the original ordering.
- `crispr.parquet`, the full benchmark including labels, chromosome, dataset, coordinates, mapped nodes and `distanceToTSS`.
- `walks_L64_p4.0_q1.0_b5.0_N8_R10.npz`, with node walks in `arr_0` and CSR edge IDs in `arr_1`. Use `--walks filename.npz` for another cache.

The existing server backup contains these files under `data/processed`; no raw BAM downloads or embedding recomputation are needed to rerun training from that backup. The loader checks anchor ordering and every walk edge against the graph. Keep the files together; independently regenerated caches are not interchangeable merely because their shapes match.

## Train and resume

Each command is independent: it does not need another model's checkpoints or predictions.

```bash
python train.py --model mamba --distance --run-dir runs/mamba-distance
python train.py --model gnn --distance --run-dir runs/gnn-distance
python train.py --model combined --run-dir runs/combined-distance
```

Omit `--distance` for the original Mamba or GNN baseline. Combined always includes distance. Defaults preserve the experiments: width 128, two layers, 30 epochs, batch 128, AdamW with learning rate 0.001 and weight decay 0.01, five chromosome-held-out folds, seed 0. Training cycles through cached walk resamples and evaluation averages logits over the first four. Distance is `log1p(abs(distanceToTSS)/1000)`.

To resume, repeat the **same command and arguments** with `--resume`. The manifest checks settings, SHA-256 input/code hashes, chromosome assignments, Python and core package versions. Completed folds contain weights, row IDs and predictions, with a checksum recorded only after the checkpoint is written. Interrupted folds restart from their beginning. Changed settings/data/code require a new run directory; old experiment checkpoints are not silently imported as resumable runs. Do not edit input files while a run is active.

Outputs are `manifest.json`, `foldN.pt`, `foldN.json`, and `predictions.parquet`. The last file contains full benchmark metadata and out-of-fold logits; uncovered pairs retain the historical score of -1e9. There is no validation-set tuning or early stopping. Different models can run in separate directories; a lock prevents two writers using the same directory.

For a small CPU/debug run, add `--device cpu --backend torch --d 8 --layers 1 --epochs 1`. The plain-PyTorch Mamba backend is a separate implementation with different checkpoint keys and initialization; it does **not** replace the CUDA backend for reproducing the A100 experiments. GNN ignores the backend option.

On Chimera, activate the environment first, then submit from the repo root:

```bash
sbatch chimera.sbatch --model gnn --distance --run-dir runs/gnn-distance
# If interrupted, resubmit the same arguments plus --resume.
```

Adjust the partition directive for your account. The batch script has no personal paths, fixed PIDs, automatic requeue, or notebook execution.

## Evaluate

```bash
python evaluate.py runs/mamba-distance/predictions.parquet \
  runs/gnn-distance/predictions.parquet runs/combined-distance/predictions.parquet \
  --ensemble mamba_distance gnn_distance --output-dir runs/comparison
```

One input is sufficient to evaluate a single model. The evaluator aligns by benchmark row ID and checks all metadata, including labels and folds, before combining files. Duplicate model columns must agree. It also reads the original experiment prediction parquets, whose benchmark row IDs were stored as their index. Use a separate output directory.

Outputs are pooled/per-dataset metrics for all pairs and covered pairs, fold metrics, merged predictions and the ensemble specification. Metrics are average precision (AUPRC) and the maximum precision at recall ≥ 70%. The optional ensemble is exactly `0.5*sigmoid(mamba_distance_logits) + 0.5*sigmoid(gnn_distance_logits)`; its weights are not fitted on test labels. Distance alone is always included as a baseline.

Original all-pair pooled AUPRC: distance **0.437696**, Mamba **0.261687**, GNN **0.262723**, Mamba+distance **0.493734**, GNN+distance **0.479541**, jointly trained combined+distance **0.468754**, probability ensemble **0.512476**. The new evaluator reproduces these from the saved predictions. They are one-seed results, not evidence that the architecture or hyperparameters are optimal.

## Verification and scope

```bash
python -m unittest discover -s tests -v
```

Tests exercise all three model families on tiny CPU fixtures, fold resume/restart, changed-data/config/code refusal, checkpoint corruption, walk-edge identity, and prediction alignment. Additional one-off checks compared all five model variants' logits and gradients against the frozen original implementation, and loaded actual baseline/distance GNN checkpoints to compare CPU inference against saved GPU predictions.

This is a behavior-preserving training refactor. The historical GNN's deduplication includes directed CSR edge IDs, so opposite traversals can retain duplicate edges. That behavior and the existing center-cropped sequence embeddings are deliberately preserved; changing them is a separate scientific experiment. GPU Mamba parity and a full new A100 training run remain untested.

`models.py` owns data loading and architectures, `train.py` owns training/resume, and `evaluate.py` owns metrics and ensembling. Historical one-off launch, transfer and notebook-patching scripts remain only in the archived server snapshot; they are not part of the active workflow. Large data, checkpoints and runs remain ignored by Git.
