#!/bin/bash
# Headline comparison: distance and ABC baselines, pair features only (pairwise head, no walks),
# attention + distance + pair features, attention + distance.
#SBATCH --job-name=e2g-headline
#SBATCH --partition=gpu,owners   # owners is preemptible: jobs requeue and resume completed folds
#SBATCH --requeue
#SBATCH -G 1
#SBATCH -C GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sherlock/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/home/groups/hanleeji/Virtual_envs/e2g/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv

for seed in 0 1 2; do
  dir=runs/headline/attention-s$seed
  resume=""; [[ -f $dir/manifest.json ]] && resume=--resume
  python -u train.py --model attention --distance --batch 64 --seed $seed --threads "$SLURM_CPUS_PER_TASK" --run-dir "$dir" $resume
  # Distance and ABC come from evaluate.py; the pair-feature runs (pairwise, attention) are experiment 7's.
  python evaluate.py runs/pairfeat/{pairwise,attention}-s$seed/predictions.parquet "$dir/predictions.parquet" \
    --output-dir runs/headline/evaluation-s$seed
done
