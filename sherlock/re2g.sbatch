#!/bin/bash
# Hierarchy rows 3-4 with rE2G-style hand-crafted features: pairwise head (no walks) and attention walks.
#SBATCH --job-name=e2g-re2g
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
  for model in pairwise attention; do
    dir=runs/re2g/$model-s$seed
    resume=""; [[ -f $dir/manifest.json ]] && resume=--resume
    python -u train.py --model $model --distance --re2g-features --batch 64 --seed $seed \
      --threads "$SLURM_CPUS_PER_TASK" --run-dir "$dir" $resume
  done
done
