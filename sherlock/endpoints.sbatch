#!/bin/bash
# Endpoint-only control (no walks), seeds 0-2, in the experiment 7 and ATAC-ablation settings.
#SBATCH --job-name=e2g-endpoints
#SBATCH --partition=gpu,owners   # owners is preemptible: jobs requeue and resume completed folds
#SBATCH --requeue
#SBATCH -G 1
#SBATCH -C GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sherlock/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/home/groups/hanleeji/Virtual_envs/e2g/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv

train() {   # train <run-dir> <args...>; resumes a run that already has a manifest
  local dir=$1; shift
  local resume=""   # a string, not an array: bash 4.2 treats an empty "${a[@]}" as unbound under set -u
  [[ -f $dir/manifest.json ]] && resume=--resume
  python -u train.py "$@" --batch 64 --threads "$SLURM_CPUS_PER_TASK" --run-dir "$dir" $resume
}

for seed in 0 1 2; do
  train runs/pairfeat/endpoints-s$seed --model endpoints --distance --pair-features --seed $seed
  python evaluate.py runs/pairfeat/{pairwise,endpoints,gnn,mamba,attention}-s$seed/predictions.parquet \
    --output-dir runs/pairfeat/evaluation-endpoints-s$seed
  train runs/atac/endpoints-s$seed --model endpoints --distance --seed $seed
  python evaluate.py runs/atac/{endpoints,gnn,mamba}-s$seed/predictions.parquet \
    --output-dir runs/atac/evaluation-endpoints-s$seed
done
