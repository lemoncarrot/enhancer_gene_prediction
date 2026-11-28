#!/bin/bash
# EXPERIMENTS.md 7 (pair features) and the ATAC ablation, one array task per (experiment, seed).
# Tasks 0-2: pair features, seeds 0-2. Tasks 3-5: ATAC ablation, seeds 0-2.
#SBATCH --job-name=e2g-exp
#SBATCH --partition=gpu,owners   # owners is preemptible: jobs requeue and resume completed folds
#SBATCH --requeue
#SBATCH -G 1
#SBATCH -C GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=sherlock/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/home/groups/hanleeji/Virtual_envs/e2g/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
seed=$(( SLURM_ARRAY_TASK_ID % 3 ))

train() {   # train <run-dir> <args...>; resumes a run that already has a manifest
  local dir=$1; shift
  local resume=""   # a string, not an array: bash 4.2 treats an empty "${a[@]}" as unbound under set -u
  [[ -f $dir/manifest.json ]] && resume=--resume
  python -u train.py "$@" --batch 64 --seed "$seed" --threads "$SLURM_CPUS_PER_TASK" --run-dir "$dir" $resume
}

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  root=runs/pairfeat
  for model in pairwise gnn mamba attention; do
    train "$root/$model-s$seed" --model "$model" --distance --pair-features
  done
  python evaluate.py "$root"/{pairwise,gnn,mamba,attention}-s$seed/predictions.parquet --output-dir "$root/evaluation-s$seed"
else
  root=runs/atac
  for model in gnn mamba; do
    train "$root/$model-s$seed" --model "$model" --distance
    train "$root/$model-noatac-s$seed" --model "$model" --distance --no-atac
  done
  python evaluate.py "$root"/{gnn,gnn-noatac,mamba,mamba-noatac}-s$seed/predictions.parquet --output-dir "$root/evaluation-s$seed"
fi
