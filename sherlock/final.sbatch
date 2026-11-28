#!/bin/bash
# Headline comparison with standardized head scalars, one array task per seed:
#   1 distance, 2 ABC (both from evaluate.py), 3 full model (attention + rE2G-like features),
#   4 full model without pair features, 5 rE2G-like features + HyenaDNA/ATAC endpoint embeddings (no walks),
#   plus rE2G-like features alone (pairwise head) so row 5's HyenaDNA contribution can be read off.
#SBATCH --job-name=e2g-final
#SBATCH --partition=gpu,owners   # owners is preemptible: jobs requeue and resume completed folds
#SBATCH --requeue
#SBATCH -G 1
#SBATCH -C GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=sherlock/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/home/groups/hanleeji/Virtual_envs/e2g/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
seed=$SLURM_ARRAY_TASK_ID
root=runs/final

train() {   # train <name> <args...>; resumes a run that already has a manifest
  local dir=$root/$1-s$seed; shift
  local resume=""   # a string, not an array: bash 4.2 treats an empty "${a[@]}" as unbound under set -u
  [[ -f $dir/manifest.json ]] && resume=--resume
  python -u train.py "$@" --distance --batch 64 --seed "$seed" --threads "$SLURM_CPUS_PER_TASK" --run-dir "$dir" $resume
}

train full --model attention --re2g-features
train full-no-pairfeat --model attention
train re2g-hyenadna --model endpoints --re2g-features
train re2g --model pairwise --re2g-features
python evaluate.py $root/{full,full-no-pairfeat,re2g-hyenadna,re2g}-s$seed/predictions.parquet \
  --output-dir $root/evaluation-s$seed
