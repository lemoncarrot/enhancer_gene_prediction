#!/bin/bash
# GNN substitution for attention: walks only select nodes; 3 rounds of message passing on the induced parent subgraph
# plus a virtual node (readout). Same walks, node/edge features, head and rE2G-like scalars as runs/final.
# Rows: subgraph GNN + rE2G-like features (row 3 analogue) and without pair features (row 4 analogue).
#SBATCH --job-name=e2g-subgraph
#SBATCH --partition=gpu,owners   # owners is preemptible: jobs requeue and resume completed folds
#SBATCH --requeue
#SBATCH -G 1
#SBATCH -C GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

train subgraph --model subgraph --layers 3 --re2g-features
train subgraph-no-pairfeat --model subgraph --layers 3
python evaluate.py $root/{full,full-no-pairfeat,re2g-hyenadna,re2g,subgraph,subgraph-no-pairfeat}-s$seed/predictions.parquet \
  --output-dir $root/evaluation-subgraph-s$seed
