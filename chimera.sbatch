#!/bin/bash
# Activate the training environment before sbatch; submit from the repo root.
# Example: sbatch chimera.sbatch --model gnn --distance --run-dir runs/gnn-distance
#SBATCH --job-name=e2g
#SBATCH --partition=zhoulab_gpu_priority,preemptible
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u train.py "$@"
