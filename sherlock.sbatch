#!/bin/bash
# Production run: HyenaDNA embeddings -> walks -> Mamba + GNN chromosome-held-out CV.
# Runs main.ipynb headless, skipping Part 1 (cells tagged "data"); expects data/processed/ already synced.
# Submit from the repo root:  sbatch sherlock.sbatch
#SBATCH --job-name=e2g
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate e2g
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"   # HyenaDNA weights; keeps $HOME quota free
nvidia-smi --query-gpu=name,memory.total --format=csv

jupyter nbconvert --to notebook --execute main.ipynb --output "main_run_${SLURM_JOB_ID}.ipynb" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["data"]'
