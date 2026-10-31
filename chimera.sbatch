#!/bin/bash
# Production run: HyenaDNA embeddings -> walks -> Mamba + GNN chromosome-held-out CV.
# Runs main.ipynb headless, skipping Part 1 (cells tagged "data"); expects data/processed/ already synced.
# Submit from the repo root:  sbatch chimera.sbatch
#SBATCH --job-name=e2g
#SBATCH --partition=zhoulab_gpu_priority,preemptible   # runs on whichever has a free GPU first
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue                                      # restart if preempted (cached walks/embeddings are reused)
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base 2>/dev/null || echo /scratch/zhoulab/kevin/miniforge3)/etc/profile.d/conda.sh"
conda activate e2g
export HF_HOME=/scratch/zhoulab/kevin/hf_cache   # HyenaDNA weights; keeps $HOME quota free
nvidia-smi --query-gpu=name,memory.total --format=csv

jupyter nbconvert --to notebook --execute main.ipynb --output "main_run_${SLURM_JOB_ID}.ipynb" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["data"]'
