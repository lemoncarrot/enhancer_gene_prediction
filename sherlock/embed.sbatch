#!/bin/bash
# HyenaDNA sequence embeddings -> seq_emb column of nodes.parquet.
#SBATCH --job-name=e2g-embed
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
export HF_HOME=$SCRATCH/.cache/huggingface
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import mamba_ssm, causal_conv1d"   # fail before experiments queue if the kernels do not load
python -u prepare_data.py embed
