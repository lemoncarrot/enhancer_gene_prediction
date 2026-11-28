#!/bin/bash
# Download raw inputs and build crispr/nodes/hic/edges parquet + walk cache (Hi-C is streamed over the network).
#SBATCH --job-name=e2g-prepare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sherlock/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/home/groups/hanleeji/Virtual_envs/e2g/bin:$PATH   # python, samtools
python -u download_data.py
python -u prepare_data.py cpu
