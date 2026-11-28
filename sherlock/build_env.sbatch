#!/bin/bash
# Build the e2g environment (data preparation + training). Submit from the repo root.
#SBATCH --job-name=e2g-env
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sherlock/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=/home/groups/hanleeji/Virtual_envs/e2g
export CONDA_PKGS_DIRS=$SCRATCH/.cache/conda_pkgs PIP_CACHE_DIR=$SCRATCH/.cache/pip
# numpy/pandas/pyarrow pinned to requirements-train.txt; scipy 1.15.3 is not on conda-forge, so pip installs scipy/scikit-learn.
[[ -x $ENV/bin/python ]] || mamba create -y -p "$ENV" -c conda-forge -c bioconda --override-channels \
  python=3.11 pip numpy=1.26.4 pandas=2.3.3 pyarrow=20.0.0 \
  matplotlib tqdm requests pysam pyfaidx samtools hic-straw packaging ninja wheel setuptools
PY="$ENV/bin/python"
# Sherlock has glibc 2.17; torch 2.7 wheels need 2.28, so use torch 2.6.0 (CUDA 12.4, manylinux2014) here.
"$PY" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
grep -v '^torch==' requirements-train.txt | "$PY" -m pip install -r /dev/stdin "pyranges>=0.1,<0.2" "transformers>=4.40,<5"
# Prebuilt mamba-ssm/causal-conv1d kernels target newer glibc; compile them against this torch.
ml gcc/12.4.0 cuda/12.4.0
export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MAX_JOBS=8   # nvcc needs several GB per job
"$PY" -m pip install --no-build-isolation -r requirements-cuda.txt
# mamba_ssm imports Triton, which needs a GPU driver, so it is checked in embed.sbatch.
"$PY" -c "import torch, torch_geometric, pysam, pyfaidx, hicstraw, pyranges, transformers; print(torch.__version__, torch.version.cuda)"
# Lockfiles kept next to the environment (not on purged scratch).
mamba list -p "$ENV" --explicit > "$ENV/e2g-conda-lock.txt"
"$PY" -m pip freeze > "$ENV/e2g-pip-freeze.txt"
"$PY" -m unittest discover -s tests -v   # -m puts the repo root on sys.path
