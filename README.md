architecture swap: GNNs to encode neighborhoods instead of mamba

sweep random walk parameters and number of mamba/gnn layers

main result: enhancer-gene regulation prediction metrics

additional: attribution maps showing what mamba is actually learning - this new architecture is used in lots of genomic language models

need to design synthetic data experiments to show what we hypothesize this architecture can capture

- when walking along a gene's regulatory graph, what can be missed? what interactions are not captured by Hi-C?

## Setup and running

### Environment
```bash
mamba env create -f environment.yml          # on a GPU node, conda picks the CUDA build of pytorch
conda activate e2g
python -m ipykernel install --user --name e2g --display-name "Python (e2g)"
```
On a login node without a GPU, prefix the create command with `CONDA_OVERRIDE_CUDA=12.6` to get the CUDA build. Optional fast Mamba kernels (Linux + GPU; otherwise a plain-PyTorch fallback is used):
```bash
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
mamba install -y -c conda-forge "cuda-nvcc=$CUDA_VER" "cuda-cudart-dev=$CUDA_VER" ninja
pip install -r requirements-cuda.txt --no-build-isolation
```

### Pipeline
1. **Data** (local, ~10 GB raw): `./download_data.sh`, then run Part 1 of `main.ipynb`. Hi-C is streamed from ENCODE. Outputs go to `data/processed/` (set `E2G_DATA_DIR` to put data elsewhere).
2. **Graph + walks**: Part 2 of `main.ipynb` (CPU; walks are cached per parameter setting).
3. **Embeddings + Mamba/GNN training** (GPU): Part 3 of `main.ipynb`, or headless on Chimera:
   ```bash
   rsync -avP --exclude 'data/raw' --exclude '.DS_Store' ./ <user>@<chimera-host>:/scratch/zhoulab/kevin/enhancer_gene_prediction/
   cd /scratch/zhoulab/kevin/enhancer_gene_prediction && sbatch chimera.sbatch
   ```
   The job skips Part 1 (cells tagged `data`) and writes the executed notebook to `main_run_<jobid>.ipynb` and metrics to `data/processed/results.csv`.