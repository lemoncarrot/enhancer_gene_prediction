architecture swap: GNNs to encode neighborhoods instead of mamba

sweep random walk parameters and number of mamba/gnn layers

main result: enhancer-gene regulation prediction metrics

additional: attribution maps showing what mamba is actually learning - this new architecture is used in lots of genomic language models

need to design synthetic data experiments to show what we hypothesize this architecture can capture

- when walking along a gene's regulatory graph, what can be missed? what interactions are not captured by Hi-C?

## Setup and running

### Environment
```bash
conda env create -f environment.yml          # on a GPU node, conda picks the CUDA build of pytorch
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
Run `main.ipynb` from the repository root with the `e2g` kernel.
Raw filenames, download URLs, and the `raw/` and `processed/` folder names are shared through `data_sources.py`. Change them there if the data layout changes. `download_data.py` reads that configuration directly.

1. **Data** (local, ~10 GB raw): run `python download_data.py`, then run Part 1 of `main.ipynb`. Hi-C is streamed from ENCODE. Processed tables go to `data/processed/` (set `E2G_DATA_DIR` to put data elsewhere).
2. **Graph + walks** (CPU): run Part 2 of `main.ipynb`. Walks are cached per parameter setting.
3. **Embeddings + Mamba/GNN training** (GPU): run Part 3 of `main.ipynb`. On Chimera, sync the repository including `data/processed/`, then submit the notebook job:
   ```bash
   rsync -avP --exclude 'data/raw' --exclude '.DS_Store' ./ kevin@chimera-login:/scratch/zhoulab/kevin/enhancer_gene_prediction/
   cd /scratch/zhoulab/kevin/enhancer_gene_prediction && sbatch chimera.sbatch
   ```
   The job skips notebook cells tagged `data`, writes the executed notebook to `main_run_<jobid>.ipynb`, and writes metrics to `data/processed/results.csv`.
