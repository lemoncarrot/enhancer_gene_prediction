"""Build the cached training inputs by running the data cells of main.ipynb headlessly.

Run from the project root after download_data.py:
    python prepare_data.py cpu     # crispr/nodes/hic/edges parquet and the walk cache (network for Hi-C)
    python prepare_data.py embed   # HyenaDNA seq_emb written into nodes.parquet (GPU)

Cells are executed from the notebook itself, so the notebook stays the single source of the pipeline.
Each stage skips work whose cached output already exists (same checks as the notebook).
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Data, Hi-C, graph and walk cells; plots and walk-setting diagnostics (20, 21, 26) are skipped.
CPU_CELLS = [1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 23, 25, 28]
EMBED_CELLS = [1, 2, 30]


def main():
    stage = sys.argv[1] if len(sys.argv) == 2 else None
    if stage not in ("cpu", "embed"):
        sys.exit(__doc__)
    # The walk cell sizes its process pool with os.cpu_count(); use the CPUs this job was given.
    os.cpu_count = lambda: len(os.sched_getaffinity(0))
    cells = json.loads(Path("main.ipynb").read_text())["cells"]
    # Run in the real __main__ namespace: the Hi-C and walk cells send their functions to process pools by name.
    namespace = vars(sys.modules["__main__"])
    for index in CPU_CELLS if stage == "cpu" else EMBED_CELLS:
        cell = cells[index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Cell {index} is not a code cell; main.ipynb changed")
        print(f"--- cell {index}", flush=True)
        exec(compile("".join(cell["source"]), f"main.ipynb[{index}]", "exec"), namespace)
        if stage == "embed" and index == 2:
            import pandas as pd
            namespace["nodes"] = pd.read_parquet(namespace["PROC"] / "nodes.parquet")
    print(f"done: {stage}", flush=True)


if __name__ == "__main__":
    main()
