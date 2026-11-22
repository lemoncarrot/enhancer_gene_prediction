"""Make controlled shorter/subsampled caches from the baseline walk cache."""

import numpy as np

from data_sources import data_directory, processed_dir
from models import DEFAULT_WALKS


processed = processed_dir(data_directory())
source = processed / DEFAULT_WALKS
settings = [(16, 8), (32, 8), (64, 4)]

with np.load(source) as saved:
    walks = saved["arr_0"]
    edges = saved["arr_1"]

if (walks.ndim != 4 or walks.shape[0] != 10 or walks.shape[2:] != (8, 65)
        or edges.shape != (*walks.shape[:-1], 64)):
    raise ValueError("Expected ten resamples of eight 64-step walks per anchor")

for length, count in settings:
    target = processed / f"walks_L{length}_p4.0_q1.0_b5.0_N{count}_R10.npz"
    node_prefix = walks[:, :, :count, :length + 1]
    edge_prefix = edges[:, :, :count, :length]
    if target.exists():
        with np.load(target) as saved:
            if not (np.array_equal(saved["arr_0"], node_prefix)
                    and np.array_equal(saved["arr_1"], edge_prefix)):
                raise ValueError(f"Existing cache does not match the expected prefix: {target}")
        print(f"Reusing {target}")
        continue
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, node_prefix, edge_prefix)
    temporary.replace(target)
    print(target)
