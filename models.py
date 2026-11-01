"""Cached graph data and the historical Mamba/GNN architectures.

Data tensors live on the requested device before model construction; they are not
checkpoint parameters. The manifest identifies their exact source files.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv

DEFAULT_WALKS = "walks_L64_p4.0_q1.0_b5.0_N8_R10.npz"


def chromosome_folds(benchmark, count):
    if count < 2 or count > benchmark.chrom.nunique():
        raise ValueError("folds must be between 2 and the number of chromosomes")
    load, fold_of = np.zeros(count), {}
    for chrom, size in benchmark.chrom.value_counts().items():
        fold_of[chrom] = int(load.argmin())
        load[fold_of[chrom]] += size
    return fold_of


@dataclass
class GraphData:
    benchmark: pd.DataFrame
    pairs: pd.DataFrame
    X: torch.Tensor
    CENTER: torch.Tensor
    ETYPE: torch.Tensor
    LOGOE: torch.Tensor
    walks: torch.Tensor
    eidx: torch.Tensor
    pe: torch.Tensor
    pg: torch.Tensor
    distance: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def load(cls, directory, walks=DEFAULT_WALKS, device="cpu"):
        directory = Path(directory)
        nodes = pd.read_parquet(directory / "nodes.parquet")
        edges = pd.read_parquet(directory / "edges.parquet")
        benchmark = pd.read_parquet(directory / "crispr.parquet").reset_index(drop=True)
        if "seq_emb" not in nodes:
            raise ValueError("nodes.parquet must contain cached sequence embeddings (seq_emb)")
        if not np.array_equal(nodes.node_id, np.arange(len(nodes))):
            raise ValueError("Node rows must match node_id")
        if not benchmark.label.isin([0, 1]).all():
            raise ValueError("Labels must be binary")
        benchmark.insert(0, "row_id", np.arange(len(benchmark)))
        pairs = benchmark[(benchmark.elem_node >= 0) & (benchmark.gene_node >= 0)].copy().reset_index(drop=True)
        anchors = np.unique(np.r_[pairs.elem_node, pairs.gene_node])
        src, dst = np.r_[edges.u, edges.v], np.r_[edges.v, edges.u]
        if min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= len(nodes):
            raise ValueError("Edge endpoints outside node table")
        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]
        with np.load(directory / walks) as cache:
            W, E = cache["arr_0"], cache["arr_1"]
        if W.ndim != 4 or E.shape != (*W.shape[:-1], W.shape[-1] - 1):
            raise ValueError("Expected walks [resample, anchor, walk, step+1] and matching edge IDs")
        if min(W.shape) == 0 or W.shape[1] != len(anchors) or W.shape[-1] < 2:
            raise ValueError("Empty walks or wrong anchor count")
        if not np.issubdtype(W.dtype, np.integer) or not np.issubdtype(E.dtype, np.integer):
            raise ValueError("Walk and edge IDs must be integers")
        # Validate each resample without allocating another full walk-sized array.
        for w, e in zip(W, E):
            if w.min() < 0 or w.max() >= len(nodes) or e.min() < 0 or e.max() >= len(src):
                raise ValueError("Walk contains invalid node/edge IDs")
            if not np.all(w[:, :, 0] == anchors[:, None]):
                raise ValueError("Cached walk anchor ordering does not match benchmark")
            if not np.array_equal(src[e], w[..., :-1]) or not np.array_equal(dst[e], w[..., 1:]):
                raise ValueError("Cached walks do not match this graph's CSR edge ordering")
        def tensor(a, dtype=None):
            return torch.as_tensor(a, dtype=dtype, device=device)
        X = np.hstack([nodes[["log_atac"]].values, np.stack(nodes.seq_emb.values)])
        distance = np.log1p(pairs.distanceToTSS.abs().to_numpy() / 1000)
        if not np.isfinite(X).all() or not np.isfinite(distance).all():
            raise ValueError("Nonfinite node features or pair distances")
        return cls(benchmark, pairs, tensor(X, torch.float32), tensor(nodes.center.values),
                   tensor(np.r_[edges.type, edges.type][order], torch.long),
                   tensor(np.nan_to_num(np.log(np.r_[edges.oe, edges.oe][order])), torch.float32),
                   tensor(W, torch.long), tensor(E, torch.long),
                   tensor(np.searchsorted(anchors, pairs.elem_node), torch.long),
                   tensor(np.searchsorted(anchors, pairs.gene_node), torch.long),
                   tensor(distance, torch.float32), tensor(pairs.label.values, torch.float32))

class TorchMamba(nn.Module):
    # Selective SSM block (Gu & Dao 2023, Alg. 2) in plain PyTorch, same interface as mamba_ssm.Mamba.
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_in, self.r = expand * d_model, math.ceil(d_model / 16)
        self.in_proj, self.out_proj = nn.Linear(d_model, 2 * d_in, bias=False), nn.Linear(d_in, d_model, bias=False)
        self.conv = nn.Conv1d(d_in, d_in, d_conv, groups=d_in, padding=d_conv - 1)
        self.x_proj, self.dt_proj = nn.Linear(d_in, self.r + 2 * d_state, bias=False), nn.Linear(self.r, d_in)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1.0)).repeat(d_in, 1))
        self.D = nn.Parameter(torch.ones(d_in))

    def forward(self, x):                                                      # (B, L, d_model)
        L, n = x.shape[1], self.A_log.shape[1]
        x, z = self.in_proj(x).chunk(2, -1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, B, C = self.x_proj(x).split([self.r, n, n], -1)
        dt = F.softplus(self.dt_proj(dt)).unsqueeze(-1)                        # input-dependent step size (selectivity)
        dA, dBx = torch.exp(-dt * torch.exp(self.A_log)), dt * B.unsqueeze(2) * x.unsqueeze(-1)
        h, ys = torch.zeros_like(dA[:, 0]), []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]; ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        return self.out_proj((torch.stack(ys, 1) + x * self.D) * F.silu(z))

class BiMamba(nn.Module):   # pre-norm residual; the reversed pass lets position 0 (the anchor) see the whole walk
    def __init__(self, d, backend):
        super().__init__()
        if backend == "cuda":
            from mamba_ssm import Mamba
            block = lambda: Mamba(d_model=d)
        elif backend == "torch":
            block = lambda: TorchMamba(d)
        else:
            raise ValueError(f"Unknown Mamba backend: {backend}")
        self.fwd, self.bwd, self.norm = block(), block(), nn.LayerNorm(d)
    def forward(self, x):
        h = self.norm(x); return x + self.fwd(h) + self.bwd(h.flip(1)).flip(1)

class Fourier(nn.Module):   # sinusoidal encoding of a scalar -> d (rho, rho')
    def __init__(self, d, n_freq=16, max_period=100.0):
        super().__init__()
        self.register_buffer("freq", 2 * math.pi / max_period ** torch.linspace(0, 1, n_freq)); self.lin = nn.Linear(2 * n_freq, d)
    def forward(self, x):
        a = x.unsqueeze(-1) * self.freq; return self.lin(torch.cat([a.sin(), a.cos()], -1))

slog = lambda d: d.float().sign() * torch.log1p(d.float().abs() / 1e3)

class WalkEncoder(nn.Module):
    def __init__(self, data, kind, d=128, n_layers=2, backend="cuda"):
        super().__init__()
        self.kind = kind
        self.data = data
        if kind not in ("mamba", "gnn"):
            raise ValueError(kind)
        self.node = nn.Sequential(nn.LayerNorm(data.X.shape[1]), nn.Linear(data.X.shape[1], d))
        self.etype, self.rho, self.rho_oe = nn.Embedding(3, d), Fourier(d), Fourier(d)   # edge types: 0 chain, 1 Hi-C, 2 anchor
        if kind == "mamba":
            self.layers = nn.ModuleList(BiMamba(d, backend) for _ in range(n_layers))
        else:
            self.layers = nn.ModuleList(GINEConv(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))) for _ in range(n_layers))
            self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.out = nn.LayerNorm(d)

    def edge_emb(self, typ, dist, loe):
        return self.etype(typ) + self.rho(dist) + self.rho_oe(loe) * (typ == 1).unsqueeze(-1)

    def forward(self, W, E):                                                   # W (U, N, L+1), E (U, N, L) -> z (U, d)
        X, CENTER = self.data.X, self.data.CENTER
        ETYPE, LOGOE = self.data.ETYPE, self.data.LOGOE
        U, N, L1 = W.shape
        if self.kind == "mamba":
            pad = lambda x, v: torch.cat([torch.full_like(x[..., :1], v), x], -1)   # step 0 = anchor token
            tok = self.node(X[W]) + self.edge_emb(pad(ETYPE[E], 2), pad(slog(CENTER[W[..., 1:]] - CENTER[W[..., :-1]]), 0), pad(LOGOE[E], 0))
            h = tok.view(U * N, L1, -1)
            for layer in self.layers: h = layer(h)
            return self.out(h[:, 0]).view(U, N, -1).mean(1)
        # gnn: union of each anchor's walk edges, kept separate per anchor
        uniq, inv = torch.unique(torch.arange(U, device=W.device).view(U, 1, 1) * len(X) + W, return_inverse=True)
        glob, s, t = uniq % len(X), inv[..., :-1].reshape(-1), inv[..., 1:].reshape(-1)
        # Preserve historical CSR-edge multiplicity for checkpoint/result compatibility.
        a, b, e = torch.unique(torch.stack([torch.minimum(s, t), torch.maximum(s, t), E.reshape(-1)]), dim=1)
        src, dst, e = torch.cat([a, b]), torch.cat([b, a]), torch.cat([e, e])
        ea = self.edge_emb(ETYPE[e], slog(CENTER[glob[dst]] - CENTER[glob[src]]), LOGOE[e])
        anchor = inv[:, 0, 0]
        h = self.node(X[glob]).index_add(0, anchor, self.etype.weight[2].expand(U, -1))
        for conv, norm in zip(self.layers, self.norms): h = h + conv(norm(h), torch.stack([src, dst]), ea)
        return self.out(h[anchor])



class PairModel(nn.Module):
    """One walk encoder, optionally augmented by log pair distance."""
    def __init__(self, data, kind, d=128, n_layers=2, distance=False, backend="cuda"):
        super().__init__()
        self.use_distance = distance
        self.enc = WalkEncoder(data, kind, d, n_layers, backend)
        self.mlp = nn.Sequential(nn.Linear(3 * d + int(distance), d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, W, E, pe, pg, pair_distance=None):
        anchors, inv = torch.unique(torch.cat([pe, pg]), return_inverse=True)
        z = self.enc(W[anchors], E[anchors])
        ze, zg = z[inv[:len(pe)]], z[inv[len(pe):]]
        features = [ze, zg, ze * zg]
        if self.use_distance:
            features.append(pair_distance.unsqueeze(-1))
        return self.mlp(torch.cat(features, -1)).squeeze(-1)


class CombinedPairModel(nn.Module):
    """Jointly trained Mamba and GNN encoders, with pair distance."""
    def __init__(self, data, d=128, n_layers=2, backend="cuda"):
        super().__init__()
        self.mamba = WalkEncoder(data, "mamba", d, n_layers, backend)
        self.gnn = WalkEncoder(data, "gnn", d, n_layers, backend)
        self.mlp = nn.Sequential(nn.Linear(6 * d + 1, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, W, E, pe, pg, pair_distance):
        anchors, inv = torch.unique(torch.cat([pe, pg]), return_inverse=True)
        wm, em = W[anchors], E[anchors]
        zm, zg = self.mamba(wm, em), self.gnn(wm, em)
        me, mg = zm[inv[:len(pe)]], zm[inv[len(pe):]]
        ge, gg = zg[inv[:len(pe)]], zg[inv[len(pe):]]
        return self.mlp(torch.cat([me, mg, me * mg, ge, gg, ge * gg, pair_distance.unsqueeze(-1)], -1)).squeeze(-1)
