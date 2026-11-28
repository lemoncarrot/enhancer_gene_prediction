"""Cached graph data, the historical Mamba/GNN architectures, and a dense-attention control.

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


def pair_feature_matrix(directory, pairs, nodes, resolution=5_000):
    """Pairwise scalars for the prediction head: log raw Hi-C contact between the element's and the TSS's bins
    (SCALE-normalized observed, 0 if no record), and log ATAC at the element and promoter nodes."""
    if not (directory / "hic").is_dir():
        raise ValueError("--pair-features needs the cached Hi-C contacts in <data-dir>/hic")
    if not {"center", "tss"}.issubset(pairs.columns):
        raise ValueError("--pair-features needs element center and gene tss columns in crispr.parquet")
    contacts = pd.read_parquet(directory / "hic").set_index(["chrom", "bin1", "bin2"]).observed
    a, b = pairs.center // resolution * resolution, pairs.tss // resolution * resolution
    contact = contacts.reindex(pd.MultiIndex.from_arrays([pairs.chrom, np.minimum(a, b), np.maximum(a, b)])).fillna(0).to_numpy()
    atac = nodes.log_atac.to_numpy()
    return np.column_stack([np.log1p(contact), atac[pairs.elem_node], atac[pairs.gene_node]])


RE2G_FEATURES = ["log_contact", "atac_elem", "atac_gene", "log_abc", "log_elements_between", "log_tss_between",
                 "log_nearby_elements", "log_nearby_atac"]


def re2g_feature_matrix(directory, pairs, nodes, nearby=10_000):
    """rE2G-style hand-crafted context: the three pair features plus the ABC score, the number of candidate elements
    and of promoter nodes strictly between element and TSS (endpoints excluded), and the count and summed ATAC of
    other candidate elements within +-nearby bp of the element. Column order is RE2G_FEATURES."""
    from baselines import abc_baseline   # deferred: baselines imports evaluate
    if not {"chrom", "is_promoter"}.issubset(nodes.columns):
        raise ValueError("--re2g-features needs chrom and is_promoter columns in nodes.parquet")
    base = pair_feature_matrix(directory, pairs, nodes)
    benchmark, abc = abc_baseline(directory)
    if len(benchmark) <= pairs.row_id.max():
        raise ValueError("crispr.parquet does not match the benchmark rows")
    activity = np.expm1(nodes.log_atac.to_numpy())
    counts = np.zeros((len(pairs), 4))
    for chrom, index in pairs.groupby("chrom").indices.items():
        on = np.flatnonzero((nodes.chrom == chrom).to_numpy())
        order = np.argsort(nodes.center.to_numpy()[on], kind="stable")
        ids, centers = on[order], nodes.center.to_numpy()[on][order]
        promoter = nodes.is_promoter.to_numpy()[ids].astype(float)
        cum_promoter, cum_activity = np.r_[0, np.cumsum(promoter)], np.r_[0, np.cumsum(activity[ids])]
        group = pairs.iloc[index]
        lo, hi = np.minimum(group.center, group.tss).to_numpy(), np.maximum(group.center, group.tss).to_numpy()
        left, right = np.searchsorted(centers, lo, "right"), np.searchsorted(centers, hi, "left")
        inside = lambda node: (np.searchsorted(centers, nodes.center.to_numpy()[node]) >= left) & \
                              (np.searchsorted(centers, nodes.center.to_numpy()[node]) < right)
        e, g = group.elem_node.to_numpy(), group.gene_node.to_numpy()
        between = right - left - inside(e) - (inside(g) & (g != e))
        tss_between = cum_promoter[right] - cum_promoter[left] - (inside(e) & nodes.is_promoter.to_numpy()[e]) \
            - (inside(g) & (g != e) & nodes.is_promoter.to_numpy()[g])
        c = group.center.to_numpy()
        a, b = np.searchsorted(centers, c - nearby, "left"), np.searchsorted(centers, c + nearby, "right")
        near_self = np.abs(nodes.center.to_numpy()[e] - c) <= nearby
        counts[index] = np.column_stack([np.maximum(between, 0), np.maximum(tss_between, 0), b - a - near_self,
                                         cum_activity[b] - cum_activity[a] - near_self * activity[e]])
    counts = np.maximum(counts, 0)
    return np.column_stack([base, np.log(abc[pairs.row_id.to_numpy()] + 1e-3), np.log1p(counts)])


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
    pair_features: torch.Tensor   # (pairs, 0) unless requested
    EDGE_PTR: torch.Tensor = None   # CSR row pointers over the sorted directed edges (index matches ETYPE/LOGOE)
    EDGE_DST: torch.Tensor = None   # target node of each sorted directed edge

    @classmethod
    def load(cls, directory, walks=DEFAULT_WALKS, device="cpu", pair_features=False, atac=True, seq=True):
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
        if not (atac or seq):
            raise ValueError("Node features need ATAC and/or the sequence embedding")
        # Ablations: atac=False leaves sequence embeddings only, seq=False leaves ATAC only.
        X = np.hstack([nodes[["log_atac"]].values if atac else np.zeros((len(nodes), 0)),
                       np.stack(nodes.seq_emb.values) if seq else np.zeros((len(nodes), 0))])
        distance = np.log1p(pairs.distanceToTSS.abs().to_numpy() / 1000)
        if pair_features == "re2g":
            extra = re2g_feature_matrix(directory, pairs, nodes)
        elif pair_features:
            extra = pair_feature_matrix(directory, pairs, nodes)
        else:
            extra = np.zeros((len(pairs), 0))
        if not np.isfinite(X).all() or not np.isfinite(distance).all() or not np.isfinite(extra).all():
            raise ValueError("Nonfinite node features, pair distances or pair features")
        return cls(benchmark, pairs, tensor(X, torch.float32), tensor(nodes.center.values),
                   tensor(np.r_[edges.type, edges.type][order], torch.long),
                   tensor(np.nan_to_num(np.log(np.r_[edges.oe, edges.oe][order])), torch.float32),
                   tensor(W, torch.long), tensor(E, torch.long),
                   tensor(np.searchsorted(anchors, pairs.elem_node), torch.long),
                   tensor(np.searchsorted(anchors, pairs.gene_node), torch.long),
                   tensor(distance, torch.float32), tensor(pairs.label.values, torch.float32),
                   tensor(extra, torch.float32),
                   tensor(np.r_[0, np.cumsum(np.bincount(src, minlength=len(nodes)))], torch.long),
                   tensor(dst, torch.long))

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

class AttentionBlock(nn.Module):
    # Dense-attention control for BiMamba: pre-norm self-attention + MLP, every token attends to every token.
    # No dropout, matching the Mamba blocks. At d=128 one block has ~0.2M parameters, close to one BiMamba block.
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d, n_heads if d % n_heads == 0 else 1, 4 * d, dropout=0.0,
                                                activation="gelu", batch_first=True, norm_first=True)
    def forward(self, x):
        return self.layer(x)

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
        if kind not in ("mamba", "gnn", "attention", "subgraph"):
            raise ValueError(kind)
        self.node = nn.Sequential(nn.LayerNorm(data.X.shape[1]), nn.Linear(data.X.shape[1], d))
        self.etype, self.rho, self.rho_oe = nn.Embedding(3, d), Fourier(d), Fourier(d)   # edge types: 0 chain, 1 Hi-C, 2 anchor
        if kind == "mamba":
            self.layers = nn.ModuleList(BiMamba(d, backend) for _ in range(n_layers))
        elif kind == "attention":
            # Attention is permutation-invariant, so walk order needs an explicit learned step embedding
            # (Mamba gets order from its recurrence). Created only here so mamba/gnn initialization is unchanged.
            self.pos = nn.Embedding(data.walks.shape[-1], d)
            self.layers = nn.ModuleList(AttentionBlock(d) for _ in range(n_layers))
        else:
            self.layers = nn.ModuleList(GINEConv(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))) for _ in range(n_layers))
            self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
            if kind == "subgraph":   # virtual node state and the edge embedding linking it to every subgraph node
                self.virtual, self.virtual_edge = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.out = nn.LayerNorm(d)

    def edge_emb(self, typ, dist, loe):
        return self.etype(typ) + self.rho(dist) + self.rho_oe(loe) * (typ == 1).unsqueeze(-1)

    def subgraph(self, W):
        """Induced subgraph of each anchor's walk nodes in the parent graph, kept separate per anchor.
        Returns global node ids of the subgraph nodes, the anchor each belongs to, and the directed parent-graph
        edges among them as (local source, local target, sorted edge id)."""
        n_nodes, ptr, target = len(self.data.X), self.data.EDGE_PTR, self.data.EDGE_DST
        U = W.shape[0]
        keys = torch.unique(torch.arange(U, device=W.device).view(U, 1, 1) * n_nodes + W)   # sorted (anchor, node)
        owner, glob = keys // n_nodes, keys % n_nodes
        degree = ptr[glob + 1] - ptr[glob]
        src = torch.repeat_interleave(torch.arange(len(keys), device=W.device), degree)
        start = torch.repeat_interleave(ptr[glob] - torch.cumsum(degree, 0) + degree, degree)
        eid = start + torch.arange(len(src), device=W.device)
        wanted = owner[src] * n_nodes + target[eid]
        pos = torch.searchsorted(keys, wanted).clamp_max(len(keys) - 1)
        keep = keys[pos] == wanted                                             # neighbour also visited by this anchor
        return glob, owner, src[keep], pos[keep], eid[keep]

    def forward(self, W, E):                                                   # W (U, N, L+1), E (U, N, L) -> z (U, d)
        X, CENTER = self.data.X, self.data.CENTER
        ETYPE, LOGOE = self.data.ETYPE, self.data.LOGOE
        U, N, L1 = W.shape
        if self.kind in ("mamba", "attention"):
            pad = lambda x, v: torch.cat([torch.full_like(x[..., :1], v), x], -1)   # step 0 = anchor token
            tok = self.node(X[W]) + self.edge_emb(pad(ETYPE[E], 2), pad(slog(CENTER[W[..., 1:]] - CENTER[W[..., :-1]]), 0), pad(LOGOE[E], 0))
            if self.kind == "attention":
                tok = tok + self.pos.weight[:L1]
            h = tok.view(U * N, L1, -1)
            for layer in self.layers: h = layer(h)
            return self.out(h[:, 0]).view(U, N, -1).mean(1)
        if self.kind == "subgraph":
            # Walks only pick the nodes; message passing runs on all original edges among them plus one virtual node
            # per anchor joined to every node of its subgraph, and the virtual node is the readout.
            glob, owner, src, dst, e = self.subgraph(W)
            M = len(glob)
            virtual = M + owner                                                # virtual node index for each subgraph node
            local = torch.arange(M, device=W.device)
            edge_index = torch.stack([torch.cat([src, local, virtual]), torch.cat([dst, virtual, local])])
            ea = torch.cat([self.edge_emb(ETYPE[e], slog(CENTER[glob[dst]] - CENTER[glob[src]]), LOGOE[e]),
                            self.virtual_edge.expand(2 * M, -1)])
            h = torch.cat([self.node(X[glob]), self.virtual.expand(U, -1)])
            for conv, norm in zip(self.layers, self.norms): h = h + conv(norm(h), edge_index, ea)
            return self.out(h[M:])
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



class EndpointEncoder(nn.Module):
    """Control for WalkEncoder: the same input projection and output norm applied to the anchor node alone
    (ATAC + sequence embedding), so the walk is the only thing it lacks."""
    def __init__(self, data, d=128):
        super().__init__()
        self.data = data
        self.node = nn.Sequential(nn.LayerNorm(data.X.shape[1]), nn.Linear(data.X.shape[1], d))
        self.out = nn.LayerNorm(d)

    def forward(self, W, E):                                                   # W (U, N, L+1) -> z (U, d)
        return self.out(self.node(self.data.X[W[:, 0, 0]]))


class PairModel(nn.Module):
    """One walk encoder, optionally augmented by log pair distance and other pairwise scalars.

    kind "pairwise" has no encoder: the same head sees only the pairwise scalars (matched control).
    kind "endpoints" embeds only the enhancer and promoter nodes themselves, without walks (context control)."""
    def __init__(self, data, kind, d=128, n_layers=2, distance=False, backend="cuda", pair_product=True):
        super().__init__()
        self.use_distance = distance
        self.use_pair_product = pair_product
        scalars = int(distance) + data.pair_features.shape[1]
        if kind == "pairwise":
            if not scalars:
                raise ValueError("pairwise needs distance and/or pair features")
            self.enc, width = None, 0
        else:
            enc = EndpointEncoder(data, d) if kind == "endpoints" else WalkEncoder(data, kind, d, n_layers, backend)
            self.enc, width = enc, (2 + int(pair_product)) * d
        self.mlp = nn.Sequential(nn.Linear(width + scalars, d),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))
        # Head scalars (distance, then pair features) are standardized with training-fold statistics set by
        # set_scalar_stats; the identity defaults reproduce the earlier unstandardized head.
        self.register_buffer("scalar_mean", torch.zeros(scalars))
        self.register_buffer("scalar_std", torch.ones(scalars))

    def scalars(self, pair_distance, pair_features):
        parts = []
        if self.use_distance:
            parts.append(pair_distance.unsqueeze(-1))
        if pair_features is not None and pair_features.shape[1]:
            parts.append(pair_features)
        return torch.cat(parts, -1) if parts else None

    @torch.no_grad()
    def set_scalar_stats(self, pair_distance, pair_features):
        values = self.scalars(pair_distance, pair_features)
        if values is not None:
            self.scalar_mean.copy_(values.mean(0))
            self.scalar_std.copy_(values.std(0).clamp_min(1e-6))

    def forward(self, W, E, pe, pg, pair_distance=None, pair_features=None):
        features = []
        if self.enc is not None:
            anchors, inv = torch.unique(torch.cat([pe, pg]), return_inverse=True)
            z = self.enc(W[anchors], E[anchors])
            ze, zg = z[inv[:len(pe)]], z[inv[len(pe):]]
            features += [ze, zg]
            if self.use_pair_product:
                features.append(ze * zg)
        scalars = self.scalars(pair_distance, pair_features)
        if scalars is not None:
            features.append((scalars - self.scalar_mean) / self.scalar_std)
        return self.mlp(torch.cat(features, -1)).squeeze(-1)


class CombinedPairModel(nn.Module):
    """Jointly trained Mamba and GNN encoders, with pair distance."""
    def __init__(self, data, d=128, n_layers=2, backend="cuda", pair_product=True):
        super().__init__()
        self.use_pair_product = pair_product
        self.mamba = WalkEncoder(data, "mamba", d, n_layers, backend)
        self.gnn = WalkEncoder(data, "gnn", d, n_layers, backend)
        self.mlp = nn.Sequential(nn.Linear((4 + 2 * int(pair_product)) * d + 1, d),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, W, E, pe, pg, pair_distance):
        anchors, inv = torch.unique(torch.cat([pe, pg]), return_inverse=True)
        wm, em = W[anchors], E[anchors]
        zm, zg = self.mamba(wm, em), self.gnn(wm, em)
        me, mg = zm[inv[:len(pe)]], zm[inv[len(pe):]]
        ge, gg = zg[inv[:len(pe)]], zg[inv[len(pe):]]
        features = [me, mg]
        if self.use_pair_product:
            features.append(me * mg)
        features.extend([ge, gg])
        if self.use_pair_product:
            features.append(ge * gg)
        features.append(pair_distance.unsqueeze(-1))
        return self.mlp(torch.cat(features, -1)).squeeze(-1)
