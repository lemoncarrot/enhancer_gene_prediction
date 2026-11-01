"""Train one model with chromosome-held-out CV; resume completed folds explicitly."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from models import DEFAULT_WALKS, CombinedPairModel, GraphData, PairModel, chromosome_folds


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_run(path, manifest, resume):
    saved = path / "manifest.json"
    if saved.exists():
        if not resume:
            raise ValueError("Run already exists; pass --resume or use a new --run-dir")
        if json.loads(saved.read_text()) != manifest:
            raise ValueError("Resume refused: config, inputs, code, folds, or environment changed. Use a new --run-dir.")
    else:
        if resume:
            raise ValueError("Cannot resume without manifest.json")
        if any(p.name != ".lock" for p in path.iterdir()):
            raise ValueError("New run directory must be empty")
        atomic_json(saved, manifest)


@contextmanager
def run_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is using this run directory") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def fit_predict(data, cfg, train, test):
    torch.manual_seed(cfg["seed"])
    if cfg["model"] == "combined":
        model = CombinedPairModel(data, cfg["d"], cfg["layers"], cfg["backend"])
    else:
        model = PairModel(data, cfg["model"], cfg["d"], cfg["layers"], cfg["distance"], cfg["backend"])
    model.to(data.X.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    positives = data.labels[train].sum()
    if positives == 0 or positives == len(train):
        raise ValueError("Each training split must contain both classes")
    pos_weight = (1 - data.labels[train]).sum() / positives

    def forward(r, i):
        return model(data.walks[r], data.eidx[r], data.pe[i], data.pg[i], data.distance[i])

    for epoch in range(cfg["epochs"]):
        model.train()
        r = epoch % len(data.walks)
        for batch in torch.randperm(len(train), device=data.X.device).split(cfg["batch"]):
            i = train[batch]
            loss = F.binary_cross_entropy_with_logits(forward(r, i), data.labels[i], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if not torch.isfinite(loss):
            raise ValueError("Training produced nonfinite loss")
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        pred = torch.stack([torch.cat([forward(r, i) for i in test.split(512)])
                            for r in range(min(4, len(data.walks)))]).mean(0).cpu().numpy()
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    return model, pred


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--walks", default=DEFAULT_WALKS, help="Cache filename relative to data-dir")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["mamba", "gnn", "combined"], required=True)
    parser.add_argument("--distance", action="store_true", help="Add pair distance (always enabled for combined)")
    parser.add_argument("--backend", choices=["cuda", "torch"], default="cuda", help="Mamba backend; torch is a separate CPU/debug implementation")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    for name, default in [("d", 128), ("layers", 2), ("epochs", 30), ("batch", 128), ("folds", 5), ("seed", 0), ("threads", 8)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed folds; restart any interrupted fold")
    args = parser.parse_args()
    if any(getattr(args, k) < 1 for k in ["d", "layers", "epochs", "batch", "threads"]) or args.lr <= 0 or args.wd < 0:
        parser.error("Sizes, epochs, threads and learning rate must be positive; wd must be nonnegative")
    if args.model != "gnn" and args.backend == "cuda" and args.device != "cuda":
        parser.error("The cuda Mamba backend requires --device cuda; use --backend torch for CPU smoke tests")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use a CUDA environment or --device cpu")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    cfg = {k: v for k, v in vars(args).items() if k not in ["data_dir", "walks", "run_dir", "resume"]}
    cfg["distance"] = args.distance or args.model == "combined"
    cfg["backend"] = "none" if args.model == "gnn" else args.backend
    packages = ["torch", "torch-geometric", "numpy", "pandas", "pyarrow"]
    if cfg["backend"] == "cuda":
        from mamba_ssm import Mamba  # fail early; never silently switch model implementations
        packages += ["mamba-ssm", "causal-conv1d"]
    tag = args.model + ("_distance" if cfg["distance"] else "")
    with run_lock(args.run_dir):
        files = {name: sha256(args.data_dir / name) for name in ["nodes.parquet", "edges.parquet", "crispr.parquet", args.walks]}
        data = GraphData.load(args.data_dir, args.walks, args.device)
        fold_of = chromosome_folds(data.benchmark, cfg["folds"])
        data.pairs["fold"] = data.pairs.chrom.map(fold_of)
        for k in range(cfg["folds"]):
            if not (data.pairs.fold == k).any():
                raise ValueError(f"Fold {k} has no covered test pairs")
        source = Path(__file__).resolve().parent
        manifest = dict(schema=1, config=cfg, inputs=files, chromosomes=fold_of,
                        code={name: sha256(source / name) for name in ["models.py", "train.py"]},
                        environment={name: version(name) for name in packages},
                        python=platform.python_version(), cuda=torch.version.cuda)
        prepare_run(args.run_dir, manifest, args.resume)
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        predictions = data.benchmark.copy()
        predictions["fold"] = predictions.chrom.map(fold_of)
        predictions[tag] = -1e9  # historical treatment of pairs outside graph coverage
        for k in range(cfg["folds"]):
            test_rows = data.pairs.loc[data.pairs.fold == k, "row_id"].to_numpy()
            checkpoint = args.run_dir / f"fold{k}.pt"
            complete = args.run_dir / f"fold{k}.json"
            if complete.exists():
                metadata = json.loads(complete.read_text())
                if metadata != dict(run=identity, sha256=sha256(checkpoint)):
                    raise ValueError(f"Fold {k} checkpoint checksum/identity mismatch")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved["run"] != identity or not np.array_equal(saved["row_id"].numpy(), test_rows):
                    raise ValueError(f"Fold {k} row identity mismatch")
                pred = saved["logits"].numpy()
                if pred.shape != test_rows.shape or not np.isfinite(pred).all():
                    raise ValueError(f"Fold {k} has invalid predictions")
                print(f"fold {k}: reusing verified checkpoint", flush=True)
            else:
                train, test = [torch.as_tensor(np.flatnonzero(mask), device=args.device)
                               for mask in [data.pairs.fold != k, data.pairs.fold == k]]
                start = time.monotonic()
                print(f"{tag} fold {k}", flush=True)
                model, pred = fit_predict(data, cfg, train, test)
                saved = dict(state_dict=model.state_dict(), config=cfg, fold=k, run=identity,
                             row_id=torch.from_numpy(test_rows), logits=torch.from_numpy(pred),
                             test_chromosomes=sorted(data.pairs.loc[data.pairs.fold == k, "chrom"].unique()),
                             seconds=time.monotonic() - start)
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(saved, temporary)
                temporary.replace(checkpoint)
                atomic_json(complete, dict(run=identity, sha256=sha256(checkpoint)))
                del model
            predictions.loc[test_rows, tag] = pred
            del saved
        temporary = args.run_dir / "predictions.tmp.parquet"
        predictions.to_parquet(temporary, index=False)
        temporary.replace(args.run_dir / "predictions.parquet")
        print(f"Saved {args.run_dir / 'predictions.parquet'}", flush=True)


if __name__ == "__main__":
    main()
