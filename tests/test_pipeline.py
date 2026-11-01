"""Small CPU tests: training/resume, cache integrity, and prediction alignment."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate, load_predictions
from models import DEFAULT_WALKS, GraphData

ROOT = Path(__file__).resolve().parents[1]


def fixture(path):
    path.mkdir()
    pd.DataFrame(dict(node_id=np.arange(8), center=np.arange(8) * 100,
                      log_atac=np.arange(8) / 8,
                      seq_emb=[np.array([i / 8, 1.0]) for i in range(8)])).to_parquet(path / "nodes.parquet")
    pd.DataFrame(dict(u=[0, 2, 4, 6], v=[1, 3, 5, 7], type=[0, 1, 0, 1], oe=[np.nan, 4., np.nan, 5.])).to_parquet(path / "edges.parquet")
    pd.DataFrame(dict(chrom=np.repeat(["chr1", "chr2", "chr3", "chr4"], 2),
                      elem_node=np.repeat([0, 2, 4, 6], 2), gene_node=np.repeat([1, 3, 5, 7], 2),
                      label=[0, 1] * 4, distanceToTSS=[100, 200] * 4, Dataset=["synthetic"] * 8)).to_parquet(path / "crispr.parquet")
    W = np.array([[[[i, i ^ 1, i]] for i in range(8)]], dtype=np.int32)
    E = np.array([[[[i, i ^ 1]] for i in range(8)]], dtype=np.int32)
    np.savez(path / DEFAULT_WALKS, W, E)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        fixture(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def train(self, kind="gnn", extra=(), success=True):
        command = [sys.executable, str(ROOT / "train.py"), "--data-dir", str(self.data),
                   "--run-dir", str(self.root / kind), "--model", kind, "--device", "cpu",
                   "--backend", "torch", "--d", "4", "--layers", "1", "--epochs", "1",
                   "--batch", "4", "--folds", "2", "--threads", "1", *extra]
        result = subprocess.run(command, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_all_models_resume_and_evaluate(self):
        for kind in ["gnn", "mamba", "combined"]:
            with self.subTest(kind=kind):
                self.train(kind, ["--distance"])
                run = self.root / kind
                before = (run / "fold0.pt").read_bytes()
                result = self.train(kind, ["--distance", "--resume"])
                self.assertIn("reusing verified checkpoint", result.stdout)
                self.assertEqual(before, (run / "fold0.pt").read_bytes())
                # Interrupted folds restart without touching other completed folds.
                (run / "fold1.json").unlink()
                self.train(kind, ["--distance", "--resume"])
                self.assertEqual(before, (run / "fold0.pt").read_bytes())
        paths = [self.root / kind / "predictions.parquet" for kind in ["gnn", "mamba", "combined"]]
        result = evaluate(paths, self.root / "evaluation", ["mamba_distance", "gnn_distance"])
        self.assertEqual(set(result.model), {"distance", "mamba_distance", "gnn_distance", "combined_distance", "ensemble"})
        self.assertTrue(np.isfinite(result.AUPRC).all())

    def test_resume_refuses_changed_config_data_code_and_corruption(self):
        self.train()
        self.assertIn("already exists", self.train(success=False).stderr)
        self.assertIn("Resume refused", self.train(extra=["--resume", "--epochs", "2"], success=False).stderr)
        manifest = self.root / "gnn" / "manifest.json"
        original = manifest.read_text()
        changed = json.loads(original)
        changed["code"]["models.py"] = "changed"
        manifest.write_text(json.dumps(changed))
        self.assertIn("Resume refused", self.train(extra=["--resume"], success=False).stderr)
        manifest.write_text(original)
        nodes = pd.read_parquet(self.data / "nodes.parquet")
        nodes.loc[0, "log_atac"] += 1
        nodes.to_parquet(self.data / "nodes.parquet")
        self.assertIn("Resume refused", self.train(extra=["--resume"], success=False).stderr)
        nodes.loc[0, "log_atac"] -= 1
        nodes.to_parquet(self.data / "nodes.parquet")
        # Restore the original hash explicitly; serialized parquet bytes may differ.
        from train import sha256
        changed = json.loads(original)
        changed["inputs"]["nodes.parquet"] = sha256(self.data / "nodes.parquet")
        manifest.write_text(json.dumps(changed))
        with (self.root / "gnn" / "fold0.pt").open("ab") as stream:
            stream.write(b"corruption")
        self.assertIn("checksum/identity mismatch", self.train(extra=["--resume"], success=False).stderr)

    def test_cache_edge_order_guard(self):
        cache = self.data / DEFAULT_WALKS
        with np.load(cache) as arrays:
            W, E = arrays["arr_0"], arrays["arr_1"]
        E[0, 0, 0, 0] = 2
        np.savez(cache, W, E)
        with self.assertRaisesRegex(ValueError, "CSR edge ordering"):
            GraphData.load(self.data)

    def test_evaluation_matches_rows_not_file_order(self):
        self.train()
        path = self.root / "gnn" / "predictions.parquet"
        frame = pd.read_parquet(path).rename(columns={"gnn": "mamba"})
        other = self.root / "other.parquet"
        frame.sample(frac=1, random_state=0).to_parquet(other, index=False)
        merged, _ = load_predictions([path, other])
        np.testing.assert_array_equal(merged.gnn, merged.mamba)
        frame.loc[0, "label"] = 1 - frame.loc[0, "label"]
        frame.to_parquet(other, index=False)
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_predictions([path, other])
        frame.loc[0, "row_id"] = frame.loc[1, "row_id"]
        frame.to_parquet(other, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_predictions([other])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
