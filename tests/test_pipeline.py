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
    pd.DataFrame(dict(node_id=np.arange(8), chrom=np.repeat(["chr1", "chr2", "chr3", "chr4"], 2), center=np.arange(8) * 100,
                      is_promoter=np.arange(8) % 2 == 1,
                      log_atac=np.arange(8) / 8,
                      seq_emb=[np.array([i / 8, 1.0]) for i in range(8)])).to_parquet(path / "nodes.parquet")
    pd.DataFrame(dict(u=[0, 2, 4, 6], v=[1, 3, 5, 7], type=[0, 1, 0, 1], oe=[np.nan, 4., np.nan, 5.])).to_parquet(path / "edges.parquet")
    (path / "hic").mkdir()
    pd.DataFrame(dict(chrom=["chr1", "chr2"], bin1=[0, 0], bin2=[0, 0], observed=[3., 7.], oe=[1., 1.])).to_parquet(path / "hic" / "all.parquet")
    pd.DataFrame(dict(chrom=np.repeat(["chr1", "chr2", "chr3", "chr4"], 2), center=[50, 60] * 4, tss=[150, 260] * 4,
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
        for kind in ["gnn", "mamba", "attention", "combined"]:
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
        paths = [self.root / kind / "predictions.parquet" for kind in ["gnn", "mamba", "attention", "combined"]]
        result = evaluate(paths, self.root / "evaluation", ["mamba_distance", "gnn_distance"])
        self.assertEqual(set(result.model), {"distance", "mamba_distance", "gnn_distance", "attention_distance", "combined_distance", "ensemble"})
        self.assertTrue(np.isfinite(result.AUPRC).all())

    def test_pair_features_and_pairwise_control(self):
        import shutil
        for kind in ["pairwise", "endpoints", "mamba"]:
            self.train(kind, ["--distance", "--pair-features"])
        paths = [self.root / kind / "predictions.parquet" for kind in ["pairwise", "endpoints", "mamba"]]
        result = evaluate(paths, self.root / "evaluation")
        self.assertEqual(set(result.model), {"distance", "pairwise_distance_pairfeat", "endpoints_distance_pairfeat",
                                             "mamba_distance_pairfeat"})
        data = GraphData.load(self.data, pair_features=True)
        self.assertEqual(tuple(data.pair_features.shape), (8, 3))
        np.testing.assert_allclose(data.pair_features[:4, 0], np.log1p([3., 3., 7., 7.]))   # chr3/chr4 have no record -> 0
        self.assertEqual(float(data.pair_features[4:, 0].abs().sum()), 0.0)
        self.assertIn("no encoder", self.train("pairwise", success=False).stderr)
        shutil.rmtree(self.data / "hic")
        shutil.rmtree(self.root / "mamba")
        self.assertIn("cached Hi-C contacts", self.train("mamba", ["--pair-features"], success=False).stderr)

    def test_endpoint_encoder_reads_only_the_anchor(self):
        from models import EndpointEncoder
        data = GraphData.load(self.data)
        encoder = EndpointEncoder(data, 4)
        W = data.walks[0]
        shuffled = W.clone()
        shuffled[..., 1:] = W[..., 1:].flip(0)   # change every non-anchor walk step
        torch.testing.assert_close(encoder(W, data.eidx[0]), encoder(shuffled, data.eidx[0]))
        anchors = torch.as_tensor(np.unique(data.pairs[["elem_node", "gene_node"]].to_numpy()))
        torch.testing.assert_close(encoder(W, data.eidx[0]), encoder.out(encoder.node(data.X[anchors])))

    def test_head_scalars_standardized_on_training_fold(self):
        from models import chromosome_folds
        self.train("pairwise", ["--distance", "--re2g-features"])
        data = GraphData.load(self.data, pair_features="re2g")
        fold = data.pairs.chrom.map(chromosome_folds(data.benchmark, 2)).to_numpy()
        saved = torch.load(self.root / "pairwise" / "fold0.pt", weights_only=True)["state_dict"]
        train = torch.as_tensor(np.flatnonzero(fold != 0))
        values = torch.cat([data.distance[train, None], data.pair_features[train]], -1)
        torch.testing.assert_close(saved["scalar_mean"], values.mean(0))
        torch.testing.assert_close(saved["scalar_std"], values.std(0).clamp_min(1e-6))
        self.train("gnn", ["--distance", "--raw-scalars"])
        saved = torch.load(self.root / "gnn" / "fold0.pt", weights_only=True)["state_dict"]
        self.assertEqual(saved["scalar_mean"].tolist(), [0.0])
        self.assertEqual(saved["scalar_std"].tolist(), [1.0])

    def test_subgraph_encoder(self):
        from models import WalkEncoder
        data = GraphData.load(self.data)
        encoder = WalkEncoder(data, "subgraph", 4, 3, "none")
        # Anchor 0 visits nodes 0 and 1 without ever stepping along their edge; anchor 1 visits only node 1.
        W = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
        glob, owner, src, dst, e = encoder.subgraph(W)
        self.assertEqual(glob.tolist(), [0, 1, 1])
        self.assertEqual(owner.tolist(), [0, 0, 1])
        self.assertEqual(sorted(zip(src.tolist(), dst.tolist())), [(0, 1), (1, 0)])   # parent edge kept, per anchor
        self.assertTrue(torch.equal(data.ETYPE[e], torch.tensor([0, 0])))
        # Only the visited node set matters, not walk order.
        walks = data.walks[0]
        shuffled = walks.flip(-1)
        torch.testing.assert_close(encoder(walks, data.eidx[0]), encoder(shuffled, data.eidx[0]))
        self.train("subgraph", ["--distance", "--re2g-features", "--layers", "3"])
        result = evaluate([self.root / "subgraph" / "predictions.parquet"], self.root / "evaluation")
        self.assertEqual(set(result.model), {"distance", "subgraph_distance_re2g"})

    def test_re2g_features(self):
        from baselines import abc_baseline
        from models import RE2G_FEATURES
        data = GraphData.load(self.data, pair_features="re2g")
        features = data.pair_features.numpy()
        self.assertEqual(features.shape, (8, len(RE2G_FEATURES)))
        np.testing.assert_allclose(features[:, :3], GraphData.load(self.data, pair_features=True).pair_features.numpy())
        _, abc = abc_baseline(self.data)
        np.testing.assert_allclose(features[:, 3], np.log(abc[data.pairs.row_id] + 1e-3), rtol=1e-5)
        # Each chromosome holds only the pair's own two nodes (element at 0/200/..., promoter at 100/300/...),
        # so nothing lies strictly between them; the promoter node is the only other node within 10 kb.
        np.testing.assert_array_equal(features[:, 4:6], 0)
        np.testing.assert_allclose(features[:, 6], np.log1p(1))
        for kind in ["pairwise", "attention"]:
            self.train(kind, ["--distance", "--re2g-features"])
        result = evaluate([self.root / kind / "predictions.parquet" for kind in ["pairwise", "attention"]], self.root / "evaluation")
        self.assertEqual(set(result.model), {"distance", "pairwise_distance_re2g", "attention_distance_re2g"})

    def test_no_seq_ablation(self):
        X = GraphData.load(self.data, seq=False).X
        np.testing.assert_allclose(X[:, 0], np.arange(8) / 8)   # log_atac is the only column
        self.assertEqual(X.shape[1], 1)
        self.train("attention", ["--distance", "--pair-features", "--no-seq"])
        result = evaluate([self.root / "attention" / "predictions.parquet"], self.root / "evaluation")
        self.assertEqual(set(result.model), {"distance", "attention_distance_pairfeat_noseq"})
        self.assertIn("--no-seq", self.train("gnn", ["--distance", "--no-seq", "--no-atac"], success=False).stderr)

    def test_no_atac_ablation(self):
        self.assertEqual(GraphData.load(self.data).X.shape[1], 3)
        X = GraphData.load(self.data, atac=False).X
        self.assertEqual(X.shape[1], 2)
        np.testing.assert_allclose(X[:, 1], 1.0)   # columns are seq_emb, not log_atac followed by seq_emb
        self.train("gnn", ["--distance", "--no-atac"])
        frame = pd.read_parquet(self.root / "gnn" / "predictions.parquet")
        self.assertIn("gnn_distance_noatac", frame.columns)
        result = evaluate([self.root / "gnn" / "predictions.parquet"], self.root / "evaluation")
        self.assertEqual(set(result.model), {"distance", "gnn_distance_noatac"})

    def test_abc_baseline_in_evaluation(self):
        self.train()
        result = evaluate([self.root / "gnn" / "predictions.parquet"], self.root / "evaluation", data_dir=self.data)
        self.assertEqual(list(result.model.unique()), ["distance", "abc", "gnn"])
        from baselines import abc_baseline
        _, abc = abc_baseline(self.data)
        self.assertTrue(np.isfinite(abc).all() and (abc >= 0).all())
        crispr = pd.read_parquet(self.data / "crispr.parquet")
        crispr.loc[0, "label"] = 1 - crispr.loc[0, "label"]
        crispr.to_parquet(self.data / "crispr.parquet")
        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluate([self.root / "gnn" / "predictions.parquet"], self.root / "evaluation", data_dir=self.data)
        self.assertIn("--no-atac", self.train("mamba", ["--distance", "--pair-features", "--no-atac"], success=False).stderr)

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
