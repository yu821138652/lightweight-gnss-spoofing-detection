from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_module(script_name: str, module_name: str):
    path = ROOT / "pipeline_total" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAINER = load_pipeline_module("46_train_band_mean_multiclass.py", "band_mean_trainer")
AGGREGATOR = load_pipeline_module("47_aggregate_band_mean_cv.py", "band_mean_aggregator")
SUMMARIZER = load_pipeline_module("58_summarize_band_mean_predictions.py", "band_mean_summarizer")


class CheckpointContractTests(unittest.TestCase):
    def test_none_causal_metadata_is_backward_compatible(self) -> None:
        self.assertEqual(
            TRAINER.causal_contract(
                {"mode": "none", "half_life_seconds": 60.0, "construction_stats": {}}
            ),
            {"mode": "none"},
        )

    def test_tensor_contract_detects_modified_prediction_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_names.json").write_text('["x"]', encoding="utf-8")
            (root / "tensor_metadata.json").write_text("{}", encoding="utf-8")
            (root / "test.npz").write_bytes(b"original-test-bytes")
            contract = {
                name: TRAINER.sha256_file(root / name)
                for name in ("feature_names.json", "tensor_metadata.json", "test.npz")
            }
            TRAINER.validate_tensor_contract(contract, root, "test")
            (root / "test.npz").write_bytes(b"modified-test-bytes")
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                TRAINER.validate_tensor_contract(contract, root, "test")


class AggregatePredictionContractTests(unittest.TestCase):
    @staticmethod
    def prediction_rows(fold: int = 1) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": fold,
                    "source_id": 10,
                    "device_id": 2,
                    "window_time_nanos": 100,
                    "true_class": 0,
                    "pred_class": 0,
                }
            ]
        )

    def test_fold_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            self.prediction_rows(fold=2).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, r"expected \[1\]"):
                AGGREGATOR.load_fold_predictions(path, expected_fold=1)

    def test_duplicate_strict_endpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            frame = pd.concat([self.prediction_rows(), self.prediction_rows()], ignore_index=True)
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate endpoint"):
                AGGREGATOR.load_fold_predictions(path, expected_fold=1)

    def test_subset_macro_f1_keeps_fixed_four_class_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            self.prediction_rows().to_csv(path, index=False)
            result = AGGREGATOR.aggregate([path], root / "summary", [1])
            self.assertEqual(result["macro_f1"], 0.25)

    def test_stage_marker_rejects_modified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"first")
            marker = root / "marker.json"
            config = {"scope": "static"}
            AGGREGATOR.write_stage_marker(marker, config, ["command"], [artifact])
            self.assertTrue(AGGREGATOR.stage_matches(marker, config, [artifact]))
            artifact.write_bytes(b"second-version")
            self.assertFalse(AGGREGATOR.stage_matches(marker, config, [artifact]))


class PredictionSummaryCompletenessTests(unittest.TestCase):
    @staticmethod
    def protocol_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        catalog = pd.DataFrame(
            [
                {
                    "recording_id": 0,
                    "Environment": "env-a",
                    "Scenario": "st_L1",
                    "Session": "session-a",
                    "test_fold": 1,
                },
                {
                    "recording_id": 1,
                    "Environment": "env-b",
                    "Scenario": "st_L5",
                    "Session": "session-b",
                    "test_fold": 2,
                },
            ]
        )
        manifests = pd.DataFrame(
            [
                {"fold": 1, "recording_id": 0, "split": "test"},
                {"fold": 2, "recording_id": 1, "split": "test"},
            ]
        )
        return manifests, catalog

    def test_missing_test_recording_is_rejected(self) -> None:
        manifests, catalog = self.protocol_tables()
        predictions = pd.DataFrame([{"fold": 1, "recording_id": 0}])
        with self.assertRaisesRegex(ValueError, "No all/static/dynamic recording-id mapping"):
            SUMMARIZER.infer_recording_mapping(predictions, manifests, catalog)

    def test_complete_test_recording_set_maps(self) -> None:
        manifests, catalog = self.protocol_tables()
        predictions = pd.DataFrame(
            [
                {"fold": 1, "recording_id": 0},
                {"fold": 2, "recording_id": 1},
            ]
        )
        mode, mapping = SUMMARIZER.infer_recording_mapping(
            predictions, manifests, catalog
        )
        self.assertIn("static", mode)
        self.assertEqual(len(mapping), 2)


if __name__ == "__main__":
    unittest.main()
