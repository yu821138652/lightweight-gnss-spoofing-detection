from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline_total" / "45_build_band_mean_window_tensors.py"
SPEC = importlib.util.spec_from_file_location("band_mean_builder", MODULE_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def make_table(l1: list[float], l5: list[float]) -> pd.DataFrame:
    count = len(l1)
    times = np.arange(count, dtype=np.int64) * 1_000_000_000
    return pd.DataFrame(
        {
            "L1_Cn0DbHz": np.asarray(l1, dtype=np.float64),
            "L5_Cn0DbHz": np.asarray(l5, dtype=np.float64),
            "utcTimeMillis": np.arange(count, dtype=np.float64) * 1000.0,
            "TOW": np.arange(count, dtype=np.float64),
        },
        index=times,
    )


def make_window_table(l1: list[float], l5: list[float]) -> pd.DataFrame:
    """Minimal epoch table accepted by ``build_windows``."""
    table = make_table(l1, l5)
    for band in ("L1", "L5"):
        table[f"{band}_AgcDb"] = -10.0
        table[f"{band}_ReceivedSvTimeUncertaintyNanos"] = 1.0
        table[f"{band}_PseudorangeRateUncertaintyMetersPerSecond"] = 0.1
    table[BUILDER.CN0_DIFF_NAME] = table["L1_Cn0DbHz"] - table["L5_Cn0DbHz"]
    table["L1Present"] = np.isfinite(table["L1_Cn0DbHz"]).astype(np.float32)
    table["L5Present"] = np.isfinite(table["L5_Cn0DbHz"]).astype(np.float32)
    return table


def manifest(count: int, split_at: int = None) -> dict[int, tuple[str, str]]:
    result = {}
    for index in range(count):
        split = "val" if split_at is not None and index >= split_at else "train"
        result[index * 1000] = (split, f"{split}-segment")
    return result


class CausalCn0FeatureTests(unittest.TestCase):
    def apply(
        self,
        table: pd.DataFrame,
        mode: str = "ema",
        gate: dict[tuple[int, int, int], float] = None,
        epoch_manifest: dict[int, tuple[str, str]] = None,
    ) -> pd.DataFrame:
        return BUILDER.add_causal_cn0_features(
            table,
            epoch_manifest or manifest(len(table)),
            source_id=7,
            device_id=3,
            mode=mode,
            half_life_seconds=1.0,
            normal_threshold=0.8,
            gate_predictions=gate or {},
        )

    def build_windows(
        self,
        table: pd.DataFrame,
        *,
        source_id: int = 7,
        recording_id: int = 11,
        intervals: list[tuple[float, float]] = None,
    ) -> tuple[dict[str, list], list[str]]:
        identity = ("test-env", "L5", f"recording-{recording_id}")
        feature_names = BUILDER.feature_names_for_mode(False, False, False, False, "ema")
        with mock.patch.object(BUILDER, "FEATURE_NAMES", feature_names):
            with mock.patch.object(BUILDER, "FEATURE_COUNT", len(feature_names)):
                with mock.patch.object(BUILDER, "CONTINUOUS_COUNT", len(feature_names) - 2):
                    parts = BUILDER.build_windows(
                        table=table,
                        identity=identity,
                        device_id=3,
                        recording_id=recording_id,
                        source_id=source_id,
                        epoch_splits=manifest(len(table)),
                        intervals={identity: intervals or []},
                        scenario_to_class={"L5": 2},
                        is_dynamic=False,
                        causal_baseline_mode="ema",
                        causal_half_life_seconds=1.0,
                        causal_normal_threshold=0.8,
                    )
        return parts, feature_names

    def test_manual_ema_uses_previous_baseline(self) -> None:
        result = self.apply(make_table([10, 12, 14], [20, 22, 24]))
        np.testing.assert_allclose(result["L1_Cn0Relative"], [0, 2, 3])
        np.testing.assert_allclose(result["L5_Cn0Relative"], [0, 2, 3])
        np.testing.assert_allclose(result["L1_Cn0AbsRelative"], [0, 2, 3])

    def test_gate_at_t_only_changes_future_feature(self) -> None:
        table = make_table([10, 10, 10, 10, 20, 30], [20, 20, 20, 20, 30, 40])
        time4 = int(table.index[4])
        frozen = self.apply(table, mode="gated", gate={(7, 3, time4): 0.1})
        updated = self.apply(table, mode="gated", gate={(7, 3, time4): 0.9})
        self.assertEqual(frozen.loc[time4, "L1_Cn0Relative"], updated.loc[time4, "L1_Cn0Relative"])
        self.assertGreater(frozen.iloc[5]["L1_Cn0Relative"], updated.iloc[5]["L1_Cn0Relative"])

    def test_missing_gate_after_warmup_freezes(self) -> None:
        table = make_table([10, 10, 10, 10, 20, 30], [20, 20, 20, 20, 30, 40])
        missing = self.apply(table, mode="gated", gate={(7, 3, int(table.index[5])): 0.9})
        explicit_freeze = self.apply(
            table,
            mode="gated",
            gate={(7, 3, int(table.index[4])): 0.0, (7, 3, int(table.index[5])): 0.9},
        )
        self.assertEqual(missing.iloc[5]["L1_Cn0Relative"], explicit_freeze.iloc[5]["L1_Cn0Relative"])

    def test_future_values_do_not_change_past_features(self) -> None:
        original = make_table([10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25])
        changed = original.copy()
        changed.iloc[4:, changed.columns.get_loc("L1_Cn0DbHz")] = [100, 200]
        before = self.apply(original)
        after = self.apply(changed)
        np.testing.assert_array_equal(
            before["L1_Cn0Relative"].iloc[:4], after["L1_Cn0Relative"].iloc[:4]
        )

    def test_split_boundary_resets_state(self) -> None:
        table = make_table([10, 12, 30, 32], [20, 22, 40, 42])
        result = self.apply(table, epoch_manifest=manifest(4, split_at=2))
        self.assertEqual(result.iloc[2]["L1_Cn0Relative"], 0.0)
        self.assertEqual(result.iloc[2]["L5_Cn0Relative"], 0.0)

    def test_segment_boundary_resets_state(self) -> None:
        table = make_table([10, 12, 30, 32], [20, 22, 40, 42])
        epoch_manifest = {
            0: ("train", "segment-a"),
            1000: ("train", "segment-a"),
            2000: ("train", "segment-b"),
            3000: ("train", "segment-b"),
        }
        result = self.apply(table, epoch_manifest=epoch_manifest)
        self.assertEqual(result.iloc[2]["L1_Cn0Relative"], 0.0)
        self.assertEqual(result.iloc[2]["L5_Cn0Relative"], 0.0)

    def test_receiver_gap_resets_state(self) -> None:
        table = make_table([10, 12, 30, 32], [20, 22, 40, 42])
        table.index = np.asarray([0, 1, 4, 5], dtype=np.int64) * 1_000_000_000
        result = self.apply(table)
        self.assertEqual(result.iloc[2]["L1_Cn0Relative"], 0.0)
        self.assertEqual(result.iloc[2]["L5_Cn0Relative"], 0.0)

    def test_band_missingness_does_not_reset_other_band(self) -> None:
        table = make_table([10, np.nan, 14], [20, 22, 24])
        result = self.apply(table)
        self.assertTrue(np.isnan(result.iloc[1]["L1_Cn0Relative"]))
        self.assertEqual(result.iloc[2]["L1_Cn0Relative"], 4.0)
        self.assertEqual(result.iloc[2]["L5_Cn0Relative"], 3.0)

    def test_separate_source_recording_builds_do_not_share_state(self) -> None:
        first = make_window_table([10, 11, 12, 13, 14], [20, 21, 22, 23, 24])
        second = make_window_table([100, 101, 102, 103, 104], [200, 201, 202, 203, 204])

        self.build_windows(first, source_id=7, recording_id=11)
        after_first, names = self.build_windows(second, source_id=8, recording_id=12)
        alone, _ = self.build_windows(second, source_id=8, recording_id=12)

        np.testing.assert_array_equal(after_first["train"]["x"][0], alone["train"]["x"][0])
        l1_relative = names.index("L1_Cn0Relative")
        l5_relative = names.index("L5_Cn0Relative")
        self.assertEqual(after_first["train"]["x"][0][0, l1_relative], 0.0)
        self.assertEqual(after_first["train"]["x"][0][0, l5_relative], 0.0)

    def test_labels_do_not_change_causal_features(self) -> None:
        table = make_window_table(
            [10, 11, 12, 13, 14, 15, 16],
            [20, 21, 22, 23, 24, 25, 26],
        )
        normal, _ = self.build_windows(table, intervals=[])
        spoofed, _ = self.build_windows(table, intervals=[(0.0, 100.0)])

        np.testing.assert_array_equal(
            np.stack(normal["train"]["x"]), np.stack(spoofed["train"]["x"])
        )
        np.testing.assert_array_equal(normal["train"]["y"], [0, 0, 0])
        np.testing.assert_array_equal(spoofed["train"]["y"], [2, 2, 2])

    def test_overlapping_windows_reuse_identical_epoch_features(self) -> None:
        table = make_window_table(
            [10, 11, 12, 13, 14, 15, 16],
            [20, 22, 21, 24, 23, 26, 25],
        )
        parts, _ = self.build_windows(table)
        windows = parts["train"]["x"]
        self.assertEqual(len(windows), 3)
        for earlier, later in zip(windows, windows[1:]):
            np.testing.assert_array_equal(earlier[1:], later[:-1])


class GatePredictionLoadingTests(unittest.TestCase):
    def write_gate_csv(self, rows: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "gate.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def test_duplicate_gate_key_is_rejected(self) -> None:
        rows = [
            {"source_id": 1, "device_id": 2, "window_time_nanos": 3, "prob_normal": 0.9},
            {"source_id": 1, "device_id": 2, "window_time_nanos": 3, "prob_normal": 0.1},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate gate prediction keys"):
            BUILDER.load_gate_predictions(self.write_gate_csv(rows))

    def test_non_numeric_gate_key_is_rejected(self) -> None:
        rows = [
            {
                "source_id": "not-a-source",
                "device_id": 2,
                "window_time_nanos": 3,
                "prob_normal": 0.9,
            }
        ]
        with self.assertRaises(ValueError):
            BUILDER.load_gate_predictions(self.write_gate_csv(rows))

    def test_fractional_gate_key_is_rejected(self) -> None:
        rows = [
            {
                "source_id": 1.2,
                "device_id": 2,
                "window_time_nanos": 3,
                "prob_normal": 0.9,
            }
        ]
        with self.assertRaisesRegex(ValueError, "must contain integers"):
            BUILDER.load_gate_predictions(self.write_gate_csv(rows))


class CausalScalerTests(unittest.TestCase):
    @staticmethod
    def datasets(validation_value: float = 10.0) -> dict[str, dict[str, np.ndarray]]:
        def split(values: list[list[float]], times: list[int]) -> dict[str, np.ndarray]:
            x = np.ones((len(values), 2, 3), dtype=np.float32)
            x[:, :, 0] = np.asarray(values, dtype=np.float32)
            return {
                "x": x,
                "device": np.zeros(len(values), dtype=np.int64),
                "source": np.ones(len(values), dtype=np.int32),
                "window_time_nanos": np.asarray(times, dtype=np.int64),
            }

        return {
            "train": split([[100.0, 1.0], [-100.0, 3.0], [5.0, 999.0]], [10, 20, 20]),
            "val": split([[validation_value, validation_value]], [30]),
            "test": split([[20.0, 20.0]], [40]),
        }

    def fit(self, validation_value: float = 10.0) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output_dir = Path(directory.name)
        with mock.patch.object(BUILDER, "FEATURE_NAMES", ["Residual", "L1Present", "L5Present"]):
            with mock.patch.object(BUILDER, "CONTINUOUS_COUNT", 1):
                BUILDER.fit_apply_scaler(
                    self.datasets(validation_value),
                    output_dir,
                    scaler_mode="global",
                    fit_unit="unique_window_endpoints",
                )
        return json.loads((output_dir / "scaler.json").read_text(encoding="utf-8"))

    def test_causal_scaler_uses_unique_train_endpoints(self) -> None:
        scaler = self.fit()
        self.assertEqual(scaler["fit_unit"], "unique_window_endpoints")
        self.assertEqual(scaler["fit_population"], "train_unique_window_endpoints_all_devices")
        np.testing.assert_allclose(scaler["global"]["mean"], [2.0])
        np.testing.assert_allclose(scaler["global"]["std"], [1.0])

    def test_validation_values_do_not_change_causal_scaler(self) -> None:
        ordinary = self.fit(validation_value=10.0)
        mutated = self.fit(validation_value=1_000_000.0)
        self.assertEqual(ordinary["global"], mutated["global"])


if __name__ == "__main__":
    unittest.main()
