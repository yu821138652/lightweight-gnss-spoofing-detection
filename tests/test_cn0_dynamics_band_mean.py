from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline_total" / "45_build_band_mean_window_tensors.py"
SPEC = importlib.util.spec_from_file_location("band_mean_cn0_dynamics", MODULE_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def table(l1: list[float], l5: list[float], times: list[int] | None = None) -> pd.DataFrame:
    count = len(l1)
    times = times or list(range(count))
    return pd.DataFrame(
        {
            "L1_Cn0DbHz": np.asarray(l1, dtype=np.float64),
            "L5_Cn0DbHz": np.asarray(l5, dtype=np.float64),
            "utcTimeMillis": np.asarray(times, dtype=np.float64) * 1000.0,
            "TOW": np.asarray(times, dtype=np.float64),
        },
        index=np.asarray(times, dtype=np.int64) * 1_000_000_000,
    )


def manifest(times: list[int], split: str = "train", segment: str = "a") -> dict[int, tuple[str, str]]:
    return {int(value * 1000): (split, segment) for value in times}


class Cn0DynamicsTests(unittest.TestCase):
    def test_w5_slope_and_spread_use_only_current_and_past(self) -> None:
        original = table([0, 1, 2, 3, 4, 5], [10, 12, 14, 16, 18, 20])
        changed_future = original.copy()
        changed_future.iloc[5, changed_future.columns.get_loc("L1_Cn0DbHz")] = 500.0
        assignments = manifest(list(range(6)))

        expected = BUILDER.add_cn0_dynamics_features(original, assignments)
        actual = BUILDER.add_cn0_dynamics_features(changed_future, assignments)

        np.testing.assert_allclose(expected["L1_Cn0W5SlopeDbHzPerSecond"].iloc[4], 1.0)
        np.testing.assert_allclose(expected["L1_Cn0W5StdDbHz"].iloc[4], np.sqrt(2.0))
        self.assertEqual(expected["L1_Cn0W5ValidCount"].iloc[4], 5.0)
        np.testing.assert_allclose(expected["L5_Cn0W5SlopeDbHzPerSecond"].iloc[4], 2.0)
        np.testing.assert_allclose(expected["L5_Cn0W5StdDbHz"].iloc[4], np.sqrt(8.0))
        self.assertEqual(expected["L5_Cn0W5ValidCount"].iloc[4], 5.0)
        np.testing.assert_array_equal(
            expected[BUILDER.CN0_DYNAMICS_NAMES].iloc[:5],
            actual[BUILDER.CN0_DYNAMICS_NAMES].iloc[:5],
        )

    def test_split_segment_and_gap_each_reset_history(self) -> None:
        frame = table([0, 2, 100, 102, 200, 202], [0, 2, 100, 102, 200, 202], [0, 1, 2, 3, 7, 8])
        assignments = {
            0: ("train", "a"),
            1000: ("train", "a"),
            2000: ("val", "b"),
            3000: ("val", "b"),
            7000: ("val", "c"),
            8000: ("val", "c"),
        }
        result = BUILDER.add_cn0_dynamics_features(frame, assignments)

        self.assertTrue(np.isnan(result["L1_Cn0W5SlopeDbHzPerSecond"].iloc[2]))
        self.assertEqual(result["L1_Cn0W5ValidCount"].iloc[2], 1.0)
        np.testing.assert_allclose(result["L1_Cn0W5SlopeDbHzPerSecond"].iloc[3], 2.0)
        self.assertTrue(np.isnan(result["L1_Cn0W5SlopeDbHzPerSecond"].iloc[4]))
        self.assertEqual(result["L1_Cn0W5ValidCount"].iloc[4], 1.0)
        np.testing.assert_allclose(result["L1_Cn0W5SlopeDbHzPerSecond"].iloc[5], 2.0)

    def test_missing_band_values_are_ignored_not_zero_filled(self) -> None:
        frame = table([10, np.nan, 14], [20, 22, 24])
        result = BUILDER.add_cn0_dynamics_features(frame, manifest([0, 1, 2]))

        np.testing.assert_allclose(result["L1_Cn0W5SlopeDbHzPerSecond"].iloc[2], 2.0)
        np.testing.assert_allclose(result["L1_Cn0W5StdDbHz"].iloc[2], 2.0)
        self.assertEqual(result["L1_Cn0W5ValidCount"].iloc[2], 2.0)
        self.assertTrue(np.isnan(result["L1_Cn0W5SlopeDbHzPerSecond"].iloc[1]))
        self.assertEqual(result["L1_Cn0W5ValidCount"].iloc[1], 1.0)

    def test_feature_names_append_four_cn0_dynamics_before_presence_flags(self) -> None:
        names = BUILDER.feature_names_for_mode(
            False, False, False, False, include_cn0_dynamics=True
        )
        self.assertEqual(names[-2:], ["L1Present", "L5Present"])
        self.assertEqual(names[-3], BUILDER.CN0_DIFF_NAME)
        self.assertEqual(names[-9:-3], BUILDER.CN0_DYNAMICS_NAMES)


if __name__ == "__main__":
    unittest.main()
