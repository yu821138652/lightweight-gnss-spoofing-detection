from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline_total" / "45_build_band_mean_window_tensors.py"
SPEC = importlib.util.spec_from_file_location("band_mean_normal_reference", MODULE_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


SCENARIO_TO_CLASS = {"L1": 1, "L5": 2, "L1+L5": 3}


def epoch_table(l1: list[float], l5: list[float], tows: list[float]) -> pd.DataFrame:
    count = len(tows)
    return pd.DataFrame(
        {
            "L1_Cn0DbHz": np.asarray(l1, dtype=np.float64),
            "L5_Cn0DbHz": np.asarray(l5, dtype=np.float64),
            "utcTimeMillis": np.arange(count, dtype=np.float64) * 1000.0,
            "TOW": np.asarray(tows, dtype=np.float64),
        },
        index=np.arange(count, dtype=np.int64) * 1_000_000_000,
    )


def source_record(
    identity: tuple[str, str, str],
    device_id: int,
    table: pd.DataFrame,
    splits: list[str] | None = None,
) -> dict:
    splits = splits or ["train"] * len(table)
    return {
        "identity": identity,
        "device_id": device_id,
        "table": table,
        "epoch_splits": {
            int(utc): (split, f"{split}-segment")
            for utc, split in zip(table["utcTimeMillis"], splits)
        },
    }


class NormalBandReferenceTests(unittest.TestCase):
    def test_target_band_semantics_for_l1_l5_and_dual_band_attacks(self) -> None:
        l1 = ("field", "L1", "l1-attack")
        l5 = ("field", "L5", "l5-attack")
        dual = ("field", "L1+L5", "dual-attack")
        intervals = {l1: [(10.0, 10.0)], l5: [(10.0, 10.0)], dual: [(10.0, 10.0)]}

        self.assertFalse(
            BUILDER.band_is_normal_at_tow(l1, 10.0, 1, intervals, SCENARIO_TO_CLASS)
        )
        self.assertTrue(
            BUILDER.band_is_normal_at_tow(l1, 10.0, 5, intervals, SCENARIO_TO_CLASS)
        )
        self.assertTrue(
            BUILDER.band_is_normal_at_tow(l5, 10.0, 1, intervals, SCENARIO_TO_CLASS)
        )
        self.assertFalse(
            BUILDER.band_is_normal_at_tow(l5, 10.0, 5, intervals, SCENARIO_TO_CLASS)
        )
        self.assertFalse(
            BUILDER.band_is_normal_at_tow(dual, 10.0, 1, intervals, SCENARIO_TO_CLASS)
        )
        self.assertFalse(
            BUILDER.band_is_normal_at_tow(dual, 10.0, 5, intervals, SCENARIO_TO_CLASS)
        )
        self.assertTrue(
            BUILDER.band_is_normal_at_tow(l1, 11.0, 1, intervals, SCENARIO_TO_CLASS)
        )
        self.assertTrue(
            BUILDER.band_is_normal_at_tow(l1, 11.0, 5, intervals, SCENARIO_TO_CLASS)
        )

    def test_fit_excludes_only_target_band_during_single_band_attack(self) -> None:
        l1 = ("field", "L1", "l1-attack")
        l5 = ("field", "L5", "l5-attack")
        dual = ("field", "L1+L5", "dual-attack")
        intervals = {l1: [(1.0, 1.0)], l5: [(1.0, 1.0)], dual: [(1.0, 1.0)]}
        records = [
            source_record(l1, 7, epoch_table([10.0, 1_000.0], [100.0, 101.0], [0.0, 1.0])),
            source_record(l5, 7, epoch_table([20.0, 21.0], [200.0, 2_000.0], [0.0, 1.0])),
            source_record(dual, 7, epoch_table([30.0, 3_000.0], [300.0, 4_000.0], [0.0, 1.0])),
        ]

        reference = BUILDER.fit_normal_band_reference(
            records, intervals, SCENARIO_TO_CLASS, minimum_epochs=1
        )

        # L1 excludes 1000 (L1 attack) and 3000 (dual-band attack), but keeps
        # 21 from the L5-only attack.  L5 follows the symmetric rule.
        self.assertEqual(reference["global"]["1"]["count"], 4)
        self.assertEqual(reference["global"]["5"]["count"], 4)
        self.assertAlmostEqual(reference["global"]["1"]["mean"], 20.25)
        self.assertAlmostEqual(reference["global"]["5"]["mean"], 175.25)
        self.assertAlmostEqual(reference["per_device"]["7"]["1"]["mean"], 20.25)
        self.assertAlmostEqual(reference["per_device"]["7"]["5"]["mean"], 175.25)

    def test_validation_and_test_epochs_do_not_change_reference(self) -> None:
        identity = ("field", "L1", "clean")
        table = epoch_table([10.0, 999.0, 8_888.0], [20.0, 777.0, 6_666.0], [0.0, 1.0, 2.0])
        records = [source_record(identity, 3, table, ["train", "val", "test"])]

        reference = BUILDER.fit_normal_band_reference(
            records, {}, SCENARIO_TO_CLASS, minimum_epochs=1
        )

        self.assertEqual(reference["global"]["1"]["count"], 1)
        self.assertEqual(reference["global"]["5"]["count"], 1)
        self.assertEqual(reference["global"]["1"]["mean"], 10.0)
        self.assertEqual(reference["global"]["5"]["mean"], 20.0)

    def test_known_device_is_preferred_unknown_device_falls_back_and_values_subtract(self) -> None:
        clean = ("field", "L1", "clean")
        records = [
            source_record(clean, 3, epoch_table([10.0], [20.0], [0.0])),
            source_record(clean, 4, epoch_table([30.0], [40.0], [0.0])),
        ]
        reference = BUILDER.fit_normal_band_reference(
            records, {}, SCENARIO_TO_CLASS, minimum_epochs=1
        )

        known, known_assignment = BUILDER.apply_normal_band_reference(
            epoch_table([15.0, 10.0], [26.0, 20.0], [0.0, 1.0]), 3, reference
        )
        unknown, unknown_assignment = BUILDER.apply_normal_band_reference(
            epoch_table([25.0, np.nan], [40.0, 50.0], [0.0, 1.0]), 999, reference
        )

        self.assertEqual(known_assignment, {"L1": "device", "L5": "device"})
        np.testing.assert_allclose(known["L1_Cn0DbHz"], [5.0, 0.0])
        np.testing.assert_allclose(known["L5_Cn0DbHz"], [6.0, 0.0])
        np.testing.assert_allclose(known[BUILDER.CN0_DIFF_NAME], [-1.0, 0.0])

        self.assertEqual(unknown_assignment, {"L1": "global", "L5": "global"})
        np.testing.assert_allclose(unknown["L1_Cn0DbHz"].iloc[0], 5.0)
        self.assertTrue(np.isnan(unknown["L1_Cn0DbHz"].iloc[1]))
        np.testing.assert_allclose(unknown["L5_Cn0DbHz"], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
