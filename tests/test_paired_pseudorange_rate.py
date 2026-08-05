from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline_total" / "45_build_band_mean_window_tensors.py"
SPEC = importlib.util.spec_from_file_location("band_mean_paired_prr", MODULE_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Two L1 code types collapse to their median before pairing.
            [100, 1_000, 1, 3, 1, 10.0, 0],
            [100, 1_000, 1, 3, 1, 14.0, 0],
            [100, 1_000, 1, 3, 5, 8.0, 0],
            [100, 1_000, 1, 4, 1, 20.0, 0],
            [100, 1_000, 1, 4, 5, 15.0, 0],
            [200, 2_000, 1, 3, 1, 50.0, 1],
            [200, 2_000, 1, 3, 5, 20.0, 0],
        ],
        columns=[
            "TimeNanos", "utcTimeMillis", "ConstellationType", "Svid", "FreqBand",
            "PseudorangeRateMetersPerSecond", "Label",
        ],
    )


class PairedPseudorangeRateTests(unittest.TestCase):
    def test_pairing_is_same_epoch_same_satellite_and_uses_band_medians(self) -> None:
        pairs = BUILDER.paired_pseudorange_rate_pairs(raw_rows())

        self.assertEqual(len(pairs), 3)
        row = pairs[(pairs["TimeNanos"] == 100) & (pairs["Svid"] == 3)].iloc[0]
        self.assertAlmostEqual(row["PrrPairDifference"], 4.0)
        self.assertEqual(row["PrrPairL1Label"], 0)
        self.assertEqual(row["PrrPairL5Label"], 0)

    def test_reference_uses_only_train_normal_pairs_and_has_global_fallback(self) -> None:
        pairs = pd.DataFrame(
            {
                "device_id": [1, 1, 2, 2, 1],
                "ConstellationType": [1, 1, 1, 1, 1],
                "PrrPairDifference": [1.0, 3.0, 7.0, 9.0, 999.0],
                # The final value represents validation/test or an attack and must not fit.
                "is_train_normal": [True, True, True, True, False],
            }
        )
        reference = BUILDER.fit_paired_pseudorange_rate_reference(pairs, minimum_pairs=2)

        self.assertAlmostEqual(reference["per_device"]["1"]["1"]["median"], 2.0)
        self.assertAlmostEqual(reference["global_by_constellation"]["1"]["median"], 5.0)
        self.assertAlmostEqual(reference["global_all"]["median"], 5.0)

        apply_rows = pd.DataFrame(
            {
                "device_id": [1, 99],
                "ConstellationType": [1, 1],
                "PrrPairDifference": [4.0, 6.0],
            }
        )
        transformed, use = BUILDER.apply_paired_pseudorange_rate_reference(
            apply_rows, reference
        )
        np.testing.assert_allclose(transformed["PrrPairResidual"], [2.0, 1.0])
        self.assertEqual(use["device_constellation"], 1)
        self.assertEqual(use["global_constellation"], 1)

    def test_epoch_aggregation_is_robust_and_records_availability(self) -> None:
        pairs = pd.DataFrame(
            {
                "TimeNanos": [100, 100, 100, 200],
                "PrrPairResidual": [-1.0, 1.0, 3.0, 2.0],
            }
        )
        summary = BUILDER.aggregate_paired_pseudorange_rate_epochs(pairs)

        self.assertAlmostEqual(summary.loc[100, "PrrPairMedianResidual"], 1.0)
        self.assertAlmostEqual(summary.loc[100, "PrrPairAbsMedianResidual"], 1.0)
        self.assertAlmostEqual(summary.loc[100, "PrrPairMadResidual"], 2.0)
        self.assertEqual(summary.loc[100, "PrrPairAvailable"], 1.0)
        self.assertAlmostEqual(summary.loc[200, "PrrPairMedianResidual"], 2.0)

    def test_pair_availability_stays_unscaled_after_continuous_features(self) -> None:
        names = BUILDER.feature_names_for_mode(
            False, False, False, False, include_paired_pseudorange_rate=True
        )
        self.assertEqual(names[-3:], ["PrrPairAvailable", "L1Present", "L5Present"])
        self.assertEqual(names[-6:-3], BUILDER.PAIRED_PRR_CONTINUOUS_NAMES)


if __name__ == "__main__":
    unittest.main()
