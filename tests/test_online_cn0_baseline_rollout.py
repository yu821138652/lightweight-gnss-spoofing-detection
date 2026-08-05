from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline_total" / "59_train_online_cn0_baseline.py"
SPEC = importlib.util.spec_from_file_location("online_cn0_baseline", MODULE_PATH)
ONLINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ONLINE
SPEC.loader.exec_module(ONLINE)


def make_data(
    l1: list[float],
    l5: list[float],
    *,
    labels: list[int] | None = None,
    single_band: list[bool] | None = None,
    stream_key: list[str] | None = None,
    l1_present: list[bool] | None = None,
    l5_present: list[bool] | None = None,
) -> ONLINE.SplitData:
    """Create overlapping W5 endpoints from an epoch-level toy stream."""
    l1_present = l1_present or [True] * len(l1)
    l5_present = l5_present or [True] * len(l5)
    l1_values = np.asarray(l1, dtype=np.float32)
    l5_values = np.asarray(l5, dtype=np.float32)
    l1_values[~np.asarray(l1_present, dtype=bool)] = np.nan
    l5_values[~np.asarray(l5_present, dtype=bool)] = np.nan
    epochs = np.column_stack(
        (
            l1_values,
            l5_values,
            np.asarray(l1_present, dtype=np.float32),
            np.asarray(l5_present, dtype=np.float32),
        )
    )
    # The tensor builder replaces absent-band NaNs by the scaled neutral value
    # after fitting, while the presence flag retains the missingness signal.
    epochs = np.nan_to_num(epochs, nan=0.0)
    windows = np.stack([epochs[index - 4:index + 1] for index in range(4, len(epochs))])
    count = len(windows)
    return ONLINE.SplitData(
        x=windows,
        y=np.asarray(labels or [0] * count, dtype=np.int64),
        single_band=np.asarray(single_band or [False] * count, dtype=bool),
        recording_id=np.zeros(count, dtype=np.int32),
        source_id=np.ones(count, dtype=np.int32),
        device_id=np.zeros(count, dtype=np.int64),
        window_time_nanos=np.arange(4, 4 + count, dtype=np.int64) * 1_000_000_000,
        endpoint_tow=np.arange(4, 4 + count, dtype=np.float64),
        stream_key=np.asarray(stream_key or ["1:train:segment-a"] * count),
    )


class OnlineBaselineStateTests(unittest.TestCase):
    def test_prediction_updates_only_future_baseline(self) -> None:
        data = make_data([10, 10, 10, 10, 20, 30], [30, 30, 30, 30, 40, 50])
        state = ONLINE.StreamState(data, ONLINE.Episode("stream", np.array([0, 1])), alpha=0.5)

        first = state.prepare_input()
        np.testing.assert_allclose(first[-1, 4:], [10.0, 30.0])
        state.advance(predicted_normal=True)

        second = state.prepare_input()
        # The first endpoint's normal prediction updates the state used by the
        # second endpoint; it must not have changed the first input itself.
        np.testing.assert_allclose(second[-1, 4:], [15.0, 35.0])

    def test_non_normal_prediction_freezes_state(self) -> None:
        data = make_data([10, 10, 10, 10, 20, 30], [30, 30, 30, 30, 40, 50])
        state = ONLINE.StreamState(data, ONLINE.Episode("stream", np.array([0, 1])), alpha=0.5)
        state.prepare_input()
        state.advance(predicted_normal=False)
        second = state.prepare_input()
        np.testing.assert_allclose(second[-1, 4:], [10.0, 30.0])

    def test_labels_do_not_affect_state_features(self) -> None:
        first_data = make_data([10, 10, 10, 10, 20, 30], [30, 30, 30, 30, 40, 50], labels=[0, 0])
        second_data = make_data([10, 10, 10, 10, 20, 30], [30, 30, 30, 30, 40, 50], labels=[3, 2])
        first = ONLINE.StreamState(first_data, ONLINE.Episode("stream", np.array([0, 1])), alpha=0.5)
        second = ONLINE.StreamState(second_data, ONLINE.Episode("stream", np.array([0, 1])), alpha=0.5)

        np.testing.assert_array_equal(first.prepare_input(), second.prepare_input())
        first.advance(predicted_normal=True)
        second.advance(predicted_normal=True)
        np.testing.assert_array_equal(first.prepare_input(), second.prepare_input())

    def test_single_band_endpoint_updates_only_present_band_when_normal(self) -> None:
        data = make_data(
            [10, 10, 10, 10, 20, 30, 40],
            [30, 30, 30, 30, 40, 50, 60],
            single_band=[False, True, False],
            l5_present=[True, True, True, True, True, False, True],
        )
        state = ONLINE.StreamState(data, ONLINE.Episode("stream", np.array([0, 1, 2])), alpha=0.5)
        state.prepare_input()
        state.advance(predicted_normal=True)
        state.prepare_input()
        state.advance(predicted_normal=True)
        third = state.prepare_input()
        np.testing.assert_allclose(third[-1, 4:], [22.5, 35.0])

    def test_rollout_uses_single_band_model_decision_but_excludes_it_from_metrics(self) -> None:
        class CaptureNormalModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[np.ndarray] = []

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                self.inputs.append(value.detach().cpu().numpy().copy())
                logits = torch.zeros((len(value), 4), dtype=value.dtype, device=value.device)
                logits[:, 0] = 1.0
                return logits

        data = make_data(
            [10, 10, 10, 10, 20, 30, 40],
            [30, 30, 30, 30, 40, 50, 60],
            single_band=[False, True, False],
            l5_present=[True, True, True, True, True, False, True],
        )
        model = CaptureNormalModel()
        result = ONLINE.rollout_split(
            model,
            data,
            ONLINE.build_episodes(data),
            alpha=0.5,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(result.metrics["samples"], 2)
        np.testing.assert_allclose(model.inputs[-1][0, -1, 4:], [22.5, 35.0])

    def test_segment_key_starts_an_independent_episode(self) -> None:
        data = make_data(
            [10, 10, 10, 10, 20, 30],
            [30, 30, 30, 30, 40, 50],
            stream_key=["1:train:segment-a", "1:train:segment-b"],
        )
        episodes = ONLINE.build_episodes(data)
        self.assertEqual(len(episodes), 2)
        self.assertTrue(all(len(episode.indices) == 1 for episode in episodes))


if __name__ == "__main__":
    unittest.main()
