# Static Response-State Experiment

This experiment promotes the device event task from a binary attack interval
label to a three-state device response label:

- `0`: normal / no observable response
- `1`: attack-associated anomaly
- `2`: direct spoof

The central CSV values are not edited.  Candidate/manual response intervals are
kept in `docs/device_response_intervals.csv`, and direct-spoof labels still
come from the existing target-band signal labels.

## Selection Principle

Watch L5 anomaly recall is a diagnostic metric, not the only target.  A useful
model must also preserve:

- overall abnormal recall
- normal false alarm rate
- direct-spoof recall
- scenario-wise recall
- device-wise recall

## Fold-6 And Fold-7 Pilot

Only `output/tensors/static_timeblock_outer_v2/fold_6` and `fold_7` are present
locally at this stage.  Both use `sparse_extreme` aggregation,
`initial_baseline_delta_with_device`, a 30-window initial baseline, and the MLP
h32 classifier.

| fold | outer scenario | Macro-F1 | FAR | abnormal recall | anomaly recall | direct recall |
|---|---|---:|---:|---:|---:|---:|
| fold_6 | st_L5 | 0.7766 | 2.83% | 90.64% | 97.58% | 47.06% |
| fold_7 | st_L1+L5 | 0.6588 | 0.22% | 97.19% | n/a | 97.19% |

Interpretation:

- The response-state formulation is not merely rescuing Watch L5.  Overall
  abnormal recall remains high on both available static outer tests.
- Fold 6 still exposes a direct-spoof weakness, especially on Pixel6 and MI8.
  This supports a tree model: first detect abnormal response, then route to
  dedicated anomaly/direct experts.
- Fold 7 suggests direct-spoof recall can be strong in L1+L5 static attacks
  when the device response is directly observable.

## Reproduction Commands

Example for fold 6:

```powershell
python pipeline_total/36_build_device_attack_event_tensors.py `
  --signal-data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/tensors/static_response_state_v1/fold_6/device_tensors_sparse_initial30_device `
  --feature-set initial_baseline_delta_with_device `
  --device-aggregate-profile sparse_extreme `
  --initial-baseline-windows 30 `
  --initial-baseline-policy exclude_stream

python pipeline_total/37_train_device_attack_event.py `
  --data-dir output/tensors/static_response_state_v1/fold_6/device_tensors_sparse_initial30_device `
  --output-dir output/hierarchical_event_v1/static_response_state_v1/fold_6/mlp_sparse_initial30_device_h32 `
  --label-key y_response_state --num-classes 3 `
  --model mlp --hidden-dim 32 --epochs 60 --batch-size 256 --patience 10

python pipeline_total/37_train_device_attack_event.py `
  --data-dir output/tensors/static_response_state_v1/fold_6/device_tensors_sparse_initial30_device `
  --output-dir output/hierarchical_event_v1/static_response_state_v1/fold_6/mlp_sparse_initial30_device_h32 `
  --label-key y_response_state `
  --checkpoint output/hierarchical_event_v1/static_response_state_v1/fold_6/mlp_sparse_initial30_device_h32/best_device_event_mlp.pt `
  --test-only
```

Summary helper:

```powershell
python pipeline_total/40_summarize_response_state_metrics.py `
  --metrics `
    output/hierarchical_event_v1/static_response_state_v1/fold_6/mlp_sparse_initial30_device_h32/test_metrics_device_event.json `
    output/hierarchical_event_v1/static_response_state_v1/fold_7/mlp_sparse_initial30_device_h32/test_metrics_device_event.json `
  --output-csv output/hierarchical_event_v1/static_response_state_v1/static_response_state_summary.csv
```
