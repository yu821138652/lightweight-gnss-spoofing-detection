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

## Direct Override Pilot

A hard two-stage tree was also tested:

1. normal vs abnormal
2. anomaly vs direct, only when stage 1 predicts abnormal

It improved fold-6 direct recall but hurt anomaly recall and degraded fold-7
direct recall, so it is not the preferred next step.  A softer variant works
better: keep the flat three-class model as the base prediction, then use a
binary direct-spoof expert only as an override when its direct probability is
high.

The current best pilot uses validation-calibrated direct thresholds with
`override_scope=all`.  The threshold is selected on validation under
`max_val_far=0.05` and `min_val_abnormal_recall=0.90`, then applied once to
the outer test.

| fold | outer scenario | Macro-F1 | FAR | abnormal recall | anomaly recall | direct recall |
|---|---|---:|---:|---:|---:|---:|
| fold_6 | st_L5 | 0.8529 | 2.85% | 91.73% | 97.52% | 65.36% |
| fold_7 | st_L1+L5 | 0.6593 | 0.28% | 97.51% | n/a | 97.51% |

Compared with the flat three-class pilot, the direct override improves fold-6
direct recall from 47.06% to 65.36% with almost unchanged FAR.  Fold 7 is not
hurt.  Across the two locally available folds, mean direct recall rises from
72.13% to 81.43%, while mean FAR remains low at 1.57%.

## Six Valid Static Folds

After rebuilding signal tensors, six folds currently have valid outer-test
response-state results under the 30-window initial-baseline protocol:
`fold_1`, `fold_2`, `fold_4`, `fold_5`, `fold_6`, and `fold_7`.

`fold_3` is not valid under this protocol because its outer test recording
(`new_building/st_L5/2025.07.29.20.36`) starts inside the same L5 attack event.
The 30-window initial baseline is therefore not independently normal, and the
builder excludes the entire test stream.  This fold needs a separate protocol
decision before it can be included.

For the six valid folds, validation-calibrated direct override gives:

```text
supported Macro-F1: 0.9485
raw Macro-F1: 0.7279
FAR: 1.28%
abnormal recall: 94.04%
direct recall: 93.39%
anomaly recall: 73.76%  # averaged only over folds with anomaly support
```

`supported Macro-F1` averages only classes present in the test split.  Raw
Macro-F1 is still retained, but it is artificially low in static folds whose
outer test contains no anomaly class.

| fold | outer scenario | supported Macro-F1 | raw Macro-F1 | FAR | abnormal recall | anomaly recall | direct recall |
|---|---|---:|---:|---:|---:|---:|---:|
| fold_1 | st_L1 | 0.9906 | 0.6604 | 3.41% | 99.88% | n/a | 99.88% |
| fold_2 | st_L5 | 0.8670 | 0.8670 | 0.36% | 76.13% | 50.00% | 98.62% |
| fold_4 | st_L1+L5 | 0.9971 | 0.6648 | 0.50% | 100.00% | n/a | 100.00% |
| fold_5 | st_L1 | 0.9941 | 0.6627 | 0.31% | 99.00% | n/a | 99.00% |
| fold_6 | st_L5 | 0.8529 | 0.8529 | 2.85% | 91.73% | 97.52% | 65.36% |
| fold_7 | st_L1+L5 | 0.9890 | 0.6593 | 0.28% | 97.51% | n/a | 97.51% |

This supports the next full experiment design:

- base model: three-class response state
- expert: binary direct-spoof detector
- inference: direct expert can override the base prediction, with the threshold
  selected on validation under FAR and abnormal-recall constraints

## Reproduction Commands

### Full Fold Runner

`pipeline_total/43_run_static_response_state_fold.py` runs one static fold from
an existing signal tensor directory:

1. build device response-state tensors
2. train the flat three-class MLP
3. evaluate the flat model on test
4. train the binary direct-spoof expert
5. calibrate the direct override threshold on validation and evaluate test

Example:

```powershell
python pipeline_total/43_run_static_response_state_fold.py `
  --fold fold_6 `
  --python-exe H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe `
  --overwrite-device-tensors
```

Use `--dry-run` first to print the exact command list without executing it.
The runner expects `output/tensors/static_timeblock_outer_v2/<fold>` to exist.
At the time of this note, only `fold_6` and `fold_7` are present locally.

For missing folds, first rebuild the signal tensors from the existing protocol
manifests.  Example for `fold_1`:

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2/fold_1/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2/fold_1/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2/fold_1 `
  --time-steps 5 `
  --block-size 256
```

Then run the response-state fold runner:

```powershell
python pipeline_total/43_run_static_response_state_fold.py `
  --fold fold_1 `
  --python-exe H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe
```

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

Direct override example:

```powershell
python pipeline_total/37_train_device_attack_event.py `
  --data-dir output/tensors/static_response_state_v1/fold_6/device_tensors_sparse_initial30_device `
  --output-dir output/hierarchical_event_v1/static_response_state_v1/fold_6/direct_expert_mlp_h32 `
  --label-key y_response_state --label-transform direct --num-classes 2 `
  --model mlp --hidden-dim 32 --epochs 60 --batch-size 256 --patience 10

python pipeline_total/42_eval_response_state_direct_override.py `
  --data-dir output/tensors/static_response_state_v1/fold_6/device_tensors_sparse_initial30_device `
  --output-dir output/hierarchical_event_v1/static_response_state_v1/fold_6/direct_override_mlp_h32_valcal_all `
  --split test `
  --flat-checkpoint output/hierarchical_event_v1/static_response_state_v1/fold_6/mlp_sparse_initial30_device_h32/best_device_event_mlp.pt `
  --direct-checkpoint output/hierarchical_event_v1/static_response_state_v1/fold_6/direct_expert_mlp_h32/best_device_event_mlp.pt `
  --calibrate-threshold-on-val `
  --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 `
  --max-val-far 0.05 `
  --min-val-abnormal-recall 0.90 `
  --override-scope all
```
