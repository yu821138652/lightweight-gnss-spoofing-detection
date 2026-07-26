# Cross-Band Hierarchical Attack Event Experiment

This is the first implementation step for the project direction in which the
deployment decision is whether a device is in an attack event, while signal
features remain available for interpretation.

## Scope

`36_build_device_attack_event_tensors.py` reads an existing static signal
tensor fold and creates one sample per `device x endpoint time`.  Its event
label is the reviewed Session attack interval independent of target frequency.
Consequently, an L5-only event is positive for a device even when its L1
signals are suppressed rather than directly spoofed.

The builder does not change `configs/preprocessing.yml`, the central CSV, or
the existing direct target-band signal labels.  It aggregates existing W5
signal statistics into L1/L5 C/N0, AGC, uncertainty, coverage, and explicit
L5-minus-L1 / L5-up-plus-L1-down features.  The aggregation scaler is fit on
train windows only.

`37_train_device_attack_event.py` provides linear and small MLP baselines.  It
selects a checkpoint on the inner validation split and requires an explicit
`--test-only` command before reading an outer test split.

## Fold-6 Diagnostic

The following uses the original fold with an inner validation split.  It must
write to a new output directory and must not overwrite historical signal
checkpoints.

```powershell
$PY = "C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"

& $PY pipeline_total\36_build_device_attack_event_tensors.py `
  --signal-data-dir output\tensors\static_timeblock_outer_v2\fold_6 `
  --output-dir output\hierarchical_event_v1\fold_6\device_tensors

& $PY pipeline_total\37_train_device_attack_event.py `
  --data-dir output\hierarchical_event_v1\fold_6\device_tensors `
  --output-dir output\hierarchical_event_v1\fold_6\linear `
  --model linear `
  --epochs 50 `
  --patience 8
```

After checkpoint selection only, the already-used Fold 6 test can be read as
an iterative diagnostic:

```powershell
& $PY pipeline_total\37_train_device_attack_event.py `
  --data-dir output\hierarchical_event_v1\fold_6\device_tensors `
  --output-dir output\hierarchical_event_v1\fold_6\linear `
  --checkpoint output\hierarchical_event_v1\fold_6\linear\best_device_event_linear.pt `
  --test-only
```

## Boundary

The initial event model can use L1 suppression as device-level attack evidence.
It does not establish that an individual L1 signal was directly spoofed.  A
later signal-level extension must keep direct-spoof labels and attack-associated
anomaly evidence separate.
