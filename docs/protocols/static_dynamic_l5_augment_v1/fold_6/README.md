# E12a: dynamic L5 augmentation for static Fold 6

This protocol is a targeted diagnostic, not a mixed static/dynamic benchmark
and not a blind test.  The outer test is the repeatedly inspected playground
static `st_L5/2025.07.30.09.48_2025.07.30.10.14` Session.

The static outer-development Sessions are listed in
`static_outer_manifest.csv`.  All ten reviewed dynamic `dy_L5` and `dy_L_15`
Sessions in `dynamic_train_manifest.csv` are train-only augmentation.  Dynamic
`dy_L1` Sessions are deliberately excluded so L1 endpoints do not dominate an
L5-target detection diagnostic.

For model selection, generate inner folds from the static manifest and hold
out one complete *static* development Session at a time.  The selected fixed
epoch count is then used to refit on all six static development Sessions plus
the ten dynamic train-only Sessions.  The outer static test stays untouched
during that selection step.

The model reports L5-only metrics and per-device L5 metrics.  It must improve
the Pixel 6 and RedMi K60 recall while reducing the Mate40 FAR before it is
considered a positive result.

## Reproduction sequence

Generate the inner static Session manifests once:

```powershell
& $PY pipeline_total\32_generate_static_inner_session_manifests.py `
  --outer-manifest docs\protocols\static_dynamic_l5_augment_v1\fold_6\static_outer_manifest.csv `
  --block-manifest output\tensors\static_timeblock_outer_v2_e9a_refit_all_dev\fold_6\block_manifest.csv `
  --output-dir output\protocols\static_dynamic_l5_augment_v1\fold_6\inner_static `
  --write-refit-all-development
```

For each L5-positive validation fold (`inner_02`, `inner_03`, `inner_04`,
`inner_06`), build tensors with `36_build_static_dynamic_l5_augmentation_tensors.py`
using that fold's `block_manifest.csv`, then run
`37_train_static_dynamic_l5_augmentation.py` without `--refit`.  The two
`st_L1` folds (`inner_01`, `inner_05`) have no L5 positives; they are useful
for checking L5-FAR but must not select the epoch by L5 Macro-F1.

Use a predeclared aggregation such as the median selected epoch from the four
L5-positive folds.  Then build tensors with
`inner_static/refit_all_development/block_manifest.csv`, run the E12a trainer
with `--refit --epochs <selected_epoch>`, and only then invoke `--test-only`
with the resulting checkpoint.
