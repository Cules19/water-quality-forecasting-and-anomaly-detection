# H00 30-day calibration

This stage compares the frozen raw H00 model with a 30-day mean-offset calibration and an affine alternative. Selection is validation-only; the held-out test is opened only according to the frozen gate implemented by the script.

## Required restored inputs

- `analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/`
- `analysis_outputs/cross_variable_hot_moment_incremental_closure_v1/`
- The frozen H00 checkpoint, effective configuration, matched predictions, residual scores, event manifest, and label-free thresholds inside those directories

## Run

```powershell
python pipeline/45_ph_h00_30day_calibration/run_h00_30day_calibration.py
```

Do not rerun the held-out test to tune the method. The script writes `analysis_outputs/ph_h00_30day_calibration_validation_v1/`, including validation comparisons, calibration parameters, selected-method records, test outputs when the frozen gate permits them, figures, reports, completion status, and SHA-256 checksums.

The retained method is the simple first-30-day mean residual offset. It improves accuracy without changing the frozen hot-moment ranking.
