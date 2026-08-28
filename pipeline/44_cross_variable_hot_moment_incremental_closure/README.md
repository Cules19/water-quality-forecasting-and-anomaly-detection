# Cross-variable hot-moment incremental closure

This stage aligns frozen H00 and historical-target predictions on identical site/timestamp keys, reconstructs label-free residual scores, joins registered events, and evaluates enrichment, lead time, lagged enrichment, alert burden, residual agreement, and compression.

## Required restored inputs

- `analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/`
- `analysis_outputs/ph_common1000_lstm_film_final_rain_ablation_teacher_aligned_v1/full/`
- `analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/`
- `analysis_outputs/ph_common1000_error_onset_do_converted_v1/`
- `analysis_outputs/ph_common1000_error_onset_final_lstm_rain_ablation_v1/`

## Run

```powershell
python pipeline/44_cross_variable_hot_moment_incremental_closure/run_incremental_closure.py
```

The script reuses frozen predictions; it does not retrain the models. On success it writes `analysis_outputs/cross_variable_hot_moment_incremental_closure_v1/` with matched keys, predictions, residual scores, event tables, bootstrap results, figures, reports, completion status, and checksums. On failure it writes `BLOCKED_OR_STOPPED_REPORT.md` and a blocked completion record.

The retained conclusion is partial support, not a universal positive hot-moment claim.
