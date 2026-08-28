# Matched recurrent architecture comparison

This stage audits the four-model GRU/LSTM x baseline/rain experiment on the
135 held-out test sites. It reports overall metrics and paired per-site RMSE
differences. Confidence intervals use a site-block bootstrap so long-record
sites do not determine uncertainty by contributing more hourly rows.

```powershell
python pipeline/32_recurrent_comparison/evaluate_recurrent_comparison.py `
  --experiment-root `
  analysis_outputs/ph_common1000_recurrent_comparison_teacher_aligned_do_converted_v1 `
  --bootstrap-replicates 20000 --seed 42
```

Negative paired RMSE differences favour the challenger. Checkpoint selection
remains validation-only; the test set is used once for the final matched
comparison.
