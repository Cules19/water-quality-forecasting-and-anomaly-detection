# Common1000 fair baseline chain

This stage adapts the curated Full50 baseline ladder from GitHub commit
`9d88648` to the exact Common1000 hourly train sample and validation/test
endpoints already used by the GRU/LSTM models.

It evaluates site mean, site-month-hour seasonal mean, one-hour pH
persistence, Ridge, and HistGradientBoosting. Ridge and HGB are each run with
and without the existing safe static metadata, then rerun after excluding
weak-label event target rows from training. Validation and test rows never
change.

Persistence uses historical pH and is reported only as a forecasting
reference. All other predictors exclude pH and event fields.

```powershell
python pipeline/13_common1000_baseline/run_common1000_baselines.py `
  --config config/common1000_baseline_v1.json `
  --mode smoke

python pipeline/13_common1000_baseline/run_common1000_baselines.py `
  --config config/common1000_baseline_v1.json `
  --mode full
```

Use `--resume` only to return immediately from an already validated run.
Outputs are written below `analysis_outputs/ph_common1000_baseline_v1/`.

Run the independent post-run audit with:

```powershell
python pipeline/13_common1000_baseline/audit_common1000_baseline.py `
  --config config/common1000_baseline_v1.json `
  --mode full
```
