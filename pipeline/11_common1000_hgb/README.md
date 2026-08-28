# Common1000 HGB baseline

This stage trains two leak-safe global HGB models on the strict, preferred,
unique-physical-site cohort produced by stage 00.

- `M0`: four scaled sensor predictors plus seven time variables.
- `M1`: M0 plus safe static registry/manifest metadata.
- Training: deterministic time-stratified sample, at most 10,000 rows per site.
- Validation/test: every row, with metrics in native pH units.
- Event fields: evaluation only; never predictors.
- Registry discharge time-series summaries are excluded because they can span
  validation/test periods.

Run the 50-site network-stratified smoke test first:

```powershell
python pipeline/11_common1000_hgb/run_common1000_hgb.py `
  --config config/common1000_hgb_v1.json `
  --mode smoke `
  --resume
```

Then run all 1,009 strict unique physical sites:

```powershell
python pipeline/11_common1000_hgb/run_common1000_hgb.py `
  --config config/common1000_hgb_v1.json `
  --mode full `
  --resume
```
