# Error onset and extreme-compression analysis

This stage turns the supervisor's four residual questions into a leakage-safe,
reproducible analysis:

1. residual-alert lead time before registered pH event windows, using the
   nearest sustained alert episode (three consecutive hours by default) as the
   primary metric and a single-hour alert as a sensitivity analysis;
2. event enrichment after shifting event windows 0--24 hours earlier;
3. attribution of large residuals to rainfall, turbidity and flow anomalies on
   the locally observed common subset;
4. extreme-compression slopes, including the requested site `01645762`.

The residual threshold is fitted once on normal rows from held-out validation
sites. Test events are never used to choose it. Event-register rows remain weak
labels, so the outputs describe enrichment and timing, not classification
accuracy.

```powershell
python pipeline/24_error_onset/run_error_onset_analysis.py `
  --config config/common1000_error_onset_v1.json
```

For the corrected-DO rainfall ablation, first build the test-site attribution
sidecars and then run the four diagnostics:

```powershell
python pipeline/24_error_onset/build_attribution_drivers.py `
  --config config/common1000_error_onset_do_converted_v1.json
python pipeline/24_error_onset/run_error_onset_analysis.py `
  --config config/common1000_error_onset_do_converted_v1.json
```
