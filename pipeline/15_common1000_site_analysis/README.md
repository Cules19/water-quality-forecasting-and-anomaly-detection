# Common1000 site hydrology and model interpretation

This stage creates the supervisor-requested post-hoc site table and figures at
the current 1,009-site scale. It derives flow descriptors only from native
`00060` observations in the training period, joins per-site model RMSE and
global-top-5% residual/event enrichment, runs Spearman analyses, and creates
four contrasting-site predictor/parity figures.

```powershell
python pipeline/15_common1000_site_analysis/run_common1000_site_analysis.py `
  --config config/common1000_site_analysis_v1.json `
  --resume
```

Event fields and model outcomes are used only after training. They never enter
any predictor matrix.
