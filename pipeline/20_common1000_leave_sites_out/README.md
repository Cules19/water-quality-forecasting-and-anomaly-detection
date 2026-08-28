# Common1000 leave-sites-out preparation

This stage creates site-disjoint pH/DO cohorts directly from the native clean
Parquet layer. It deliberately does not reuse per-site-scaled Common1000 arrays.

```powershell
python pipeline/20_common1000_leave_sites_out/prepare_leave_sites_out.py `
  --config config/common1000_leave_sites_out_v1.json `
  --mode smoke
```

After the smoke output passes, run `--mode full --resume`. Persistence is defined
as the same site's observed target at exactly `t - horizon`; it is reported as a
history-using reference, not as an exogenous baseline.
