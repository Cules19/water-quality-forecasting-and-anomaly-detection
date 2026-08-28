# Teacher Archive precipitation features for all sites

This stage applies the precipitation feature definition in
`data/Archive/data_and_metadata_snippets.py::add_precip_channels` to every
unique station in the downloaded precipitation archive, not only to the
Common1000 pH cohort.

For each station, the complete canonical hourly rainfall record is used:

```python
r = precipitation.fillna(0.0)
precip_api = r.ewm(halflife=48.0, adjust=False).mean()
precip_6h = r.rolling(6, min_periods=1).sum()
```

The output preserves both the Archive input name (`precip_mm`) and the model
parameter-code alias (`00045`). Explicit observation and source-missing flags
are retained even though the current 1,750 canonical records have no internal
missing hours.

Repeated exports are resolved to one file per station using the latest export
timestamp embedded in the filename. This is more reliable than filesystem
modification time after DVC checkout.

## Commands

Smoke test:

```powershell
.\.venv\Scripts\python.exe pipeline\28_teacher_archive_precip_all_sites\build_all_site_precip_features.py `
  --config config\all_precip_sites_teacher_archive_v1.json --smoke-per-source 3
```

Independent validation:

```powershell
.\.venv\Scripts\python.exe pipeline\28_teacher_archive_precip_all_sites\validate_all_site_precip_features.py `
  --config config\all_precip_sites_teacher_archive_v1.json `
  --run-root analysis_outputs\all_precip_sites_teacher_archive_v1\smoke_3_per_source
```

Omit `--smoke-per-source` and validate the `full` run root for all 1,750
stations. Source CSV files remain read-only and no model is trained.
