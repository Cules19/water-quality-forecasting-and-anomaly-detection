# Common1000 dynamic rainfall sidecars

This stage converts the audited canonical precipitation CSV for each of the 910
primary sites into a model-time-aligned hourly sidecar. It is intentionally
non-destructive: raw precipitation CSVs, water-quality Parquet files, model
inputs, checkpoints, and predictions are never modified.

Each sidecar contains:

- `timestamp`: the existing pH model hourly timestamp;
- `site_id`: site identifier;
- `00045`: observed precipitation in mm/hour;
- `00045_observed_hour`: whether rainfall was observed at that hour;
- `00045_missing`: 1 when unavailable and 0 when observed.

Observed zero rainfall remains `0.0`. Missing rainfall remains null and is never
silently replaced by zero.

Canonical precipitation files are taken from the completed audit manifest,
which selects the latest export using the timestamp embedded in the filename.
Filesystem modification time is not used.

Outputs are written under:

`analysis_outputs/ph_common1000_dynamic_rainfall_v1`

The output contains per-site Parquet sidecars plus mapping, alignment-QA, source
summary, network summary, split summary, and JSON summary reports.
