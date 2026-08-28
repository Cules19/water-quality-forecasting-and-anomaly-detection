# Generated files and storage policy

## Why outputs are not committed

The audited local workspace contains roughly 89 GB of generated analysis files and more than 110 GB under `data/`. Committing these files would make the Git repository difficult to clone and would risk redistributing restricted source material. The repository tracks the code and configurations needed to interpret them; the USB archive holds the evidence separately.

## Main generated locations

| Location | Produced content | Keep outside GitHub? |
|---|---|---|
| `data/clean/` | Harmonised source observations by site | Yes; essential private archive |
| `data/external/upstream1_primary/` | Hydrography90m/Environment90m archives and indices | Yes; large licensed source data |
| `analysis_outputs/ph_common1000_processed_v1/` | Canonical processed cohort and manifests | Yes; essential derived data |
| `analysis_outputs/ph_common1000_*` | Baselines, recurrent models, predictions, reports | Yes |
| `analysis_outputs/ph_h00_30day_calibration_validation_v1/` | Offset parameters, frozen predictions, summaries | Yes; essential final evidence |
| `analysis_outputs/cross_variable_hot_moment_incremental_closure_v1/` | Matched residual/event analysis | Yes; essential final evidence |
| `analysis_outputs/ph_h00_reproducibility_audit_v1/` | Hashes, reproductions, histories, audit report | Yes; essential final evidence |
| `analysis_outputs/ph_h00_solar_dynamic1_v1/` | 910-site derived solar sidecars and probe results | Yes; preserve as bounded negative result |
| `analysis_outputs/ph_h00_upstream1_v1/` | Spatial matches, UPSTREAM13 table, QA and decision | Yes; preserve as data product/negative gate |
| `figures/methodology/` | Small publication-ready methodology figures | No; these are intentionally tracked |

## Typical file types

- `.parquet`: hourly features, predictions, residual scores, or per-site sidecars.
- `.pt`: model checkpoints; large and regenerated only with substantial compute.
- `.csv`/`.json`: audit tables, manifests, hashes, decisions, and metrics.
- `.gpkg`/`.tif`: geospatial vectors and rasters.
- `.png`/`.svg`/`.pdf`: figures and reports.

## Safe rerun pattern

Run commands from the repository root. Restore the required private inputs first, use a new versioned output directory, and never overwrite a frozen result unless the stage explicitly supports resumption. Smoke-test directories and temporary caches are disposable only after their final summaries and hashes have been preserved.

The USB manifests record byte sizes and hashes for the selected archive. Local cleanup was completed only after source/destination verification; the retained archive state is recorded in `manifests/BACKUP_COMPLETE.md`.
