# Stage 00: common-predictor ~1000-site cohort

This stage prepares a larger reduced-predictor cohort without changing the
historical strict 200-site experiment.

The proposed common schema contains processed pH (`00400`) as the target and
four exogenous predictors: water temperature (`00010`), specific conductance
(`00095`), dissolved oxygen (`00300`), and turbidity (`63680`). Historical pH
is not an input feature.

## Step 1: site manifest

Run:

```powershell
python pipeline/00_common1000/01_build_site_manifest.py
```

The script uses `data/work/analysis_work/inventory.json`, selects only latest
clean snapshots, deduplicates physical CSV/Parquet copies, and chooses one
representative logical dataset per site. It writes generated artefacts to:

```text
analysis_outputs/ph_common1000_step1_v1/
```

Key outputs:

- `candidate_sites_1200.csv`: all sites with the complete common schema, with
  separate 14-day and 30-day cohort flags.
- `excluded_sites.csv`: sites missing one or more required variables.
- `variable_availability.csv`: one auditable row for every latest clean site.
- `step1_summary.json`: reconciled counts and network summaries.
- `STEP1_REPORT.md`: concise human-readable interpretation.

All `selected_file` paths in the manifests are project-relative and begin with
`data/`, so downstream scripts can resolve them from the repository root.

This is a schema/time-span audit only. Observation-level missingness, units,
physical ranges, duplicates, sampling regularity, and imputation requirements
must be audited before bulk preprocessing or modelling.

## Full processing entry point

After the step-1 manifest has been reviewed, run the complete reduced-predictor
processing pipeline with:

```powershell
python pipeline/00_common1000/run_common1000_processing.py `
  --config config/common1000_processing_v1.json `
  --resume
```

The command performs a four-network smoke test and then processes every
candidate site independently. It writes a native-unit cleaned layer, a
chronologically split leak-safe model-ready layer, per-site scaler metadata,
cohort manifests, hashes, quality statistics, event-match audits, logs, and
restartable checkpoints under:

```text
analysis_outputs/ph_common1000_processed_v1/
```

The existing `data/processed` directory is never modified. Predictor gaps are
linearly filled only when internally bracketed by observations no more than six
hours apart. Target pH is never imputed in model-ready rows. Conductance
standardisation is fitted on training rows only, and regulatory event fields
are evaluation-only columns that are forbidden from the declared feature list.
