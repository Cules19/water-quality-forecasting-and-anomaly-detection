# Cross-device large-site pH modelling plan

## 1. Purpose of this document

This document is a complete handover instruction for Codex on another computer. The objective is to reproduce the present computer's pH hot-moment workflow from data audit to final interpretation, and then apply it to a larger number of sites without changing the scientific definition of the experiment.

The main study is:

> **Multi-site prediction-based detection of pH hot moments using high-frequency water-quality sensor data and hydrological metadata.**

The workflow predicts processed pH (`00400`) from exogenous sensor variables, calculates prediction residuals, and evaluates whether low- and high-pH weak-label event windows are enriched in the appropriate signed residual tails. Static hydrological metadata are used to describe differences between sites. Historical pH is deliberately excluded from the primary models.

This is an execution plan, not only a methodological outline. Codex on the other device should inspect the transferred project, implement any parameterisation needed, run each phase in order, verify every checkpoint, and produce the required files. It must not silently alter the cohort, predictors, metadata, event definition, split logic, or model architecture merely to make a larger run succeed.

---

## 2. Non-negotiable scientific rules

1. Use `data/processed` as the primary time-series source for modelling.
2. The target is processed/standardised pH column `00400`.
3. The dynamic predictors are exactly:
   - `00010`: water temperature
   - `00095`: specific conductance
   - `63680`: turbidity
   - `00060`: discharge
   - `00065`: gage height
   - `sin_hour`, `cos_hour`, `sin_day`, `cos_day`, `month`, `hour`, `day_of_year`
4. Do not use current or historical pH as an input to the main HGB, GRU, or LSTM models.
5. Define residual as:

   ```text
   residual = observed processed pH - predicted processed pH
   ```

6. Evaluate low-pH events in the **negative** residual tail and high-pH events in the **positive** residual tail. Also retain absolute-residual results for comparison.
7. Regulatory event windows are incomplete weak labels. They are used for evaluation and for controlled non-event training exclusions only. They are never predictors and must not be described as complete pollution-event ground truth.
8. Static metadata, derived event variables, residuals, predictions, validation/test summaries, and model-performance metrics must never leak into dynamic predictors.
9. Any train-derived metadata and all scalers/encoders must be fitted using training rows only.
10. Use a chronological `70%/15%/15%` train/validation/test split within each site after timestamp sorting and duplicate removal.
11. The same-site chronological split tests forecasting at already monitored sites. It does not establish generalisation to unseen sites. Leave-site-out testing is a separate experiment.
12. Do not interpret processed `00400` values as native regulatory pH units unless the exact inverse transformation has been verified from the saved preprocessing object.
13. Keep model-prediction performance and event-detection performance as separate result families.
14. Never select sites using event frequency, test performance, residual behaviour, or the desired result direction.
15. Never overwrite or delete prior `analysis_outputs`. Every run gets a unique versioned output directory.

---

## 3. Cohort strategy: how to process more sites without changing the experiment

The present processed inventory contains approximately 336 site IDs, all matchable to `sites_register.json`, but only approximately 201 sites have the exact common schema containing:

```text
site_id, timestamp, 00400, 00010, 00095, 63680, 00060, 00065
```

These counts are reference values from the present computer, not constants to force on the other computer. Recalculate them from the transferred data and explain any difference.

Use two clearly separated cohorts:

### 3.1 Primary strict cohort

- Use sites that genuinely contain the full common target and predictor schema.
- The primary large experiment should use 200 sites because this preserves exact comparability with the existing HGB200 work and leaves one site as an audit margin in the current inventory.
- Also define deterministic nested site sets of 50, 100, 150, and 200 sites for the scaling curve.
- The 50-site set must be a subset of the 100-site set; the 100-site set a subset of 150; and the 150-site set a subset of 200.
- If the new processed inventory contains more than 200 fully eligible sites, add `N250` and `N300` only when those sites meet exactly the same rules.

### 3.2 Extended cohort above the strict limit

Do not label a 300-site experiment as the same full-predictor experiment if only about 201 sites have the required schema.

For sites outside the strict cohort:

1. Inspect parquet schemas, preprocessing JSON, parameter mappings, and small samples.
2. Build an explicit canonical alias table for each required variable.
3. Accept an alias only when its physical meaning, unit/preprocessing definition, and temporal field are unambiguous.
4. Save the mapping as `extended_parameter_alias_map.csv` with evidence and source file.
5. Reject a site when a required signal is absent, rather than replacing an entire missing channel with a constant.
6. If a reduced-predictor cohort is necessary, treat it as a separate experiment with a different name and separate tables. Do not place it on the strict scaling curve.

The report must therefore distinguish:

- **Strict common-predictor scaling**: directly comparable.
- **Alias-normalised extension**: comparable only after a passed equivalence audit.
- **Reduced-predictor extension**: exploratory and not directly comparable.

---

## 4. Files to transfer to the other computer

Transfer the project while preserving relative paths. At minimum, transfer:

```text
Pollution_ID/
  data/
    processed/
    reference/
      sites_register.json
      regulatory_events.csv
  analysis_outputs/
    ph_processed_metadata_availability_check_v1/
    ph_hgb_processed200_metadata_v1/
    ph_processed100_step4_0_4_1_v1/
    ph_sequence100_gru_lstm_metadata_v1/
    ph_v2_metadata_rerun_v1/
    ph_signed_residual_tail_analysis_v1/
    ph_site_hydrology_performance_analysis_v1/
    ph_current_machine_supervisor_tasks_v1/
```

The current scripts may refer to `~/Downloads/regulatory_events.csv`. Before transfer, place the event file at:

```text
data/reference/regulatory_events.csv
```

New scripts must use a configurable project-relative event path. They must not depend on a particular user's Downloads directory.

Optional transfer:

- Transfer `data/clean` only if the native-unit discharge/hydrological post-hoc analysis will be rerun.
- Do not use `data/clean` as a substitute for the processed modelling data.
- Transfer existing model artefacts if storage allows. They are useful for reconstruction and parity checks, but the new large run must write new outputs.

Generate SHA-256 hashes for the registry, event file, configuration, site manifest, and each source script used. Store them in `00_environment/input_hashes.csv`.

---

## 5. Environment and hardware preflight

### 5.1 Recommended resources

- Windows or Linux with Python 3.11 or 3.12.
- At least 32 GB RAM; 64 GB is preferred for 200-site sequence work.
- At least 80 GB free working space; 100 GB or more is preferred.
- NVIDIA GPU with at least 12 GB VRAM is recommended for GRU/LSTM training.
- CPU-only execution is acceptable for audits and HGB, but full sequence experiments will be much slower.

The present computer used Python 3.13.5 with NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, PyTorch 2.12.1+cpu, PyArrow 24.0.0, SciPy 1.17.1, Matplotlib 3.10.9, and Joblib 1.5.3. Exact replication of these unusually recent versions is not required. Compatibility and recorded versions are required.

### 5.2 Create an isolated environment

Example PowerShell commands:

```powershell
cd "C:\path\to\Pollution_ID"
py -3.12 -m venv .venv_largeN
.\.venv_largeN\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow scikit-learn scipy matplotlib seaborn joblib tqdm psutil pyyaml
```

Install the appropriate PyTorch build from the official PyTorch instructions for that device's CUDA version. Then verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import pandas, pyarrow, sklearn, scipy; print('imports OK')"
python -m pip freeze > analysis_outputs\ph_cross_device_largeN_v1\00_environment\pip_freeze.txt
```

Save CPU count, RAM, GPU model, VRAM, operating system, Python version, package versions, disk free space, and execution start time in `environment_report.md`.

### 5.3 Resource stop conditions

Before a heavy phase, stop and report rather than starting if:

- free disk space is below 40 GB;
- the event file or registry is missing;
- source parquet files cannot be opened consistently;
- fewer sites than expected pass the declared eligibility rule;
- GPU out-of-memory persists after one deterministic batch-size reduction;
- a data-leakage assertion fails;
- a test-set key comparison fails between controlled variants.

---

## 6. New output structure

Create a new root and do not write large new files into old result folders:

```text
analysis_outputs/ph_cross_device_largeN_v1/
  00_environment/
  01_inventory/
  02_cohort/
  03_processedN/
  04_hgb/
  05_sequence_data/
  06_sequence_models/
  07_nonevent/
  08_signed_residual/
  09_leave_site_out/
  10_hydrology/
  11_figures/
  12_final_report/
  checkpoints/
  logs/
  run_config.json
  run_manifest.json
```

Each phase must create:

- a timestamped log;
- a copy of the effective configuration;
- input and output row counts;
- warnings and rejected-site reasons;
- a completion marker named `_SUCCESS` only after validation passes.

Large intermediate datasets should be partitioned by `site_id` or stored in manageable parquet shards with compression. Avoid a single giant in-memory DataFrame.

---

## 7. Freeze the run configuration before heavy processing

Create `run_config.json` with at least the following content. Paths may be changed, but scientific fields may not be silently changed:

```json
{
  "run_id": "ph_cross_device_largeN_v1",
  "random_seed": 42,
  "target": "00400",
  "base_dynamic_features": ["00010", "00095", "63680", "00060", "00065"],
  "time_features": ["sin_hour", "cos_hour", "sin_day", "cos_day", "month", "hour", "day_of_year"],
  "required_raw_columns": ["site_id", "timestamp", "00400", "00010", "00095", "63680", "00060", "00065"],
  "split_fractions": [0.70, 0.15, 0.15],
  "strict_site_counts": [50, 100, 150, 200],
  "sequence_length_hours": 24,
  "sequence_step_hours": 1,
  "residual_tail_fractions": [0.01, 0.05, 0.10],
  "processed_dir": "data/processed",
  "registry_path": "data/reference/sites_register.json",
  "events_path": "data/reference/regulatory_events.csv",
  "use_ph_history": false,
  "event_columns_are_predictors": false,
  "seed_site_selection_from_events": false
}
```

Do not edit the configuration after model training begins. If a change is necessary, create `v2` rather than altering the evidence trail.

---

## 8. Phase 1: processed-data inventory and metadata audit

This phase is lightweight. Inspect filenames, parquet footers/schemas, JSON keys, and small samples before reading full datasets.

### Tasks

1. Inventory every file under `data/processed` with path, extension, size, and inferred site ID.
2. Read parquet schemas without loading full tables where possible.
3. Record column names, dtypes, row counts from parquet metadata, and schema variants.
4. Normalise site IDs to zero-padded eight-character strings for matching only; retain original values in the audit.
5. Inspect `sites_register.json` keys and flatten only metadata fields needed later.
6. Match all processed sites to the registry.
7. Identify strict sites containing the full required schema.
8. Inspect a small head/tail sample from representative files in every schema class.
9. Detect duplicate files, obviously empty files, unreadable files, and conflicting site IDs.

### Required outputs

```text
01_inventory/processed_file_inventory.csv
01_inventory/processed_schema_summary.csv
01_inventory/processed_site_schema_matrix.csv
01_inventory/processed_static_metadata_presence.csv
01_inventory/registry_metadata_field_summary.csv
01_inventory/processed_to_registry_site_match.csv
01_inventory/inventory_report.md
```

### Validation checkpoint

The present-device reference is approximately 1,066 processed files, 365 parquet files, 336 processed site IDs, 201 strict full-schema sites, and 336/336 registry matches. Differences are allowed only when explained by different transferred inputs.

Do not proceed until the strict-site count and registry match rate are known.

---

## 9. Phase 2: deterministic cohort construction

### 9.1 Row-level eligibility audit

For strict-schema sites, stream only the required columns and calculate per-site eligibility summaries:

- source row count;
- valid row count after cleaning;
- first and last timestamp;
- coverage duration;
- duplicate `site_id + timestamp` count;
- non-finite and sentinel count by variable;
- proportion with complete target and five core exogenous variables;
- median sampling interval and interval distribution;
- registry-match status.

Cleaning rules used by the current workflow:

- convert timestamps consistently and reject invalid timestamps;
- values `<= -999` are invalid sentinels;
- pH values outside `0-14` are invalid before using the processed target;
- replace infinities with missing;
- sort by site and timestamp;
- remove duplicate site-timestamp records using a documented deterministic rule;
- retain rows needed by the declared model only after the audit counts are saved.

### 9.2 Deterministic site ranking

Rank eligible sites without reference to event labels or model outcomes. Use, in order:

1. full required schema;
2. sufficient complete valid observations;
3. sufficient temporal coverage;
4. stable dominant interval/usable hourly coverage;
5. `site_id` as the final deterministic tie-breaker.

Freeze the ordered list before any model is trained. Save:

```text
02_cohort/site_eligibility_audit.csv
02_cohort/site_exclusion_reasons.csv
02_cohort/strict_site_ranking.csv
02_cohort/site_set_N50.csv
02_cohort/site_set_N100.csv
02_cohort/site_set_N150.csv
02_cohort/site_set_N200.csv
02_cohort/cohort_manifest.json
02_cohort/cohort_report.md
```

If more fully eligible sites exist, create additional nested manifests. If alias normalisation is attempted, save the alias-normalised cohort separately and never replace the strict files.

---

## 10. Phase 3: build the model-ready processed-N data

Use the current HGB200 builder as the behavioural reference:

```text
pipeline/02_hgb200/run_processed200_hgb_metadata_experiment.py
```

Do not edit the old script in place. Create a parameterised new builder that accepts the run configuration, site manifest, and output directory.

### 10.1 Time-series preparation

For each selected site:

1. Read only required columns.
2. Apply the cleaning rules above.
3. generate the 12 dynamic features;
4. sort and de-duplicate;
5. assign chronological train/validation/test rows within site;
6. preserve a stable row key consisting of site ID and timestamp;
7. write partitioned parquet by site or split.

The split boundary must be determined from rows available for the same declared modelling task. No random row split is allowed.

### 10.2 Event-state coding

Match regulatory event intervals to rows using site ID and timestamp. Create at least:

```text
in_ph_event
ph_event_label
ph_event_direction
event_id
```

Use explicit low/high direction rules from the event source. Preserve unmatched and ambiguous labels in the audit. Event fields remain evaluation/training-control columns and are excluded from all feature lists.

### 10.3 Static metadata source

Static site metadata come from:

```text
data/reference/sites_register.json
```

Processed time-series files generally do not contain the hydrological static metadata needed by the experiment. Join metadata once per site and audit match and missingness before broadcasting it to model rows.

### 10.4 Compact hydrological V2 metadata

Use a frozen compact V2 set, not the earlier 73-feature full metadata set, for the main larger-site rerun.

Candidate official registry fields:

```text
latitude
longitude
elevation
drainage_area
official_average_discharge
catchment_precipitation
evapotranspiration
population
number_of_dams
dam_capacity
irrigation_area
irrigation_percentage
dominant_land_cover_code
dominant_land_cover_percentage
site_type_code
basin_code
```

Candidate train-only summaries, calculated only from each site's training rows:

```text
train_median_00010, train_std_00010
train_median_00095, train_std_00095
train_median_63680, train_std_63680
train_median_00060, train_std_00060
train_median_00065, train_std_00065
train_coverage_days
train_dominant_interval
```

This gives up to 28 compact V2 features. Resolve registry aliases explicitly, measure coverage, and freeze the final list in `v2_metadata_feature_definition.json`. Do not add fields merely because they are available.

Exclude by design:

- pH summaries;
- event counts or rates;
- residual or prediction summaries;
- validation/test/future summaries;
- model metrics;
- administrative source/status/unit strings;
- broad missingness-indicator families;
- redundant train mean, percentile, IQR, and missing-rate families used in the older 73-feature version.

Fit numeric imputation values on training sites/rows as applicable and save them. Fit categorical code mappings without looking at model outcomes; reserve an `UNKNOWN` category.

### Required outputs

```text
03_processedN/model_ready_by_site/*.parquet
03_processedN/site_metadata_v2.parquet
03_processedN/v2_metadata_feature_definition.json
03_processedN/metadata_imputation_values.json
03_processedN/event_match_audit.csv
03_processedN/split_counts_by_site.csv
03_processedN/data_quality_summary.csv
03_processedN/dataset_manifest.json
```

### Leakage assertions

The builder must fail if any predictor name contains or equals pH target/history, event, residual, prediction, validation/test outcome, or model metric fields. It must also verify that every train-derived metadata statistic used only training timestamps.

---

## 11. Phase 4: HGB benchmark on the strict site sets

Run HistGradientBoostingRegressor as the scalable non-sequential benchmark.

### Controlled models

- `HGB_N_no_metadata`: 12 dynamic features plus deterministic site code.
- `HGB_N_V2`: the same inputs plus compact V2 metadata.

Use identical rows, splits, target, seed, and HGB parameters between the two variants:

```json
{
  "loss": "squared_error",
  "learning_rate": 0.05,
  "max_iter": 200,
  "max_leaf_nodes": 31,
  "min_samples_leaf": 20,
  "l2_regularization": 0.0,
  "random_state": 42,
  "early_stopping": false
}
```

Run at least N=50, 100, 150, and 200 with the frozen nested cohorts. Use all valid training rows. Stream into preallocated arrays or memory-mapped arrays; scikit-learn HGB internally validates float64, so budget disk and RAM accordingly.

### Required metrics

- overall and per-site RMSE, MAE, and R-squared;
- absolute residual-tail event enrichment at 1%, 5%, and 10%;
- negative-tail enrichment for low events;
- positive-tail enrichment for high events;
- event-window hit rate;
- event row/window counts and coverage;
- runtime and peak memory.

### Required outputs per model

```text
model.joblib
feature_list.json
training_config.json
overall_metrics.csv
site_metrics.csv
test_predictions.parquet
residual_event_metrics.csv
resource_usage.json
```

Current HGB200 reference, for same-data parity only:

- without metadata: RMSE about `0.017215`, R-squared about `0.786936`;
- compact metadata: RMSE about `0.016119`, R-squared about `0.813207`.

Do not force the new result to match these values. Use them only to detect gross implementation errors when inputs are unchanged.

---

## 12. Phase 5: build the sequence dataset once and reuse it

Use the current sequence script as behavioural reference:

```text
pipeline/04_sequence_gru/run_sequence100_gru_lstm_metadata.py
```

### 12.1 Hourly aggregation

For each site:

1. floor timestamps to the hour;
2. average numeric target and sensor predictors within site-hour;
3. recompute all cyclical/calendar features from the hourly timestamp;
4. assign the split by the majority of source rows in the hour, while rejecting boundary ambiguity if needed;
5. preserve event status conservatively for evaluation;
6. write partitioned hourly parquet.

Do not average arbitrary precomputed time features across rows.

### 12.2 Valid 24-hour windows

A valid prediction window must contain:

- 24 consecutive hourly rows;
- one site only;
- one split only;
- no timestamp gap;
- complete 12-feature dynamic input;
- target at the final hour;
- inputs from `t-23 h` through `t`;
- no pH history.

Create a compact window index rather than materialising every 24 x feature tensor. The index should include site ID, start row/offset, target timestamp, split, event fields, and stable test key.

### 12.3 Scaling and sampling

- Fit the dynamic `StandardScaler` on training windows only.
- The current processed `00400` target was not scaled again. Preserve that rule unless inverse preprocessing proves a different target definition.
- Fit metadata scaling on training sites only, with non-finite values filled using training-site means.
- Use site-balanced training sampling so the largest sites do not dominate solely by row count.
- Prefer all training windows on the larger device when feasible.
- If a deterministic cap is required, generate and save the exact sampled window indices once, and reuse them for every controlled GRU/LSTM variant.
- Validation and test sets must not be downsampled for final metrics.

### Required outputs

```text
05_sequence_data/hourly_by_site/*.parquet
05_sequence_data/window_index_by_site/*.parquet
05_sequence_data/window_counts_by_site_split.csv
05_sequence_data/dynamic_scaler.joblib
05_sequence_data/metadata_scaler.joblib
05_sequence_data/site_code_mapping.json
05_sequence_data/training_window_sample.parquet
05_sequence_data/sequence_data_manifest.json
```

Current 100-site reference counts were approximately 6,094,947 train windows, 1,282,428 validation windows, and 1,307,486 test windows. Recalculate for the new cohorts.

---

## 13. Phase 6: GRU and LSTM experiments

### 13.1 Primary controlled models

For the largest strict cohort that the device can complete, train:

1. `GRU_no_metadata`
2. `GRU_V2_metadata`
3. `LSTM_V2_metadata`

For the scaling curve, prioritise `GRU_V2_metadata` at N=50, 100, 150, and 200. Run the full three-model comparison at N=200. If resources allow, repeat all variants at each N.

### 13.2 Architecture

Use the same basic architecture as the current 100-site work:

- sequence encoder: GRU or LSTM;
- input size: 12;
- hidden size: 64;
- recurrent layers: 1;
- recurrent dropout: 0;
- site embedding size: `min(16, ceil(sqrt(number_of_sites)) + 1)`;
- concatenate final recurrent hidden state, site embedding, and V2 metadata vector;
- MLP hidden layer: 64 with ReLU;
- scalar processed-pH output.

The no-metadata model omits the metadata vector but keeps the otherwise comparable architecture. Preserve the site embedding for the same-site chronological experiment.

### 13.3 Training controls

Use:

```text
seed = 42
optimizer = Adam
learning_rate = 1e-3
weight_decay = 1e-5
maximum_epochs = 8
early_stopping_patience = 2
selection_metric = validation RMSE or validation loss, fixed before training
```

Use mixed precision on a compatible GPU:

- prefer `bfloat16` when supported;
- otherwise use safe `float16` AMP;
- otherwise use float32.

Start with batch size 4096 on GPU or 2048 on CPU. Run a short memory probe. If GPU out-of-memory occurs, halve the batch size once, record the final batch size, and then freeze it for all controlled model variants. Do not let different models use different sampled windows or different test rows.

Enable checkpointing after each epoch and resume from the latest complete checkpoint after interruption. Save best and final checkpoints separately.

### 13.4 Fair-comparison assertions

Before calculating comparison tables, assert:

- identical validation and test row keys;
- identical target arrays;
- identical split assignments;
- identical dynamic scaler and site mapping where applicable;
- identical training sample for controlled variants;
- no event columns in model inputs.

### Required outputs per model

```text
best_model.pt
final_model.pt
training_history.csv
training_config.json
feature_manifest.json
test_predictions.parquet
overall_metrics.csv
site_metrics.csv
residual_event_metrics.csv
resource_usage.json
```

An optional FiLM conditioning experiment may be added only after the concatenation baseline is complete. Label it as an architectural extension and do not mix it into the primary metadata comparison.

---

## 14. Phase 7: controlled non-event sequence training

Use the best completed metadata sequence model, expected to be `GRU_V2_metadata`, as the baseline architecture.

Train an exact controlled pair:

- `all_data`: all eligible training windows;
- `non_event_target_only`: exclude a training window when its target row has `in_ph_event == 1`.

Everything else must remain identical: cohort, window index, architecture, initialisation seed, optimiser, learning rate, batch size, epoch rule, validation set, test set, and evaluation code.

Required verification:

1. reconstruct baseline predictions on a fixed batch and check numerical agreement;
2. report the number and percentage of excluded training windows by site and event direction;
3. assert that validation and test keys are exactly identical;
4. compare prediction metrics and signed event-tail metrics side by side.

After the target-only experiment is complete, an optional sensitivity run may exclude any training window overlapping an event during all 24 input hours. Label this `non_event_strict_24h`; it is not the primary comparison.

The current 100-site controlled result is a useful qualitative reference: target-only non-event training improved RMSE, R-squared, absolute top-5% enrichment, low-event negative-tail enrichment, and event-window hit rate, while high-event positive-tail enrichment changed little. Do not assume that the larger cohort must reproduce the same direction.

---

## 15. Phase 8: signed residual-tail analysis

Parameterise a new model manifest rather than editing hard-coded old output paths. Use the current script as the calculation reference:

```text
pipeline/09_signed_residual/run_signed_residual_tail_analysis.py
```

For each model and each tail fraction `1%`, `5%`, and `10%`, calculate:

- absolute-residual threshold and enrichment;
- negative-residual threshold and low-event enrichment;
- positive-residual threshold and high-event enrichment;
- expected-sign rate within low/high event rows;
- event-row recall/hit rate;
- event-window hit rate;
- event and non-event denominators;
- site-level metrics where event counts are sufficient.

The tail threshold should be derived from the declared evaluation population using one documented method applied consistently. Do not use event labels to tune the threshold.

Construct event windows by site, event label/ID, and temporal continuity. Keep HGB native-row outputs and hourly sequence outputs separate because their temporal resolution and row counts differ. Do not pool them into one denominator.

Recommended confidence extension: site-stratified bootstrap confidence intervals for the primary top-5% metrics, while preserving the original point-estimate table.

Required outputs:

```text
08_signed_residual/model_manifest.csv
08_signed_residual/overall_signed_tail_metrics.csv
08_signed_residual/site_signed_tail_metrics.csv
08_signed_residual/event_window_hit_metrics.csv
08_signed_residual/direction_consistency.csv
08_signed_residual/bootstrap_intervals.csv
08_signed_residual/signed_residual_report.md
```

---

## 16. Phase 9: leave-site-out generalisation experiment

This phase answers a different question from the chronological same-site split and must have separate result tables.

Use fixed five-fold grouped cross-validation by site. Freeze the fold assignment before training and balance folds approximately by number of sites, usable windows, broad geography/hydrology, and event availability without using model performance.

For unseen-site evaluation:

- remove the site embedding from the primary leave-site-out model;
- use compact V2 metadata as the site descriptor;
- fit metadata imputation/scaling and all dynamic scaling on training folds only;
- map unseen categorical levels to `UNKNOWN`;
- train on four folds, validate within training sites using chronological validation rows, and test only on the held-out sites;
- report prediction and event metrics by fold and pooled by held-out site.

Compare:

1. `LOSO_GRU_no_metadata_no_site_embedding`
2. `LOSO_GRU_V2_no_site_embedding`

Do not compare the numerical level of LOSO performance to same-site performance as though only metadata changed; the generalisation task itself is harder.

---

## 17. Phase 10: hydrological metadata and site-performance analysis

This is a post-hoc explanatory analysis. It does not retrain the models.

If native `data/clean` discharge data are available, calculate physical-flow metrics from native/unscaled `00060`, restricted to each site's training-period dates:

- mean flow: arithmetic mean of training-period daily mean discharge;
- flow CV: sample standard deviation divided by positive mean;
- Richards-Baker flashiness: sum of absolute consecutive daily changes divided by sum of daily flows, using consecutive-day pairs only;
- Q50: median daily discharge;
- Q95 under hydrological exceedance convention: fifth percentile daily discharge;
- Q95/Q50 ratio using that convention.

Do not calculate physical hydrological statistics from processed standardised discharge.

Relate site metadata to:

- site RMSE/MAE/R-squared;
- absolute top-5% event enrichment;
- low-event negative-tail enrichment;
- high-event positive-tail enrichment;
- event-window hit rate;
- all-data versus non-event improvement.

Use Spearman correlations. Calculate p-values only with at least 10 valid sites and apply false-discovery-rate correction across the declared correlation family. Classify event reliability as:

```text
>= 10 event windows: reliable
3-9 event windows: limited
< 3 event windows: insufficient
```

Describe these as associations, not causal effects.

---

## 18. Figures and final reporting

Create publication-quality figures only from frozen result tables. At minimum:

1. site-count scaling curve for RMSE and R-squared;
2. metadata versus no-metadata model comparison;
3. GRU versus LSTM comparison;
4. all-data versus non-event training comparison;
5. absolute, negative, and positive residual-tail enrichment;
6. low/high event direction-consistency figure;
7. site-level performance distribution;
8. hydrological metadata versus reliable site-level event detectability;
9. same-site versus leave-site-out performance, clearly labelled as different tasks.

Use consistent model names, colours, tail definitions, and units. Do not imply native pH units when plotting processed target/residual values.

The final report should contain:

- data inventory and strict/extended site counts;
- exact cohort selection logic;
- preprocessing, split, metadata, and event definitions;
- HGB results;
- GRU/LSTM results;
- non-event training results;
- signed low/high residual-tail results;
- site hydrology/performance associations;
- leave-site-out findings if completed;
- limitations and a clear statement that event windows are weak labels;
- a reproducibility appendix listing configurations, hashes, versions, seeds, and output paths.

---

## 19. Execution order and decision gates

Follow this order:

1. Transfer and hash inputs.
2. Create environment and record versions/hardware.
3. Run schema/registry inventory.
4. Freeze strict and extended cohort manifests.
5. Run a two-site end-to-end smoke test.
6. Run an N=10 or N=50 resource/performance smoke test.
7. Build the strict N=200 processed dataset.
8. Run HGB scaling models.
9. Build hourly data and sequence window indices once.
10. Run GRU scaling models and the N=200 controlled model set.
11. Run controlled non-event training.
12. Run signed residual-tail analysis.
13. Run leave-site-out only after the same-site models pass.
14. Run native hydrological post-hoc analysis if clean data are present.
15. Generate figures and final report.

Do not start all expensive jobs simultaneously. After each gate, inspect counts, missingness, leakage assertions, and a small prediction sample.

### Raw-only new sites

If the additional sites exist only as raw or clean files and not in `data/processed`, stop before adding them to the primary cohort. Locate the original preprocessing scripts and saved transformation objects. Process at least three sites that already exist in both raw and processed form, then compare schema, row keys, distributions, and transformed values. Only after near-exact equivalence is demonstrated may the same preprocessing be applied to new raw sites. If equivalence cannot be demonstrated, keep those sites out of the directly comparable experiment and document the limitation.

---

## 20. Reproducibility and restartability requirements

1. Set Python, NumPy, PyTorch, and CUDA seeds to 42 and record deterministic settings.
2. Record the exact ordered site list, not only the number of sites.
3. Save train/validation/test row or window keys.
4. Save fitted scalers, category maps, metadata imputers, and feature order.
5. Save the exact sampled training-window IDs when sampling is used.
6. Write checkpoints atomically and resume only from complete checkpoints.
7. Use `_SUCCESS` markers and input/config hashes to decide whether a phase may be reused.
8. Never treat a partially written parquet or checkpoint as complete.
9. Capture stdout/stderr to timestamped logs, for example:

   ```powershell
   python -u path\to\script.py --config analysis_outputs\ph_cross_device_largeN_v1\run_config.json 2>&1 |
     Tee-Object analysis_outputs\ph_cross_device_largeN_v1\logs\phase_name.log
   ```

10. Before reusing an existing phase, verify that the input hash, site-manifest hash, feature list, configuration hash, and source-code hash match.

---

## 21. Final acceptance checklist

The large-site run is complete only when all applicable checks pass:

- [ ] Every selected site has the declared target and five core exogenous variables.
- [ ] The exact ordered site manifests are saved and nested correctly.
- [ ] No site was selected using events or model outcomes.
- [ ] Registry metadata matches and missingness are reported.
- [ ] All train-derived metadata use training rows only.
- [ ] No pH history, event, residual, prediction, or model-metric field enters predictors.
- [ ] Chronological within-site split counts are saved.
- [ ] Sequence windows are 24 consecutive hours within one site and one split.
- [ ] Scalers and imputers are fitted on training data only.
- [ ] Controlled model variants use identical validation/test keys.
- [ ] Non-event and all-data models differ only in the declared training-window exclusion.
- [ ] Low events are evaluated in the negative tail and high events in the positive tail.
- [ ] Event denominators and reliability categories are shown.
- [ ] Processed pH is not reported as native pH units without verified inversion.
- [ ] Same-site and leave-site-out results are not conflated.
- [ ] Strict, alias-normalised, and reduced-predictor cohorts are not mixed.
- [ ] Logs, hashes, versions, seeds, model artefacts, predictions, and reports are present.
- [ ] The final report describes failures, excluded sites, and incomplete phases honestly.

---

## 22. Reference scripts and artefacts on the present computer

Use these as behavioural references, not as hard-coded path dependencies:

```text
pipeline/00_baseline/README.md
pipeline/00_baseline/01_build_baseline_dataset.py
pipeline/00_baseline/02_prepare_baseline_ready.py
pipeline/00_baseline/03_clean_baseline_v3.py
pipeline/00_baseline/04_train_full50_baselines_v3.py
pipeline/00_baseline/05_validate_residual_events_v3.py
pipeline/00_baseline/06_train_nonevent_baselines_v3.py
pipeline/02_hgb200/run_processed200_hgb_metadata_experiment.py
pipeline/03_processed100/run_processed100_step4_0_4_1.py
pipeline/04_sequence_gru/run_sequence100_gru_lstm_metadata.py
pipeline/05_lstm/run_lstm100_metadata.py
pipeline/06_metadata_v2/run_v2_metadata_rerun.py
pipeline/07_non_event/run_nonevent_gru100.py
pipeline/08_event_analysis/run_ph_low_high_event_analysis.py
pipeline/09_signed_residual/run_signed_residual_tail_analysis.py
pipeline/10_site_hydrology/run_site_hydrology_performance_analysis.py
config/v2_metadata_feature_definition.json
```

The new implementation should convert hard-coded constants and personal paths into explicit configuration/command-line arguments while preserving the existing calculations.

---

## 23. Direct instruction to Codex on the other device

When this file is provided to Codex, use the following instruction:

> Read this entire Markdown file and inspect the transferred project before making changes. Implement and execute the workflow phase by phase. Begin with the environment, file inventory, and strict-site eligibility audit; do not begin a heavy model run until the site manifest and leakage checks pass. Use the existing scripts as behavioural references, create new parameterised scripts under a new versioned output/work directory, and never overwrite old results. Report progress and important counts as you work. If the data cannot support more strict full-predictor sites, do not lower the standard silently: finish the strict experiment, create a separately labelled extended-cohort audit, and explain exactly what prevents direct expansion.

The first response from the other device should report:

1. detected project root;
2. available disk, RAM, CPU, and GPU;
3. presence and hashes of `data/processed`, `sites_register.json`, and `regulatory_events.csv`;
4. Python and package environment status;
5. the proposed new output path;
6. confirmation that it will start with the lightweight inventory rather than model training.
