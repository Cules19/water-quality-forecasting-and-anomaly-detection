# Other-device Codex: start here

## 1. Your task

You are working on the continuation of an MSc dissertation project concerning
multi-site prediction-based detection of pH hot moments from high-frequency
water-quality sensor data and hydrological metadata.

Your immediate task is **not** to start heavy model training. First inspect the
new computer, the available data, and the curated code in this repository. Then
decide and report what the scientifically valid next step should be for
processing more sites with the same methodology.

Read this file completely, inspect the repository with tools, and then read:

```text
documentation/ph_cross_device_full_pipeline_plan_v1.md
pipeline/README.md
README.md
config/v2_metadata_feature_definition.json
```

Do not respond with only a generic plan. Verify the local files, schemas,
hardware, and paths first, then make a recommendation grounded in what is
actually present on this device.

---

## 2. Scientific objective

The primary modelling approach predicts processed/standardised pH (`00400`)
from exogenous variables and interprets the prediction residual:

```text
residual = observed processed pH - predicted processed pH
```

Core exogenous sensor variables:

```text
00010  water temperature
00095  specific conductance
63680  turbidity
00060  discharge
00065  gage height
```

Time features:

```text
sin_hour, cos_hour, sin_day, cos_day,
month, hour, day_of_year
```

The principal models deliberately exclude current and historical pH from their
inputs. The intention is for the residual to represent pH behaviour not
explained by the exogenous sensor and temporal variables.

Low-pH weak-label events are evaluated in the negative residual tail. High-pH
weak-label events are evaluated in the positive residual tail. Absolute
residual tails are retained as a direction-agnostic comparison.

---

## 3. Non-negotiable rules

1. Use `data/processed` as the main modelling time-series source.
2. Do not rebuild the modelling data from `data/clean` unless a separate
   preprocessing-equivalence audit proves that this is necessary and valid.
3. Do not use pH history in the main HGB, GRU, or LSTM predictors.
4. Regulatory event windows are incomplete weak labels, not complete pollution
   ground truth.
5. Event labels may be used for evaluation and controlled training exclusion,
   but never as model predictors.
6. Static metadata must not include pH, event, residual, prediction, validation,
   test, future, or model-performance information.
7. Train-derived metadata, scalers, imputers, and encoders must be fitted using
   training rows only.
8. Use chronological `70%/15%/15%` train/validation/test splits within each site
   after sorting and duplicate removal.
9. Do not select sites using event counts, model performance, residual behaviour,
   or a desired result direction.
10. Do not silently relax the predictor requirements merely to obtain 300 sites.
11. Do not overwrite or delete previous outputs. Create a new versioned output
    directory.
12. Do not describe processed `00400` values as native pH units unless the exact
    inverse transformation has been verified.
13. Same-site chronological testing does not demonstrate generalisation to
    unseen sites. Leave-site-out analysis is a separate task.
14. Prediction metrics and event-detection metrics must be reported separately.

---

## 4. Expected local data

Look for the following project-relative inputs:

```text
data/processed/
data/reference/sites_register.json
data/reference/regulatory_events.csv
```

Optional input for native-unit hydrological post-hoc analysis:

```text
data/clean/
```

The GitHub repository intentionally does not contain the data. If these paths
are missing, locate the transferred data elsewhere on the device and report the
path. Do not duplicate several gigabytes before checking available disk space.

If `regulatory_events.csv` is still in a Downloads folder, copy or point it to
the project-relative path above before event-dependent phases. Do not hard-code
a personal Windows user path in new code.

---

## 5. Current-device reference values

The previous device found approximately:

- 336 processed site IDs;
- 336/336 matches to `sites_register.json`;
- 201 sites with the strict common schema:

  ```text
  site_id, timestamp, 00400, 00010, 00095, 63680, 00060, 00065
  ```

- about 4.5 GB in `data/processed`.

These values are reference checks, not values to force. Recalculate them from
the data on this computer and explain any differences.

The strict directly comparable experiment currently supports about 200 sites.
More than 200 sites are scientifically acceptable only when the new device has
additional full-schema processed sites, or when an explicit and verified alias
normalisation creates equivalent canonical variables. A reduced-predictor
extension must be labelled as a separate exploratory experiment.

---

## 6. First actions: inspection only

### 6.1 Repository check

Report:

- repository root;
- current Git branch and commit;
- whether the working tree is clean;
- presence of the baseline chain and all 10 numbered current pipeline stages;
- presence of the main plan and V2 metadata definition.

Do not modify the curated scripts during this first inspection.

### 6.2 Hardware and environment check

Report:

- operating system;
- Python version;
- CPU model/core count;
- total and available RAM;
- GPU model and VRAM;
- PyTorch version and CUDA availability;
- available disk space in the project/output location;
- installed versions of NumPy, pandas, PyArrow, scikit-learn, SciPy,
  Matplotlib, Seaborn, Joblib, and PyTorch.

If dependencies are missing, propose an isolated Python 3.11 or 3.12
environment. Do not alter the system Python environment unnecessarily.

### 6.3 Lightweight processed-data check

Inspect only filenames, parquet metadata/footers, schemas, JSON keys, and small
samples initially. Do not fully load every parquet or large CSV.

Determine:

- number and total size of processed files;
- number of parquet files;
- distinct processed site IDs;
- schema variants and dtypes;
- strict full-schema site count;
- registry match count and match rate;
- availability of regulatory events;
- whether the additional data are already processed or exist only as raw/clean
  files;
- obvious empty, unreadable, duplicate, or conflicting files.

The existing lightweight reference script is:

```text
pipeline/01_inventory/run_processed_metadata_availability_check.py
```

Inspect the script and paths before running it. Running this lightweight audit
is allowed after confirming it will write only to a new or appropriate
versioned output folder. Do not start stages 02-10 yet.

---

## 7. Decide which expansion is valid

After the audit, classify the available data into one of the following cases.

### Case A: at least 200 strict full-schema sites

Use a deterministic nested strict cohort, initially:

```text
N50 subset of N100 subset of N150 subset of N200
```

Run the strict N200 experiment as the primary comparable study. Additional
strict sites may support N250/N300 if all eligibility rules are unchanged.

### Case B: more than 200 sites only after verified aliases

Build an explicit canonical alias table. For every accepted alias, verify:

- physical meaning;
- unit or preprocessing definition;
- timestamp semantics;
- transformed-value comparability;
- evidence source.

Keep the alias-normalised extension separate from the strict cohort until the
equivalence audit passes.

### Case C: additional sites lack one or more required predictor channels

Do not fill an entirely absent channel with a constant and call it the same
experiment. Either exclude those sites or define a separately named
reduced-predictor exploratory experiment. It must not be placed on the strict
site-count scaling curve.

### Case D: additional sites exist only as raw or clean data

Locate the original preprocessing scripts and saved transformation objects.
Choose at least three sites present in both raw/clean and processed forms.
Reprocess them and compare:

- schema;
- timestamps and row keys;
- row counts;
- target and predictor distributions;
- transformed values;
- missing/sentinel handling.

Only apply the preprocessing to new sites after near-exact equivalence is
demonstrated. Otherwise keep the new sites outside the directly comparable
study.

---

## 8. What to recommend after inspection

Your first substantive report must contain the following sections:

### A. Machine readiness

State whether this device is suitable for:

- lightweight audit;
- HGB200;
- 100/200-site sequence preparation;
- GRU/LSTM training;
- leave-site-out or larger-site extensions.

Give disk, RAM, GPU, and expected runtime risks.

### B. Data readiness

State:

- total processed sites;
- strict eligible sites;
- registry matches;
- event-file status;
- metadata availability;
- whether additional sites are processed, alias-compatible, reduced-schema, or
  raw-only.

### C. Recommended cohort

Recommend the exact primary cohort size and explain why. Distinguish strict,
alias-normalised, and reduced-predictor cohorts.

### D. Recommended next implementation

Identify which existing scripts can be reused unchanged, which require path or
parameter changes, and which new parameterised orchestration is needed.

The preferred direction for a larger run is to create new parameterised scripts
under a new versioned work/output area. Do not edit historical calculations
without documenting the change.

### E. Proposed execution order

Provide a short, concrete order with decision gates, beginning with:

1. inventory;
2. eligibility and frozen site manifests;
3. two-site smoke test;
4. N10 or N50 resource smoke test;
5. strict HGB scaling;
6. reusable hourly/window dataset;
7. controlled GRU/LSTM models;
8. compact V2 metadata comparison;
9. controlled non-event training;
10. signed residual and site hydrology analysis;
11. leave-site-out only after same-site results pass.

### F. Stop/go recommendation

End the first report with one of:

```text
GO: ready for lightweight audit only
GO: ready to implement the strict larger-site pipeline
HOLD: missing required data or environment
HOLD: cohort definition needs user confirmation
```

Do not start heavy training until the audit and recommendation have been shown
to the user.

---

## 9. Suggested first output directory

Use a new versioned root, for example:

```text
analysis_outputs/ph_cross_device_largeN_v1/
```

The full expected folder structure is defined in:

```text
documentation/ph_cross_device_full_pipeline_plan_v1.md
```

Every completed phase should save its effective configuration, counts, warnings,
log, input hashes, output hashes, and `_SUCCESS` marker. Partial outputs must not
be treated as complete.

---

## 10. Exact instruction to follow now

> Inspect this repository and the local data using the rules above. Read the
> full cross-device plan. Perform only the repository, hardware, environment,
> and lightweight processed-data availability checks. Do not train a model and
> do not build a full 200/300-site dataset yet. Then give the user a concise,
> evidence-based report of what is present, what is missing, how many sites are
> strictly eligible, whether more sites can be added without changing the
> experiment, and the exact next action you recommend.
