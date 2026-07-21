# pH Hot-Moment Modelling

Code-only repository for multi-site prediction-based detection of pH hot
moments from high-frequency water-quality sensor data and hydrological
metadata.

The primary models predict processed pH (`00400`) from exogenous variables:

- water temperature (`00010`)
- specific conductance (`00095`)
- turbidity (`63680`)
- discharge (`00060`)
- gage height (`00065`)
- calendar and cyclical time features

Historical pH and regulatory event labels are deliberately excluded from the
main predictors. Regulatory low/high pH windows are incomplete weak labels used
for residual-tail evaluation and controlled non-event training only.

## Start here

On a new computer, give Codex
[the other-device start file](documentation/START_HERE_OTHER_DEVICE.md) first.
It instructs Codex to inspect the machine and data and recommend the next valid
step before any heavy processing.

Read [the cross-device execution plan](documentation/ph_cross_device_full_pipeline_plan_v1.md)
before running the scripts. It defines the cohort rules, leakage controls,
chronological splits, metadata construction, HGB and sequence experiments,
signed residual analysis, and acceptance checks.

## Repository layout

```text
pipeline/                Ten curated stages in execution order
config/                  Compact V2 metadata definition
documentation/           Full cross-device execution and validation plan
analysis_outputs/        Generated locally when scripts run; never versioned
```

Only the latest scripts that contributed to the current modelling conclusions
are included. Superseded development versions, duplicated reruns, 50-site
prototype scripts, report generators, and figure-only utilities are excluded.
See [the pipeline index](pipeline/README.md) for the purpose and dependencies of
each retained stage.

## Local data layout

Data and model artefacts are intentionally not included. On a working machine,
provide at least:

```text
data/processed/
data/reference/sites_register.json
data/reference/regulatory_events.csv
```

Install dependencies in an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the lightweight inventory and cohort audit before any heavy model training.
The present data supported approximately 201 sites with the exact common
predictor schema, so the directly comparable primary experiment uses a strict
200-site cohort. Any larger alias-normalised or reduced-predictor cohort must be
reported separately.

## Data policy

This private repository contains source code and documentation only. It does
not contain raw/processed observations, registry records, event CSV files,
predictions, model weights, figures, Word/PDF reports, or environment secrets.
