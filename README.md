# Pollution_ID: Cross-site Water-Quality Forecasting

This repository is the curated, audited code snapshot for a multi-network water-quality forecasting project spanning Great Britain and the United States. It focuses on pH and dissolved oxygen, unseen-site transfer, low-cost deployment, calibration, and pollution-relevant residual “hot moments”.

The repository intentionally contains **source code, configuration, compact documentation, and publication figures only**. Raw monitoring data, geospatial rasters, model checkpoints, and generated predictions are stored separately because the complete local evidence base exceeds 100 GB and includes sources that must not be redistributed.

## What the project does

- Harmonises hourly observations across 910 sites: 638 training, 137 validation, and 135 held-out test sites.
- Builds causal 48-hour sequence inputs from water-quality covariates, rainfall, and calendar features.
- Compares HGB, GRU, LSTM, and LSTM-FiLM models under matched splits.
- Tests a no-history horizon-0 pH model (`H00`) that excludes historical pH.
- Evaluates one-hour historical-target forecasting (`H01`), low-cost sensor subsets, site warm-up, and conformal uncertainty.
- Audits static site context (geology, baseflow tendency, imperviousness, and permitted-discharge proximity), solar context, upstream catchment context, and official flow/stage availability.
- Separates promoted results from negative or under-supported experiments.

## Main audited findings

The frozen no-history H00 benchmark had useful pooled held-out skill (RMSE 0.406864; NSE 0.512871) but inconsistent site transfer. A preregistered 30-day site offset calibration was promoted: on its strict held-out comparison it reduced pooled RMSE from 0.412487 to 0.377248 and macro-site MAE from 0.242124 to 0.180789. The cross-variable hot-moment hypothesis received partial support, not a universal positive conclusion. Static, solar, upstream, flow, and stage extensions are retained only at the evidential level each audit supports.

## Repository map

| Path | Purpose |
|---|---|
| [`pipeline/`](pipeline/) | Ordered processing, modelling, calibration, transfer, and audit stages |
| [`models/lstm_film/`](models/lstm_film/) | Reusable LSTM-FiLM implementation |
| [`core/`](core/) | Download, standardisation, station, registry, and event utilities |
| [`config/`](config/) | Versioned experiment configurations |
| [`analysis/`](analysis/) | Final frozen-output analyses and supervisor checks |
| [`tools/solar_context/`](tools/solar_context/) | Solar-position feature generation and validation |
| [`scripts/`](scripts/) | Data acquisition, merging, regulatory-event, and figure entry points |
| [`figures/methodology/`](figures/methodology/) | Publication methodology figures and source manifest |
| [`docs/`](docs/) | Project overview, provenance, generated-file guide, and exclusions |

Start with [the project overview](docs/PROJECT_OVERVIEW.md), then use [the reproducibility guide](docs/REPRODUCIBILITY.md) to restore the private archive and [the repository map](docs/REPOSITORY_MAP.md) to identify the relevant workflow. The Markdown files render under GitHub’s **Preview** tab; the Python files remain under **Code**, matching the requested explanation/code separation.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install `requirements-geospatial.txt` for upstream/static spatial processing, or `requirements-reporting.txt` for Word/PDF reporting. Most workflows expect the private data archive to be restored as `data/` and selected evidence to be restored as `analysis_outputs/`; see the [reproducibility guide](docs/REPRODUCIBILITY.md) and [USB backup guide](docs/USB_BACKUP_GUIDE.md).

## Reproducibility boundary

Configuration files and code are public-repository material. Data licences still apply independently. In particular, raw EA/NRW permit extracts and identifying permit-holder fields must not be published. Large outputs are described in [generated files](docs/GENERATED_FILES.md), while discarded experimental branches are recorded in [excluded experiments](docs/EXCLUDED_EXPERIMENTS.md) so negative results are not silently lost.
