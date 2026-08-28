# Repository map

## Primary workflow

| Stage | Purpose |
|---|---|
| `pipeline/00_common1000` | Canonical processing and split preparation |
| `pipeline/20_common1000_leave_sites_out` | Unseen-site cohort preparation |
| `pipeline/21_common1000_leave_sites_out_models` | Matched leave-sites-out models |
| `pipeline/24_error_onset` | Error-onset and residual-event analysis |
| `pipeline/25_precipitation_audit` | Precipitation source/coverage audit |
| `pipeline/28_teacher_archive_precip_all_sites` | Archived precipitation ingestion |
| `pipeline/30_common1000_teacher_archive_features` | Rainfall feature materialisation |
| `pipeline/31_common1000_rainfall_ablation` | Rainfall and target-history ablations |
| `pipeline/32_recurrent_comparison` | Recurrent architecture comparison |
| `pipeline/33_lstm_final_rain_ablation` | Final LSTM rainfall experiment |
| `pipeline/34_final_recurrent_benchmark` | Frozen recurrent benchmark |
| `pipeline/35_warmup_feasibility` | Site warm-up feasibility |
| `pipeline/36_warmup_adaptation` | Warm-up adaptation experiments |
| `pipeline/37_lowcost_deployment` | Low-cost deployment evaluation |
| `pipeline/38_lowcost_conformal` | Conformal uncertainty analysis |
| `pipeline/42_h0_site_transfer_final` | Frozen H00 unseen-site transfer |
| `pipeline/44_cross_variable_hot_moment_incremental_closure` | Hot-moment matched-key closure |
| `pipeline/45_ph_h00_30day_calibration` | Retained 30-day offset calibration |
| `pipeline/50_ph_h00_reproducibility_audit` | Checkpoint and pipeline reproducibility |
| `pipeline/56_ph_h00_upstream1` | Upstream data acquisition and spatial QA; model not run |

`pipeline/01_inventory` is the input inventory entry point. The earlier `pipeline/11`–`18`, `22`, `26`, `27`, `29`, `40`, and `41` directories are retained as valid supplementary analyses or report builders rather than the shortest final execution chain.

## Reusable code

| Path | Contents |
|---|---|
| `models/lstm_film` | Dataset configuration, sequence assembly, FiLM model, training, evaluation, sweep, and horizon scan |
| `core/data` | EA/USGS/NEON acquisition and clean-Parquet construction |
| `core/utilities` | Datetime, registry, USGS, NEON-position, and consolidation helpers |
| `core/tools` | Regulatory threshold event detector |
| `scripts` | Small operational entry points and methodology-figure generation |
| `analysis` | Three final analyses recovered from the generated-output tree and made path-portable |
| `tools/solar_context` | Solar sidecar generation, independent validation, and the bounded probe |

## Configuration and outputs

The `config/` directory holds the 48 configurations needed by the retained code. Scripts resolve the repository root from their file location where possible. Generated artefacts belong under `analysis_outputs/`, while source datasets belong under `data/`; both are ignored by Git. See [generated files](GENERATED_FILES.md) for the expected layout.

## Preview versus code

GitHub renders every `README.md` or other Markdown file under **Preview** and exposes its source under **Code**. This repository therefore places human explanation in Markdown beside, but not inside, the executable Python modules.
