# Curated pipeline index

The numbered directories follow the current analysis logic. Generated files
are written to `analysis_outputs/`, keeping source code separate from results.

`00_baseline/` contains the six-step Full50 baseline ladder from simple site and
seasonal means through persistence, Ridge, HGB, residual validation and
non-event training. Its own README explains the important pH-history boundary.

| Stage | Script | Purpose |
| --- | --- | --- |
| 01 | `01_inventory/run_processed_metadata_availability_check.py` | Lightweight processed-file, schema and registry audit |
| 02 | `02_hgb200/run_processed200_hgb_metadata_experiment.py` | Strict 200-site HGB benchmark with/without metadata |
| 03 | `03_processed100/run_processed100_step4_0_4_1.py` | Nested 100-site event-state coding and leakage-controlled metadata merge |
| 04 | `04_sequence_gru/run_sequence100_gru_lstm_metadata.py` | Hourly 24-step sequence construction and GRU models |
| 05 | `05_lstm/run_lstm100_metadata.py` | Controlled LSTM comparison using the same sequence data |
| 06 | `06_metadata_v2/run_v2_metadata_rerun.py` | Compact hydrological V2 metadata rerun and ablation |
| 07 | `07_non_event/run_nonevent_gru100.py` | Controlled non-event target-row GRU training |
| 08 | `08_event_analysis/run_ph_low_high_event_analysis.py` | Regulatory low/high pH event behaviour analysis |
| 09 | `09_signed_residual/run_signed_residual_tail_analysis.py` | Absolute and direction-specific residual-tail evaluation |
| 10 | `10_site_hydrology/run_site_hydrology_performance_analysis.py` | Site hydrology and model-performance association analysis |

Stages 01-05 form the direct processed-data modelling chain. Stage 06 is the
historical compact-metadata reproduction script and may consume existing HGB50
artefacts in addition to the current HGB200/sequence outputs. For a new larger-N
run, use `config/v2_metadata_feature_definition.json` and the cross-device plan
to parameterise one common V2 feature set instead of recreating obsolete full
metadata branches.

Stages 07-10 are controlled extensions and post-hoc analyses. Event labels are
never model predictors. Historical pH remains excluded from the primary model
inputs.
