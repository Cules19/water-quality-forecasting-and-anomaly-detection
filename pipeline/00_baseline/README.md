# Baseline chain

This folder contains the latest useful Full50 baseline chain. It shows how the
exogenous HGB/GRU work was compared with progressively stronger reference
models without retaining superseded dev10, v2, or duplicate rerun scripts.

## Execution order

1. `01_build_baseline_dataset.py`
   - audits processed parquet files;
   - selects 50 complete-predictor sites using event-independent coverage and
     row-count criteria;
   - adds chronological splits, time features and evaluation-only event labels.
2. `02_prepare_baseline_ready.py`
   - constructs the declared baseline columns and leakage audit.
3. `03_clean_baseline_v3.py`
   - applies the final sentinel/non-finite cleaning used by the v3 baseline.
4. `04_train_full50_baselines_v3.py`
   - evaluates M0 site mean, M1 seasonal mean, M2 1-hour persistence, M3 Ridge
     and M4 HistGradientBoosting.
5. `05_validate_residual_events_v3.py`
   - evaluates residual-event enrichment and event-window hit rates.
6. `06_train_nonevent_baselines_v3.py`
   - trains controlled non-event Ridge and HGB comparisons.

## Interpretation boundary

`M2_persistence_1h` uses previous pH and is therefore a forecasting benchmark,
not a valid primary model for the exogenous residual interpretation. The main
HGB/GRU/LSTM models exclude historical pH so that residuals represent pH
variation unexplained by the exogenous sensor variables.

The original early dataset builder prioritised event-rich sites. That ranking
has been replaced here by deterministic event-independent selection to avoid
cohort-selection leakage. Consequently, a newly rerun baseline may not reproduce
the exact historical Full50 numbers if the eligible site inventory changes.
