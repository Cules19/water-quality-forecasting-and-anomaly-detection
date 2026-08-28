# H00 reproducibility audit

This audit verifies that the official frozen H00 checkpoint and evaluation path can be reproduced, identifies training-path drift, performs two seed-42 reproduction runs, and confirms determinism. Test time series remain sealed throughout.

## Required restored inputs

- `analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/`
- `analysis_outputs/ph_common1000_teacher_archive_features_do_converted_v1/full/`
- `analysis_outputs/ph_h00_hydrological_memory_validation_v1/`
- `analysis_outputs/ph_h00_site_balanced_validation_v1/`
- A writable `analysis_outputs/ph_h00_reproducibility_audit_v1/`

GPU training is strongly preferred for the two reproduction runs. Preserve the archived completed audit before starting a new run.

## Run in this order

```powershell
python pipeline/50_ph_h00_reproducibility_audit/stage_a_preserve_evidence.py
python pipeline/50_ph_h00_reproducibility_audit/stage_b_reevaluate_official.py
python pipeline/50_ph_h00_reproducibility_audit/stage_c_to_f_audit.py
python pipeline/50_ph_h00_reproducibility_audit/stage_g_h_reproduce.py --run primary
python pipeline/50_ph_h00_reproducibility_audit/stage_g_h_reproduce.py --run repeat
python pipeline/50_ph_h00_reproducibility_audit/finalize_audit.py
```

Every command must succeed before continuing. Stage A hashes and inventories the evidence. Stage B must reproduce the official evaluation. Stages C–F compare checkpoints, data, scalers, windows, configurations, histories, and prediction paths. The repeat run is allowed only after the primary reproduction passes. Finalisation verifies that preserved evidence did not change and writes the final reports, root-cause summary, completion record, and checksums.

The retained diagnosis is training-configuration drift plus optimizer/training-budget drift, not data or evaluation drift.
