# Frozen H00 site-transfer and warm-up finalisation

This stage audits the frozen H00/H01 deployment evidence, freezes validation-only choices, and compiles the final pH and dissolved-oxygen warm-up package. It does not turn H01 into a like-for-like replacement for H00: H01 consumes recent target history.

## Required restored inputs

- `analysis_outputs/ph_common1000_lstm_nohistory_h0_rain_final_v1/full/`
- `analysis_outputs/ph_common1000_lstm_film_ph_sensor_ablation_v1/full/`
- `analysis_outputs/do_common1000_warmup_adaptation_lowcost_mg_l_validation_v1/full/`
- `analysis_outputs/do_common1000_warmup_adaptation_lowcost_mg_l_test_v1/full/`
- Stage-specific checkpoints restored under `analysis_outputs/h0_site_transfer_warmup_final_audit_v1/checkpoints/`
- The three `config/h0_final_*.json` configurations

## Audit and finalisation order

Run from the repository root:

```powershell
python pipeline/42_h0_site_transfer_final/run_h0_final_audit.py
python pipeline/42_h0_site_transfer_final/freeze_warmup_knee.py
python pipeline/42_h0_site_transfer_final/freeze_h01_input_selection.py
python pipeline/42_h0_site_transfer_final/finalize_h0_warmup_results.py
```

`run_h0_final_audit.py` is fail-closed: a leakage, temporal-causality, checkpoint-freeze, or dissolved-oxygen unit failure writes a blocked report and exits non-zero. Stop if that happens. The freeze scripts require validation outputs that already exist in the restored evidence; they do not train missing checkpoints.

## Output

The stage writes `analysis_outputs/h0_site_transfer_warmup_final_audit_v1/`, including frozen selections, site metrics, summaries, figures, reports, completion status, and SHA-256 checksums. Preserve the archived original and use a new versioned directory for exploratory reruns.
