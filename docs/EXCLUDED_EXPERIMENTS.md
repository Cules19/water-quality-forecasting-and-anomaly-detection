# Excluded experiment code

This curated repository omits branches that were superseded, administrative-only, failed a preregistered gate, or did not produce a promoted method. Their conclusions are recorded here so omission is not mistaken for missing evidence.

## Legacy and superseded

- `pipeline/02_hgb200` through `pipeline/10_site_hydrology`: early 100/200-site prototypes superseded by the Common1000 chain.
- `pipeline/19_remote_machine_audit` and `pipeline/23_remote_final_synthesis`: machine-transfer/administrative stages, not scientific source modules.
- `pipeline/37_ph_warmup_failure_diagnosis`: diagnostic branch superseded by the retained warm-up and deployment analyses.
- `pipeline/38_ph_h01_phase_b_safe_selection`: Phase-B selection attempt not promoted to the final method.

## Frozen-gate negative results

- `pipeline/43_ph_h01_phase_d_annual_calendar`: annual-calendar gate failed.
- `pipeline/46_ph_h00_robust_offset_validation`: robust offset did not replace the retained mean offset.
- `pipeline/47_ph_h00_site_centred_validation`: site-centred target was not promoted; comparison also requires sampler-aware interpretation.
- `pipeline/48_ph_h00_hydrological_memory_validation`: hydrological-memory extension did not pass its matched comparison.
- `pipeline/49_ph_h00_site_balanced_validation`: site-balanced extension did not pass its within-pipeline comparison.
- `pipeline/51_ph_h01_static3_controlled_ablation`: static-context H01 gate failed.
- `pipeline/52_ph_h00_static3_additive_validation`: static-context H00 gate failed.
- `pipeline/53_ph_h00_static_interaction_validation`: static interaction was not promoted.
- `pipeline/54_ph_h00_calibration_aware_residual`: calibration-aware residual model was not promoted.
- `pipeline/57_ph_h00_upstream13_complete_case`: complete-case upstream modelling was not authorised as a final result.
- `pipeline/58_ph_h00_upstream13_available_case`: available-case upstream alternative was not promoted.
- `pipeline/59_ph_h00_flow_asinh_adapter`: official-flow adapter lacked adequate supported coverage.
- `pipeline/60_ph_h00_stage00065_adapter`: USGS stage (`00065`) adapter lacked adequate supported coverage.

## Selectively retained tools from negative branches

The complete `pipeline/55_ph_h00_solar_dynamic1` experimental branch is omitted because its adapter was not promoted. Its reusable, outcome-independent feature generator and validator are preserved under `tools/solar_context/`.

The upstream spatial/data audit is different: it created a valuable, reproducible data product even though the model gate failed. Therefore `pipeline/56_ph_h00_upstream1` is retained with explicit “model not run” labelling.

Generated reports from these experiments remain in the private USB evidence archive. Removing their code from the public repository does not delete the underlying negative-result record.
