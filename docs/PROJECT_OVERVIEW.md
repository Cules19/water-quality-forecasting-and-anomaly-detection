# Project overview

## Research objective

The project asks how well water-quality conditions can be predicted or diagnosed at monitoring sites that were not used to train the model, and whether large cross-variable residuals contain pollution-relevant information. The core targets are pH (`00400`) and dissolved oxygen (`00300`) from British and United States monitoring networks.

## Frozen cohort

The final cross-site split contains 910 sites:

| Split | Sites | Role |
|---|---:|---|
| Training | 638 | Fitting and train-only preprocessing |
| Validation | 137 | Model selection and preregistered gates |
| Test | 135 | One-time held-out evaluation |

All sequence models use causal windows. The main H00 pH benchmark uses 48 hours of temperature, specific conductance, dissolved oxygen, turbidity, hourly rainfall, and calendar features while deliberately excluding historical pH. H01 experiments answer a different question because they use historical target information to predict one hour ahead.

## Modelling path

1. Harmonise and audit site-level hourly data.
2. Establish tabular and recurrent baselines on identical site splits.
3. Test metadata conditioning with FiLM.
4. Add rainfall and evaluate target-history ablations.
5. Freeze a no-history H00 benchmark before held-out testing.
6. Quantify unseen-site failure, hot-moment enrichment, and low-cost deployment behaviour.
7. Test bounded calibration and environmental-context extensions under explicit stop rules.
8. Reproduce the retained checkpoint and diagnose any pipeline drift.

## Retained conclusions

- The frozen no-history H00 model has useful pooled skill but inconsistent transfer across individual held-out sites.
- A simple first-30-day mean residual offset is the retained calibration method. It improved held-out accuracy without changing the frozen hot-moment ranking.
- The historical-target H01 model is much more accurate, but it is not a like-for-like replacement for H00 because it consumes recent pH.
- Cross-variable residual hot moments show partial support: event enrichment is present, but uncertainty does not support a universal claim across sites.
- The official H00 checkpoint is reproducible. An apparent alternative-training discrepancy was traced to sampler and optimizer-budget drift, not to data or evaluation drift.
- Static site context, solar context, upstream catchment context, flow, and stage did not produce a promoted replacement for the official H00 model under their frozen gates.

## Static and environmental context

The external static audit assembled a 910-site table containing broad point geology, a baseflow-tendency proxy, local imperviousness or a labelled Alaska built-surface proxy, and straight-line distance to an active permitted discharge location. Coverage was high, but interpretation is deliberately narrow: point geology is not upstream lithology; gridded BFI is not a measured catchment BFI; permit proximity is not hydrological connectivity.

The upstream audit downloaded and verified approximately 39.3 GB of Hydrography90m/Environment90m material and produced a reproducible 13-feature table. It failed the preregistered completeness gate (94.357% for seven climate/soil fields versus 95% required), so no upstream candidate model was trained.

Solar fields (`solar_hour_sin`, `solar_hour_cos`, and `solar_potential`) were calculated from UTC timestamps and audited coordinates with `pvlib`/NREL SPA. They are derived features, not downloaded solar-radiation observations. The bounded solar adapter was not promoted.

## Reading results safely

An experiment directory being retained does not mean its hypothesis was supported. Status is stated in each stage README and in [excluded experiments](EXCLUDED_EXPERIMENTS.md). The large local `analysis_outputs/` tree is evidence, not source code, and is intentionally excluded from GitHub.
