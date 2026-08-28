# Models

`lstm_film/` is the reusable recurrent modelling framework: configuration, causal dataset assembly, FiLM conditioning, training, evaluation, horizon scans, and sweeps. The snapshot includes the observed-target split alignment correction used by the final audited work.

Experiment-specific choices live in `config/` and the numbered pipeline stages. Checkpoints and predictions are generated under `analysis_outputs/` and are not committed.
