# Common1000 FiLM, site scaling and non-event sequence experiment

This stage completes the large-machine sequence tasks on the existing
Common1000 disk-backed arrays:

1. At N=200, train GRU/LSTM concatenation controls and GRU/LSTM FiLM models.
2. Select the FiLM recurrent cell using validation RMSE only.
3. Train the selected FiLM architecture on nested N=50/200/500/1009 cohorts.
4. At N=1009, train a target-row non-event counterpart with identical model
   initialisation and unchanged validation/test rows.
5. Report native-pH prediction and signed event-tail metrics, resource usage,
   and the site-count scaling curve.

FiLM maps compact V2 hydrological and train-only exogenous summaries to
feature-wise `gamma` and `beta`, then modulates every recurrent output state.
Event fields and pH history are never model inputs.

```powershell
python pipeline/14_common1000_film/run_common1000_film_scaling.py `
  --config config/common1000_film_scaling_v1.json
```

Results are local under `analysis_outputs/ph_common1000_film_scaling_v1/`.

After training, evaluate the scaling models on fixed common N50 and N200 test
cohorts so that changing test-site difficulty is not mistaken for a training
scale effect:

```powershell
python pipeline/14_common1000_film/evaluate_common_site_scaling.py `
  --config config/common1000_film_scaling_v1.json
```

Finally run the independent prediction, common-key, initialisation and leakage
audit:

```powershell
python pipeline/14_common1000_film/audit_common1000_film.py `
  --config config/common1000_film_scaling_v1.json
```
