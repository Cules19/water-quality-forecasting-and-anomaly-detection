# Common1000 GRU/LSTM sequence models

This stage consumes the stage-00 model-ready site files and the safe static
metadata produced for stage 11. It aggregates exogenous variables and observed
pH to hourly rows, constructs valid 24-hour windows, and trains three controlled
GPU models on exactly the same window sample:

- `GRU_S0`: dynamic sequence plus site embedding;
- `GRU_S1`: dynamic sequence plus site embedding and static metadata;
- `LSTM_S1`: dynamic sequence plus site embedding and static metadata.

Current and historical pH and all event fields are excluded from predictors.
Event fields are carried only on the target hour for post-prediction analysis.

Run the 50-site smoke cohort first:

```powershell
python pipeline/12_common1000_sequence/run_common1000_sequence_models.py `
  --config config/common1000_sequence_v1.json `
  --mode smoke `
  --resume
```

After the smoke run passes, use `--mode full` for all 1,009 sites.
