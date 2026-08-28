# Site-disjoint recurrent FiLM models

This stage trains GRU/LSTM models on mutually exclusive site sets. Unlike the
earlier Common1000 sequence models, it has no site embedding. Static metadata is
provided only through a per-layer FiLM generator, with and without coordinates.

```powershell
python pipeline/21_common1000_leave_sites_out_models/run_leave_sites_out_models.py `
  --config config/common1000_leave_sites_out_models_v1.json `
  --mode smoke
```

The full mode supports resumable per-model checkpoints. Prediction keys are
audited against the preparation-stage baseline keys.
