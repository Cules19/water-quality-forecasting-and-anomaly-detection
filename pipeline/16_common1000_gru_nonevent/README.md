# Common1000 full-metadata GRU non-event pair

This stage closes the original GRU100 supervisor task at the current scale. It
reuses the exact 1,009-site `GRU_S1` architecture, full 44-column safe metadata,
dynamic scaler, window keys and seed, while removing only event target windows
from the sampled training endpoints. Validation and test data are unchanged.

```powershell
python pipeline/16_common1000_gru_nonevent/run_common1000_gru_fullmeta_nonevent.py `
  --config config/common1000_gru_fullmeta_nonevent_v1.json `
  --resume
```
