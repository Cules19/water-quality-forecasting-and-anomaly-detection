# Validation-only new-site warm-up adaptation

This stage starts from the frozen teacher-aligned LSTM-FiLM rainfall checkpoint
and compares:

- zero-shot transfer;
- a 130-parameter site-specific FiLM-offset plus output-affine adapter;
- full checkpoint fine-tuning plus the same adapter.

Every site-duration model uses only labels before the warm-up cutoff. The
primary evaluation timestamps are fixed after the 12-month boundary. This
configuration is validation-only; a separate test configuration must not be
created until the adaptation protocol is frozen.

Smoke run:

```powershell
.\.venv\Scripts\python.exe `
  pipeline\36_warmup_adaptation\run_warmup_adaptation.py `
  --config config\common1000_warmup_adaptation_validation_v1.json `
  --mode smoke
```

Full validation run:

```powershell
.\.venv\Scripts\python.exe `
  pipeline\36_warmup_adaptation\run_warmup_adaptation.py `
  --config config\common1000_warmup_adaptation_validation_v1.json `
  --mode full `
  --resume
```

Freeze and interpret the validation protocol:

```powershell
.\.venv\Scripts\python.exe `
  pipeline\36_warmup_adaptation\evaluate_warmup_validation.py `
  --config config\common1000_warmup_adaptation_validation_v1.json `
  --mode full
```

After freezing the protocol, run the held-out test once:

```powershell
.\.venv\Scripts\python.exe `
  pipeline\36_warmup_adaptation\run_warmup_adaptation.py `
  --config config\common1000_warmup_adaptation_test_v1.json `
  --mode full `
  --resume

.\.venv\Scripts\python.exe `
  pipeline\36_warmup_adaptation\evaluate_warmup_test.py
```
