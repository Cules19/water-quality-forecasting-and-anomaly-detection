# Common1000 30-minute resolution experiment

This stage follows site-count scaling with a finer temporal resolution while
holding the 1,009-site cohort, 24-hour history duration, GRU-FiLM V2 model,
training cap and chronological splits fixed. Per-site 30-minute Parquet shards
are built first and then assembled into disk-backed arrays; full windows are
never materialised in RAM.

```powershell
python pipeline/17_common1000_resolution/run_common1000_resolution30m.py `
  --config config/common1000_resolution30m_v1.json `
  --resume
```
