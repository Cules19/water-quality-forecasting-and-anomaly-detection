# Common1000 teacher-Archive rainfall ablation

This stage performs the primary rainfall experiment requested by the teacher.
It uses only the 910 direct canonical Archive sites and preserves their
original train/validation/test site assignments.

The three non-event-training models are:

1. `BASE_NONEVENT`: the established pH GRU-history + FiLM model;
2. `RAIN_NONEVENT`: baseline plus dynamic hourly precipitation (`00045`);
3. `RAIN_API_NONEVENT`: baseline plus `00045`, the teacher-Archive 48-hour
   normalized EWMA (`precip_api`), and the teacher-Archive rolling 6-hour
   precipitation sum (`precip_6h`).

All three models use exactly the same target timestamps. A target timestamp is
eligible only if its full 24-hour causal input window has teacher-Archive
precipitation features. All feature and target standardization parameters are
fitted on training sites only. Rainfall variables are transformed with
`log1p` after the exact Archive features have been constructed.

Run:

```powershell
python pipeline/31_common1000_rainfall_ablation/run_rainfall_ablation.py `
  --config config/common1000_rainfall_ablation_v1.json --mode smoke

python pipeline/31_common1000_rainfall_ablation/run_rainfall_ablation.py `
  --config config/common1000_rainfall_ablation_v1.json --mode full
```

## Matched GRU/LSTM comparison

Model definitions may set `recurrent_type` to `gru` or `lstm`. Both use the
same site-agnostic FiLM wrapper and never use a learned site embedding. The
teacher-aligned comparison uses a 48-hour input window, effective dropout,
AdamW with ReduceLROnPlateau, gradient clipping, and validation-only early
stopping:

```powershell
python pipeline/31_common1000_rainfall_ablation/run_rainfall_ablation.py `
  --config config/common1000_recurrent_comparison_teacher_aligned_do_converted_v1.json `
  --mode smoke

python pipeline/31_common1000_rainfall_ablation/run_rainfall_ablation.py `
  --config config/common1000_recurrent_comparison_teacher_aligned_do_converted_v1.json `
  --mode full
```

The four matched runs are GRU/LSTM crossed with baseline/dynamic-rainfall
features. The API feature set is intentionally excluded from this architecture
comparison.
