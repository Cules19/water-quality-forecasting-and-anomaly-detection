# Antecedent precipitation index (API)

This stage derives causal, hourly antecedent precipitation indices from the
canonical rainfall files selected and validated in stage 26.

For each configured half-life `H`:

```text
alpha = 0.5 ** (1 / H)
API[t] = rainfall[t] + alpha * API[t - 1]
```

The recurrence is evaluated on the complete canonical rainfall record before
it is aligned to the prepared pH time axis. This preserves rainfall history
that predates the first pH hour.

## Missingness and warm-up

- An observed rainfall value of zero is a real no-rain hour. API decays.
- Missing rainfall is never converted to zero.
- A missing hour breaks the recurrence. The next observed segment restarts.
- API is withheld for five half-lives after each segment starts. This limits
  sensitivity to the unknown initial state to at most `1 / 32` (3.125%).
- Hours outside the source rainfall period remain null.

The output retains explicit `valid`, `missing`, and `warmup` columns for every
API scale. Downstream models must filter on the valid flag or use an explicit
missingness mask.

## Commands

Smoke test:

```powershell
.\.venv\Scripts\python.exe pipeline\27_antecedent_precipitation_index\build_api_sidecars.py `
  --config config\common1000_api_v1.json --smoke-per-source 3
```

Independent validation:

```powershell
.\.venv\Scripts\python.exe pipeline\27_antecedent_precipitation_index\validate_api_sidecars.py `
  --config config\common1000_api_v1.json `
  --run-root analysis_outputs\ph_common1000_api_v1\smoke_3_per_source
```

Omit `--smoke-per-source` and use the `full` run root for all 910 sites.
This stage does not train a model and does not modify source rainfall or
prepared water-quality files.
