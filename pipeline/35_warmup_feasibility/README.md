# New-site warm-up feasibility audit

This stage audits whether the teacher-Archive cohort can support the proposed
2-week, 1-, 2-, 3-, 6-, 9- and 12-month new-site adaptation curves. It uses the
exact common target timestamps from the frozen final LSTM-FiLM rainfall run.
It does not train a model or compare performance.

Run:

```powershell
.\.venv\Scripts\python.exe `
  pipeline\35_warmup_feasibility\audit_warmup_feasibility.py `
  --config config\common1000_warmup_feasibility_v1.json
```

The primary controlled design uses nested adaptation keys and one fixed
post-12-month evaluation key set per site. A separate immediate 30-day
eligibility count is retained for an operational sensitivity curve.

Outputs are written below
`analysis_outputs/ph_common1000_warmup_feasibility_v1/reports`.
