# Dynamic precipitation audit

This read-only stage inventories the DVC precipitation batch and matches it to
the Common1000 water-quality cohort. It does not modify rainfall CSV files,
water-quality shards, model inputs, checkpoints, or predictions.

## Rules

- Reconstruct hourly timestamps from `year`, `month`, `day`, and `hour`.
- Treat `datetime` as date-only metadata.
- When a site has multiple exports, select the latest export timestamp embedded
  in the filename. Do not select by filesystem modification time after DVC
  checkout.
- Define downloaded-data preliminary eligibility as:
  exact site-ID match, at least 30 days of time overlap, varying rainfall, and
  at least 80% nonmissing hourly coverage in the overlap.
- Report existing varying `00045` separately, then take the union for combined
  preliminary coverage.

## Outputs

The default audit output directory used for this run is:

`analysis_outputs/ph_common1000_precipitation_audit_v1`

It contains:

- `precipitation_file_inventory.csv`
- `precipitation_site_inventory.csv`
- `common1000_precipitation_match.csv`
- `common1000_precipitation_unresolved.csv`
- `network_summary.csv`
- `source_summary.csv`
- `match_status_summary.csv`
- `summary.json`

The human-readable workbook is:

`outputs/precipitation_audit_v1/precipitation_audit_final.xlsx`
