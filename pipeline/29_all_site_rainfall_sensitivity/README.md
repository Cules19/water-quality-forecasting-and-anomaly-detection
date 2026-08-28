# All-site rainfall and API sensitivity sidecars

This stage extends the earlier Common1000 rainfall outputs from 910 matched pH
sites to all 1,750 unique canonical precipitation stations.

It creates a prior-format hourly rainfall sidecar for every station and then
reuses stage 27 to calculate the earlier API sensitivity variants:

- `api_24h`
- `api_72h`
- `api_168h`

For stations without pH observations, the sidecar uses the complete canonical
rainfall time axis. Existing 910-site outputs are not overwritten.

The teacher-Archive `precip_api` (48-hour half-life, normalized EWMA) and
`precip_6h` remain separate in stage 28 because their definitions differ from
the sensitivity API recurrence above.
