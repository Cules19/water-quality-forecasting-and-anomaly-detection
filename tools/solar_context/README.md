# Solar context tools

`build_and_validate_solar_features.py` derives hourly solar phase and potential from UTC timestamps plus audited site coordinates and independently checks the computation. `run_solar_probe.py` reproduces the bounded solar probe. Shared constants and helpers are in `common.py`.

The fields are derived astronomical context, not downloaded irradiation. The associated H00 solar adapter was not promoted; these files are retained because the feature construction and validation are reusable and the negative result should remain reproducible.
