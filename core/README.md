# Core data utilities

This folder contains the small reusable layer beneath the experiment pipeline:

- `data/`: official-source downloaders, station representation, parameter mapping, preprocessing, and clean Parquet construction.
- `utilities/`: datetime, registry, source-specific, NEON-position, and consolidation helpers.
- `tools/`: regulatory threshold/event detection.

These modules are a curated snapshot of the reference implementation used by the project. No raw data are bundled here.
