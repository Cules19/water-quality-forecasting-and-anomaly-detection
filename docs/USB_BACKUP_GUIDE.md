# USB backup guide

The private archive is designed to be copied as one top-level folder. It is separate from the Git repository so that code remains cloneable and licensed/large evidence remains private.

## Planned layout

```text
Pollution_ID_USB_ARCHIVE_2026-08-25/
├── code_snapshot/          # This curated repository, without .git
├── data/                   # Essential clean/reference/upstream source data
├── evidence/               # Selected final and scientifically important outputs
├── reports/                # Retained reports, figures, and project inventories
└── manifests/              # Paths, sizes, hashes, source locations, and status
```

## Restore

1. Copy `code_snapshot/` to a writable location.
2. Create a Python environment and install the relevant requirements file.
3. Restore each archived data group using `manifests/data_selection.csv`. In particular, restore `data/reference_project/` as repository `data/reference/`.
4. Copy each first-level directory under archived `evidence/` into repository `analysis_outputs/`.
5. Check `manifests/data_copy_verification.csv`, `manifests/evidence_copy_verification.csv`, and `manifests/archive_sections.csv` before rerunning a pipeline.
6. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup, path mapping, and execution boundaries.
7. Keep the archive private unless each source licence has been reviewed.

## Verification

The archive manifest records file counts and byte sizes. Important compact files receive SHA-256 values; extremely large collections are verified through source manifests where available plus copy counts/sizes. The archive should be copied to an NTFS or exFAT drive because many individual files and model/data files exceed the practical limits of FAT32.

## Deletion policy

No original is deleted merely because it has been copied. Local cleanup begins only after:

1. archive creation completes without copy errors;
2. the archive inventory is reviewed;
3. exact deletion candidates and reclaimable sizes are presented; and
4. the owner explicitly confirms deletion.
