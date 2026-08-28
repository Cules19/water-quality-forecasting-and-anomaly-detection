# Reproducibility guide

This repository preserves code, configurations, tests, and compact figures. The private USB archive preserves large or restricted data, checkpoints, predictions, and audit evidence. A complete recovery requires both.

## Reproducibility levels

1. **Code validation:** clone GitHub, install dependencies, parse configurations, and run unit tests. Private data is not required.
2. **Frozen-result verification:** restore `data/` and `analysis_outputs/` from the private archive, then rerun audit and summary scripts without retraining.
3. **Training reproduction:** restore the frozen cohort, preprocessing objects, checkpoints, and training inputs. This is compute-intensive and must preserve the frozen split, seeds, sampler, optimizer budget, and held-out-test rules.

GitHub alone cannot reproduce the scientific results because the private archive is approximately 198.772 GiB and includes files that must not be redistributed.

## Environment

Python 3.12 was used for the curated-code validation. From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Install `requirements-geospatial.txt` before running upstream/static spatial workflows, and `requirements-reporting.txt` before rebuilding Word or PDF outputs. The requirements record supported minimum versions rather than a byte-for-byte historical environment lock; preserve a working environment separately when exact numerical reproduction is required.

## Restore the private archive

Assume the cloned repository is `<repo>` and the archive root is `<archive>`.

| Archive location | Restore location | Purpose |
|---|---|---|
| `data/clean/` | `<repo>/data/clean/` | Harmonised site observations |
| `data/processed/` | `<repo>/data/processed/` | Reusable derived inputs |
| `data/training/` | `<repo>/data/training/` | Reusable training inputs |
| `data/external/` | `<repo>/data/external/` | External/upstream source material |
| `data/reference_project/` | `<repo>/data/reference/` | Project registry and regulatory references |
| `data/reference_teacher/` | `<repo>/data/reference_teacher/` | Read-only reference snapshot |
| `data/teacher_raw/` | `<repo>/data/teacher_raw/` | Read-only archived source observations |
| `evidence/<name>/` | `<repo>/analysis_outputs/<name>/` | Frozen outputs and audit evidence |

Do not copy the archive root itself inside the repository. Copy the contents according to the table so that paths do not become `data/data/...` or `analysis_outputs/evidence/...`.

Use these archive manifests after copying:

- `manifests/data_selection.csv`: authoritative source-to-archive mapping.
- `manifests/data_copy_verification.csv`: data group file counts and byte totals.
- `manifests/evidence_selection.txt`: selected evidence directories.
- `manifests/evidence_copy_verification.csv`: source/destination counts and byte totals.
- `manifests/code_sha256.csv` and `report_sha256.csv`: compact-file SHA-256 hashes.
- `manifests/archive_sections.csv`: final section sizes.

Historical audit tables may record paths from the original workstation. Treat those as provenance strings; executable code in this repository resolves its root from the script location. If a script reads a path stored inside an old CSV or JSON, rebase that field to the restored repository before running it.

## Validation before scientific execution

```powershell
python -m unittest discover -s tests
python -m unittest discover -s pipeline -p "test_*.py"
```

Run from the repository root. Verify that the required input directories exist, retain the original frozen outputs, and direct experimental reruns to a new versioned output directory whenever the script supports it.

## Main execution chain

The shortest retained chain is:

1. `pipeline/00_common1000`: canonical cohort processing when rebuilding from clean observations.
2. `pipeline/31_common1000_rainfall_ablation` through the frozen H00 checkpoint and recurrent benchmark stages.
3. `pipeline/42_h0_site_transfer_final`: frozen low-cost warm-up audit and finalisation.
4. `pipeline/44_cross_variable_hot_moment_incremental_closure`: matched residual/event closure.
5. `pipeline/45_ph_h00_30day_calibration`: retained calibration comparison.
6. `pipeline/50_ph_h00_reproducibility_audit`: checkpoint and training-path reproducibility audit.

Each of stages 42, 44, 45, and 50 has a local README with exact commands and prerequisites. Earlier or negative experiments remain documented but are not part of the shortest final chain.

## Held-out and overwrite rules

- Do not use test data for model selection, threshold selection, or debugging.
- Do not rerun a one-time held-out test merely to improve a result.
- Do not overwrite a frozen evidence directory in place. Preserve it and use a new output version for exploratory work.
- A retained directory is evidence, not proof that its candidate model was promoted.
- Keep raw permit extracts and other restricted source material out of GitHub.

## Expected limitations

External APIs and source portals can change. Re-downloading from scratch may not return byte-identical raw files. Frozen-result verification should therefore use the private archive; downloaders are retained for future acquisition, not as a guarantee that third-party services remain unchanged.
