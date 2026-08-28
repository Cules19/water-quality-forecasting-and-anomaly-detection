# Methodology figure QA

Status: PASS
Audit date: 2026-08-11

## Scope

Exactly two figures were generated:

- `Figure_1_study_design` in PDF, SVG and 600 dpi PNG.
- `Figure_2_information_design` in PDF, SVG and 600 dpi PNG.

No third figure, results panel, manuscript caption, accessibility text or additional model analysis was produced.

## Source and identity checks

- Final primary cohort: 910 sites; EA 696, USGS 158 and NEON 56; fixed split 638 train, 137 validation and 135 test. Verified from `primary_archive_experiment_cohort.csv` before drawing.
- Exact retained model checkpoints exist for `ph_LSTM_RAIN_H0_H00`, `ph_LSTM_RAIN_FINAL_H01`, `ph_LSTM_LOW_COST_H01` and `do_LSTM_LOW_COST_H01`.
- H00 identity: pH code 00400, 48-hour input, horizon 0 and target history excluded.
- Historical-target H01 identity: pH code 00400, 48-hour input, horizon 1 and earlier pH present in the verified historical feature order.
- Low-cost H01 identity: 48-hour input, horizon 1 and target history excluded for both pH and DO; final DO target unit is mg/L.
- H00 local-support denominator: 128 validation and 126 test sites; the 30-day offset is explicitly shown as conditional on local target support.
- Conformal split: 45 calibration and 90 evaluation sites per target.
- Flow context: 61 sites and 39 groups, labelled training-only available-case; Stage is not shown.
- Warm-up labels exactly match the frozen protocol: 2 weeks, 1, 2, 3, 6, 9 and 12 months; the fixed query begins after 12 months and the 12-month test estimate is labelled descriptive.
- Familiar-site temporal design exactly matches the local split protocol: history through `Ti-90 d`, isolated 30-day support and final 60-day hold-out.

The source script performs these identity assertions before it writes any figure.

## Export and editability checks

| Check | Figure 1 | Figure 2 |
|---|---:|---:|
| Page size | 183 x 120 mm | 183 x 130 mm |
| PDF pages | 1 | 1 |
| PNG pixels | 4322 x 2834 | 4322 x 3070 |
| PNG resolution metadata | 599.9988 dpi | 599.9988 dpi |
| Extractable PDF text | 1656 characters | 1127 characters |
| Embedded PDF fonts | Arial regular and bold, Type 0 with embedded FontFile2 | Arial regular and bold, Type 0 with embedded FontFile2 |
| Raster image XObjects in PDF | 0 | 0 |
| Editable SVG text nodes | 61 | 61 |

Both SVG files parse successfully. Both PDFs contain vector drawing objects and selectable text rather than a flattened page image. Visible type is 5-7 pt except for bold 8 pt panel labels. Visible strokes are 0.5-0.8 pt.

## Visual checks

- Both final PDFs were rendered with Poppler at 200 dpi and visually inspected.
- No clipping, text collision, broken arrow, unintended overflow or blank page was found.
- Panel labels, target symbols, inference cutoffs, permitted/prohibited input markers and support/query boundaries are consistent across panels.
- Support and query periods do not overlap in either evaluation schematic.
- Figure 1 distinguishes upstream sensor QC from later residual/event analysis.
- Figure 1 uses task-specific denominators and does not present 910 as a universal model denominator.
- Neither figure contains a performance metric, comparative result, model ranking or conclusion claim.

## Caption draft

`METHODOLOGY_FINAL_DRAFT.md` was not found anywhere in the project and therefore was not used. No replacement caption was authored because captions were outside the requested deliverables.
