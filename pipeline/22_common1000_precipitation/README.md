# Available precipitation and flow ablation

The register points to a missing external rainfall batch, but some local raw
files carry real `00045`. This stage performs a strictly common-site/common-key
screening on sites where both `00045` and `00060` vary. It compares core,
core+rainfall, core+flow and core+flow+rainfall with Ridge and HGB.

It is intentionally labelled an available-subset result; it does not claim that
the absent register-linked rainfall batch was processed.
