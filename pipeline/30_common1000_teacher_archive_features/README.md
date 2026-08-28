# Common1000 pH plus teacher-Archive rainfall features

This stage creates non-destructive, model-ready pH files containing:

- `00045`: hourly precipitation in mm/hour;
- `precip_api`: the teacher Archive normalized EWMA with a 48-hour half-life;
- `precip_6h`: the teacher Archive rolling six-hour rainfall sum;
- explicit observation and missingness flags.

Two scientific cohorts are retained:

1. **Primary Archive cohort** — 910 pH sites directly matched to the downloaded
   canonical precipitation archive.
2. **Extended cohort** — the primary cohort plus 66 sites with a varying local
   USGS `00045` series. These legacy values are hourly sums in inches converted
   to millimetres using 25.4 mm/inch and are explicitly marked as supplemental.

The remaining 33 pH sites are listed with exclusion reasons. Existing pH,
rainfall, model, split and prediction files are never overwritten.
