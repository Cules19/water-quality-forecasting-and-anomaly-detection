#!/usr/bin/env python3
"""
Regulatory Threshold Event Detector
=====================================
Flags Events of Interest (EoI) in high-frequency water quality timeseries
against EPA / EA absolute regulatory thresholds.

Produces a tidy events DataFrame:
    site_id | parameter | label | severity | incident_start | incident_end | duration_hrs | n_points

USGS parameter codes supported:
    00300  Dissolved Oxygen (mg/L)         — inverted threshold (low = bad)
    00400  pH (standard units)
    00618  Nitrate (mg/L as N)
    00665  Phosphorus (mg/L as P)
    32211  Chlorophyll-a (µg/L)

E. coli (31648 / 61218) is discrete / lab-grab; pass as column if available.

References:
    DO     — EPA Table 1 (life-stage dependent; stress proxy used here)
    pH     — EPA: 6.5–9.5 acceptable range
    Chl-a  — Spaulding et al. trophic tiers
    Nitrate— EPA NRSA Table 7.1
    Phos   — EPA NRSA Table 7.1
    E. coli— EPA recreational criteria (30-day / single-sample)

Author: TSLR
Date: 2026-06-23
"""

import pandas as pd
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Threshold configuration
# Each parameter entry: list of dicts, checked in order (most severe first).
# "direction": "low"  → exceedance when value < threshold
#              "high" → exceedance when value > threshold
#              "range"→ exceedance when value outside [low, high]
# ---------------------------------------------------------------------------

REGULATORY_THRESHOLDS: dict = {
    "00300": {  # Dissolved Oxygen, mg/L
        "name": "Dissolved Oxygen",
        "units": "mg/L",
        "tiers": [
            {
                "label": "DO_critical",
                "severity": "critical",
                "direction": "low",
                "threshold": 3.0,
                "description": "Severe hypoxia — acute stress risk",
            },
            {
                "label": "DO_warning",
                "severity": "warning",
                "direction": "low",
                "threshold": 5.0,
                "description": "Sub-optimal — stress risk for sensitive species",
            },
        ],
    },
    "00400": {  # pH
        "name": "pH",
        "units": "s.u.",
        "tiers": [
            {
                "label": "pH_exceedance_high",
                "severity": "warning",
                "direction": "high",
                "threshold": 9.5,
                "description": "pH above EPA upper bound (9.5)",
            },
            {
                "label": "pH_exceedance_low",
                "severity": "warning",
                "direction": "low",
                "threshold": 6.5,
                "description": "pH below EPA lower bound (6.5)",
            },
        ],
    },
    "00618": {  # Nitrate as N, mg/L  — EPA NRSA ecoregion benchmarks vary;
        "name": "Nitrate",  #   using a general concern threshold pending site-specific lookup
        "units": "mg/L as N",
        "tiers": [
            {
                "label": "nitrate_high",
                "severity": "warning",
                "direction": "high",
                "threshold": 2.0,
                "description": "Elevated nitrate — exceeds typical reference-condition benchmark",
            },
        ],
    },
    "00665": {  # Total Phosphorus as P, mg/L — EPA NRSA
        "name": "Phosphorus",
        "units": "mg/L as P",
        "tiers": [
            {
                "label": "phosphorus_high",
                "severity": "warning",
                "direction": "high",
                "threshold": 0.1,
                "description": "Elevated phosphorus — exceeds EPA eutrophication concern level",
            },
        ],
    },
    "32211": {  # Chlorophyll-a, µg/L — Spaulding et al.
        "name": "Chlorophyll-a",
        "units": "µg/L",
        "tiers": [
            {
                "label": "chl_bloom",
                "severity": "critical",
                "direction": "high",
                "threshold": 50.0,
                "description": "High trophic state — bloom risk (Spaulding >50 µg/L)",
            },
            {
                "label": "chl_moderate",
                "severity": "warning",
                "direction": "high",
                "threshold": 25.0,
                "description": "Moderate trophic state (Spaulding 25–50 µg/L)",
            },
            {
                "label": "chl_low",
                "severity": "info",
                "direction": "high",
                "threshold": 10.0,
                "description": "Low-moderate trophic state (Spaulding 10–25 µg/L)",
            },
        ],
    },
    "31648": {  # E. coli, cfu/100 mL — EPA recreational criteria (single-sample)
        "name": "E. coli",
        "units": "cfu/100 mL",
        "tiers": [
            {
                "label": "ecoli_beach_closure",
                "severity": "critical",
                "direction": "high",
                "threshold": 575,
                "description": "Single-sample — beach closure threshold (EPA 575 cfu/100 mL)",
            },
            {
                "label": "ecoli_high",
                "severity": "warning",
                "direction": "high",
                "threshold": 235,
                "description": "Single-sample — elevated recreational risk (EPA 235 cfu/100 mL)",
            },
        ],
    },
}

SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}


def _check_exceedance(value: float, tier: dict) -> bool:
    """Return True if value exceeds the threshold in the specified direction."""
    if pd.isna(value):
        return False
    direction = tier["direction"]
    thresh = tier["threshold"]
    if direction == "low":
        return value < thresh
    if direction == "high":
        return value > thresh
    return False


def detect_regulatory_events(
    df: pd.DataFrame,
    site_id: str,
    timestamp_col: str = "TIMESTAMP",
    min_duration_steps: int = 2,
    gap_tolerance_steps: int = 4,
) -> pd.DataFrame:
    """
    Detect regulatory threshold exceedances in a single-site timeseries.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned timeseries. Must contain `timestamp_col` and at least one
        recognised USGS parameter code as a column.
    site_id : str
        Site identifier written into the output.
    timestamp_col : str
        Name of the datetime column (default 'TIMESTAMP').
    min_duration_steps : int
        Minimum consecutive flagged rows to constitute an event (default 1).
    gap_tolerance_steps : int
        Number of non-flagged rows allowed inside an event before splitting it
        (bridging short data gaps, default 0 = no bridging).

    Returns
    -------
    pd.DataFrame with columns:
        site_id, parameter, label, severity, description,
        incident_start, incident_end, duration_hrs, n_points
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    events = []

    present_params = [c for c in REGULATORY_THRESHOLDS if c in df.columns]

    for param_code in present_params:
        cfg = REGULATORY_THRESHOLDS[param_code]
        series = df[param_code]

        # For each row, find the highest-severity tier that fires
        row_label = pd.Series("", index=df.index, dtype=str)
        row_severity = pd.Series("", index=df.index, dtype=str)
        row_rank = pd.Series(0, index=df.index, dtype=int)
        row_desc = pd.Series("", index=df.index, dtype=str)

        for tier in cfg["tiers"]:
            mask = series.apply(lambda v: _check_exceedance(v, tier))
            tier_rank = SEVERITY_RANK.get(tier["severity"], 0)
            upgrade = mask & (tier_rank > row_rank)
            row_label[upgrade] = tier["label"]
            row_severity[upgrade] = tier["severity"]
            row_rank[upgrade] = tier_rank
            row_desc[upgrade] = tier["description"]

        flagged = row_rank > 0

        # Group consecutive flagged rows into incidents (with optional gap bridging)
        events += _group_into_incidents(
            df[timestamp_col],
            flagged,
            row_label,
            row_severity,
            row_desc,
            site_id,
            param_code,
            cfg["name"],
            min_duration_steps,
            gap_tolerance_steps,
        )

    if not events:
        return pd.DataFrame(
            columns=[
                "site_id",
                "parameter",
                "label",
                "severity",
                "description",
                "incident_start",
                "incident_end",
                "duration_hrs",
                "n_points",
            ]
        )

    return pd.DataFrame(events).sort_values("incident_start").reset_index(drop=True)


def _group_into_incidents(
    timestamps,
    flagged,
    labels,
    severities,
    descriptions,
    site_id,
    param_code,
    param_name,
    min_duration_steps,
    gap_tolerance_steps,
):
    """Convert a boolean mask into discrete incident records."""
    incidents = []
    in_event = False
    gap_count = 0
    start_idx = None
    event_indices = []

    for i in range(len(flagged)):
        if flagged.iloc[i]:
            if not in_event:
                in_event = True
                start_idx = i
                event_indices = []
            gap_count = 0
            event_indices.append(i)
        else:
            if in_event:
                gap_count += 1
                if gap_count > gap_tolerance_steps:
                    # Close event
                    incident = _build_incident(
                        timestamps,
                        labels,
                        severities,
                        descriptions,
                        event_indices,
                        site_id,
                        param_code,
                        param_name,
                        min_duration_steps,
                    )
                    if incident:
                        incidents.append(incident)
                    in_event = False
                    gap_count = 0
                    event_indices = []

    # Close any trailing event
    if in_event and event_indices:
        incident = _build_incident(
            timestamps,
            labels,
            severities,
            descriptions,
            event_indices,
            site_id,
            param_code,
            param_name,
            min_duration_steps,
        )
        if incident:
            incidents.append(incident)

    return incidents


def _build_incident(
    timestamps,
    labels,
    severities,
    descriptions,
    indices,
    site_id,
    param_code,
    param_name,
    min_duration_steps,
):
    if len(indices) < min_duration_steps:
        return None

    idx = indices
    t_start = timestamps.iloc[idx[0]]
    t_end = timestamps.iloc[idx[-1]]
    duration_hrs = (t_end - t_start).total_seconds() / 3600

    # Dominant label = highest-severity label at most time steps
    event_labels = labels.iloc[idx]
    label_counts = event_labels.value_counts()
    dominant_label = label_counts.index[0]
    dominant_sev = severities.iloc[idx[0]]
    dominant_desc = descriptions.iloc[idx[0]]

    return {
        "site_id": site_id,
        "parameter": f"{param_name} ({param_code})",
        "label": dominant_label,
        "severity": dominant_sev,
        "description": dominant_desc,
        "incident_start": t_start,
        "incident_end": t_end,
        "duration_hrs": round(duration_hrs, 2),
        "n_points": len(idx),
    }


def run_on_file(
    filepath: str,
    site_id: Optional[str] = None,
    timestamp_col: str = "TIMESTAMP",
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: load a clean CSV and return the events DataFrame.

    Parameters
    ----------
    filepath : str
        Path to a clean CSV file.
    site_id : str, optional
        Overrides site_id; if None, inferred from filename stem.
    """
    from pathlib import Path

    path = Path(filepath)
    if site_id is None:
        site_id = path.stem.split("_")[0]

    df = pd.read_csv(filepath)
    if timestamp_col not in df.columns and "timestamp" in df.columns:
        timestamp_col = "timestamp"

    return detect_regulatory_events(
        df, site_id=site_id, timestamp_col=timestamp_col, **kwargs
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Detect EPA/EA regulatory threshold events in a clean CSV."
    )
    parser.add_argument("filepath", help="Path to clean water quality CSV")
    parser.add_argument(
        "--site-id", default=None, help="Site identifier (default: infer from filename)"
    )
    parser.add_argument(
        "--timestamp-col", default="TIMESTAMP", help="Timestamp column name"
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Min consecutive flagged steps for an event (default 1)",
    )
    parser.add_argument(
        "--gap-tolerance",
        type=int,
        default=0,
        help="Non-flagged steps tolerated inside an event (default 0)",
    )
    parser.add_argument("--out", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    events = run_on_file(
        args.filepath,
        site_id=args.site_id,
        timestamp_col=args.timestamp_col,
        min_duration_steps=args.min_steps,
        gap_tolerance_steps=args.gap_tolerance,
    )

    if events.empty:
        print("No regulatory threshold events detected.")
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"  Regulatory Events of Interest — {len(events)} incident(s) found")
    print(f"{'='*70}\n")
    print(events.to_string(index=False))

    if args.out:
        events.to_csv(args.out, index=False)
        print(f"\n✓ Events written to {args.out}")
