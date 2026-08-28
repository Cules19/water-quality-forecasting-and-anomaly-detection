from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/supervisor_frozen_quick_wins_v1"
APPENDIX = OUT / "appendix"

CLOSURE = ROOT / "analysis_outputs/cross_variable_hot_moment_incremental_closure_v1"
CONFORMAL = ROOT / "analysis_outputs/common1000_lowcost_conformal_v1"
DO_ROOT = ROOT / "analysis_outputs/do_common1000_lstm_film_sensor_ablation_mg_l_v1/full"
PH_LOWCOST_ROOT = ROOT / "analysis_outputs/ph_common1000_lstm_film_ph_sensor_ablation_v1/full"
PREREG = APPENDIX / "LOW_DO_THRESHOLD_PREREGISTRATION.md"

SEED = 73129
BOOTSTRAPS = 10_000
PERSISTENCE_HOURS = 3
EPISODE_GAP_HOURS = 1.5
LEAD_WINDOW_HOURS = 48
LOW_DO_THRESHOLD_MG_L = 2.0
LOW_DO_CONFORMAL_NOMINAL = 0.90
LOW_DO_MIN_SITES = 10
LOW_DO_MIN_EVENTS = 30
LOW_DO_MIN_BOOTSTRAP_SITES = 10

MODEL_INFO = {
    "ph_h00_30day_offset": {
        "short": "cross",
        "display": "pH H00 (30-day offset)",
        "historical_target": False,
        "horizon_hours": 0,
    },
    "ph_historical_target_h01": {
        "short": "hist",
        "display": "Historical-target pH H01",
        "historical_target": True,
        "horizon_hours": 1,
    },
}

EXPECTED_HASHES = {
    "matched_residuals": (
        CLOSURE / "residual_scores/matched_residuals.parquet",
        "b3b2c9ece5c24d80c3e03a34add19cd6d8cc10f91404dda2d940f22ec89717ec",
    ),
    "matched_event_manifest": (
        CLOSURE / "data/matched_event_manifest.csv",
        "26bb04c7f3b97e2d9b80234c3b6a6667dcbbc60a6b32715e2999aed8c30b42dd",
    ),
    "label_free_thresholds": (
        CLOSURE / "calibration/label_free_thresholds.json",
        "5b2842c8698a0980d9b1afd34b41469b323e03ddb8ac61a5a3ec585304e54750",
    ),
    "model_manifest": (
        CLOSURE / "configs/model_manifest.json",
        "74171b25130891ad3acae1147bc731734ab375e1df969c8301e77892da71a028",
    ),
    "saved_event_enrichment": (
        CLOSURE / "event_metrics/event_enrichment.csv",
        "39fa02e9aa6b5b9d07b5cc96baef791dd13e4e92c97194c0b4e646e450507a9a",
    ),
    "saved_overlap_categories": (
        CLOSURE / "event_metrics/residual_overlap_categories.csv",
        "88f78b069c243fcaf3352f0a3345559b0e4142ac43b467f83ce8450ddb5724a2",
    ),
    "conformal_config": (
        ROOT / "config/common1000_lowcost_conformal_v1.json",
        "b4c015f282081b22f2f35d94290ebdbe2ba23999afadb42fa986e8d084868d8b",
    ),
    "conformal_summary": (
        CONFORMAL / "reports/conformal_summary.csv",
        "04bb10c378fd1f6d601ac78a6ebec7bd4bc5f28ffc720df42f7bee9766dd8cd7",
    ),
    "conformal_site_split": (
        CONFORMAL / "reports/conformal_site_split.csv",
        "2ccb271b5dbfa34a920f15872c484f0ec39b84528cc71befff41f1faa1dd9120",
    ),
    "conformal_site_coverage": (
        CONFORMAL / "reports/conformal_site_coverage.csv",
        "33fbf3ef423f9f368cdc4e2ee489f395121c9e0538cb3287e43c8075b475c90e",
    ),
    "ph_lowcost_config": (
        PH_LOWCOST_ROOT / "reports/effective_config.json",
        "9197d5cf97849e307f3ac205921ed4f410cb115be9cd0e081d3c2a553abd4e5a",
    ),
    "do_native_config": (
        DO_ROOT / "reports/effective_config.json",
        "5a90f2d4cdc8c7789740fc31ea98836e1d37bbfbafb7720fade2067839331f15",
    ),
    "do_unit_audit": (
        DO_ROOT / "reports/CHECKPOINT_UNIT_RESCALING_AUDIT.md",
        "f6460f2b529b422a41f3c261b84c7d371c67f01486ae2c007f1096d23978a7be",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task0-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def seeded_rng(key: str) -> np.random.Generator:
    seed = int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_frozen_test() -> pd.DataFrame:
    columns = [
        "site_id", "target_timestamp", "split", "observed_ph", "predicted_cross",
        "predicted_hist", "in_event", "event_direction", "evaluation_period",
        "signed_score_cross", "absolute_score_cross", "signed_score_hist",
        "absolute_score_hist", "top5_cross", "top5_hist", "positive_tail_cross",
        "negative_tail_cross", "positive_tail_hist", "negative_tail_hist",
    ]
    frame = pd.read_parquet(
        EXPECTED_HASHES["matched_residuals"][0], columns=columns,
        filters=[("split", "==", "test")],
    )
    frame = frame.loc[frame["evaluation_period"]].copy()
    frame["site_id"] = frame["site_id"].astype(str)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"])
    return frame.sort_values(["site_id", "target_timestamp"], kind="mergesort").reset_index(drop=True)


def pooled_enrichment(frame: pd.DataFrame, event_mask: pd.Series, alert: pd.Series) -> dict[str, Any]:
    event_mask = event_mask.to_numpy(dtype=bool)
    alert_array = alert.to_numpy(dtype=bool)
    other_mask = ~frame["in_event"].to_numpy(dtype=bool)
    event_alerts = int(alert_array[event_mask].sum())
    event_rows = int(event_mask.sum())
    other_alerts = int(alert_array[other_mask].sum())
    other_rows = int(other_mask.sum())
    event_rate = event_alerts / event_rows if event_rows else np.nan
    other_rate = other_alerts / other_rows if other_rows else np.nan
    enrichment = event_rate / other_rate if other_rate > 0 else np.nan

    site_rows = []
    for site_id, site in frame.groupby("site_id", sort=False):
        site_event = event_mask[site.index]
        site_other = ~site["in_event"].to_numpy(dtype=bool)
        if not site_event.sum() or not site_other.sum():
            continue
        site_alert = alert_array[site.index]
        site_rows.append(
            [int(site_alert[site_event].sum()), int(site_event.sum()), int(site_alert[site_other].sum()), int(site_other.sum())]
        )
    values = np.asarray(site_rows, dtype=float)
    if len(values):
        rng = seeded_rng(f"directional|{alert.name}|{event_rows}")
        sample = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        selected = values[sample].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = (selected[:, 0] / selected[:, 1]) / (selected[:, 2] / selected[:, 3])
        ci_low, ci_high = np.nanquantile(boot, [0.025, 0.975])
    else:
        ci_low = ci_high = np.nan
    return {
        "event_alert_hours": event_alerts,
        "event_hours": event_rows,
        "background_alert_hours": other_alerts,
        "background_hours": other_rows,
        "event_alert_rate": event_rate,
        "background_alert_rate": other_rate,
        "enrichment": enrichment,
        "site_bootstrap_ci_low": float(ci_low),
        "site_bootstrap_ci_high": float(ci_high),
        "bootstrap_site_count": int(len(values)),
        "bootstrap_replicates": BOOTSTRAPS,
    }


def build_episodes(frame: pd.DataFrame, alert_column: str, label: str) -> pd.DataFrame:
    episodes: list[dict[str, Any]] = []
    for site_id, site in frame.groupby("site_id", sort=False):
        site = site.sort_values("target_timestamp", kind="mergesort")
        hot = site.loc[site[alert_column].astype(bool), ["target_timestamp"]].copy()
        if hot.empty:
            continue
        hot["episode_id"] = hot["target_timestamp"].diff().gt(pd.Timedelta(hours=EPISODE_GAP_HOURS)).fillna(True).cumsum()
        for _, episode in hot.groupby("episode_id", sort=False):
            if len(episode) < PERSISTENCE_HOURS:
                continue
            start = episode["target_timestamp"].min()
            end = episode["target_timestamp"].max()
            episodes.append(
                {
                    "analysis": label,
                    "site_id": str(site_id),
                    "episode_start": start,
                    "episode_end": end,
                    "consecutive_flagged_hours": int(len(episode)),
                    "wall_duration_hours": float((end - start).total_seconds() / 3600.0 + 1.0),
                }
            )
    return pd.DataFrame(episodes)


def build_boolean_windows(frame: pd.DataFrame, active_column: str, direction: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for site_id, site in frame.groupby("site_id", sort=False):
        site = site.sort_values("target_timestamp", kind="mergesort")
        active = site.loc[site[active_column].astype(bool), ["target_timestamp"]].copy()
        if active.empty:
            continue
        active["window_number"] = active["target_timestamp"].diff().gt(pd.Timedelta(hours=EPISODE_GAP_HOURS)).fillna(True).cumsum()
        for number, window in active.groupby("window_number", sort=False):
            start = window["target_timestamp"].min()
            end = window["target_timestamp"].max()
            rows.append(
                {
                    "site_id": str(site_id),
                    "event_window_id": f"{site_id}:{start.isoformat()}",
                    "event_start": start,
                    "event_end": end,
                    "event_rows": int(len(window)),
                    "event_direction": direction,
                }
            )
    return pd.DataFrame(rows)


def event_detection_details(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    alert_column: str,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    by_site = {str(site): group.sort_values("target_timestamp", kind="mergesort") for site, group in frame.groupby("site_id", sort=False)}
    episode_by_site = {
        str(site): group.sort_values("episode_start", kind="mergesort")
        for site, group in episodes.groupby("site_id", sort=False)
    } if len(episodes) else {}
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        site = by_site[str(event.site_id)]
        inside = site.loc[
            site["target_timestamp"].ge(event.event_start)
            & site["target_timestamp"].le(event.event_end)
            & site["in_event"].astype(bool)
        ]
        inside_any = bool(inside[alert_column].astype(bool).any()) if len(inside) else False
        site_episodes = episode_by_site.get(str(event.site_id), pd.DataFrame())
        episode_overlap = bool(
            ((site_episodes["episode_start"] <= event.event_end) & (site_episodes["episode_end"] >= event.event_start)).any()
        ) if len(site_episodes) else False

        lower = event.event_start - pd.Timedelta(hours=LEAD_WINDOW_HOURS)
        pre = site.loc[
            site["target_timestamp"].ge(lower)
            & site["target_timestamp"].lt(event.event_start)
            & ~site["in_event"].astype(bool)
        ].sort_values("target_timestamp", kind="mergesort")
        hot = pre.loc[pre[alert_column].astype(bool), ["target_timestamp"]].copy()
        qualifying = []
        if len(hot):
            hot["episode_id"] = hot["target_timestamp"].diff().gt(pd.Timedelta(hours=EPISODE_GAP_HOURS)).fillna(True).cumsum()
            for _, episode in hot.groupby("episode_id", sort=False):
                if len(episode) >= PERSISTENCE_HOURS:
                    qualifying.append((episode["target_timestamp"].min(), episode["target_timestamp"].max(), len(episode)))
        selected = sorted(qualifying, key=lambda item: item[1])[-1] if qualifying else None
        lead = (event.event_start - selected[0]).total_seconds() / 3600.0 if selected else np.nan
        rows.append(
            {
                "site_id": str(event.site_id),
                "event_window_id": str(event.event_window_id),
                "event_direction": str(event.event_direction),
                "event_start": event.event_start,
                "event_end": event.event_end,
                "event_rows": int(event.event_rows),
                "inside_row_alert": inside_any,
                "persistent_episode_overlap": episode_overlap,
                "pre_event_rows": int(len(pre)),
                "lead_eligible": bool(len(pre) > 0),
                "pre_event_persistent_alert": bool(selected is not None),
                "lead_hours": float(lead) if np.isfinite(lead) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def outside_burden(frame: pd.DataFrame, episodes: pd.DataFrame, all_events: pd.DataFrame) -> dict[str, Any]:
    starts = {
        str(site): pd.to_datetime(group["event_start"]).sort_values()
        for site, group in all_events.groupby("site_id", sort=False)
    }
    outside = 0
    if len(episodes):
        outside_flags = []
        for episode in episodes.itertuples(index=False):
            onset = starts.get(str(episode.site_id), pd.Series([], dtype="datetime64[ns]"))
            overlaps = bool(
                ((onset - pd.Timedelta(hours=48) <= episode.episode_end)
                 & (onset + pd.Timedelta(hours=24) >= episode.episode_start)).any()
            ) if len(onset) else False
            outside_flags.append(not overlaps)
        outside = int(sum(outside_flags))
    site_months = float(len(frame) / (24 * 30.4375))
    return {
        "alert_episode_count": int(len(episodes)),
        "alert_episodes_per_site_month": float(len(episodes) / site_months) if site_months else np.nan,
        "outside_register_episodes": outside,
        "evaluation_site_months": site_months,
        "outside_register_episodes_per_site_month": float(outside / site_months) if site_months else np.nan,
    }


def summarize_detection(details: pd.DataFrame) -> dict[str, Any]:
    eligible = details["lead_eligible"].astype(bool)
    lead_alerted = details.loc[eligible & details["pre_event_persistent_alert"].astype(bool), "lead_hours"].dropna()
    return {
        "in_window_alerted_events": int(details["inside_row_alert"].sum()),
        "in_window_alerted_event_fraction": float(details["inside_row_alert"].mean()) if len(details) else np.nan,
        "persistent_episode_detected_events": int(details["persistent_episode_overlap"].sum()),
        "persistent_episode_detection_fraction": float(details["persistent_episode_overlap"].mean()) if len(details) else np.nan,
        "lead_eligible_events": int(eligible.sum()),
        "pre_event_alerted_events": int(details.loc[eligible, "pre_event_persistent_alert"].sum()),
        "pre_event_alerted_fraction_eligible": float(details.loc[eligible, "pre_event_persistent_alert"].mean()) if eligible.any() else np.nan,
        "median_lead_hours": float(lead_alerted.median()) if len(lead_alerted) else np.nan,
        "lead_hours_q25": float(lead_alerted.quantile(0.25)) if len(lead_alerted) else np.nan,
        "lead_hours_q75": float(lead_alerted.quantile(0.75)) if len(lead_alerted) else np.nan,
    }


def verify_hashes() -> pd.DataFrame:
    rows = []
    failures = []
    for role, (path, expected) in EXPECTED_HASHES.items():
        actual = sha256(path)
        passed = actual == expected
        rows.append({"role": role, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "size_bytes": path.stat().st_size, "expected_sha256": expected, "actual_sha256": actual, "pass": passed})
        if not passed:
            failures.append(role)
    if failures:
        raise RuntimeError(f"FROZEN_STARTING_POINT_REPRODUCTION_FAILED: hash mismatch {failures}")
    return pd.DataFrame(rows)


def task0_reproduction(test: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = load_json(EXPECTED_HASHES["model_manifest"][0])
    thresholds = load_json(EXPECTED_HASHES["label_free_thresholds"][0])
    cross = manifest["cross_variable_h00"]
    hist = manifest["historical_target_h01"]
    identity_checks = {
        "cross_target": cross["target"] == "pH (00400)",
        "cross_horizon_0": cross["forecast_horizon_hours"] == 0,
        "cross_no_historical_ph": cross["historical_ph_present"] is False,
        "cross_checkpoint_hash": cross["checkpoint_sha256"] == "2781f6440c122d591b3264d7e9ac8115329d4f4d3b0f15b77dc9db8a17a69afb",
        "hist_target": hist["target"] == "pH (00400)",
        "hist_horizon_1": hist["forecast_horizon_hours"] == 1,
        "hist_uses_historical_ph": hist["historical_ph_present"] is True,
        "hist_checkpoint_hash": hist["checkpoint_sha256"] == "558a559bf196c00f8d8de3ee15656e0e5cd13b450320509347395be441ada8c1",
        "thresholds_validation_only": all(value["fit_split"] == "validation" for value in thresholds["models"].values()),
        "thresholds_event_label_free": thresholds["event_labels_used"] is False,
        "persistence_3h": thresholds["persistence_consecutive_hours"] == 3,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"FROZEN_STARTING_POINT_REPRODUCTION_FAILED: identity {identity_checks}")

    duplicate_count = int(test.duplicated(["site_id", "target_timestamp"]).sum())
    cross_flag = test["top5_cross"].astype(bool)
    hist_flag = test["top5_hist"].astype(bool)
    both = int((cross_flag & hist_flag).sum())
    union = int((cross_flag | hist_flag).sum())
    counts = {
        "evaluation_rows": int(len(test)),
        "sites": int(test["site_id"].nunique()),
        "duplicate_site_target_timestamp_keys": duplicate_count,
        "h00_only_top5_rows": int((cross_flag & ~hist_flag).sum()),
        "historical_h01_only_top5_rows": int((~cross_flag & hist_flag).sum()),
        "shared_top5_rows": both,
        "union_top5_rows": union,
        "jaccard": float(both / union),
        "spearman_absolute_score": float(test[["absolute_score_cross", "absolute_score_hist"]].corr(method="spearman").iloc[0, 1]),
    }
    expected_counts = {
        "evaluation_rows": 1_493_309,
        "sites": 126,
        "duplicate_site_target_timestamp_keys": 0,
        "h00_only_top5_rows": 132_097,
        "historical_h01_only_top5_rows": 47_993,
        "shared_top5_rows": 12_967,
        "union_top5_rows": 193_057,
    }
    if any(counts[key] != value for key, value in expected_counts.items()):
        raise RuntimeError(f"FROZEN_STARTING_POINT_REPRODUCTION_FAILED: counts {counts}")
    if not np.isclose(counts["jaccard"], 0.06716669170049156, atol=1e-15) or not np.isclose(counts["spearman_absolute_score"], 0.1026145445, atol=1e-10):
        raise RuntimeError(f"FROZEN_STARTING_POINT_REPRODUCTION_FAILED: overlap metrics {counts}")

    saved_enrichment = pd.read_csv(EXPECTED_HASHES["saved_event_enrichment"][0])
    signed_rows = []
    for canonical_model, info in MODEL_INFO.items():
        source_model = "cross_variable_h00" if info["short"] == "cross" else "historical_target_h01"
        for direction in ["positive", "negative"]:
            alert_col = f"{direction}_tail_{info['short']}"
            event_mask = test["in_event"].astype(bool)
            calculated = pooled_enrichment(test, event_mask, test[alert_col].rename(alert_col))
            saved = saved_enrichment.loc[
                saved_enrichment["model"].eq(source_model)
                & saved_enrichment["score_definition"].eq(f"{direction}_tail")
            ].iloc[0]
            passed = all(
                np.isclose(calculated[key], float(saved[saved_key]), atol=1e-12, rtol=1e-12)
                for key, saved_key in [
                    ("event_alert_rate", "pooled_event_high_rate"),
                    ("background_alert_rate", "pooled_other_high_rate"),
                    ("enrichment", "pooled_enrichment"),
                ]
            )
            signed_rows.append(
                {
                    "model": canonical_model,
                    "alert_direction": direction,
                    "reproduced_event_alert_rate": calculated["event_alert_rate"],
                    "saved_event_alert_rate": float(saved["pooled_event_high_rate"]),
                    "reproduced_background_alert_rate": calculated["background_alert_rate"],
                    "saved_background_alert_rate": float(saved["pooled_other_high_rate"]),
                    "reproduced_enrichment": calculated["enrichment"],
                    "saved_enrichment": float(saved["pooled_enrichment"]),
                    "pass": passed,
                }
            )
    signed = pd.DataFrame(signed_rows)
    if not signed["pass"].all():
        raise RuntimeError("FROZEN_STARTING_POINT_REPRODUCTION_FAILED: signed-tail enrichment")

    all_events = pd.read_csv(EXPECTED_HASHES["matched_event_manifest"][0], dtype={"site_id": str})
    all_events["event_start"] = pd.to_datetime(all_events["event_start"])
    all_events["event_end"] = pd.to_datetime(all_events["event_end"])
    included = all_events.loc[all_events["event_evaluation_included"].astype(bool)].copy()
    availability = included.groupby("event_direction").agg(
        evaluated_events=("event_window_id", "size"),
        evaluated_sites=("site_id", "nunique"),
        event_hours=("common_event_rows_in_evaluation", "sum"),
    ).reset_index()
    availability_dict = {row.event_direction: (int(row.evaluated_events), int(row.evaluated_sites), int(row.event_hours)) for row in availability.itertuples(index=False)}
    if availability_dict.get("low") != (1177, 18, 40029) or availability_dict.get("high") != (150, 10, 8025):
        raise RuntimeError(f"FROZEN_STARTING_POINT_REPRODUCTION_FAILED: event availability {availability_dict}")

    result = {
        "decision": "FROZEN_STARTING_POINT_REPRODUCED",
        "identity_checks": identity_checks,
        "counts": counts,
        "event_availability": availability_dict,
        "signed_tail_reproduction_all_pass": True,
    }
    return result, signed, all_events, thresholds


def directional_ph(
    test: pd.DataFrame,
    all_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included = all_events.loc[all_events["event_evaluation_included"].astype(bool)].copy()
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    result_rows = []
    detail_parts = []
    episode_parts = []
    for model, info in MODEL_INFO.items():
        short = info["short"]
        alert_map = {
            "negative": f"negative_tail_{short}",
            "positive": f"positive_tail_{short}",
            "absolute": f"top5_{short}",
        }
        for alert_direction, alert_col in alert_map.items():
            test[alert_col] = test[alert_col].astype(bool)
            episodes = build_episodes(test, alert_col, f"{model}|{alert_direction}")
            cache[(model, alert_direction)] = episodes
            if len(episodes):
                episode_export = episodes.copy()
                episode_export["model"] = model
                episode_export["alert_direction"] = alert_direction
                episode_parts.append(episode_export)
        for event_direction in ["low", "high"]:
            events = included.loc[included["event_direction"].eq(event_direction)].copy()
            event_mask = test["in_event"].astype(bool) & test["event_direction"].eq(event_direction)
            for alert_direction in ["negative", "positive", "absolute"]:
                alert_col = alert_map[alert_direction]
                result_role = (
                    "primary_directional"
                    if (event_direction, alert_direction) in [("low", "negative"), ("high", "positive")]
                    else "specificity_control"
                    if alert_direction != "absolute"
                    else "absolute_context"
                )
                enrichment = pooled_enrichment(test, event_mask, test[alert_col].rename(f"{model}|{event_direction}|{alert_direction}"))
                details = event_detection_details(test, events, alert_col, cache[(model, alert_direction)])
                details["model"] = model
                details["alert_direction"] = alert_direction
                detail_parts.append(details)
                detection = summarize_detection(details)
                burden = outside_burden(test, cache[(model, alert_direction)], all_events)
                result_rows.append(
                    {
                        "analysis_id": f"{model}|{event_direction}|{alert_direction}",
                        "target": "pH",
                        "task": "H00 contemporaneous" if info["horizon_hours"] == 0 else "historical-target H01 one-hour-ahead",
                        "model": model,
                        "model_uses_historical_target": info["historical_target"],
                        "event_direction": event_direction,
                        "alert_direction": alert_direction,
                        "result_role": result_role,
                        "evaluated_rows": len(test),
                        "evaluated_sites": int(events["site_id"].nunique()),
                        "eligible_sites": int(test["site_id"].nunique()),
                        "event_sites": int(events["site_id"].nunique()),
                        "evaluated_events": int(len(events)),
                        **enrichment,
                        **detection,
                        **burden,
                        "detector_definition": alert_col,
                        "analysis_status": "COMPLETE",
                    }
                )
    return (
        pd.DataFrame(result_rows),
        pd.concat(detail_parts, ignore_index=True),
        pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame(),
    )


def per_site_union_metrics(
    test: pd.DataFrame,
    events: pd.DataFrame,
    alert_columns: dict[str, str],
    details_by_model: dict[str, pd.DataFrame],
    episodes_by_model: dict[str, pd.DataFrame],
    all_events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    all_event_starts = {
        str(site): pd.to_datetime(group["event_start"]).sort_values()
        for site, group in all_events.groupby("site_id", sort=False)
    }
    for model, alert_col in alert_columns.items():
        details = details_by_model[model]
        episodes = episodes_by_model[model]
        detail_groups = {str(site): group for site, group in details.groupby("site_id", sort=False)}
        episode_groups = {str(site): group for site, group in episodes.groupby("site_id", sort=False)} if len(episodes) else {}
        for site_id, site in test.groupby("site_id", sort=False):
            event_mask = site["in_event"].astype(bool).to_numpy()
            alert = site[alert_col].astype(bool).to_numpy()
            detail = detail_groups.get(str(site_id), pd.DataFrame())
            eps = episode_groups.get(str(site_id), pd.DataFrame())
            starts = all_event_starts.get(str(site_id), pd.Series([], dtype="datetime64[ns]"))
            outside = 0
            if len(eps):
                for episode in eps.itertuples(index=False):
                    overlaps = bool(
                        ((starts - pd.Timedelta(hours=48) <= episode.episode_end)
                         & (starts + pd.Timedelta(hours=24) >= episode.episode_start)).any()
                    ) if len(starts) else False
                    outside += int(not overlaps)
            rows.append(
                {
                    "model": model,
                    "site_id": str(site_id),
                    "event_alert_hours": int(alert[event_mask].sum()),
                    "event_hours": int(event_mask.sum()),
                    "background_alert_hours": int(alert[~event_mask].sum()),
                    "background_hours": int((~event_mask).sum()),
                    "events": int(len(detail)),
                    "inside_alerted_events": int(detail["inside_row_alert"].sum()) if len(detail) else 0,
                    "episode_detected_events": int(detail["persistent_episode_overlap"].sum()) if len(detail) else 0,
                    "lead_eligible_events": int(detail["lead_eligible"].sum()) if len(detail) else 0,
                    "pre_event_alerted_events": int(detail.loc[detail["lead_eligible"], "pre_event_persistent_alert"].sum()) if len(detail) else 0,
                    "episodes": int(len(eps)),
                    "outside_episodes": int(outside),
                    "site_months": float(len(site) / (24 * 30.4375)),
                }
            )
    return pd.DataFrame(rows)


def metric_from_site_counts(values: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = (values[..., 0] / values[..., 1]) / (values[..., 2] / values[..., 3])
        in_window = values[..., 5] / values[..., 4]
        episode = values[..., 6] / values[..., 4]
        pre_event = values[..., 8] / values[..., 7]
        burden = values[..., 10] / values[..., 11]
    return {
        "enrichment": enrichment,
        "in_window_alerted_event_fraction": in_window,
        "persistent_episode_detection_fraction": episode,
        "pre_event_alerted_fraction_eligible": pre_event,
        "outside_register_episodes_per_site_month": burden,
    }


def union_analysis(
    test: pd.DataFrame,
    all_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included = all_events.loc[all_events["event_evaluation_included"].astype(bool)].copy()
    test["union_alert"] = test["top5_cross"].astype(bool) | test["top5_hist"].astype(bool)
    alert_columns = {
        "ph_h00_30day_offset": "top5_cross",
        "ph_historical_target_h01": "top5_hist",
        "ph_h00_or_historical_h01_union": "union_alert",
    }
    episodes_by_model = {model: build_episodes(test, column, model) for model, column in alert_columns.items()}
    details_by_model = {
        model: event_detection_details(test, included, column, episodes_by_model[model])
        for model, column in alert_columns.items()
    }
    result_rows = []
    for model, alert_col in alert_columns.items():
        event_mask = test["in_event"].astype(bool)
        enrichment = pooled_enrichment(test, event_mask, test[alert_col].rename(f"union|{model}"))
        detection = summarize_detection(details_by_model[model])
        burden = outside_burden(test, episodes_by_model[model], all_events)
        result_rows.append(
            {
                "model": model,
                "target": "pH",
                "task": "common target-time intersection",
                "historical_target_input": model in ["ph_historical_target_h01", "ph_h00_or_historical_h01_union"],
                "evaluated_rows": int(len(test)),
                "evaluated_sites": int(test["site_id"].nunique()),
                "alerted_hours": int(test[alert_col].sum()),
                "row_alert_rate": float(test[alert_col].mean()),
                **enrichment,
                **detection,
                **burden,
                "detector_definition": alert_col,
            }
        )
    results = pd.DataFrame(result_rows)

    detail_export = []
    for model, details in details_by_model.items():
        current = details.copy()
        current["model"] = model
        detail_export.append(current)
    detail_frame = pd.concat(detail_export, ignore_index=True)
    episode_export = []
    for model, episodes in episodes_by_model.items():
        current = episodes.copy()
        current["model"] = model
        episode_export.append(current)
    episode_frame = pd.concat(episode_export, ignore_index=True)

    pivot_inside = detail_frame.pivot(index="event_window_id", columns="model", values="inside_row_alert")
    pivot_episode = detail_frame.pivot(index="event_window_id", columns="model", values="persistent_episode_overlap")
    pivot_pre = detail_frame.pivot(index="event_window_id", columns="model", values="pre_event_persistent_alert")
    union_model = "ph_h00_or_historical_h01_union"
    h00_model = "ph_h00_30day_offset"
    h01_model = "ph_historical_target_h01"
    incremental = {
        "union_incremental_in_window_events_beyond_h00": int((pivot_inside[union_model] & ~pivot_inside[h00_model]).sum()),
        "union_incremental_in_window_events_beyond_h01": int((pivot_inside[union_model] & ~pivot_inside[h01_model]).sum()),
        "union_incremental_episode_detected_events_beyond_h00": int((pivot_episode[union_model] & ~pivot_episode[h00_model]).sum()),
        "union_incremental_episode_detected_events_beyond_h01": int((pivot_episode[union_model] & ~pivot_episode[h01_model]).sum()),
        "union_incremental_pre_event_alerted_events_beyond_h00": int((pivot_pre[union_model] & ~pivot_pre[h00_model]).sum()),
        "union_incremental_pre_event_alerted_events_beyond_h01": int((pivot_pre[union_model] & ~pivot_pre[h01_model]).sum()),
    }
    for key, value in incremental.items():
        results.loc[results["model"].eq(union_model), key] = value

    per_site = per_site_union_metrics(test, included, alert_columns, details_by_model, episodes_by_model, all_events)
    metric_columns = [
        "event_alert_hours", "event_hours", "background_alert_hours", "background_hours",
        "events", "inside_alerted_events", "episode_detected_events", "lead_eligible_events",
        "pre_event_alerted_events", "episodes", "outside_episodes", "site_months",
    ]
    models = list(alert_columns)
    site_order = sorted(test["site_id"].unique())
    arrays = {
        model: per_site.loc[per_site["model"].eq(model)].set_index("site_id").loc[site_order, metric_columns].to_numpy(dtype=float)
        for model in models
    }
    rng = seeded_rng("union_paired_site_bootstrap")
    paired_rows = []
    chunk = 250
    for start in range(0, BOOTSTRAPS, chunk):
        stop = min(start + chunk, BOOTSTRAPS)
        sample = rng.integers(0, len(site_order), size=(stop - start, len(site_order)))
        metrics = {model: metric_from_site_counts(arrays[model][sample].sum(axis=1)) for model in models}
        for offset in range(stop - start):
            replicate = start + offset + 1
            for metric in metrics[union_model]:
                union_value = metrics[union_model][metric][offset]
                paired_rows.append({"replicate": replicate, "comparison": "union_minus_h00", "metric": metric, "difference": union_value - metrics[h00_model][metric][offset]})
                paired_rows.append({"replicate": replicate, "comparison": "union_minus_h01", "metric": metric, "difference": union_value - metrics[h01_model][metric][offset]})
    paired = pd.DataFrame(paired_rows)
    paired_summary = paired.groupby(["comparison", "metric"], as_index=False).agg(
        mean_difference=("difference", "mean"),
        ci_low=("difference", lambda x: x.quantile(0.025)),
        ci_high=("difference", lambda x: x.quantile(0.975)),
        finite_replicates=("difference", lambda x: int(np.isfinite(x).sum())),
    )
    return results, detail_frame, episode_frame, paired_summary


def load_conformal_predictions(target: str, split: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[Path]]:
    spec = next(item for item in config["targets"] if item["target"] == target)
    root = ROOT / spec["prediction_root"]
    sites = sorted(split.loc[split["target"].eq(target) & split["conformal_split"].eq("evaluation"), "site_id"].astype(str))
    paths = [root / f"{site}.parquet" for site in sites]
    frames = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["site_id", "timestamp", "split", "target", "horizon_hours", "model_id", "observed_native", "predicted_native", "in_event", "event_direction", "event_id"],
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["site_id"] = result["site_id"].astype(str)
    result["target_timestamp"] = pd.to_datetime(result.pop("timestamp"))
    result = result.sort_values(["site_id", "target_timestamp"], kind="mergesort").reset_index(drop=True)
    if result.duplicated(["site_id", "target_timestamp"]).any():
        raise RuntimeError(f"Duplicate conformal evaluation key for {target}")
    if result["site_id"].nunique() != 90 or set(result["site_id"].unique()) != set(sites):
        raise RuntimeError(f"Conformal evaluation identity failure for {target}")
    return result, paths


def conformal_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], list[Path]]:
    config = load_json(EXPECTED_HASHES["conformal_config"][0])
    split = pd.read_csv(EXPECTED_HASHES["conformal_site_split"][0], dtype={"site_id": str})
    saved = pd.read_csv(EXPECTED_HASHES["conformal_summary"][0])
    rows = []
    site_parts = []
    episode_parts = []
    data_by_target: dict[str, pd.DataFrame] = {}
    prediction_paths: list[Path] = []
    for target in ["pH", "DO"]:
        frame, paths = load_conformal_predictions(target, split, config)
        prediction_paths.extend(paths)
        data_by_target[target] = frame
        for summary in saved.loc[saved["target"].eq(target)].sort_values("nominal_coverage").itertuples(index=False):
            nominal = float(summary.nominal_coverage)
            radius = float(summary.interval_radius)
            label = f"{target}|{nominal:.2f}"
            frame["lower"] = frame["predicted_native"] - radius
            frame["upper"] = frame["predicted_native"] + radius
            frame["lower_noncoverage"] = frame["observed_native"] < frame["lower"]
            frame["upper_noncoverage"] = frame["observed_native"] > frame["upper"]
            frame["noncoverage"] = frame["lower_noncoverage"] | frame["upper_noncoverage"]
            site = frame.groupby("site_id", as_index=False).agg(
                n=("observed_native", "size"),
                noncoverage_rows=("noncoverage", "sum"),
                lower_noncoverage_rows=("lower_noncoverage", "sum"),
                upper_noncoverage_rows=("upper_noncoverage", "sum"),
            )
            site["coverage"] = 1 - site["noncoverage_rows"] / site["n"]
            site["noncoverage_rate"] = site["noncoverage_rows"] / site["n"]
            site["lower_noncoverage_rate"] = site["lower_noncoverage_rows"] / site["n"]
            site["upper_noncoverage_rate"] = site["upper_noncoverage_rows"] / site["n"]
            site.insert(0, "target", target)
            site.insert(1, "nominal_coverage", nominal)
            site.insert(2, "interval_radius", radius)
            site_parts.append(site)

            episodes = build_episodes(frame, "noncoverage", label)
            if len(episodes):
                episodes.insert(0, "target", target)
                episodes.insert(1, "nominal_coverage", nominal)
                episodes.insert(2, "interval_radius", radius)
                episode_parts.append(episodes)
            pooled_coverage = float(1 - frame["noncoverage"].mean())
            macro_coverage = float(site["coverage"].mean())
            if not np.isclose(pooled_coverage, float(summary.pooled_empirical_coverage), atol=1e-12, rtol=1e-12) or not np.isclose(macro_coverage, float(summary.macro_site_empirical_coverage), atol=1e-12, rtol=1e-12):
                raise RuntimeError(f"Conformal reproduction failure {target} {nominal}: {pooled_coverage} {macro_coverage}")
            site_months = float(len(frame) / (24 * 30.4375))
            event_fields: dict[str, Any] = {
                "external_ph_event_alert_rate": np.nan,
                "external_ph_background_alert_rate": np.nan,
                "external_ph_event_enrichment": np.nan,
                "external_ph_event_enrichment_ci_low": np.nan,
                "external_ph_event_enrichment_ci_high": np.nan,
                "external_ph_event_rows": np.nan,
            }
            if target == "pH":
                external = pooled_enrichment(frame, frame["in_event"].astype(bool), frame["noncoverage"].rename(f"conformal|{label}"))
                event_fields = {
                    "external_ph_event_alert_rate": external["event_alert_rate"],
                    "external_ph_background_alert_rate": external["background_alert_rate"],
                    "external_ph_event_enrichment": external["enrichment"],
                    "external_ph_event_enrichment_ci_low": external["site_bootstrap_ci_low"],
                    "external_ph_event_enrichment_ci_high": external["site_bootstrap_ci_high"],
                    "external_ph_event_rows": external["event_hours"],
                }
            rows.append(
                {
                    "target": target,
                    "task": "no-target-history low-cost H01",
                    "nominal_coverage": nominal,
                    "interval_radius": radius,
                    "unit": summary.unit,
                    "evaluation_sites": int(frame["site_id"].nunique()),
                    "evaluation_rows": int(len(frame)),
                    "pooled_empirical_coverage": pooled_coverage,
                    "macro_site_coverage": macro_coverage,
                    "overall_noncoverage_rate": float(frame["noncoverage"].mean()),
                    "lower_noncoverage_rate": float(frame["lower_noncoverage"].mean()),
                    "upper_noncoverage_rate": float(frame["upper_noncoverage"].mean()),
                    "site_coverage_min": float(site["coverage"].min()),
                    "site_coverage_q05": float(site["coverage"].quantile(0.05)),
                    "site_coverage_q25": float(site["coverage"].quantile(0.25)),
                    "site_coverage_median": float(site["coverage"].median()),
                    "site_coverage_q75": float(site["coverage"].quantile(0.75)),
                    "site_coverage_q95": float(site["coverage"].quantile(0.95)),
                    "site_coverage_max": float(site["coverage"].max()),
                    "noncoverage_episode_count_min3h": int(len(episodes)),
                    "median_episode_flagged_hours": float(episodes["consecutive_flagged_hours"].median()) if len(episodes) else np.nan,
                    "episode_flagged_hours_q95": float(episodes["consecutive_flagged_hours"].quantile(0.95)) if len(episodes) else np.nan,
                    "noncoverage_episodes_per_site_month": float(len(episodes) / site_months) if site_months else np.nan,
                    **event_fields,
                }
            )
    return (
        pd.DataFrame(rows),
        pd.concat(site_parts, ignore_index=True),
        pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame(),
        data_by_target,
        prediction_paths,
    )


def do_unit_spot_check(do_frame: pd.DataFrame) -> pd.DataFrame:
    config = load_json(EXPECTED_HASHES["do_native_config"][0])
    if config["target"] != "do" or config["horizon_hours"] != 1 or config["target_unit"] != "mg/L" or config["target_source_multiplier"] != 20.0 or config["include_target_history"] is not False:
        raise RuntimeError("DO native-unit configuration identity failure")
    source_root = ROOT / config["merged_input_root"]
    rows = []
    for site_id in sorted(do_frame["site_id"].unique())[:3]:
        prediction = do_frame.loc[do_frame["site_id"].eq(site_id)].head(3).copy()
        source = pd.read_parquet(source_root / f"{site_id}.parquet", columns=["timestamp", "x_00300", "x_00300_observed_hour"])
        source["target_timestamp"] = pd.to_datetime(source.pop("timestamp"))
        merged = prediction.merge(source, on="target_timestamp", how="left", validate="one_to_one")
        for row in merged.itertuples(index=False):
            expected = float(row.x_00300) * 20.0
            rows.append(
                {
                    "site_id": site_id,
                    "target_timestamp": row.target_timestamp,
                    "source_x_00300_div20": float(row.x_00300),
                    "source_observed_hour": bool(row.x_00300_observed_hour),
                    "source_times_20_mg_l": expected,
                    "prediction_observed_native_mg_l": float(row.observed_native),
                    "absolute_difference": abs(expected - float(row.observed_native)),
                    "prediction_target": row.target,
                    "prediction_model_id": row.model_id,
                }
            )
    audit = pd.DataFrame(rows)
    if not audit["source_observed_hour"].all() or audit["absolute_difference"].max() > 1e-5 or set(audit["prediction_target"].astype(str).str.lower()) != {"do"}:
        raise RuntimeError("DO source-to-output native-unit spot check failed")
    return audit


def low_do_analysis(
    do_frame: pd.DataFrame,
    conformal_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not PREREG.exists():
        raise RuntimeError("LOW_DO_ANALYSIS_BLOCKED_NO_INDEPENDENT_THRESHOLD")
    summary = conformal_results.loc[
        conformal_results["target"].eq("DO")
        & conformal_results["nominal_coverage"].eq(LOW_DO_CONFORMAL_NOMINAL)
    ].iloc[0]
    radius = float(summary["interval_radius"])
    do_frame = do_frame.copy()
    do_frame["low_do_event"] = do_frame["observed_native"] < LOW_DO_THRESHOLD_MG_L
    do_frame["lower_noncoverage_90"] = do_frame["observed_native"] < (do_frame["predicted_native"] - radius)
    do_frame["in_event"] = do_frame["low_do_event"]
    do_frame["event_direction"] = np.where(do_frame["low_do_event"], "low_DO", "none")
    events = build_boolean_windows(do_frame, "low_do_event", "low_DO")
    event_sites = int(events["site_id"].nunique()) if len(events) else 0
    background_sites = set(do_frame.loc[~do_frame["low_do_event"], "site_id"])
    bootstrap_sites = len(set(events["site_id"]) & background_sites) if len(events) else 0
    gate = {
        "threshold_mg_l": LOW_DO_THRESHOLD_MG_L,
        "detector": "saved 90% nominal lower conformal noncoverage",
        "saved_radius_mg_l": radius,
        "eligible_rows": int(len(do_frame)),
        "eligible_sites": int(do_frame["site_id"].nunique()),
        "low_do_event_hours": int(do_frame["low_do_event"].sum()),
        "low_do_sites": int(do_frame.loc[do_frame["low_do_event"], "site_id"].nunique()),
        "low_do_events": int(len(events)),
        "bootstrap_sites": int(bootstrap_sites),
        "minimum_sites": LOW_DO_MIN_SITES,
        "minimum_events": LOW_DO_MIN_EVENTS,
        "minimum_bootstrap_sites": LOW_DO_MIN_BOOTSTRAP_SITES,
    }
    support = event_sites >= LOW_DO_MIN_SITES and len(events) >= LOW_DO_MIN_EVENTS and bootstrap_sites >= LOW_DO_MIN_BOOTSTRAP_SITES
    gate["support_gate_pass"] = bool(support)
    if not support:
        gate["decision"] = "INSUFFICIENT_LOW_DO_SUPPORT"
        return pd.DataFrame(), events, pd.DataFrame(), gate

    episodes = build_episodes(do_frame, "lower_noncoverage_90", "do_h01|lower_noncoverage_90")
    details = event_detection_details(do_frame, events, "lower_noncoverage_90", episodes)
    enrichment = pooled_enrichment(do_frame, do_frame["low_do_event"], do_frame["lower_noncoverage_90"].rename("do_h01|low_do|lower90"))
    detection = summarize_detection(details)
    burden = outside_burden(do_frame, episodes, events)
    result = pd.DataFrame(
        [{
            "analysis_id": "do_no_target_history_h01|low_DO|lower_conformal_noncoverage_90",
            "target": "DO",
            "task": "no-target-history low-cost H01",
            "model": "do_lowcost_no_target_history_h01",
            "model_uses_historical_target": False,
            "event_direction": "low_DO",
            "alert_direction": "lower_conformal_noncoverage_90",
            "result_role": "conditional_primary_directional",
            "evaluated_rows": int(len(do_frame)),
            "evaluated_sites": int(events["site_id"].nunique()),
            "eligible_sites": int(do_frame["site_id"].nunique()),
            "event_sites": int(events["site_id"].nunique()),
            "evaluated_events": int(len(events)),
            **enrichment,
            **detection,
            **burden,
            "detector_definition": f"observed < predicted - {radius:.12g} mg/L; saved 90% radius",
            "analysis_status": "COMPLETE",
        }]
    )
    gate["decision"] = "LOW_DO_ANALYSIS_COMPLETE"
    return result, events, details, gate


def make_directional_figure(results: pd.DataFrame) -> None:
    ph = results.loc[results["target"].eq("pH") & ~results["alert_direction"].eq("absolute")].copy()
    ph["label"] = ph["event_direction"].str.replace("low", "Low-pH", regex=False).str.replace("high", "High-pH", regex=False) + " × " + ph["alert_direction"].str.title()
    order = ["Low-pH × Negative", "Low-pH × Positive", "High-pH × Positive", "High-pH × Negative"]
    low_do = results.loc[results["target"].eq("DO") & results["analysis_status"].eq("COMPLETE")].copy()
    if len(low_do):
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [4, 1.25]})
        axis, do_axis = axes
    else:
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        do_axis = None
    colors = {"ph_h00_30day_offset": "#276FBF", "ph_historical_target_h01": "#F28E2B"}
    offsets = {"ph_h00_30day_offset": -0.16, "ph_historical_target_h01": 0.16}
    for model in MODEL_INFO:
        current = ph.loc[ph["model"].eq(model)].set_index("label").reindex(order)
        x = np.arange(len(order)) + offsets[model]
        y = current["enrichment"].to_numpy(float)
        lower = y - current["site_bootstrap_ci_low"].to_numpy(float)
        upper = current["site_bootstrap_ci_high"].to_numpy(float) - y
        axis.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color=colors[model], label=MODEL_INFO[model]["display"])
    axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(order)), order, rotation=15, ha="right")
    axis.set_ylabel("Event-window enrichment vs non-event background")
    axis.set_title("Directional pH")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    if do_axis is not None:
        row = low_do.iloc[0]
        y = float(row["enrichment"])
        lower = y - float(row["site_bootstrap_ci_low"])
        upper = float(row["site_bootstrap_ci_high"]) - y
        do_axis.errorbar([0], [y], yerr=np.array([[lower], [upper]]), fmt="o", capsize=5, color="#59A14F")
        do_axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
        do_axis.set_xticks([0], ["Low DO ×\nlower noncoverage"])
        do_axis.set_title("Conditional low DO")
        do_axis.grid(axis="y", alpha=0.25)
        do_axis.text(0, y, f"  {y:.1f}", va="center", fontsize=9)
    figure.suptitle("Directional enrichment (site-clustered 95% CI)")
    figure.tight_layout()
    figure.savefig(OUT / "figure_directional_enrichment.png", dpi=180)
    plt.close(figure)


def make_union_figure(results: pd.DataFrame) -> None:
    labels = ["H00", "Historical H01", "OR union"]
    models = ["ph_h00_30day_offset", "ph_historical_target_h01", "ph_h00_or_historical_h01_union"]
    current = results.set_index("model").loc[models]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    colors = ["#276FBF", "#F28E2B", "#59A14F"]
    axes[0].bar(labels, current["enrichment"], color=colors)
    axes[0].errorbar(np.arange(3), current["enrichment"], yerr=np.vstack([current["enrichment"] - current["site_bootstrap_ci_low"], current["site_bootstrap_ci_high"] - current["enrichment"]]), fmt="none", ecolor="#222222", capsize=4)
    axes[0].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title("Event enrichment")
    axes[1].bar(labels, current["pre_event_alerted_fraction_eligible"], color=colors)
    axes[1].set_title("Pre-event coverage")
    axes[1].set_ylim(0, max(0.25, float(current["pre_event_alerted_fraction_eligible"].max()) * 1.2))
    axes[2].bar(labels, current["outside_register_episodes_per_site_month"], color=colors)
    axes[2].set_title("Outside-register burden")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Fixed H00 / historical-target H01 OR-union trade-off", y=1.02)
    figure.tight_layout()
    figure.savefig(OUT / "figure_union_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_conformal_figure(results: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    for column, target in enumerate(["pH", "DO"]):
        current = results.loc[results["target"].eq(target)].sort_values("nominal_coverage")
        x = current["nominal_coverage"].to_numpy(float)
        coverage_axis = axes[0, column]
        noncoverage_axis = axes[1, column]
        coverage_axis.plot(x, current["pooled_empirical_coverage"], marker="o", label="Pooled coverage", color="#276FBF")
        coverage_axis.plot(x, current["macro_site_coverage"], marker="s", label="Macro-site coverage", color="#59A14F")
        coverage_axis.plot([0.78, 0.97], [0.78, 0.97], linestyle="--", color="#555555", label="Nominal")
        coverage_axis.set_title(target)
        coverage_axis.grid(alpha=0.22)
        noncoverage_axis.plot(x, current["lower_noncoverage_rate"], marker="v", label="Lower noncoverage", color="#4E79A7")
        noncoverage_axis.plot(x, current["upper_noncoverage_rate"], marker="^", label="Upper noncoverage", color="#E15759")
        noncoverage_axis.set_xlabel("Nominal coverage")
        noncoverage_axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[1, 0].set_ylabel("Noncoverage rate")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("Model-defined conformal coverage and noncoverage")
    figure.tight_layout()
    figure.savefig(OUT / "figure_conformal_noncoverage.png", dpi=180)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    rows = []
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    reproduction: dict[str, Any],
    directional: pd.DataFrame,
    union: pd.DataFrame,
    conformal: pd.DataFrame,
    low_do_gate: dict[str, Any],
) -> str:
    primary = directional.loc[directional["result_role"].isin(["primary_directional", "specificity_control", "conditional_primary_directional"])].copy()
    pht = primary.loc[primary["target"].eq("pH")].copy()
    pht["pair"] = pht["event_direction"] + " × " + pht["alert_direction"]
    dir_table = markdown_table(
        pht,
        ["model", "pair", "evaluated_events", "evaluated_sites", "event_hours", "event_alert_rate", "background_alert_rate", "enrichment", "site_bootstrap_ci_low", "site_bootstrap_ci_high", "pre_event_alerted_fraction_eligible", "median_lead_hours", "outside_register_episodes_per_site_month"],
        {key: "{:.4f}" for key in ["event_alert_rate", "background_alert_rate", "enrichment", "site_bootstrap_ci_low", "site_bootstrap_ci_high", "pre_event_alerted_fraction_eligible", "median_lead_hours", "outside_register_episodes_per_site_month"]},
    )
    union_table = markdown_table(
        union,
        ["model", "alerted_hours", "row_alert_rate", "alert_episode_count", "enrichment", "site_bootstrap_ci_low", "site_bootstrap_ci_high", "pre_event_alerted_events", "lead_eligible_events", "pre_event_alerted_fraction_eligible", "median_lead_hours", "outside_register_episodes_per_site_month"],
        {key: "{:.4f}" for key in ["row_alert_rate", "enrichment", "site_bootstrap_ci_low", "site_bootstrap_ci_high", "pre_event_alerted_fraction_eligible", "median_lead_hours", "outside_register_episodes_per_site_month"]},
    )
    conf_table = markdown_table(
        conformal,
        ["target", "nominal_coverage", "interval_radius", "evaluation_rows", "pooled_empirical_coverage", "macro_site_coverage", "overall_noncoverage_rate", "lower_noncoverage_rate", "upper_noncoverage_rate", "site_coverage_median", "noncoverage_episode_count_min3h", "median_episode_flagged_hours"],
        {key: "{:.4f}" for key in ["nominal_coverage", "interval_radius", "pooled_empirical_coverage", "macro_site_coverage", "overall_noncoverage_rate", "lower_noncoverage_rate", "upper_noncoverage_rate", "site_coverage_median", "median_episode_flagged_hours"]},
    )
    union_row = union.loc[union["model"].eq("ph_h00_or_historical_h01_union")].iloc[0]
    h00_row = union.loc[union["model"].eq("ph_h00_30day_offset")].iloc[0]
    h01_row = union.loc[union["model"].eq("ph_historical_target_h01")].iloc[0]
    h00_low = pht.loc[pht["model"].eq("ph_h00_30day_offset") & pht["event_direction"].eq("low") & pht["alert_direction"].eq("negative")].iloc[0]
    h00_high = pht.loc[pht["model"].eq("ph_h00_30day_offset") & pht["event_direction"].eq("high") & pht["alert_direction"].eq("positive")].iloc[0]
    h01_low = pht.loc[pht["model"].eq("ph_historical_target_h01") & pht["event_direction"].eq("low") & pht["alert_direction"].eq("negative")].iloc[0]
    h01_high = pht.loc[pht["model"].eq("ph_historical_target_h01") & pht["event_direction"].eq("high") & pht["alert_direction"].eq("positive")].iloc[0]
    h01_high_control = pht.loc[pht["model"].eq("ph_historical_target_h01") & pht["event_direction"].eq("high") & pht["alert_direction"].eq("negative")].iloc[0]
    ph90 = conformal.loc[conformal["target"].eq("pH") & conformal["nominal_coverage"].eq(0.9)].iloc[0]
    do90 = conformal.loc[conformal["target"].eq("DO") & conformal["nominal_coverage"].eq(0.9)].iloc[0]
    low_do_complete = low_do_gate.get("decision") == "LOW_DO_ANALYSIS_COMPLETE"
    decision = "PHASE1_COMPLETE" if low_do_complete else "PHASE1_COMPLETE_LOW_DO_BLOCKED"
    if low_do_complete:
        low_do = directional.loc[directional["target"].eq("DO")].iloc[0]
        low_do_text = (
            f"There were {low_do_gate['eligible_rows']:,} eligible rows across {low_do_gate['eligible_sites']} sites. "
            f"The predeclared `<2.0 mg/L` definition yielded {low_do_gate['low_do_event_hours']:,} event-hours in "
            f"{low_do_gate['low_do_events']:,} contiguous events across {low_do_gate['low_do_sites']} event sites. "
            f"The saved 90% lower conformal noncoverage detector had enrichment {low_do['enrichment']:.3f} "
            f"(site-bootstrap 95% CI {low_do['site_bootstrap_ci_low']:.3f} to {low_do['site_bootstrap_ci_high']:.3f}). "
            f"It alerted within {low_do['in_window_alerted_event_fraction']:.1%} of events and persistently detected {low_do['persistent_episode_detection_fraction']:.1%}. "
            "This is a no-target-history DO H01 result in native mg/L, not DO H00 and not an independently validated pollution detector."
        )
    else:
        low_do_text = f"Task 4 stopped as `{low_do_gate.get('decision', 'LOW_DO_ANALYSIS_BLOCKED_NO_INDEPENDENT_THRESHOLD')}` after reporting fixed support counts; no threshold was broadened."

    text = f"""# Final Frozen-Output Quick-Wins Report

Audit/analysis date: 2026-08-11
Scope: frozen predictions and previously fixed thresholds only; no training, inference rerun, tuning, Flow/Stage work, checkpoint change, or dissertation edit

## 1. Executive result

**Directional pH.** H00 showed the expected sign for both low-pH and high-pH events: enrichment was {h00_low['enrichment']:.3f} for low-pH × negative and {h00_high['enrichment']:.3f} for high-pH × positive. Historical-target H01 showed low-pH × negative enrichment of {h01_low['enrichment']:.3f}, but its high-pH × positive result was null ({h01_high['enrichment']:.3f}) while the opposite-sign control was elevated ({h01_high_control['enrichment']:.3f}). Directionality therefore did not generalize consistently across the two tasks; the high-pH analysis has only 150 events/10 sites and wide uncertainty.

**Fixed OR union.** The row-level rule is exactly `H00 top-5 OR historical-target H01 top-5`. Pre-event coverage rose from {h00_row['pre_event_alerted_fraction_eligible']:.3%} (H00) and {h01_row['pre_event_alerted_fraction_eligible']:.3%} (H01) to {union_row['pre_event_alerted_fraction_eligible']:.3%}; enrichment fell to {union_row['enrichment']:.3f} from {h00_row['enrichment']:.3f}/{h01_row['enrichment']:.3f}, and outside-register burden rose to {union_row['outside_register_episodes_per_site_month']:.3f} episodes/site-month from {h00_row['outside_register_episodes_per_site_month']:.3f}/{h01_row['outside_register_episodes_per_site_month']:.3f}. The union is a recall/burden trade-off, not an automatic improvement.

**Conformal noncoverage.** Saved pH and DO low-cost H01 radii reproduce all six canonical values on the fixed 90-site evaluation splits. At 90% nominal, pH pooled coverage was {ph90['pooled_empirical_coverage']:.1%} versus macro-site {ph90['macro_site_coverage']:.1%}, while DO was {do90['pooled_empirical_coverage']:.1%} pooled and {do90['macro_site_coverage']:.1%} macro-site. Lower and upper exceedances are model-defined noncoverage, not independently verified pollution events.

**Low DO.** {low_do_text}

## 2. Frozen identities and reproduction checks

- Decision: `{reproduction['decision']}`.
- Exact common evaluation support: {reproduction['counts']['evaluation_rows']:,} rows, {reproduction['counts']['sites']} sites, zero duplicate `site_id + target_timestamp` keys.
- H00-only/H01-only/shared/OR rows: {reproduction['counts']['h00_only_top5_rows']:,} / {reproduction['counts']['historical_h01_only_top5_rows']:,} / {reproduction['counts']['shared_top5_rows']:,} / {reproduction['counts']['union_top5_rows']:,}.
- Jaccard: {reproduction['counts']['jaccard']:.9f}; continuous absolute-score Spearman: {reproduction['counts']['spearman_absolute_score']:.9f}.
- Existing positive/negative tail enrichments reproduced to saved numeric precision before stratification.
- pH H00 is contemporaneous, has no historical pH, and uses the retained 30-day residual calibration. The H01 comparator here is `ph_LSTM_RAIN_FINAL_H01`, a causal one-hour model **with historical pH**. It is not the no-target-history deployment branch.

## 3. Directional pH results

{dir_table}

Primary scientific pairs are low-pH × negative tail and high-pH × positive tail. The opposite-sign pairs are specificity controls. Event-window rates use the original non-event background definition; 95% intervals use 10,000 fixed-seed site-cluster bootstrap replicates. Alerted-event and lead-time summaries use the existing 48-hour lookback, >=3 consecutive flagged hours, and 1.5-hour gap rule. Absolute-tail context remains in `directional_event_results.csv` and is not substituted for directionality.

## 4. H00/historical-H01 union results

{union_table}

The union adds {int(union_row.get('union_incremental_pre_event_alerted_events_beyond_h00', 0))} pre-event alerted events beyond H00 alone and {int(union_row.get('union_incremental_pre_event_alerted_events_beyond_h01', 0))} beyond H01 alone. It also changes enrichment and outside-register episode burden; these trade-offs, not recall alone, define its usefulness. Low overlap is evidence of complementarity on this exact 126-site intersection, not statistical independence. Paired site-bootstrap differences are in `appendix/union_paired_bootstrap_summary.csv`.

## 5. Conformal noncoverage results

{conf_table}

The intervals are symmetric around frozen low-cost H01 predictions. A noncoverage row is below `prediction-radius` or above `prediction+radius`; lower and upper rates are reported separately. Episodes use the pre-existing >=3 flagged-hour/1.5-hour gap rule. The pH external event overlap columns in `conformal_noncoverage_results.csv` use the existing independent pH event mask and are separate from coverage. DO noncoverage is not evaluated against pH events.

## 6. Low-DO result or explicit stop reason

The threshold was preregistered before DO test values were read in `appendix/LOW_DO_THRESHOLD_PREREGISTRATION.md`. NOAA describes hypoxia as typically below 2 mg/L, and EPA places hypoxic levels below 2-3 mg/L:

- https://coastalscience.noaa.gov/crp/hypoxia/
- https://www.ncei.noaa.gov/archive/archive-management-system/OAS/bin/prd/jquery/datatype/details/743
- https://www.epa.gov/ms-htf/hypoxia-101

{low_do_text}

The native-unit gate passed: the canonical config says H01, no target history, `target_source_multiplier=20`, and `target_unit=mg/L`; nine deterministic source/output rows satisfy `20*x_00300 == observed_native` within 1e-5. No additional conversion was applied.

## 7. Interpretation limits

- Residual convention is `observed-predicted`: negative is unexpectedly low and positive unexpectedly high.
- Directional enrichment is association with a weak external event register, not causal source attribution.
- H00 and H01 are not ranked as interchangeable prediction tasks.
- Site-clustered uncertainty does not remove the small 10-site high-pH limitation or temporal dependence within sites.
- OR-union membership is fixed; no score combination or test-set optimization was attempted.
- Conformal noncoverage is model-defined. Evaluating noncoverage against itself would be circular.
- The 2 mg/L DO threshold is an operational hypoxia definition, not a universal species-specific standard.

## 8. Recommended thesis placement

**Main text:** one compact directional-pH result; one OR-union trade-off paragraph/figure; the statement that low overlap indicates complementarity, not independence; and the retained H00/historical-H01 identity boundary.

**Appendix:** full specificity controls, all denominators and bootstrap intervals, paired union differences, site-level conformal distributions, episode-duration tables, pH external conformal overlap, DO unit spot check, and the conditional low-DO analysis/support gate.

The conformal result belongs in the uncertainty/model-diagnostic section. Low DO, if retained, belongs in an exploratory appendix because it uses model-defined lower noncoverage and a general operational threshold rather than a project-specific event register.

## 9. Reproducibility

- Runner: `appendix/run_frozen_quick_wins.py`.
- Python environment and command are recorded in `appendix/run_manifest.json`.
- Canonical hashes are in `appendix/input_hashes.csv`; the exact 180 conformal evaluation prediction files are in `appendix/prediction_input_manifest.csv`.
- Reproduction checks are in `appendix/frozen_starting_point_reproduction.json` and `appendix/signed_tail_reproduction.csv`.
- Output hashes are in `appendix/output_hashes.csv`.
- No canonical input was modified. No validation/test-derived threshold, tail, persistence, event window, subgroup, or union rule was selected.

## Overall decision

`{decision}`
"""
    (OUT / "FINAL_FROZEN_QUICK_WINS_REPORT.md").write_text(text, encoding="utf-8")
    return decision


def write_manifests(input_hashes: pd.DataFrame, prediction_paths: list[Path], decision: str) -> None:
    prediction_rows = []
    for path in prediction_paths:
        prediction_rows.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(APPENDIX / "prediction_input_manifest.csv", pd.DataFrame(prediction_rows))
    extra = [
        {"role": "low_do_threshold_preregistration", "path": str(PREREG.relative_to(ROOT)).replace("\\", "/"), "size_bytes": PREREG.stat().st_size, "expected_sha256": "", "actual_sha256": sha256(PREREG), "pass": True},
    ]
    complete_inputs = pd.concat([input_hashes, pd.DataFrame(extra)], ignore_index=True)
    write_csv(APPENDIX / "input_hashes.csv", complete_inputs)
    manifest = {
        "decision": decision,
        "audit_only": True,
        "models_trained": 0,
        "inference_reruns": 0,
        "thresholds_tuned": 0,
        "canonical_inputs_modified": False,
        "python": str(ROOT / ".venv/Scripts/python.exe"),
        "command": "python analysis/run_frozen_quick_wins.py",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "persistence_hours": PERSISTENCE_HOURS,
        "episode_gap_hours": EPISODE_GAP_HOURS,
        "lead_window_hours": LEAD_WINDOW_HOURS,
    }
    write_json(APPENDIX / "run_manifest.json", manifest)


def write_output_hashes() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "output_hashes.csv" or "__pycache__" in path.parts:
            continue
        rows.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(APPENDIX / "output_hashes.csv", pd.DataFrame(rows))


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    APPENDIX.mkdir(parents=True, exist_ok=True)
    input_hashes = verify_hashes()
    test = load_frozen_test()
    reproduction, signed, all_events, _ = task0_reproduction(test)
    write_json(APPENDIX / "frozen_starting_point_reproduction.json", reproduction)
    write_csv(APPENDIX / "signed_tail_reproduction.csv", signed)
    if args.task0_only:
        print(json.dumps(reproduction, indent=2))
        return

    directional, directional_details, directional_episodes = directional_ph(test, all_events)
    union, union_details, union_episodes, union_paired = union_analysis(test, all_events)
    conformal, conformal_site, conformal_episodes, conformal_data, prediction_paths = conformal_analysis()
    unit_spot = do_unit_spot_check(conformal_data["DO"])
    low_do, low_do_events, low_do_details, low_do_gate = low_do_analysis(conformal_data["DO"], conformal)
    if len(low_do):
        directional = pd.concat([directional, low_do], ignore_index=True, sort=False)

    write_csv(OUT / "directional_event_results.csv", directional)
    write_csv(OUT / "h00_h01_union_results.csv", union)
    write_csv(OUT / "conformal_noncoverage_results.csv", conformal)
    write_csv(APPENDIX / "directional_event_details.csv", directional_details)
    write_csv(APPENDIX / "directional_alert_episodes.csv", directional_episodes)
    write_csv(APPENDIX / "union_event_details.csv", union_details)
    write_csv(APPENDIX / "union_alert_episodes.csv", union_episodes)
    write_csv(APPENDIX / "union_paired_bootstrap_summary.csv", union_paired)
    write_csv(APPENDIX / "conformal_site_noncoverage.csv", conformal_site)
    write_csv(APPENDIX / "conformal_noncoverage_episodes.csv", conformal_episodes)
    write_csv(APPENDIX / "do_unit_spot_check.csv", unit_spot)
    write_csv(APPENDIX / "low_do_event_windows.csv", low_do_events)
    write_csv(APPENDIX / "low_do_event_details.csv", low_do_details)
    write_json(APPENDIX / "low_do_support_gate.json", low_do_gate)

    make_directional_figure(directional)
    make_union_figure(union)
    make_conformal_figure(conformal)
    decision = write_report(reproduction, directional, union, conformal, low_do_gate)
    write_manifests(input_hashes, prediction_paths, decision)
    write_output_hashes()
    print(json.dumps({"decision": decision, "rows": {"directional": len(directional), "union": len(union), "conformal": len(conformal)}, "low_do": low_do_gate}, indent=2))


if __name__ == "__main__":
    main()
