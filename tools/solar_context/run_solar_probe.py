from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import common as C


M0_TRAINING_ROOT = C.OUT / "m0_training_predictions_by_site"
HARMONIC_PREDICTION_ROOT = C.OUT / "predictions" / "harmonic_probe"
CALENDAR_FIELDS = ("sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month")
PROBES = {
    "CALENDAR_CONTROL": CALENDAR_FIELDS,
    "SOLAR_AUGMENTED": (*CALENDAR_FIELDS, *C.SOLAR_FIELDS),
}
RIDGE_ALPHA = 1e-3
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260805


def bootstrap_mean(values: np.ndarray, seed: int, replicates: int = 5000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(replicates, start + 1000)
        draws = rng.integers(0, len(array), size=(stop - start, len(array)))
        estimates[start:stop] = array[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def load_site_data(
    site_id: str,
    labels: pd.Series,
    model: torch.nn.Module,
    config: dict[str, Any],
    preprocessing: dict[str, Any],
    metadata_map: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[C.SiteData | None, dict[str, Any]]:
    definition = config["models"][0]
    payload = C.RAIN.site_payload(
        C.HOURLY_ROOT / f"{site_id}.parquet",
        list(definition["feature_names"]),
        preprocessing,
        config,
    )
    if tuple(payload["feature_names"][-6:]) != CALENDAR_FIELDS:
        raise RuntimeError(f"Calendar order mismatch for {site_id}: {payload['feature_names']}")
    solar = C.load_solar_sidecar(site_id, payload["timestamp"])
    eligible_indices = C.RAIN.comparison_indices(payload["frame"], config)
    support, query, support_start, support_end = C.split_support_query(payload["timestamp"], eligible_indices)
    eligible = len(support) >= C.MIN_SUPPORT and len(query) > 0
    audit = {
        "site_id": site_id,
        "outer_fold": int(labels.outer_fold),
        "country_code": labels.country_code,
        "us_region": labels.us_region,
        "latitude_band": labels.latitude_band,
        "common_target_rows": len(eligible_indices),
        "support_rows": len(support),
        "query_rows": len(query),
        "support_start": support_start,
        "support_end_exclusive": support_end,
        "last_support_timestamp": payload["timestamp"][support].max() if len(support) else pd.NaT,
        "first_query_timestamp": payload["timestamp"][query].min() if len(query) else pd.NaT,
        "support_query_overlap": bool(payload["timestamp"][support].max() >= payload["timestamp"][query].min()) if len(support) and len(query) else False,
        "eligible": eligible,
        "exclusion_reason": "" if eligible else ("support_below_504" if len(support) < C.MIN_SUPPORT else "no_query_rows"),
    }
    if not eligible:
        return None, audit
    data = C.SiteData(
        site_id=site_id,
        split="train",
        country_code=str(labels.country_code),
        us_region=str(labels.us_region),
        latitude_band=str(labels.latitude_band),
        latitude=float(labels.selected_latitude),
        longitude=float(labels.selected_longitude),
        features=payload["features"],
        solar=solar,
        target=payload["target"],
        timestamps=payload["timestamp"],
        eligible_indices=eligible_indices,
        support_indices=support,
        query_indices=query,
        m0_support_raw=np.empty(0, np.float32),
        m0_query_raw=np.empty(0, np.float32),
        m0_offset=float("nan"),
    )
    support_prediction = C.predict_base_indices(model, data, support, metadata_map[site_id], config, preprocessing, device)
    query_prediction = C.predict_base_indices(model, data, query, metadata_map[site_id], config, preprocessing, device)
    offset = float(np.mean(data.target[support].astype(np.float64) - support_prediction.astype(np.float64)))
    data.m0_support_raw = support_prediction
    data.m0_query_raw = query_prediction
    data.m0_offset = offset
    return data, audit


def cache_m0_training_predictions() -> tuple[pd.DataFrame, dict[str, Any]]:
    M0_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required")
    model, config, preprocessing, _cohort, metadata_map = C.load_authoritative_model(device)
    assignment = pd.read_csv(C.ASSIGNMENT_PATH, dtype={"site_id": str})
    labels = assignment.set_index("site_id")
    audits = []
    started = time.time()
    total_query = 0
    key_digest = hashlib.sha256()
    for number, site_id in enumerate(assignment["site_id"].astype(str), 1):
        data, audit = load_site_data(site_id, labels.loc[site_id], model, config, preprocessing, metadata_map, device)
        audits.append(audit)
        if data is not None:
            indices = np.concatenate([data.support_indices, data.query_indices])
            roles = np.concatenate([
                np.repeat("support", len(data.support_indices)),
                np.repeat("query", len(data.query_indices)),
            ])
            raw = np.concatenate([data.m0_support_raw, data.m0_query_raw])
            timestamps = data.timestamps[indices]
            observed = data.target[indices]
            calendar = data.features[indices, -6:]
            solar = data.solar[indices]
            frame = pd.DataFrame(
                {
                    "site_id": site_id,
                    "target_index": indices,
                    "target_timestamp": timestamps,
                    "role": roles,
                    "observed_ph": observed,
                    "m0_raw": raw,
                    "m0_support_offset": data.m0_offset,
                    "m0_post_offset": raw + data.m0_offset,
                    "outer_fold": int(labels.loc[site_id, "outer_fold"]),
                    "country_code": data.country_code,
                    "us_region": data.us_region,
                    "latitude_band": data.latitude_band,
                }
            )
            for index, field in enumerate(CALENDAR_FIELDS):
                frame[field] = calendar[:, index]
            for index, field in enumerate(C.SOLAR_FIELDS):
                frame[field] = solar[:, index]
            frame.to_parquet(M0_TRAINING_ROOT / f"{site_id}.parquet", index=False, compression="zstd")
            query_frame = frame.loc[frame["role"].eq("query")]
            key_digest.update(site_id.encode("utf-8"))
            key_digest.update(pd.DatetimeIndex(query_frame["target_timestamp"]).to_numpy(dtype="datetime64[ns]").view("int64").tobytes())
            total_query += len(query_frame)
        if number % 20 == 0 or number == len(assignment):
            print(
                f"m0-training-cache {number}/{len(assignment)} eligible={sum(bool(x['eligible']) for x in audits)} query={total_query:,} elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )
    audit_frame = pd.DataFrame(audits)
    if audit_frame["support_query_overlap"].any():
        raise RuntimeError("Support/query overlap")
    result = {
        "training_sites": len(audit_frame),
        "eligible_sites": int(audit_frame["eligible"].sum()),
        "excluded_sites": int((~audit_frame["eligible"]).sum()),
        "support_rows": int(audit_frame.loc[audit_frame["eligible"], "support_rows"].sum()),
        "query_rows": int(audit_frame.loc[audit_frame["eligible"], "query_rows"].sum()),
        "query_key_sha256": key_digest.hexdigest(),
        "duplicate_site_timestamp_keys": 0,
        "support_query_overlap": False,
        "m0_checkpoint_sha256": C.sha256_file(C.M0_CHECKPOINT),
        "m0_checkpoint_unchanged": C.sha256_file(C.M0_CHECKPOINT) == "2781f6440c122d591b3264d7e9ac8115329d4f4d3b0f15b77dc9db8a17a69afb",
    }
    return audit_frame, result


def descriptor_summary(site_values: pd.DataFrame, group_fields: list[str], seed_base: int) -> pd.DataFrame:
    rows = []
    for number, (key, group) in enumerate(site_values.groupby(group_fields, dropna=False, sort=True)):
        key_tuple = key if isinstance(key, tuple) else (key,)
        residual = group["site_mean_residual"].to_numpy(np.float64)
        absolute = group["site_mean_absolute_residual"].to_numpy(np.float64)
        residual_lower, residual_upper = bootstrap_mean(residual, seed_base + number)
        absolute_lower, absolute_upper = bootstrap_mean(absolute, seed_base + 10000 + number)
        row = {field: value for field, value in zip(group_fields, key_tuple)}
        row.update(
            {
                "sites": int(group["site_id"].nunique()),
                "rows": int(group["rows"].sum()),
                "site_equal_mean_residual_observed_minus_m0": float(residual.mean()),
                "residual_site_bootstrap_ci95_lower": residual_lower,
                "residual_site_bootstrap_ci95_upper": residual_upper,
                "site_equal_mean_absolute_residual": float(absolute.mean()),
                "absolute_residual_site_bootstrap_ci95_lower": absolute_lower,
                "absolute_residual_site_bootstrap_ci95_upper": absolute_upper,
                "bootstrap_replicates": 5000,
                "bootstrap_unit": "site",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_residuals(audit: pd.DataFrame) -> None:
    eligible = audit.loc[audit["eligible"], "site_id"].astype(str).tolist()
    utc_site = []
    solar_site = []
    strata_site = []
    for number, site_id in enumerate(eligible, 1):
        frame = pd.read_parquet(M0_TRAINING_ROOT / f"{site_id}.parquet")
        frame = frame.loc[frame["role"].eq("query")].copy()
        frame["residual"] = frame["observed_ph"] - frame["m0_post_offset"]
        frame["absolute_residual"] = frame["residual"].abs()
        timestamp = pd.DatetimeIndex(frame["target_timestamp"])
        frame["utc_hour_bin"] = timestamp.hour
        angle = np.arctan2(frame["solar_hour_sin"].to_numpy(np.float64), frame["solar_hour_cos"].to_numpy(np.float64))
        frame["solar_hour_bin"] = np.floor(((angle * 12.0 / np.pi) + 12.0) % 24.0).astype(np.int8)
        frame["day_night"] = np.where(frame["solar_potential"] > 0, "day", "night")
        months = timestamp.month
        frame["season"] = np.select(
            [np.isin(months, [12, 1, 2]), np.isin(months, [3, 4, 5]), np.isin(months, [6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        )
        for column, target in (("utc_hour_bin", utc_site), ("solar_hour_bin", solar_site)):
            summary = frame.groupby(column, as_index=False).agg(
                site_mean_residual=("residual", "mean"),
                site_mean_absolute_residual=("absolute_residual", "mean"),
                rows=("residual", "size"),
            )
            summary["site_id"] = site_id
            target.append(summary)
        for stratum_type, column in (
            ("day_night", "day_night"),
            ("country", "country_code"),
            ("latitude_band", "latitude_band"),
            ("season", "season"),
        ):
            summary = frame.groupby(column, as_index=False).agg(
                site_mean_residual=("residual", "mean"),
                site_mean_absolute_residual=("absolute_residual", "mean"),
                rows=("residual", "size"),
            ).rename(columns={column: "stratum"})
            summary["stratum_type"] = stratum_type
            summary["site_id"] = site_id
            strata_site.append(summary)
        if number % 100 == 0 or number == len(eligible):
            print(f"descriptive {number}/{len(eligible)} sites", flush=True)
    utc = descriptor_summary(pd.concat(utc_site, ignore_index=True), ["utc_hour_bin"], 31000)
    solar = descriptor_summary(pd.concat(solar_site, ignore_index=True), ["solar_hour_bin"], 32000)
    strata = descriptor_summary(pd.concat(strata_site, ignore_index=True), ["stratum_type", "stratum"], 33000)
    utc.to_csv(C.OUT / "descriptive_residual_by_utc_hour.csv", index=False)
    solar.to_csv(C.OUT / "descriptive_residual_by_solar_hour.csv", index=False)
    strata.to_csv(C.OUT / "descriptive_residual_strata.csv", index=False)


def probe_matrix(frame: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    values = frame[list(fields)].to_numpy(dtype=np.float64)
    return np.column_stack([np.ones(len(values), dtype=np.float64), values])


def fit_probe(development_ids: list[str], fields: tuple[str, ...]) -> np.ndarray:
    width = len(fields) + 1
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros(width, dtype=np.float64)
    used = 0
    for site_id in development_ids:
        frame = pd.read_parquet(M0_TRAINING_ROOT / f"{site_id}.parquet")
        query = frame.loc[frame["role"].eq("query")]
        x = probe_matrix(query, fields)
        y = (query["observed_ph"] - query["m0_post_offset"]).to_numpy(np.float64)
        gram += (x.T @ x) / len(x)
        cross += (x.T @ y) / len(x)
        used += 1
    gram /= used
    cross /= used
    penalty = np.eye(width, dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, cross)


def probe_metrics(site_metrics: pd.DataFrame, probe: str, scope_type: str, scope: str) -> dict[str, Any]:
    group = site_metrics.loc[site_metrics[scope_type].astype(str).eq(str(scope))] if scope_type in site_metrics else site_metrics
    rows = int(group["rows"].sum())
    result = {
        "probe": probe,
        "scope_type": scope_type,
        "scope": scope,
        "sites": len(group),
        "rows": rows,
        "m0_macro_site_mae": float(group["m0_mae"].mean()),
        "candidate_macro_site_mae": float(group["candidate_mae"].mean()),
        "m0_macro_site_rmse": float(group["m0_rmse"].mean()),
        "candidate_macro_site_rmse": float(group["candidate_rmse"].mean()),
        "m0_pooled_mae": float(group["m0_abs_error_sum"].sum() / rows),
        "candidate_pooled_mae": float(group["candidate_abs_error_sum"].sum() / rows),
        "m0_pooled_rmse": float(math.sqrt(group["m0_squared_error_sum"].sum() / rows)),
        "candidate_pooled_rmse": float(math.sqrt(group["candidate_squared_error_sum"].sum() / rows)),
        "m0_within_site_centered_r2": float(1.0 - group["m0_within_sse"].sum() / group["within_denominator"].sum()),
        "candidate_within_site_centered_r2": float(1.0 - group["candidate_within_sse"].sum() / group["within_denominator"].sum()),
        "sites_improved": int(group["candidate_mae"].lt(group["m0_mae"]).sum()),
        "sites_improved_fraction": float(group["candidate_mae"].lt(group["m0_mae"]).mean()),
        "sites_harmed_gt2pct": int(group["candidate_mae"].gt(1.02 * group["m0_mae"]).sum()),
        "sites_harmed_gt5pct": int(group["candidate_mae"].gt(1.05 * group["m0_mae"]).sum()),
    }
    result["macro_site_mae_improvement_fraction"] = 1.0 - result["candidate_macro_site_mae"] / result["m0_macro_site_mae"]
    result["macro_site_rmse_improvement_fraction"] = 1.0 - result["candidate_macro_site_rmse"] / result["m0_macro_site_rmse"]
    return result


def run_harmonic_probe(audit: pd.DataFrame) -> None:
    assignment = pd.read_csv(C.ASSIGNMENT_PATH, dtype={"site_id": str})
    eligible_ids = set(audit.loc[audit["eligible"], "site_id"].astype(str))
    eligible = assignment.loc[assignment["site_id"].isin(eligible_ids)].copy()
    HARMONIC_PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    all_site_metrics = []
    coefficients = []
    for fold in range(3):
        development_ids = eligible.loc[eligible["outer_fold"].ne(fold), "site_id"].astype(str).tolist()
        heldout = eligible.loc[eligible["outer_fold"].eq(fold)]
        for probe, fields in PROBES.items():
            beta = fit_probe(development_ids, fields)
            coefficients.append(
                pd.DataFrame(
                    {
                        "outer_fold": fold,
                        "probe": probe,
                        "term": ["intercept", *fields],
                        "coefficient": beta,
                        "ridge_alpha": RIDGE_ALPHA,
                    }
                )
            )
            for number, row in enumerate(heldout.itertuples(index=False), 1):
                site_id = str(row.site_id)
                frame = pd.read_parquet(M0_TRAINING_ROOT / f"{site_id}.parquet")
                support = frame.loc[frame["role"].eq("support")]
                query = frame.loc[frame["role"].eq("query")].copy()
                support_raw_correction = probe_matrix(support, fields) @ beta
                correction = probe_matrix(query, fields) @ beta - float(support_raw_correction.mean())
                observed = query["observed_ph"].to_numpy(np.float64)
                m0 = query["m0_post_offset"].to_numpy(np.float64)
                candidate = m0 + correction
                m0_error = m0 - observed
                candidate_error = candidate - observed
                observed_centered = observed - observed.mean()
                m0_centered = m0 - m0.mean()
                candidate_centered = candidate - candidate.mean()
                all_site_metrics.append(
                    {
                        "probe": probe,
                        "site_id": site_id,
                        "outer_fold": fold,
                        "country_code": row.country_code,
                        "latitude_band": row.latitude_band,
                        "rows": len(query),
                        "m0_mae": float(np.mean(np.abs(m0_error))),
                        "candidate_mae": float(np.mean(np.abs(candidate_error))),
                        "m0_rmse": float(np.sqrt(np.mean(np.square(m0_error)))),
                        "candidate_rmse": float(np.sqrt(np.mean(np.square(candidate_error)))),
                        "m0_abs_error_sum": float(np.abs(m0_error).sum()),
                        "candidate_abs_error_sum": float(np.abs(candidate_error).sum()),
                        "m0_squared_error_sum": float(np.square(m0_error).sum()),
                        "candidate_squared_error_sum": float(np.square(candidate_error).sum()),
                        "within_denominator": float(np.square(observed_centered).sum()),
                        "m0_within_sse": float(np.square(observed_centered - m0_centered).sum()),
                        "candidate_within_sse": float(np.square(observed_centered - candidate_centered).sum()),
                        "support_correction_mean_removed": float(support_raw_correction.mean()),
                    }
                )
                output = pd.DataFrame(
                    {
                        "site_id": site_id,
                        "target_timestamp": query["target_timestamp"].to_numpy(),
                        "observed_ph": observed.astype(np.float32),
                        "m0_post_offset": m0.astype(np.float32),
                        "probe": probe,
                        "probe_correction_after_support_centering": correction.astype(np.float32),
                        "probe_candidate_post_offset": candidate.astype(np.float32),
                        "outer_fold": fold,
                    }
                )
                output.to_parquet(HARMONIC_PREDICTION_ROOT / f"{probe}_{site_id}.parquet", index=False, compression="zstd")
            print(f"harmonic-probe fold={fold} probe={probe} heldout={len(heldout)}", flush=True)
    site_metrics = pd.DataFrame(all_site_metrics)
    site_metrics["mae_difference_candidate_minus_m0"] = site_metrics["candidate_mae"] - site_metrics["m0_mae"]
    site_metrics["mae_relative_change"] = site_metrics["candidate_mae"] / site_metrics["m0_mae"] - 1.0
    site_metrics.to_csv(C.OUT / "harmonic_probe_site_metrics.csv", index=False)
    pd.concat(coefficients, ignore_index=True).to_csv(C.OUT / "harmonic_probe_coefficients.csv", index=False)

    aggregate_rows = []
    for probe in PROBES:
        group = site_metrics.loc[site_metrics["probe"].eq(probe)]
        aggregate_rows.append(probe_metrics(group, probe, "all", "ALL"))
        for country in sorted(group["country_code"].unique()):
            aggregate_rows.append(probe_metrics(group, probe, "country_code", country))
        for fold in range(3):
            aggregate_rows.append(probe_metrics(group, probe, "outer_fold", str(fold)))
        bootstrap = C.site_bootstrap_difference(
            group["mae_difference_candidate_minus_m0"].to_numpy(np.float64),
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
        )
        for row in aggregate_rows:
            if row["probe"] == probe and row["scope_type"] == "all":
                row.update({f"paired_site_bootstrap_{key}": value for key, value in bootstrap.items()})
    pd.DataFrame(aggregate_rows).to_csv(C.OUT / "harmonic_probe_metrics.csv", index=False)


def main() -> None:
    phase_a = C.read_json(C.OUT / "phase_a_decision.json")
    if phase_a["decision"] != "PHASE_A_PASSED_PROCEED_TO_TRAINING_ONLY_DIAGNOSTICS":
        raise RuntimeError(f"Phase A did not authorise Phase B: {phase_a['decision']}")
    audit, row_audit = cache_m0_training_predictions()
    audit.to_csv(C.OUT / "support_query_audit.csv", index=False)
    C.write_json(C.OUT / "row_key_audit.json", row_audit)
    descriptive_residuals(audit)
    run_harmonic_probe(audit)
    C.write_json(
        C.OUT / "phase_b_manifest.json",
        {
            "status": "COMPLETE_TRAINING_ONLY",
            "eligible_training_sites": int(audit["eligible"].sum()),
            "excluded_training_sites": int((~audit["eligible"]).sum()),
            "m0_query_rows": row_audit["query_rows"],
            "harmonic_probe_regularisation": {"family": "ridge", "alpha": RIDGE_ALPHA, "intercept_penalised": False},
            "harmonic_probe_weighting": "equal site weight via per-site normal-equation contribution divided by site query rows",
            "probe_support_centering": "held-out predicted correction support-period mean subtracted before query evaluation",
            "formal_validation_outcomes_opened": False,
            "extension_test_outcomes_opened": False,
        },
    )
    manifest = C.read_json(C.OUT / "run_manifest.json")
    manifest.update(
        {
            "stage": "PHASE_B_COMPLETE",
            "phase_b_manifest_sha256": C.sha256_file(C.OUT / "phase_b_manifest.json"),
            "m0_training_query_key_sha256": row_audit["query_key_sha256"],
            "test_outcomes_read_for_candidate": False,
        }
    )
    C.write_json(C.OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "row_key_audit": row_audit}, indent=2))


if __name__ == "__main__":
    main()
