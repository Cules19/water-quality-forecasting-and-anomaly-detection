"""Site-disjoint post-hoc split-conformal intervals for low-cost pH and DO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            PROJECT_ROOT / "config" / "common1000_lowcost_conformal_v1.json"
        ),
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def site_hash(seed: int, target: str, site_id: str) -> str:
    return hashlib.sha256(f"{seed}|{target}|{site_id}".encode()).hexdigest()


def finite_sample_radius(scores: np.ndarray, coverage: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    rank = int(math.ceil((len(scores) + 1) * coverage))
    return float(scores[min(max(rank, 1), len(scores)) - 1])


def read_sites(root: Path) -> dict[str, pd.DataFrame]:
    result = {}
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=["site_id", "observed_native", "predicted_native"],
        )
        if not len(frame):
            continue
        site_id = str(frame["site_id"].iloc[0])
        result[site_id] = frame
    return result


def main() -> None:
    args = parse_args()
    config = load_json(resolve(args.config))
    output = resolve(config["output_root"])
    reports = output / "reports"
    figures = output / "figures"
    summary_rows = []
    site_rows = []
    split_rows = []

    for spec in config["targets"]:
        target = str(spec["target"])
        sites = read_sites(resolve(spec["prediction_root"]))
        ordered = sorted(
            sites,
            key=lambda site_id: site_hash(
                int(config["random_seed"]), target, site_id
            ),
        )
        calibration_count = int(config["calibration_sites_per_target"])
        calibration_sites = ordered[:calibration_count]
        evaluation_sites = ordered[calibration_count:]
        if not calibration_sites or not evaluation_sites:
            raise RuntimeError(f"Invalid conformal split for {target}")
        for site_id in ordered:
            split_rows.append(
                {
                    "target": target,
                    "site_id": site_id,
                    "conformal_split": (
                        "calibration"
                        if site_id in set(calibration_sites)
                        else "evaluation"
                    ),
                    "split_hash": site_hash(
                        int(config["random_seed"]), target, site_id
                    ),
                }
            )

        calibration_scores = np.concatenate(
            [
                np.abs(
                    frame["predicted_native"].to_numpy(dtype=np.float64)
                    - frame["observed_native"].to_numpy(dtype=np.float64)
                )
                for site_id, frame in sites.items()
                if site_id in set(calibration_sites)
            ]
        )
        for nominal in map(float, config["nominal_coverages"]):
            radius = finite_sample_radius(calibration_scores, nominal)
            target_site_rows = []
            for site_id in evaluation_sites:
                frame = sites[site_id]
                error = np.abs(
                    frame["predicted_native"].to_numpy(dtype=np.float64)
                    - frame["observed_native"].to_numpy(dtype=np.float64)
                )
                row = {
                    "target": target,
                    "site_id": site_id,
                    "nominal_coverage": nominal,
                    "interval_radius": radius,
                    "interval_width": 2.0 * radius,
                    "n": len(error),
                    "empirical_coverage": float((error <= radius).mean()),
                    "operational_tolerance": float(
                        spec["operational_tolerance"]
                    ),
                    "within_operational_tolerance": float(
                        (
                            error
                            <= float(spec["operational_tolerance"])
                        ).mean()
                    ),
                }
                site_rows.append(row)
                target_site_rows.append(row)
            target_site = pd.DataFrame(target_site_rows)
            weights = target_site["n"].to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    "target": target,
                    "unit": str(spec["unit"]),
                    "calibration_sites": len(calibration_sites),
                    "evaluation_sites": len(evaluation_sites),
                    "calibration_rows": len(calibration_scores),
                    "evaluation_rows": int(target_site["n"].sum()),
                    "nominal_coverage": nominal,
                    "interval_radius": radius,
                    "interval_width": 2.0 * radius,
                    "pooled_empirical_coverage": float(
                        np.average(
                            target_site["empirical_coverage"], weights=weights
                        )
                    ),
                    "macro_site_empirical_coverage": float(
                        target_site["empirical_coverage"].mean()
                    ),
                    "sites_meeting_nominal_coverage": int(
                        target_site["empirical_coverage"].ge(nominal).sum()
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    site_coverage = pd.DataFrame(site_rows)
    split = pd.DataFrame(split_rows)
    write_csv(reports / "conformal_summary.csv", summary)
    write_csv(reports / "conformal_site_coverage.csv", site_coverage)
    write_csv(reports / "conformal_site_split.csv", split)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for axis, target in zip(axes, summary["target"].drop_duplicates()):
        group = summary.loc[summary["target"].eq(target)].sort_values(
            "nominal_coverage"
        )
        axis.plot(
            group["nominal_coverage"],
            group["pooled_empirical_coverage"],
            marker="o",
            label="Pooled empirical",
        )
        axis.plot(
            group["nominal_coverage"],
            group["macro_site_empirical_coverage"],
            marker="s",
            label="Macro site empirical",
        )
        axis.plot([0.75, 1.0], [0.75, 1.0], linestyle="--", color="#444444")
        axis.set_xlim(0.77, 0.98)
        axis.set_ylim(0.77, 1.0)
        axis.set_title(target)
        axis.set_xlabel("Nominal coverage")
        axis.set_ylabel("Empirical coverage")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "conformal_coverage.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Site-disjoint low-cost conformal interval report",
        "",
        "The held-out sites were split once by a fixed hash into 45 calibration "
        "sites and the remaining evaluation sites. The low-cost point models "
        "were not refit and calibration/evaluation sites do not overlap.",
        "",
        "| Target | Nominal | Radius | Pooled coverage | Macro-site coverage | Sites meeting nominal |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.nominal_coverage:.0%} | "
            f"{row.interval_radius:.4f} {row.unit} | "
            f"{row.pooled_empirical_coverage:.1%} | "
            f"{row.macro_site_empirical_coverage:.1%} | "
            f"{row.sites_meeting_nominal_coverage}/"
            f"{row.evaluation_sites} |"
        )
    lines.extend(
        [
            "",
            "These are post-hoc symmetric split-conformal intervals around the "
            "LSTM-FiLM point prediction. Hourly observations within a station "
            "are temporally dependent, so the table is reported as empirical "
            "marginal coverage on site-disjoint data, not as an IID "
            "finite-sample guarantee. A future quantile head can reduce interval "
            "width by modelling heteroscedasticity directly.",
            "",
        ]
    )
    (reports / "FINAL_CONFORMAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    calibration_sets = {
        target: set(
            split.loc[
                split["target"].eq(target)
                & split["conformal_split"].eq("calibration"),
                "site_id",
            ]
        )
        for target in split["target"].unique()
    }
    evaluation_sets = {
        target: set(
            split.loc[
                split["target"].eq(target)
                & split["conformal_split"].eq("evaluation"),
                "site_id",
            ]
        )
        for target in split["target"].unique()
    }
    audit = {
        "status": "pass",
        "targets": sorted(summary["target"].unique().tolist()),
        "summary_rows": len(summary),
        "site_rows": len(site_coverage),
        "calibration_evaluation_site_overlap": {
            target: sorted(calibration_sets[target] & evaluation_sets[target])
            for target in calibration_sets
        },
        "model_refit_during_conformal_calibration": False,
        "claim_scope": "empirical marginal coverage under temporal dependence",
    }
    if (
        len(summary) != len(config["targets"])
        * len(config["nominal_coverages"])
        or any(audit["calibration_evaluation_site_overlap"].values())
    ):
        audit["status"] = "fail"
    write_json(reports / "independent_audit.json", audit)
    if audit["status"] != "pass":
        raise RuntimeError(audit)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
