from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "pipeline" / "00_common1000"))

import run_common1000_processing as pipeline  # noqa: E402


def config() -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / "common1000_processing_v1.json").read_text(encoding="utf-8")
    )


class Common1000ProcessingTests(unittest.TestCase):
    def test_normalise_site_id_preserves_leading_zero_contract(self) -> None:
        self.assertEqual(pipeline.normalise_site_id("1302020"), "01302020")
        self.assertEqual(pipeline.normalise_site_id(1302020.0), "01302020")
        self.assertEqual(pipeline.normalise_site_id("NPOSES2"), "NPOSES2")

    def test_interpolation_fills_only_short_internal_gaps(self) -> None:
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=8, freq="h"))
        short = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        filled, imputed, max_used = pipeline.interpolate_short_internal_gaps(timestamps, short, 6.0)
        self.assertEqual(int(imputed.sum()), 2)
        self.assertAlmostEqual(float(filled.iloc[1]), 2.0)
        self.assertAlmostEqual(float(filled.iloc[2]), 3.0)
        self.assertLessEqual(max_used, 6.0)

        long_gap = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0])
        unfilled, imputed_long, _ = pipeline.interpolate_short_internal_gaps(
            timestamps, long_gap, 6.0
        )
        self.assertEqual(int(imputed_long.sum()), 0)
        self.assertTrue(unfilled.iloc[1:7].isna().all())

    def test_qc_and_network_specific_do_ranges(self) -> None:
        cfg = config()
        raw = pd.Series([10.0, 150.0, -1.0, np.nan])
        qc = pd.Series([0.0, 1.0, 0.0, np.nan])
        cleaned, qc_status, range_flag, stats = pipeline.clean_numeric_column(
            raw, qc, "00300", "EA 临时站点", cfg
        )
        self.assertEqual(float(cleaned.iloc[0]), 10.0)
        self.assertTrue(pd.isna(cleaned.iloc[1]))
        self.assertTrue(pd.isna(cleaned.iloc[2]))
        self.assertEqual(int(qc_status[1]), 1)
        self.assertEqual(int(range_flag[2]), 1)
        self.assertEqual(stats["00300_qc_fail"], 1)

        non_ea, _, non_ea_range, _ = pipeline.clean_numeric_column(
            pd.Series([19.0, 21.0]), None, "00300", "USGS（8位数字站点）", cfg
        )
        self.assertEqual(float(non_ea.iloc[0]), 19.0)
        self.assertTrue(pd.isna(non_ea.iloc[1]))
        self.assertEqual(int(non_ea_range[1]), 1)

    def test_event_overlap_is_explicitly_mixed(self) -> None:
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=5, freq="h"))
        events = [
            {
                "event_id": 1,
                "incident_start": pd.Timestamp("2026-01-01 01:00"),
                "incident_end": pd.Timestamp("2026-01-01 03:00"),
                "direction": "low",
            },
            {
                "event_id": 2,
                "incident_start": pd.Timestamp("2026-01-01 02:00"),
                "incident_end": pd.Timestamp("2026-01-01 04:00"),
                "direction": "high",
            },
        ]
        fields, stats = pipeline.assign_event_fields(timestamps, events)
        self.assertEqual(fields.loc[1, "ph_event_direction"], "low")
        self.assertEqual(fields.loc[2, "ph_event_direction"], "mixed")
        self.assertEqual(fields.loc[4, "ph_event_direction"], "high")
        self.assertEqual(stats["matched_mixed_rows"], 2)

    def test_model_target_is_observed_and_conductance_is_train_scaled(self) -> None:
        cfg = config()
        timestamps = list(pd.date_range("2026-01-01", periods=20, freq="h"))
        timestamps.insert(5, timestamps[5])
        p_h = [7.0 + index * 0.01 for index in range(21)]
        p_h[8] = np.nan
        raw = pd.DataFrame(
            {
                "Site_ID": ["01302020"] * 21,
                "TIMESTAMP": timestamps,
                "00400": p_h,
                "00010": [10.0] * 21,
                "00095": np.arange(100.0, 121.0),
                "00300": [8.0] * 21,
                "63680": [4.0] * 21,
            }
        )
        mapping = {
            pipeline.SITE_COL: "Site_ID",
            pipeline.TIME_COL: "TIMESTAMP",
            **{code: code for code in pipeline.REQUIRED_CODES},
        }
        native, model, stats, scaler = pipeline.build_native_and_model(
            raw, mapping, {}, "01302020", "USGS（8位数字站点）", [], cfg
        )
        self.assertTrue(native["timestamp"].is_unique)
        self.assertTrue(model["observed_00400"].all())
        self.assertTrue(model[["00400", *pipeline.PREDICTOR_CODES]].notna().all(axis=None))
        train = model.loc[model["split"].eq("train"), "00095"]
        self.assertAlmostEqual(float(train.mean()), 0.0, places=5)
        self.assertEqual(
            scaler["00095_standard_scaler"]["fit_scope"], "rows where split == train only"
        )
        self.assertEqual(stats["duplicate_timestamp_rows_removed"], 1)


if __name__ == "__main__":
    unittest.main()
