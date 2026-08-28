from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pvlib import solarposition


warnings.filterwarnings(
    "ignore",
    message="RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "ph_h00_solar_dynamic1_v1"
PIPELINE = Path(__file__).resolve().parent

RAIN_CODE = ROOT / "pipeline" / "31_common1000_rainfall_ablation" / "run_rainfall_ablation.py"
M0_ROOT = ROOT / "analysis_outputs" / "ph_common1000_lstm_nohistory_h0_rain_final_v1" / "full"
M0_CONFIG = M0_ROOT / "reports" / "effective_config.json"
M0_PREPROCESSING = M0_ROOT / "reports" / "train_only_preprocessing.json"
M0_METADATA_TRANSFORM = M0_ROOT / "metadata" / "ph_no_coordinates_train_only_transform.json"
M0_CHECKPOINT = M0_ROOT / "models" / "ph_LSTM_RAIN_H0_H00.pt"
M0_SUCCESS = M0_ROOT / "models" / "ph_LSTM_RAIN_H0_H00_SUCCESS.json"
COHORT_PATH = (
    ROOT
    / "analysis_outputs"
    / "ph_common1000_teacher_archive_features_do_converted_v1"
    / "full"
    / "reports"
    / "primary_archive_experiment_cohort.csv"
)
HOURLY_ROOT = (
    ROOT
    / "analysis_outputs"
    / "ph_common1000_teacher_archive_features_do_converted_v1"
    / "full"
    / "hourly_ph_rainfall_by_site"
)
STATIC_PATH = ROOT / "analysis_outputs" / "ph_h01_static3_controlled_ablation_v1" / "spec" / "frozen_static3_910.csv"
COORDINATE_PATH = OUT / "coordinate_audit.csv"
ASSIGNMENT_PATH = OUT / "train_internal_fold_assignment.csv"
INNER_ASSIGNMENT_PATH = OUT / "inner_early_stopping_assignment.csv"
SOLAR_ROOT = OUT / "solar_features_by_site"

SEEDS = (42, 43, 44)
SOLAR_FIELDS = ("solar_hour_sin", "solar_hour_cos", "solar_potential")
LATITUDE_BANDS = (
    (-90.0, 25.0, "LT25"),
    (25.0, 35.0, "25_TO_LT35"),
    (35.0, 45.0, "35_TO_LT45"),
    (45.0, 55.0, "45_TO_LT55"),
    (55.0, 90.000001, "GE55"),
)
SUPPORT_DAYS = 30
MIN_SUPPORT = 504
TRAIN_CAP = 2500
INNER_VALIDATION_CAP = 1000
MAX_EPOCHS = 20
MIN_EPOCHS = 3
EARLY_PATIENCE = 4
MIN_DELTA = 0.0001
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
L2_COEFFICIENT = 1e-4
PREDICTION_BATCH_SIZE = 8192
TRAIN_BATCH_SIZE = 3072


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RAIN = load_module("h00_solar_dynamic1_rain", RAIN_CODE)
BASE = RAIN.BASE


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_frame_bytes(timestamps: np.ndarray, features: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(timestamps, dtype="datetime64[ns]").view("int64").tobytes())
    digest.update(np.ascontiguousarray(features, dtype=np.float32).tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latitude_band(latitude: float) -> str:
    for lower, upper, label in LATITUDE_BANDS:
        if lower <= latitude < upper:
            return label
    raise ValueError(latitude)


def utc_index(values: Any) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def solar_features(
    timestamps: Any,
    latitude: float,
    longitude: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pinned pvlib SPA geometric solar features from timezone-aware UTC."""
    times = utc_index(timestamps)
    spa = solarposition.spa_python(
        times,
        latitude=float(latitude),
        longitude=float(longitude),
        altitude=0.0,
        pressure=0.0,
        temperature=12.0,
        delta_t=None,
        atmos_refract=0.0,
        how="numpy",
    )
    hour_angle_degrees = solarposition.hour_angle(
        times,
        float(longitude),
        spa["equation_of_time"],
    ).to_numpy(dtype=np.float64)
    hour_angle_radians = np.deg2rad(hour_angle_degrees)
    geometric_zenith_degrees = spa["zenith"].to_numpy(dtype=np.float64)
    features = np.column_stack(
        [
            np.sin(hour_angle_radians),
            np.cos(hour_angle_radians),
            np.maximum(0.0, np.cos(np.deg2rad(geometric_zenith_degrees))),
        ]
    ).astype(np.float32)
    diagnostics = {
        "geometric_zenith_degrees": geometric_zenith_degrees,
        "hour_angle_degrees": hour_angle_degrees,
        "equation_of_time_minutes": spa["equation_of_time"].to_numpy(dtype=np.float64),
    }
    return features, diagnostics


def load_solar_sidecar(site_id: str, expected_timestamps: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(SOLAR_ROOT / f"{site_id}.parquet")
    found = pd.DatetimeIndex(frame["timestamp"]).to_numpy(dtype="datetime64[ns]")
    expected = np.asarray(expected_timestamps, dtype="datetime64[ns]")
    if not np.array_equal(found, expected):
        raise RuntimeError(f"Solar timestamp key mismatch for {site_id}")
    values = frame[list(SOLAR_FIELDS)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite solar feature for {site_id}")
    return values


def load_authoritative_model(device: torch.device):
    config = read_json(M0_CONFIG)
    preprocessing = read_json(M0_PREPROCESSING)
    cohort = pd.read_csv(COHORT_PATH, dtype={"site_id": str})
    metadata_source = BASE.merge_metadata(config)
    metadata_fields = [
        field for field in config["metadata_fields"] if field not in set(config["coordinate_fields"])
    ]
    metadata_map, metadata_parameters = BASE.fit_metadata(
        metadata_source,
        cohort,
        metadata_fields,
        set(config["metadata_log1p_fields"]),
    )
    official_metadata = read_json(M0_METADATA_TRANSFORM)
    if json.dumps(metadata_parameters, sort_keys=True) != json.dumps(official_metadata, sort_keys=True):
        raise RuntimeError("Recomputed metadata transform is not exactly the authoritative transform")
    definition = config["models"][0]
    first_site = str(cohort.iloc[0]["site_id"])
    sample = RAIN.site_payload(
        HOURLY_ROOT / f"{first_site}.parquet",
        list(definition["feature_names"]),
        preprocessing,
        config,
    )
    spec = {
        "model_id": definition["model_id"],
        "recurrent_type": definition["recurrent_type"],
        "recurrent_layers": int(config.get("recurrent_layers", 1)),
        "conditioning": "film",
        "metadata_variant": "no_coordinates",
        "target_history": False,
        "train_non_event": False,
    }
    model = BASE.SiteAgnosticFiLMRNN(
        spec,
        sample["features"].shape[1],
        len(metadata_map[first_site]),
        config,
    ).to(device)
    model.load_state_dict(torch.load(M0_CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, preprocessing, cohort, metadata_map


class SolarInjectedM0(nn.Module):
    """Frozen M0 with one isolated [4H, 3] solar gate matrix."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model
        if len(self.base.layers) != 1 or not isinstance(self.base.layers[0], nn.LSTM):
            raise RuntimeError("Frozen candidate requires the authoritative one-layer LSTM")
        hidden = int(self.base.layers[0].hidden_size)
        self.W_solar = nn.Parameter(torch.zeros((4 * hidden, 3), dtype=torch.float32))
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def expanded_forward(
        self,
        dynamic: torch.Tensor,
        solar: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.base.layers[0]
        value = torch.cat([dynamic, solar], dim=2)
        batch = value.shape[0]
        hidden = int(layer.hidden_size)
        h0 = torch.zeros((1, batch, hidden), dtype=value.dtype, device=value.device)
        c0 = torch.zeros_like(h0)
        weight_ih = torch.cat([layer.weight_ih_l0, self.W_solar], dim=1)
        flat_weights = [
            weight_ih,
            layer.weight_hh_l0,
            layer.bias_ih_l0,
            layer.bias_hh_l0,
        ]
        output, _h, _c = torch._VF.lstm(
            value,
            (h0, c0),
            flat_weights,
            True,
            1,
            0.0,
            self.training,
            False,
            True,
        )
        gamma, beta = self.base.film(metadata)
        output = gamma[:, 0].unsqueeze(1) * output + beta[:, 0].unsqueeze(1)
        return self.base.head(output[:, -1]).squeeze(1)

    def forward(
        self,
        dynamic: torch.Tensor,
        solar: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        expanded = self.expanded_forward(dynamic, solar, metadata)
        if torch.count_nonzero(self.W_solar).item() == 0:
            # Exact-value anchor with the gradient of the mathematically
            # equivalent expanded kernel. This avoids a GEMM-width-dependent
            # sub-ULP CUDA difference while preserving the correct first step.
            base_value = self.base(dynamic, metadata)
            return base_value.detach() + (expanded - expanded.detach())
        return expanded


@dataclass
class SiteData:
    site_id: str
    split: str
    country_code: str
    us_region: str
    latitude_band: str
    latitude: float
    longitude: float
    features: np.ndarray
    solar: np.ndarray
    target: np.ndarray
    timestamps: np.ndarray
    eligible_indices: np.ndarray
    support_indices: np.ndarray
    query_indices: np.ndarray
    m0_support_raw: np.ndarray
    m0_query_raw: np.ndarray
    m0_offset: float


def split_support_query(
    timestamps: np.ndarray,
    eligible_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.datetime64, np.datetime64]:
    values = timestamps[eligible_indices]
    if not len(values):
        return np.empty(0, np.int64), np.empty(0, np.int64), np.datetime64("NaT"), np.datetime64("NaT")
    start = values.min()
    end = start + np.timedelta64(SUPPORT_DAYS, "D")
    support = eligible_indices[values < end]
    query = eligible_indices[values >= end]
    return support, query, start, end


def array_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def window_arrays(
    data: SiteData,
    indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence = int(config["sequence_length_hours"])
    horizon = int(config["horizon_hours"])
    offsets = np.arange(sequence - 1, -1, -1, dtype=np.int64)
    end = indices - horizon
    positions = end[:, None] - offsets[None, :]
    return (
        np.ascontiguousarray(data.features[positions]),
        np.ascontiguousarray(data.solar[positions]),
        np.ascontiguousarray(data.target[indices]),
    )


def predict_indices(
    model: nn.Module,
    data: SiteData,
    indices: np.ndarray,
    metadata: np.ndarray,
    config: dict[str, Any],
    preprocessing: dict[str, Any],
    device: torch.device,
    batch_size: int = PREDICTION_BATCH_SIZE,
    grad: bool = False,
) -> torch.Tensor | np.ndarray:
    pieces: list[torch.Tensor] = []
    mean = float(preprocessing["target"]["mean"])
    scale = float(preprocessing["target"]["scale"])
    context = torch.enable_grad() if grad else torch.inference_mode()
    with context:
        for positions in array_batches(indices, batch_size):
            dynamic, solar, _ = window_arrays(data, positions, config)
            dynamic_t = torch.from_numpy(dynamic).to(device)
            solar_t = torch.from_numpy(solar).to(device)
            metadata_t = torch.from_numpy(
                np.ascontiguousarray(np.tile(metadata, (len(positions), 1)))
            ).to(device)
            value = model(dynamic_t, solar_t, metadata_t) * scale + mean
            pieces.append(value if grad else value.float().cpu())
    if not pieces:
        if grad:
            return torch.empty(0, dtype=torch.float32, device=device)
        return np.empty(0, np.float32)
    if grad:
        return torch.cat(pieces)
    return torch.cat(pieces).numpy().astype(np.float32, copy=False)


def predict_base_indices(
    model: nn.Module,
    data: SiteData,
    indices: np.ndarray,
    metadata: np.ndarray,
    config: dict[str, Any],
    preprocessing: dict[str, Any],
    device: torch.device,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    mean = float(preprocessing["target"]["mean"])
    scale = float(preprocessing["target"]["scale"])
    with torch.inference_mode():
        for positions in array_batches(indices, batch_size):
            dynamic, _solar, _ = window_arrays(data, positions, config)
            dynamic_t = torch.from_numpy(dynamic).to(device)
            metadata_t = torch.from_numpy(
                np.ascontiguousarray(np.tile(metadata, (len(positions), 1)))
            ).to(device)
            value = model(dynamic_t, metadata_t).float().cpu().numpy() * scale + mean
            pieces.append(value.astype(np.float32, copy=False))
    return np.concatenate(pieces) if pieces else np.empty(0, np.float32)


def basic_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def within_site_centered_r2(frame: pd.DataFrame, prediction_col: str) -> float:
    observed_centered = frame["observed_ph"] - frame.groupby("site_id", sort=False)["observed_ph"].transform("mean")
    predicted_centered = frame[prediction_col] - frame.groupby("site_id", sort=False)[prediction_col].transform("mean")
    denominator = float(np.square(observed_centered.to_numpy(np.float64)).sum())
    numerator = float(np.square((observed_centered - predicted_centered).to_numpy(np.float64)).sum())
    return float(1.0 - numerator / denominator)


def pooled_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    denominator = float(np.square(y - y.mean()).sum())
    return float(1.0 - np.square(y - p).sum() / denominator)


def site_bootstrap_difference(
    differences: np.ndarray,
    seed: int = 20260805,
    replicates: int = 10_000,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(replicates, start + 1000)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_candidate_better": float(np.mean(estimates < 0)),
        "replicates": replicates,
        "sites": len(values),
        "seed": seed,
    }
