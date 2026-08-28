"""Train site-disjoint recurrent FiLM models without a site embedding."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_PATH = PROJECT_ROOT / "pipeline" / "20_common1000_leave_sites_out" / "prepare_leave_sites_out.py"
SPEC = importlib.util.spec_from_file_location("leave_site_prepare", PREPARE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREPARE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--target", action="append", choices=("ph", "do"), default=[])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FiLMGenerator(nn.Module):
    def __init__(self, metadata_size: int, hidden_size: int, layers: int, film_hidden: int) -> None:
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(metadata_size, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * layers * hidden_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.net(metadata).view(-1, self.layers, 2, self.hidden_size)
        return 1.0 + value[:, :, 0], value[:, :, 1]


class SiteAgnosticFiLMRNN(nn.Module):
    def __init__(self, spec: dict[str, Any], dynamic_size: int, metadata_size: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        layers = int(spec["recurrent_layers"])
        recurrent_type = str(spec["recurrent_type"]).lower()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")
        recurrent = nn.GRU if recurrent_type == "gru" else nn.LSTM
        dropout = float(config.get("dropout", 0.0))
        self.layers = nn.ModuleList(
            [recurrent(dynamic_size if index == 0 else hidden, hidden, batch_first=True) for index in range(layers)]
        )
        self.inter_layer_dropout = nn.Dropout(dropout)
        self.conditioning = spec["conditioning"]
        self.film = (
            FiLMGenerator(metadata_size, hidden, layers, int(config["film_hidden_size"]))
            if self.conditioning == "film"
            else None
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(config["head_hidden_size"])),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(config["head_hidden_size"]), 1),
        )

    def forward(self, dynamic: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        if self.film is not None:
            if metadata is None:
                raise ValueError("FiLM metadata is required")
            gamma, beta = self.film(metadata)
        value = dynamic
        for index, layer in enumerate(self.layers):
            value, _ = layer(value)
            if self.film is not None:
                value = gamma[:, index].unsqueeze(1) * value + beta[:, index].unsqueeze(1)
            if index < len(self.layers) - 1:
                value = self.inter_layer_dropout(value)
        return self.head(value[:, -1]).squeeze(1)


def merge_metadata(config: dict[str, Any]) -> pd.DataFrame:
    metadata = pd.read_csv(resolve(config["metadata_input"]), dtype={"site_id": "string"})
    metadata["site_id"] = metadata["site_id"].map(PREPARE.normalise_site_id)
    flow = pd.read_csv(resolve(config["flow_metadata_input"]), dtype={"site_id": "string"})
    flow["site_id"] = flow["site_id"].map(PREPARE.normalise_site_id)
    keep = ["site_id", "log10_mean_flow", "flow_cv", "richards_baker_flashiness", "q95_q50_ratio"]
    metadata = metadata.merge(flow[keep], on="site_id", how="left", validate="one_to_one")
    return metadata.drop_duplicates("site_id").set_index("site_id")


def fit_metadata(
    metadata: pd.DataFrame,
    split: pd.DataFrame,
    fields: list[str],
    log_fields: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_ids = split.loc[split["split"].eq("train"), "site_id"].astype(str).tolist()
    all_ids = split["site_id"].astype(str).tolist()
    parameters: dict[str, Any] = {"fit_split": "train_sites_only", "fields": []}
    matrices = {site_id: [] for site_id in all_ids}
    for field in fields:
        values = pd.to_numeric(metadata.reindex(all_ids)[field], errors="coerce") if field in metadata else pd.Series(np.nan, index=all_ids)
        values.index = all_ids
        if field in log_fields:
            values = np.log1p(values.where(values >= 0))
        train = values.reindex(train_ids).dropna().to_numpy(dtype=np.float64)
        mean = float(train.mean()) if len(train) else 0.0
        scale = float(train.std()) if len(train) else 1.0
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        for site_id in all_ids:
            raw = values.get(site_id, np.nan)
            missing = not np.isfinite(raw)
            matrices[site_id].extend([0.0 if missing else (float(raw) - mean) / scale, float(missing)])
        parameters["fields"].append(
            {"field": field, "mean": mean, "scale": scale, "training_nonmissing_sites": len(train), "output": [field, f"{field}_missing"]}
        )
    return {key: np.asarray(value, dtype=np.float32) for key, value in matrices.items()}, parameters


def site_payload(
    path: Path,
    target_spec: dict[str, Any],
    preprocessing: dict[str, Any],
    history: bool,
    history_fill_limit: int,
) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    columns = []
    names = []
    for code in target_spec["dynamic_codes"]:
        values = pd.to_numeric(frame[f"x_{code}"], errors="coerce").to_numpy(dtype=np.float32)
        stat = preprocessing["features"][code]
        missing = ~np.isfinite(values)
        values = np.where(missing, float(stat["mean"]), values)
        columns.extend([((values - float(stat["mean"])) / float(stat["scale"])).astype(np.float32), missing.astype(np.float32)])
        names.extend([f"x_{code}_standardized", f"x_{code}_missing"])
    if "x_00300_percent_family" in frame:
        columns.append(frame["x_00300_percent_family"].to_numpy(dtype=np.float32))
        names.append("x_00300_percent_family")
    for field in PREPARE.CALENDAR_FEATURES:
        columns.append(frame[field].to_numpy(dtype=np.float32))
        names.append(field)
    target_column = f"target_{target_spec['target_code']}_native"
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float32)
    if history:
        filled = pd.Series(target).ffill(limit=history_fill_limit).to_numpy(dtype=np.float32)
        missing = ~np.isfinite(filled)
        mean = float(preprocessing["target"]["mean"])
        scale = float(preprocessing["target"]["scale"])
        filled = np.where(missing, mean, filled)
        columns.extend([((filled - mean) / scale).astype(np.float32), missing.astype(np.float32)])
        names.extend(["target_history_standardized", "target_history_missing"])
    return {
        "features": np.column_stack(columns).astype(np.float32),
        "feature_names": names,
        "target": target,
        "timestamp": pd.DatetimeIndex(frame["timestamp"]).to_numpy(dtype="datetime64[ns]"),
        "in_event": frame["in_event"].to_numpy(dtype=bool),
        "event_direction": frame["event_direction"].astype(str).to_numpy(),
        "event_id": frame["event_id"].to_numpy(dtype=np.int64),
        "frame": frame,
    }


def select_indices(
    payload: dict[str, Any],
    target_spec: dict[str, Any],
    preparation_config: dict[str, Any],
    horizon: int,
    non_event: bool,
    cap: int,
    seed: int,
) -> np.ndarray:
    index = PREPARE.valid_window_targets(
        payload["frame"],
        target_spec["target_code"],
        list(target_spec["dynamic_codes"]),
        int(preparation_config["sequence_length_hours"]),
        horizon,
        float(preparation_config["maximum_dynamic_missing_fraction_per_window"]),
    )
    if non_event:
        index = index[~payload["in_event"][index]]
    if cap > 0 and len(index) > cap:
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(index, size=cap, replace=False))
    return index


def window_batches(
    payload: dict[str, Any],
    indices: np.ndarray,
    metadata: np.ndarray | None,
    sequence_length: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
    order = indices.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    target_mean_dummy = 0.0
    for start in range(0, len(order), batch_size):
        target_index = order[start : start + batch_size]
        input_end = target_index - horizon
        positions = input_end[:, None] - offsets[None, :]
        dynamic = np.ascontiguousarray(payload["features"][positions])
        meta = None if metadata is None else np.ascontiguousarray(np.tile(metadata, (len(target_index), 1)))
        target = np.ascontiguousarray(payload["target"][target_index])
        yield dynamic, meta, target, target_index


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return PREPARE.metric_values(observed.astype(np.float64), predicted.astype(np.float64))


def evaluate_validation(
    model: nn.Module,
    site_ids: list[str],
    target_name: str,
    target_spec: dict[str, Any],
    spec: dict[str, Any],
    horizon: int,
    metadata_map: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    preparation_config: dict[str, Any],
    model_config: dict[str, Any],
    prepared: Path,
    mode_settings: dict[str, Any],
    device: torch.device,
) -> float:
    model.eval()
    squared = 0.0
    count = 0
    mean = float(preprocessing["target"]["mean"])
    scale = float(preprocessing["target"]["scale"])
    with torch.no_grad():
        for number, site_id in enumerate(site_ids):
            payload = site_payload(
                prepared / "hourly_native_by_site" / target_name / f"{site_id}.parquet",
                target_spec,
                preprocessing,
                bool(spec["target_history"]),
                int(model_config["target_history_fill_limit_hours"]),
            )
            index = select_indices(
                payload,
                target_spec,
                preparation_config,
                horizon,
                False,
                int(mode_settings["validation_window_cap_per_site"]),
                int(model_config["random_seed"]) + number,
            )
            metadata = metadata_map[site_id] if spec["conditioning"] == "film" else None
            for dynamic, meta, target, _ in window_batches(
                payload, index, metadata, int(preparation_config["sequence_length_hours"]), horizon,
                int(mode_settings["prediction_batch_size"]), False, 0,
            ):
                dynamic_tensor = torch.from_numpy(dynamic).to(device)
                meta_tensor = torch.from_numpy(meta).to(device) if meta is not None else None
                prediction = model(dynamic_tensor, meta_tensor).float().cpu().numpy() * scale + mean
                squared += float(np.square(prediction - target).sum())
                count += len(target)
    return math.sqrt(squared / count) if count else float("inf")


def train_model(
    model: nn.Module,
    train_ids: list[str],
    validation_ids: list[str],
    target_name: str,
    target_spec: dict[str, Any],
    spec: dict[str, Any],
    horizon: int,
    metadata_map: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    preparation_config: dict[str, Any],
    config: dict[str, Any],
    prepared: Path,
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    loss_function = nn.MSELoss()
    target_mean = float(preprocessing["target"]["mean"])
    target_scale = float(preprocessing["target"]["scale"])
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    history = []
    event_removed = 0
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        order = train_ids.copy()
        random.Random(int(config["random_seed"]) + epoch).shuffle(order)
        for site_number, site_id in enumerate(order):
            payload = site_payload(
                prepared / "hourly_native_by_site" / target_name / f"{site_id}.parquet",
                target_spec,
                preprocessing,
                bool(spec["target_history"]),
                int(config["target_history_fill_limit_hours"]),
            )
            all_index = select_indices(payload, target_spec, preparation_config, horizon, False, 0, 0)
            index = all_index[~payload["in_event"][all_index]] if bool(spec["train_non_event"]) else all_index
            event_removed += int(len(all_index) - len(index)) if epoch == 1 else 0
            cap = int(settings["train_window_cap_per_site"])
            if cap > 0 and len(index) > cap:
                index = np.random.default_rng(int(config["random_seed"]) + epoch * 100003 + site_number).choice(index, cap, replace=False)
            metadata = metadata_map[site_id] if spec["conditioning"] == "film" else None
            for dynamic, meta, target, _ in window_batches(
                payload, index, metadata, int(preparation_config["sequence_length_hours"]), horizon,
                int(settings["train_batch_size"]), True, int(config["random_seed"]) + epoch + site_number,
            ):
                dynamic_tensor = torch.from_numpy(dynamic).to(device)
                meta_tensor = torch.from_numpy(meta).to(device) if meta is not None else None
                target_tensor = torch.from_numpy(((target - target_mean) / target_scale).astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction = model(dynamic_tensor, meta_tensor)
                    loss = loss_function(prediction, target_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(target)
                epoch_n += len(target)
        validation_rmse = evaluate_validation(
            model, validation_ids, target_name, target_spec, spec, horizon, metadata_map,
            preprocessing, preparation_config, config, prepared, settings, device,
        )
        history.append({"epoch": epoch, "train_mse_scaled": epoch_loss / max(epoch_n, 1), "train_windows": epoch_n, "validation_rmse_native": validation_rmse})
        print(f"{target_name} {spec['model_id']} h={horizon} epoch={epoch} train_n={epoch_n} val_rmse={validation_rmse:.5f}", flush=True)
        if validation_rmse < best_validation - 1e-8:
            best_validation = validation_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_validation_rmse_native": best_validation, "train_event_windows_removed": event_removed}


def predict_model(
    model: nn.Module,
    split_frame: pd.DataFrame,
    target_name: str,
    target_spec: dict[str, Any],
    spec: dict[str, Any],
    horizon: int,
    metadata_map: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    preparation_config: dict[str, Any],
    config: dict[str, Any],
    prepared: Path,
    settings: dict[str, Any],
    output: Path,
    run_id: str,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    overall_values: dict[str, list[np.ndarray]] = {"observed": [], "predicted": []}
    site_rows = []
    target_mean = float(preprocessing["target"]["mean"])
    target_scale = float(preprocessing["target"]["scale"])
    for split_name in ("validation", "test"):
        for site_id in split_frame.loc[split_frame["split"].eq(split_name), "site_id"].astype(str):
            payload = site_payload(
                prepared / "hourly_native_by_site" / target_name / f"{site_id}.parquet",
                target_spec,
                preprocessing,
                bool(spec["target_history"]),
                int(config["target_history_fill_limit_hours"]),
            )
            index = select_indices(payload, target_spec, preparation_config, horizon, False, 0, 0)
            metadata = metadata_map[site_id] if spec["conditioning"] == "film" else None
            predicted_parts = []
            index_parts = []
            with torch.no_grad():
                for dynamic, meta, _target, target_index in window_batches(
                    payload, index, metadata, int(preparation_config["sequence_length_hours"]), horizon,
                    int(settings["prediction_batch_size"]), False, 0,
                ):
                    dynamic_tensor = torch.from_numpy(dynamic).to(device)
                    meta_tensor = torch.from_numpy(meta).to(device) if meta is not None else None
                    predicted_parts.append(model(dynamic_tensor, meta_tensor).float().cpu().numpy() * target_scale + target_mean)
                    index_parts.append(target_index)
            if not index_parts:
                continue
            index = np.concatenate(index_parts)
            predicted = np.concatenate(predicted_parts)
            observed = payload["target"][index]
            prediction = pd.DataFrame(
                {
                    "site_id": site_id,
                    "timestamp": payload["timestamp"][index],
                    "split": split_name,
                    "target": target_name,
                    "horizon_hours": horizon,
                    "model_id": spec["model_id"],
                    "observed_native": observed,
                    "predicted_native": predicted.astype(np.float32),
                    "in_event": payload["in_event"][index],
                    "event_direction": payload["event_direction"][index],
                    "event_id": payload["event_id"][index],
                }
            )
            write_parquet(output / "predictions_by_site" / run_id / split_name / f"{site_id}.parquet", prediction)
            metric = prediction_metrics(observed, predicted)
            site_rows.append({"run_id": run_id, "target": target_name, "split": split_name, "site_id": site_id, "horizon_hours": horizon, **metric})
            overall_values.setdefault(f"{split_name}_observed", []).append(observed)
            overall_values.setdefault(f"{split_name}_predicted", []).append(predicted)
    overall_rows = []
    for split_name in ("validation", "test"):
        observed_parts = overall_values.get(f"{split_name}_observed", [])
        predicted_parts = overall_values.get(f"{split_name}_predicted", [])
        if not observed_parts:
            continue
        metric = prediction_metrics(np.concatenate(observed_parts), np.concatenate(predicted_parts))
        subset = pd.DataFrame(site_rows)
        subset = subset[subset["split"].eq(split_name)]
        overall_rows.append(
            {
                "run_id": run_id,
                "model_id": spec["model_id"],
                "target": target_name,
                "split": split_name,
                "horizon_hours": horizon,
                "recurrent_type": spec["recurrent_type"],
                "conditioning": spec["conditioning"],
                "metadata_variant": spec["metadata_variant"],
                "target_history": bool(spec["target_history"]),
                "train_non_event": bool(spec["train_non_event"]),
                **metric,
                "sites": int(subset["site_id"].nunique()),
                "macro_site_rmse": float(subset["rmse"].mean()),
                "macro_site_mae": float(subset["mae"].mean()),
                "macro_site_nse": float(subset["nse"].mean()),
            }
        )
    return pd.DataFrame(overall_rows), pd.DataFrame(site_rows)


def audit_prediction_keys(output: Path, prepared: Path, run_rows: pd.DataFrame) -> list[str]:
    problems = []
    for row in run_rows.itertuples(index=False):
        run_id = row.run_id
        horizon = int(row.horizon_hours)
        target = row.target
        for split_name in ("validation", "test"):
            for path in (output / "predictions_by_site" / run_id / split_name).glob("*.parquet"):
                site_id = path.stem
                model = pd.read_parquet(path, columns=["site_id", "timestamp"])
                baseline_path = prepared / "baseline_predictions_by_site" / target / f"h{horizon:02d}" / split_name / f"{site_id}.parquet"
                baseline = pd.read_parquet(baseline_path, columns=["site_id", "timestamp"])
                same_site = model["site_id"].astype(str).to_numpy().tolist() == baseline["site_id"].astype(str).to_numpy().tolist()
                model_time = pd.to_datetime(model["timestamp"]).to_numpy(dtype="datetime64[ns]")
                baseline_time = pd.to_datetime(baseline["timestamp"]).to_numpy(dtype="datetime64[ns]")
                same_time = len(model_time) == len(baseline_time) and np.array_equal(model_time, baseline_time)
                if not (same_site and same_time):
                    problems.append(f"prediction keys differ from baseline: {run_id}/{split_name}/{site_id}")
                    return problems
    return problems


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = resolve(args.config)
    config = load_json(config_path)
    preparation_config = load_json(resolve(config["preparation_config"]))
    prepared = resolve(config["prepared_root"]) / args.mode
    if not (prepared / "_SUCCESS").exists():
        raise FileNotFoundError(f"Prepared leave-sites-out data is incomplete: {prepared}")
    output = resolve(config["output_root"]) / args.mode
    for directory in ("models", "reports", "predictions_by_site", "metadata"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Large-machine sequence experiments require CUDA")
    set_seed(int(config["random_seed"]))
    metadata_source = merge_metadata(config)
    settings = config[args.mode]
    selected_models = set(args.model)
    selected_targets = set(args.target)
    all_overall = []
    all_sites = []
    run_records = []
    for target_name, target_spec in preparation_config["targets"].items():
        if selected_targets and target_name not in selected_targets:
            continue
        split = pd.read_csv(prepared / "reports" / f"site_split_{target_name}.csv", dtype={"site_id": "string"})
        split["site_id"] = split["site_id"].map(PREPARE.normalise_site_id)
        preprocessing = load_json(prepared / "reports" / f"train_only_preprocessing_{target_name}.json")
        metadata_variants = {}
        for variant in ("no_coordinates", "with_coordinates"):
            fields = list(config["metadata_fields"])
            if variant == "no_coordinates":
                fields = [field for field in fields if field not in set(config["coordinate_fields"])]
            matrix, parameters = fit_metadata(metadata_source, split, fields, set(config["metadata_log1p_fields"]))
            metadata_variants[variant] = matrix
            write_json(output / "metadata" / f"{target_name}_{variant}_train_only_transform.json", parameters)
        for spec in config["models"]:
            if selected_models and spec["model_id"] not in selected_models:
                continue
            if target_name not in spec["targets"]:
                continue
            horizons = list(spec["horizons"])
            if args.mode == "smoke":
                horizons = [value for value in horizons if int(value) == 1]
            for horizon in horizons:
                horizon = int(horizon)
                run_id = f"{target_name}_{spec['model_id']}_H{horizon:02d}"
                success = output / "models" / f"{run_id}_SUCCESS.json"
                model_path = output / "models" / f"{run_id}.pt"
                metadata_map = metadata_variants.get(spec["metadata_variant"], {site_id: np.empty(0, dtype=np.float32) for site_id in split["site_id"]})
                first_site = str(split.iloc[0]["site_id"])
                sample_payload = site_payload(
                    prepared / "hourly_native_by_site" / target_name / f"{first_site}.parquet",
                    target_spec, preprocessing, bool(spec["target_history"]), int(config["target_history_fill_limit_hours"]),
                )
                metadata_size = len(metadata_map[first_site]) if spec["conditioning"] == "film" else 0
                # Reuse the same target/horizon seed across architecture and
                # conditioning variants. In particular, all/non-event pairs
                # begin from identical weights and see the same site order.
                run_seed = int(config["random_seed"]) + (0 if target_name == "ph" else 100000) + horizon * 1000
                set_seed(run_seed)
                model = SiteAgnosticFiLMRNN(spec, sample_payload["features"].shape[1], metadata_size, config).to(device)
                if args.resume and success.exists() and model_path.exists():
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint)
                    training_info = load_json(success)
                    history_path = output / "reports" / f"training_history_{run_id}.csv"
                    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
                else:
                    model, history, training_info = train_model(
                        model,
                        split.loc[split["split"].eq("train"), "site_id"].astype(str).tolist(),
                        split.loc[split["split"].eq("validation"), "site_id"].astype(str).tolist(),
                        target_name, target_spec, spec, horizon, metadata_map, preprocessing,
                        preparation_config, config, prepared, settings, device,
                    )
                    torch.save(model.state_dict(), model_path)
                    write_csv(output / "reports" / f"training_history_{run_id}.csv", history)
                    training_info.update(
                        {
                            "run_id": run_id,
                            "feature_names": sample_payload["feature_names"],
                            "metadata_size": metadata_size,
                            "no_site_embedding": True,
                            "bf16": torch.cuda.is_bf16_supported(),
                            "site_split_sha256": sha256(prepared / "reports" / f"site_split_{target_name}.csv"),
                        }
                    )
                    write_json(success, training_info)
                overall, site = predict_model(
                    model, split, target_name, target_spec, spec, horizon, metadata_map, preprocessing,
                    preparation_config, config, prepared, settings, output, run_id, device,
                )
                overall["train_event_windows_removed"] = int(training_info.get("train_event_windows_removed", 0))
                all_overall.append(overall)
                all_sites.append(site)
                run_records.append({"run_id": run_id, "target": target_name, "horizon_hours": horizon, "model_id": spec["model_id"]})
                write_csv(output / "reports" / f"overall_metrics_{run_id}.csv", overall)
                write_csv(output / "reports" / f"site_metrics_{run_id}.csv", site)
    overall = pd.concat(all_overall, ignore_index=True) if all_overall else pd.DataFrame()
    sites = pd.concat(all_sites, ignore_index=True) if all_sites else pd.DataFrame()
    runs = pd.DataFrame(run_records)
    write_csv(output / "reports" / "model_metrics_all.csv", overall)
    write_csv(output / "reports" / "site_metrics_all.csv", sites)
    write_csv(output / "reports" / "run_manifest.csv", runs)
    problems = audit_prediction_keys(output, prepared, runs)
    if not overall.empty and (overall[["rmse", "mae"]].isna().any().any() or ~np.isfinite(overall[["rmse", "mae"]].to_numpy()).all()):
        problems.append("non-finite overall metrics")
    test = overall[overall["split"].eq("test")].sort_values(["target", "horizon_hours", "rmse"])
    report = "\n".join(
        [
            "# Site-disjoint GRU/LSTM-FiLM report",
            "",
            "All test sites are absent from training and validation. Models contain no site embedding.",
            "Metadata preprocessing is fitted on training sites only. Coordinates are an explicit sensitivity variant.",
            "Target-history models and exogenous-only models are reported separately.",
            "",
            PREPARE.frame_to_markdown(test[["target", "horizon_hours", "model_id", "target_history", "train_non_event", "metadata_variant", "n", "sites", "rmse", "mae", "nse", "macro_site_rmse", "macro_site_nse"]]),
        ]
    ) + "\n"
    (output / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "pass" if not problems else "fail",
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(),
        "no_site_embedding": True,
        "prediction_keys_match_baseline": not problems,
        "problems": problems,
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output / "reports" / "independent_audit.json", audit)
    if problems:
        raise RuntimeError(problems)
    write_json(output / "_SUCCESS", audit)
    print(report)


if __name__ == "__main__":
    main()
