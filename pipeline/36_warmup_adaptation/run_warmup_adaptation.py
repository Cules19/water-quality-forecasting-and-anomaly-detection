"""Run new-site warm-up adaptation curves under a frozen protocol.

The generic recurrent model and preprocessing are frozen artefacts from the
teacher-aligned LSTM-FiLM rainfall experiment. Each held-out site is adapted
independently using labels strictly before its warm-up cutoff. All warm-up
durations are evaluated on the site's exact same post-12-month target keys.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAIN_PATH = (
    PROJECT_ROOT
    / "pipeline"
    / "31_common1000_rainfall_ablation"
    / "run_rainfall_ablation.py"
)
AUDIT_PATH = (
    PROJECT_ROOT
    / "pipeline"
    / "35_warmup_feasibility"
    / "audit_warmup_feasibility.py"
)


def import_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


RAIN = import_module("warmup_rainfall_runner", RAIN_PATH)
BASE = RAIN.BASE
FEASIBILITY = import_module("warmup_feasibility_audit", AUDIT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            PROJECT_ROOT
            / "config"
            / "common1000_warmup_adaptation_validation_v1.json"
        ),
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--resume", action="store_true")
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: Any) -> int:
    token = "|".join(str(value) for value in values).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


class SiteFiLMAffineAdapter(nn.Module):
    """Add site-specific FiLM offsets and a positive output affine transform."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        if base_model.film is None:
            raise ValueError("Warm-up adapter requires a FiLM-conditioned model")
        self.base_model = base_model
        layers = len(base_model.layers)
        hidden = int(base_model.film.hidden_size)
        self.film_gamma_delta = nn.Parameter(torch.zeros(layers, hidden))
        self.film_beta_delta = nn.Parameter(torch.zeros(layers, hidden))
        self.output_log_scale = nn.Parameter(torch.zeros(()))
        self.output_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self, dynamic: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.base_model.film(metadata)
        gamma = gamma + self.film_gamma_delta.unsqueeze(0)
        beta = beta + self.film_beta_delta.unsqueeze(0)
        value = dynamic
        for index, layer in enumerate(self.base_model.layers):
            value, _ = layer(value)
            value = (
                gamma[:, index].unsqueeze(1) * value
                + beta[:, index].unsqueeze(1)
            )
            if index < len(self.base_model.layers) - 1:
                value = self.base_model.inter_layer_dropout(value)
        prediction = self.base_model.head(value[:, -1]).squeeze(1)
        return torch.exp(self.output_log_scale) * prediction + self.output_bias


def configure_strategy(
    model: SiteFiLMAffineAdapter, strategy: str
) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if strategy == "film_affine":
        for name in (
            "film_gamma_delta",
            "film_beta_delta",
            "output_log_scale",
            "output_bias",
        ):
            getattr(model, name).requires_grad = True
    elif strategy == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif strategy != "zero_shot":
        raise ValueError(f"Unknown adaptation strategy: {strategy}")
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def apply_saved_metadata_transform(
    raw_metadata: pd.DataFrame,
    site_ids: list[str],
    parameters: dict[str, Any],
    log_fields: set[str],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for site_id in site_ids:
        values: list[float] = []
        for field in parameters["fields"]:
            name = str(field["field"])
            raw = (
                pd.to_numeric(
                    pd.Series([raw_metadata[name].get(site_id, np.nan)]),
                    errors="coerce",
                ).iloc[0]
                if name in raw_metadata
                else np.nan
            )
            if name in log_fields and np.isfinite(raw):
                raw = np.log1p(raw) if raw >= 0 else np.nan
            missing = not np.isfinite(raw)
            standardized = (
                0.0
                if missing
                else (float(raw) - float(field["mean"])) / float(field["scale"])
            )
            values.extend([standardized, float(missing)])
        result[site_id] = np.asarray(values, dtype=np.float32)
    return result


def temporal_train_validation_split(
    indices: np.ndarray,
    fraction: float,
    minimum_validation: int,
    maximum_validation: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    validation_count = int(math.ceil(len(ordered) * fraction))
    validation_count = max(validation_count, int(minimum_validation))
    validation_count = min(validation_count, int(maximum_validation))
    validation_count = min(validation_count, len(ordered) - 1)
    if validation_count <= 0:
        raise ValueError("Not enough adaptation targets for temporal validation")
    return ordered[:-validation_count], ordered[-validation_count:]


def make_model(
    model_config: dict[str, Any],
    dynamic_size: int,
    metadata_size: int,
    checkpoint: Path,
    device: torch.device,
    strategy: str,
) -> SiteFiLMAffineAdapter:
    specification = {
        "model_id": "LSTM_RAIN_FINAL",
        "recurrent_type": "lstm",
        "recurrent_layers": int(model_config.get("recurrent_layers", 1)),
        "conditioning": "film",
        "metadata_variant": "no_coordinates",
        "target_history": True,
        "train_non_event": False,
    }
    base_model = BASE.SiteAgnosticFiLMRNN(
        specification, dynamic_size, metadata_size, model_config
    )
    base_model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model = SiteFiLMAffineAdapter(base_model).to(device)
    configure_strategy(model, strategy)
    return model


def set_training_mode(model: SiteFiLMAffineAdapter, strategy: str) -> None:
    if strategy == "full_finetune":
        model.train()
    else:
        # Gradients still flow to the site-specific adapter parameters while
        # the frozen recurrent/head dropout remains disabled.
        model.eval()


def predict_scaled(
    model: SiteFiLMAffineAdapter,
    payload: dict[str, Any],
    indices: np.ndarray,
    metadata: np.ndarray,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(preprocessing["target"]["mean"])
    scale = float(preprocessing["target"]["scale"])
    model.eval()
    predicted_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []
    with torch.no_grad():
        for dynamic, meta, target, _ in RAIN.window_batches(
            payload,
            np.asarray(indices, dtype=np.int64),
            metadata,
            model_config,
            int(batch_size),
            False,
            0,
        ):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    torch.from_numpy(dynamic).to(device),
                    torch.from_numpy(meta).to(device),
                )
            predicted_parts.append(
                prediction.float().cpu().numpy() * scale + mean
            )
            observed_parts.append(target.astype(np.float64))
    if not predicted_parts:
        raise RuntimeError("No prediction windows")
    return np.concatenate(observed_parts), np.concatenate(predicted_parts)


def validation_rmse(
    model: SiteFiLMAffineAdapter,
    payload: dict[str, Any],
    indices: np.ndarray,
    metadata: np.ndarray,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> float:
    observed, predicted = predict_scaled(
        model,
        payload,
        indices,
        metadata,
        model_config,
        preprocessing,
        batch_size,
        device,
    )
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def train_epoch(
    model: SiteFiLMAffineAdapter,
    payload: dict[str, Any],
    indices: np.ndarray,
    metadata: np.ndarray,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    settings: dict[str, Any],
    device: torch.device,
    seed: int,
    strategy: str,
) -> tuple[float, int]:
    set_training_mode(model, strategy)
    loss_function = nn.MSELoss()
    target_mean = float(preprocessing["target"]["mean"])
    target_scale = float(preprocessing["target"]["scale"])
    loss_sum = 0.0
    count = 0
    for dynamic, meta, target, _ in RAIN.window_batches(
        payload,
        np.asarray(indices, dtype=np.int64),
        metadata,
        model_config,
        int(settings["batch_size"]),
        True,
        seed,
    ):
        dynamic_tensor = torch.from_numpy(dynamic).to(device)
        metadata_tensor = torch.from_numpy(meta).to(device)
        target_tensor = torch.from_numpy(
            ((target - target_mean) / target_scale).astype(np.float32)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(dynamic_tensor, metadata_tensor)
            loss = loss_function(prediction, target_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(settings["gradient_clip_norm"]),
        )
        optimizer.step()
        loss_sum += float(loss.item()) * len(target)
        count += len(target)
    return loss_sum / max(count, 1), count


def fit_adapter(
    strategy: str,
    payload: dict[str, Any],
    adaptation_indices: np.ndarray,
    metadata: np.ndarray,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    experiment_config: dict[str, Any],
    settings: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[SiteFiLMAffineAdapter, dict[str, Any], list[dict[str, Any]]]:
    train_indices, local_validation_indices = temporal_train_validation_split(
        adaptation_indices,
        float(experiment_config["local_validation_fraction"]),
        int(experiment_config["minimum_local_validation_targets"]),
        int(experiment_config["maximum_local_validation_targets"]),
    )
    learning_rate = float(settings[f"{strategy}_learning_rate"])
    model = make_model(
        model_config,
        payload["features"].shape[1],
        len(metadata),
        checkpoint,
        device,
        strategy,
    )
    trainable_parameters = configure_strategy(model, strategy)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(settings["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        loss, train_count = train_epoch(
            model,
            payload,
            train_indices,
            metadata,
            model_config,
            preprocessing,
            optimizer,
            settings,
            device,
            seed + epoch,
            strategy,
        )
        score = validation_rmse(
            model,
            payload,
            local_validation_indices,
            metadata,
            model_config,
            preprocessing,
            int(settings["prediction_batch_size"]),
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": loss,
                "train_targets": train_count,
                "local_validation_targets": len(local_validation_indices),
                "local_validation_rmse_native": score,
            }
        )
        if score < best_rmse - 1e-6:
            best_rmse = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if (
            epoch >= int(settings["minimum_epochs"])
            and stale >= int(settings["early_stopping_patience"])
        ):
            break

    # Refit from the generic checkpoint on every allowed adaptation target for
    # the locally selected number of epochs. No post-cutoff target is involved.
    model = make_model(
        model_config,
        payload["features"].shape[1],
        len(metadata),
        checkpoint,
        device,
        strategy,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(settings["weight_decay"]),
    )
    refit_history = []
    for epoch in range(1, max(best_epoch, 1) + 1):
        loss, train_count = train_epoch(
            model,
            payload,
            adaptation_indices,
            metadata,
            model_config,
            preprocessing,
            optimizer,
            settings,
            device,
            seed + 10000 + epoch,
            strategy,
        )
        refit_history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": loss,
                "train_targets": train_count,
            }
        )

    fitting = {
        "learning_rate": learning_rate,
        "trainable_parameters": trainable_parameters,
        "selection_train_targets": len(train_indices),
        "selection_validation_targets": len(local_validation_indices),
        "best_local_validation_rmse_native": best_rmse,
        "best_epoch": max(best_epoch, 1),
        "selection_epochs_ran": len(history),
        "refit_targets": len(adaptation_indices),
        "output_scale": float(torch.exp(model.output_log_scale).detach().cpu()),
        "output_bias_scaled": float(model.output_bias.detach().cpu()),
        "film_gamma_delta_l2": float(
            torch.linalg.vector_norm(model.film_gamma_delta).detach().cpu()
        ),
        "film_beta_delta_l2": float(
            torch.linalg.vector_norm(model.film_beta_delta).detach().cpu()
        ),
    }
    combined_history = [
        {"phase": "selection", **row} for row in history
    ] + [{"phase": "refit", **row} for row in refit_history]
    return model, fitting, combined_history


def metric_payload(
    observed: np.ndarray,
    predicted: np.ndarray,
    tolerance: float | None = None,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    metrics = RAIN.prediction_metrics(observed, predicted)
    payload = {
        **metrics,
        "sse": float(np.square(residual).sum()),
        "sae": float(np.abs(residual).sum()),
        "observed_sum": float(observed.sum()),
        "observed_sum_sq": float(np.square(observed).sum()),
    }
    if tolerance is not None:
        within = np.abs(residual) <= float(tolerance)
        payload.update(
            {
                "absolute_error_tolerance": float(tolerance),
                "within_tolerance_count": int(within.sum()),
                "within_tolerance_fraction": float(within.mean()),
            }
        )
    return payload


def bootstrap_mean_ci(
    values: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        sample = generator.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[sample].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate_results(
    site_results: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    for (order, label, strategy), group in site_results.groupby(
        ["warmup_order", "warmup_label", "strategy"], sort=False
    ):
        total_n = int(group["n"].sum())
        total_sse = float(group["sse"].sum())
        total_sae = float(group["sae"].sum())
        observed_sum = float(group["observed_sum"].sum())
        observed_sum_sq = float(group["observed_sum_sq"].sum())
        denominator = observed_sum_sq - observed_sum**2 / total_n
        row = {
                "warmup_order": order,
                "warmup_label": label,
                "strategy": strategy,
                "sites": group["site_id"].nunique(),
                "evaluated_targets": total_n,
                "pooled_rmse": math.sqrt(total_sse / total_n),
                "pooled_mae": total_sae / total_n,
                "pooled_nse": (
                    1.0 - total_sse / denominator
                    if denominator > 0
                    else float("nan")
                ),
                "macro_site_rmse": float(group["rmse"].mean()),
                "macro_site_mae": float(group["mae"].mean()),
                "macro_site_nse": float(group["nse"].mean()),
                "median_best_epoch": (
                    float(group["best_epoch"].median())
                    if strategy != "zero_shot"
                    else 0.0
                ),
                "trainable_parameters": int(
                    group["trainable_parameters"].max()
                ),
            }
        if "within_tolerance_count" in group:
            row["absolute_error_tolerance"] = float(
                group["absolute_error_tolerance"].iloc[0]
            )
            row["pooled_within_tolerance_fraction"] = float(
                group["within_tolerance_count"].sum() / total_n
            )
            row["macro_site_within_tolerance_fraction"] = float(
                group["within_tolerance_fraction"].mean()
            )
        curve_rows.append(row)
    curve = pd.DataFrame(curve_rows).sort_values(
        ["warmup_order", "strategy"]
    )

    paired_rows = []
    baseline = site_results.loc[
        site_results["strategy"].eq("zero_shot"),
        ["site_id", "warmup_order", "warmup_label", "rmse"],
    ].rename(columns={"rmse": "zero_shot_rmse"})
    for strategy in ("film_affine", "full_finetune"):
        challenger = site_results.loc[
            site_results["strategy"].eq(strategy),
            ["site_id", "warmup_order", "warmup_label", "rmse"],
        ].rename(columns={"rmse": "adapted_rmse"})
        paired = baseline.merge(
            challenger,
            on=["site_id", "warmup_order", "warmup_label"],
            validate="one_to_one",
        )
        paired["delta_rmse"] = (
            paired["adapted_rmse"] - paired["zero_shot_rmse"]
        )
        for (order, label), group in paired.groupby(
            ["warmup_order", "warmup_label"], sort=False
        ):
            delta = group["delta_rmse"].to_numpy(dtype=np.float64)
            lower, upper = bootstrap_mean_ci(
                delta,
                replicates,
                stable_seed(seed, strategy, label),
            )
            paired_rows.append(
                {
                    "warmup_order": order,
                    "warmup_label": label,
                    "strategy": strategy,
                    "sites": len(group),
                    "mean_site_delta_rmse_adapted_minus_zero": float(
                        delta.mean()
                    ),
                    "median_site_delta_rmse_adapted_minus_zero": float(
                        np.median(delta)
                    ),
                    "bootstrap_95_ci_lower": lower,
                    "bootstrap_95_ci_upper": upper,
                    "sites_improved": int((delta < 0).sum()),
                    "sites_worsened": int((delta > 0).sum()),
                }
            )
    paired_summary = pd.DataFrame(paired_rows).sort_values(
        ["warmup_order", "strategy"]
    )
    return curve, paired_summary


def build_report(
    config: dict[str, Any],
    curve: pd.DataFrame,
    paired: pd.DataFrame,
    audit: dict[str, Any],
) -> str:
    split = str(audit["evaluation_split"])
    if split == "validation":
        status_text = (
            f"This run uses **{audit['sites']} strict validation sites** only. "
            "The held-out test-site files were not opened."
        )
        closing_text = (
            "These are validation results. Learning rates, epoch policy and the "
            "preferred adaptation method must be frozen before any test run."
        )
        title = "# Validation-only new-site warm-up adaptation report"
    else:
        status_text = (
            f"This is the single frozen-protocol evaluation on "
            f"**{audit['sites']} held-out test sites**."
        )
        closing_text = (
            "These are held-out test results. The adaptation protocol was frozen "
            "from validation before these post-cutoff labels were evaluated."
        )
        title = "# Held-out test new-site warm-up adaptation report"
    lines = [
        title,
        "",
        "## Status",
        "",
        status_text,
        "",
        "The generic LSTM-FiLM recurrent core comes from the frozen "
        "teacher-aligned rainfall model. All adaptation labels precede the "
        "site-specific warm-up cutoff, and every duration uses the same "
        "post-12-month evaluation keys.",
        "",
        "## Curve",
        "",
        "| Warm-up | Strategy | Pooled RMSE | Macro site RMSE | Median epoch |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in curve.itertuples(index=False):
        lines.append(
            f"| {row.warmup_label} | {row.strategy} | "
            f"{row.pooled_rmse:.6f} | {row.macro_site_rmse:.6f} | "
            f"{row.median_best_epoch:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Paired site-level change from zero-shot",
            "",
            "Negative RMSE deltas favour adaptation.",
            "",
            "| Warm-up | Strategy | Mean delta | 95% site-bootstrap CI | Improved sites |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.warmup_label} | {row.strategy} | "
            f"{row.mean_site_delta_rmse_adapted_minus_zero:.6f} | "
            f"[{row.bootstrap_95_ci_lower:.6f}, "
            f"{row.bootstrap_95_ci_upper:.6f}] | "
            f"{row.sites_improved}/{row.sites} |"
        )
    lines.extend(
        [
            "",
            "## Adaptation definitions",
            "",
            "- `zero_shot`: the generic checkpoint is unchanged.",
            "- `film_affine`: the generic recurrent core, metadata FiLM generator "
            "and prediction head are frozen; 128 site-specific FiLM scale/shift "
            "offsets plus a two-parameter positive output affine transform are updated.",
            "- `full_finetune`: all generic-model weights and the same site-specific "
            "adapter parameters are updated.",
            "",
            closing_text,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    # Allow controlled experiment configs to inherit the frozen warm-up
    # protocol while overriding only their checkpoint/payload definition.
    # Configs without `base_config` retain identical behaviour.
    config = RAIN.load_config(config_path)
    settings = config[args.mode]
    split = str(config["evaluation_split"])
    if split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    if split == "test" and args.mode != "full":
        raise ValueError("Held-out test evaluation is permitted in full mode only")

    model_config = RAIN.load_config(resolve(config["base_model_config"]))
    feasibility_config = load_json(resolve(config["feasibility_config"]))
    preprocessing = load_json(resolve(config["preprocessing_path"]))
    metadata_parameters = load_json(resolve(config["metadata_transform_path"]))
    checkpoint = resolve(config["base_checkpoint"])
    checkpoint_digest = sha256(checkpoint)
    output = resolve(config["output_root"]) / args.mode
    frozen_protocol: dict[str, Any] | None = None
    if split == "test":
        if "frozen_protocol_path" not in config:
            raise ValueError("Test configuration requires frozen_protocol_path")
        frozen_protocol = load_json(resolve(config["frozen_protocol_path"]))
        if frozen_protocol.get("status") != "frozen_on_validation":
            raise ValueError("Adaptation protocol is not frozen on validation")
        frozen_checks = {
            "base_checkpoint_sha256": checkpoint_digest,
            "warmup_labels": list(settings["warmup_labels"]),
            "local_validation_fraction": config["local_validation_fraction"],
            "minimum_local_validation_targets": config[
                "minimum_local_validation_targets"
            ],
            "maximum_local_validation_targets": config[
                "maximum_local_validation_targets"
            ],
            "maximum_epochs": settings["maximum_epochs"],
            "minimum_epochs": settings["minimum_epochs"],
            "early_stopping_patience": settings["early_stopping_patience"],
            "film_affine_learning_rate": settings[
                "film_affine_learning_rate"
            ],
            "full_finetune_learning_rate": settings[
                "full_finetune_learning_rate"
            ],
            "weight_decay": settings["weight_decay"],
            "gradient_clip_norm": settings["gradient_clip_norm"],
        }
        mismatches = {
            key: {
                "test_config": value,
                "frozen_protocol": frozen_protocol.get(key),
            }
            for key, value in frozen_checks.items()
            if value != frozen_protocol.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Test configuration differs from frozen protocol: {mismatches}"
            )

    cohort = pd.read_csv(
        resolve(model_config["primary_cohort_path"]),
        dtype={"site_id": "string", "split": "string"},
    )
    strict_cohort = pd.read_csv(
        resolve(config["strict_cohort_path"]),
        dtype={"site_id": "string", "split": "string"},
    )
    feasibility = pd.read_csv(
        resolve(config["site_period_feasibility_path"]),
        dtype={"site_id": "string", "split": "string"},
        parse_dates=[
            "deployment_start",
            "warmup_cutoff_exclusive",
            "fixed_evaluation_start",
        ],
    )
    sites = sorted(
        strict_cohort.loc[
            strict_cohort["split"].eq(split)
            & strict_cohort["full_fixed_curve_eligible"].eq(True),
            "site_id",
        ].astype(str)
    )
    limit = int(settings["site_limit"])
    if limit > 0:
        sites = sites[:limit]
    warmup_labels = list(settings["warmup_labels"])
    feasibility = feasibility.loc[
        feasibility["split"].eq(split)
        & feasibility["site_id"].isin(sites)
        & feasibility["warmup_label"].isin(warmup_labels)
    ].copy()

    raw_metadata = BASE.merge_metadata(model_config)
    metadata_map = apply_saved_metadata_transform(
        raw_metadata,
        sites,
        metadata_parameters,
        set(model_config["metadata_log1p_fields"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Warm-up adaptation requires CUDA")
    input_root = resolve(model_config["merged_input_root"])
    strategies = list(config["strategies"])

    results: list[dict[str, Any]] = []
    key_mismatches: list[str] = []
    overlap_count = 0
    for site_number, site_id in enumerate(sites, start=1):
        payload = RAIN.site_payload(
            input_root / f"{site_id}.parquet",
            list(config["feature_names"]),
            preprocessing,
            model_config,
        )
        common_indices = RAIN.comparison_indices(payload["frame"], model_config)
        common_timestamps = pd.DatetimeIndex(payload["timestamp"][common_indices])
        site_manifest = strict_cohort.loc[
            strict_cohort["site_id"].eq(site_id)
            & strict_cohort["split"].eq(split)
        ].iloc[0]
        if (
            FEASIBILITY.key_digest(site_id, common_timestamps)
            != str(site_manifest["common_key_sha256"])
        ):
            key_mismatches.append(f"{site_id}:common")
        zero_prediction: tuple[np.ndarray, np.ndarray] | None = None

        site_periods = feasibility.loc[
            feasibility["site_id"].eq(site_id)
        ].sort_values("warmup_order")
        for period in site_periods.itertuples(index=False):
            cutoff = pd.Timestamp(period.warmup_cutoff_exclusive)
            fixed_start = pd.Timestamp(period.fixed_evaluation_start)
            adaptation_indices = common_indices[common_timestamps < cutoff]
            fixed_indices = common_indices[common_timestamps >= fixed_start]
            adaptation_timestamps = pd.DatetimeIndex(
                payload["timestamp"][adaptation_indices]
            )
            fixed_timestamps = pd.DatetimeIndex(
                payload["timestamp"][fixed_indices]
            )
            if (
                FEASIBILITY.key_digest(site_id, adaptation_timestamps)
                != str(period.adaptation_key_sha256)
            ):
                key_mismatches.append(
                    f"{site_id}:{period.warmup_label}:adaptation"
                )
            if (
                FEASIBILITY.key_digest(site_id, fixed_timestamps)
                != str(period.fixed_evaluation_key_sha256)
            ):
                key_mismatches.append(
                    f"{site_id}:{period.warmup_label}:evaluation"
                )
            overlap = len(
                np.intersect1d(adaptation_indices, fixed_indices)
            )
            overlap_count += int(overlap)

            for strategy in strategies:
                shard = (
                    output
                    / "shards"
                    / split
                    / site_id
                    / f"{int(period.warmup_order):02d}_{period.warmup_label}_{strategy}.json"
                )
                if args.resume and shard.exists():
                    results.append(load_json(shard))
                    continue

                seed = stable_seed(
                    config["random_seed"],
                    site_id,
                    period.warmup_label,
                    strategy,
                )
                BASE.set_seed(seed)
                if strategy == "zero_shot":
                    if zero_prediction is None:
                        model = make_model(
                            model_config,
                            payload["features"].shape[1],
                            len(metadata_map[site_id]),
                            checkpoint,
                            device,
                            strategy,
                        )
                        zero_prediction = predict_scaled(
                            model,
                            payload,
                            fixed_indices,
                            metadata_map[site_id],
                            model_config,
                            preprocessing,
                            int(settings["prediction_batch_size"]),
                            device,
                        )
                    observed, predicted = zero_prediction
                    fitting = {
                        "learning_rate": 0.0,
                        "trainable_parameters": 0,
                        "selection_train_targets": 0,
                        "selection_validation_targets": 0,
                        "best_local_validation_rmse_native": float("nan"),
                        "best_epoch": 0,
                        "selection_epochs_ran": 0,
                        "refit_targets": 0,
                        "output_scale": 1.0,
                        "output_bias_scaled": 0.0,
                        "film_gamma_delta_l2": 0.0,
                        "film_beta_delta_l2": 0.0,
                    }
                    history = []
                else:
                    model, fitting, history = fit_adapter(
                        strategy,
                        payload,
                        adaptation_indices,
                        metadata_map[site_id],
                        model_config,
                        preprocessing,
                        config,
                        settings,
                        checkpoint,
                        device,
                        seed,
                    )
                    observed, predicted = predict_scaled(
                        model,
                        payload,
                        fixed_indices,
                        metadata_map[site_id],
                        model_config,
                        preprocessing,
                        int(settings["prediction_batch_size"]),
                        device,
                    )

                result = {
                    "site_id": site_id,
                    "split": split,
                    "warmup_order": int(period.warmup_order),
                    "warmup_label": str(period.warmup_label),
                    "strategy": strategy,
                    "adaptation_targets": len(adaptation_indices),
                    "fixed_evaluation_targets": len(fixed_indices),
                    "maximum_adaptation_target": (
                        adaptation_timestamps.max().isoformat()
                    ),
                    "minimum_fixed_evaluation_target": (
                        fixed_timestamps.min().isoformat()
                    ),
                    "adaptation_key_sha256": str(
                        period.adaptation_key_sha256
                    ),
                    "fixed_evaluation_key_sha256": str(
                        period.fixed_evaluation_key_sha256
                    ),
                    "checkpoint_sha256": checkpoint_digest,
                    **fitting,
                    **metric_payload(
                        observed,
                        predicted,
                        (
                            float(config["absolute_error_tolerance"])
                            if "absolute_error_tolerance" in config
                            else None
                        ),
                    ),
                    "history": history,
                }
                write_json(shard, result)
                results.append(result)
                print(
                    f"[{site_number}/{len(sites)}] {site_id} "
                    f"{period.warmup_label} {strategy} "
                    f"rmse={result['rmse']:.6f} "
                    f"epoch={result['best_epoch']}",
                    flush=True,
                )

    site_rows = []
    history_rows = []
    for result in results:
        site_rows.append(
            {key: value for key, value in result.items() if key != "history"}
        )
        for row in result["history"]:
            history_rows.append(
                {
                    "site_id": result["site_id"],
                    "warmup_order": result["warmup_order"],
                    "warmup_label": result["warmup_label"],
                    "strategy": result["strategy"],
                    **row,
                }
            )
    site_results = pd.DataFrame(site_rows).sort_values(
        ["warmup_order", "site_id", "strategy"]
    )
    history = pd.DataFrame(history_rows)
    curve, paired = aggregate_results(
        site_results,
        int(config["bootstrap_replicates"]),
        int(config["random_seed"]),
    )
    expected_results = len(sites) * len(warmup_labels) * len(strategies)
    audit = {
        "status": "pass",
        "mode": args.mode,
        "evaluation_split": split,
        "sites": len(sites),
        "warmup_periods": len(warmup_labels),
        "strategies": len(strategies),
        "result_rows": len(site_results),
        "expected_result_rows": expected_results,
        "common_or_period_key_mismatches": key_mismatches,
        "adaptation_evaluation_overlap_targets": overlap_count,
        "base_checkpoint_sha256": checkpoint_digest,
        "test_site_files_opened": split == "test",
        "pre_cutoff_test_targets_used_for_site_adaptation": split == "test",
        "post_cutoff_test_targets_used_for_model_or_epoch_selection": False,
        "frozen_protocol_path": (
            str(resolve(config["frozen_protocol_path"]))
            if split == "test"
            else None
        ),
        "frozen_protocol_verified": split != "test" or frozen_protocol is not None,
    }
    if (
        len(site_results) != expected_results
        or key_mismatches
        or overlap_count
    ):
        audit["status"] = "fail"

    reports = output / "reports"
    write_csv(reports / "warmup_site_metrics.csv", site_results)
    write_csv(reports / "warmup_training_history.csv", history)
    write_csv(reports / "warmup_curve_summary.csv", curve)
    write_csv(reports / "warmup_paired_site_summary.csv", paired)
    write_json(reports / "warmup_adaptation_audit.json", audit)
    report_name = (
        "WARMUP_ADAPTATION_VALIDATION_REPORT.md"
        if split == "validation"
        else "WARMUP_ADAPTATION_TEST_REPORT.md"
    )
    (reports / report_name).write_text(
        build_report(config, curve, paired, audit), encoding="utf-8"
    )
    if audit["status"] != "pass":
        raise RuntimeError("Warm-up adaptation audit failed")
    print(
        f"Warm-up {split} complete: sites={len(sites)} "
        f"periods={len(warmup_labels)} results={len(site_results)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
