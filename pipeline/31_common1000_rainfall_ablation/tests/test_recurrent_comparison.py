from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "run_rainfall_ablation.py"
SPEC = importlib.util.spec_from_file_location("rainfall_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def model_config() -> dict[str, float | int]:
    return {
        "hidden_size": 64,
        "film_hidden_size": 64,
        "head_hidden_size": 32,
        "dropout": 0.1,
    }


def model_spec(recurrent_type: str) -> dict[str, str | int | bool]:
    return {
        "recurrent_type": recurrent_type,
        "recurrent_layers": 1,
        "conditioning": "film",
        "metadata_variant": "no_coordinates",
        "target_history": True,
    }


def test_gru_and_lstm_have_matched_interfaces() -> None:
    dynamic = torch.randn(4, 48, 18)
    metadata = torch.randn(4, 28)
    for recurrent_type in ("gru", "lstm"):
        model = MODULE.BASE.SiteAgnosticFiLMRNN(
            model_spec(recurrent_type),
            dynamic_size=18,
            metadata_size=28,
            config=model_config(),
        )
        assert model(dynamic, metadata).shape == (4,)
        assert not any(
            isinstance(layer, torch.nn.Embedding) for layer in model.modules()
        )


def test_lstm_parameter_count_exceeds_matched_hidden_gru() -> None:
    gru = MODULE.BASE.SiteAgnosticFiLMRNN(
        model_spec("gru"), 18, 28, model_config()
    )
    lstm = MODULE.BASE.SiteAgnosticFiLMRNN(
        model_spec("lstm"), 18, 28, model_config()
    )
    gru_parameters = sum(value.numel() for value in gru.parameters())
    lstm_parameters = sum(value.numel() for value in lstm.parameters())
    assert lstm_parameters > gru_parameters


def test_teacher_aligned_config_defines_balanced_two_by_two_design() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "common1000_recurrent_comparison_teacher_aligned_do_converted_v1.json"
    )
    config = MODULE.load_config(config_path)
    definitions = {
        (
            str(model["recurrent_type"]),
            "rain" if "rain_00045" in model["feature_names"] else "base",
        )
        for model in config["models"]
    }
    assert definitions == {
        ("gru", "base"),
        ("gru", "rain"),
        ("lstm", "base"),
        ("lstm", "rain"),
    }
    assert config["sequence_length_hours"] == 48
    assert config["learning_rate_scheduler_enabled"]
    assert config["gradient_clip_norm"] == 1.0


def test_lstm_lr_sweep_changes_only_learning_rate() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "common1000_lstm_film_rain_lr_sweep_teacher_aligned_v1.json"
    )
    config = MODULE.load_config(config_path)
    models = config["models"]
    assert len(models) == 3
    assert {
        float(model["training_overrides"]["learning_rate"]) for model in models
    } == {1e-3, 3e-4, 1e-4}
    assert {str(model["recurrent_type"]) for model in models} == {"lstm"}
    assert {
        tuple(model["feature_names"]) for model in models
    } == {
        ("x_00010", "x_00095", "x_00300", "x_63680", "rain_00045")
    }
    assert {bool(model["train_non_event"]) for model in models} == {False}
    assert config["hyperparameter_selection"]["selection_split"] == "validation"
    assert not config["hyperparameter_selection"][
        "test_predictions_allowed_during_selection"
    ]


def test_window_sweep_uses_one_96_hour_common_key_anchor() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "common1000_lstm_film_rain_window_sweep_teacher_aligned_v1.json"
    )
    config = MODULE.load_config(config_path)
    models = config["models"]
    assert {
        int(model["sequence_length_hours"]) for model in models
    } == {24, 48, 96}
    assert config["comparison_anchor_sequence_length_hours"] == 96
    assert {str(model["recurrent_type"]) for model in models} == {"lstm"}
    assert {
        tuple(model["feature_names"]) for model in models
    } == {
        ("x_00010", "x_00095", "x_00300", "x_63680", "rain_00045")
    }

    rows = 150
    frame = pd.DataFrame(
        {
            "target_00400_native": np.ones(rows),
            "x_00010_missing": np.zeros(rows, dtype=np.int8),
            "x_00095_missing": np.zeros(rows, dtype=np.int8),
            "x_00300_missing": np.zeros(rows, dtype=np.int8),
            "x_63680_missing": np.zeros(rows, dtype=np.int8),
            "precip_feature_available": np.ones(rows, dtype=bool),
        }
    )
    reference = None
    for model in models:
        model_config = {
            **config,
            "sequence_length_hours": model["sequence_length_hours"],
        }
        index = MODULE.comparison_indices(frame, model_config)
        assert index[0] == 96
        if reference is None:
            reference = index
        else:
            assert np.array_equal(reference, index)


def test_final_lstm_ablation_is_frozen_and_paired() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "common1000_lstm_film_final_rain_ablation_teacher_aligned_v1.json"
    )
    config = MODULE.load_config(config_path)
    models = config["models"]
    assert len(models) == 2
    assert {str(model["recurrent_type"]) for model in models} == {"lstm"}
    assert {bool(model["train_non_event"]) for model in models} == {False}
    assert {
        "rain_00045" in model["feature_names"] for model in models
    } == {False, True}
    assert config["learning_rate"] == 3e-4
    assert config["sequence_length_hours"] == 48
    assert config["comparison_anchor_sequence_length_hours"] == 48
    assert config["hidden_size"] == 64
    assert config["dropout"] == 0.1
    assert "once" in config["frozen_selection"]["test_policy"].lower()


def test_matched_gru_benchmark_changes_only_recurrent_cell() -> None:
    root = Path(__file__).resolve().parents[3]
    lstm = MODULE.load_config(
        root
        / "config"
        / "common1000_lstm_film_final_rain_ablation_teacher_aligned_v1.json"
    )
    gru = MODULE.load_config(
        root
        / "config"
        / "common1000_gru_film_final_rain_ablation_matched_v1.json"
    )
    for field in (
        "merged_input_root",
        "primary_cohort_path",
        "sequence_length_hours",
        "comparison_anchor_sequence_length_hours",
        "hidden_size",
        "film_hidden_size",
        "head_hidden_size",
        "dropout",
        "learning_rate",
        "weight_decay",
        "early_stopping_patience",
        "gradient_clip_norm",
        "full",
    ):
        assert gru[field] == lstm[field]
    assert {model["recurrent_type"] for model in gru["models"]} == {"gru"}
    assert {
        "rain_00045" in model["feature_names"] for model in gru["models"]
    } == {False, True}
    assert gru["benchmark_status"]["role"] == "post_hoc_architecture_benchmark"


def test_target_values_can_require_a_real_observation() -> None:
    frame = pd.DataFrame(
        {
            "x_00300": [8.1, 8.2, 8.3],
            "x_00300_observed_hour": [True, False, True],
        }
    )
    values = MODULE.target_values(
        frame,
        {
            "target_code": "00300",
            "target_source_column": "x_00300",
            "target_observed_column": "x_00300_observed_hour",
            "target_source_multiplier": 20.0,
        },
    )
    assert np.allclose(values[[0, 2]], [162.0, 166.0])
    assert np.isnan(values[1])


def test_site_payload_can_exclude_target_history() -> None:
    rows = 6
    frame = pd.DataFrame(
        {
            "x_00010": np.arange(rows, dtype=float),
            "target_00400_native": np.full(rows, 7.0),
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "in_event": np.zeros(rows, dtype=bool),
            "event_direction": ["none"] * rows,
            "event_id": np.full(rows, -1, dtype=np.int64),
            **{
                name: np.zeros(rows, dtype=np.float32)
                for name in MODULE.CALENDAR_FEATURES
            },
        }
    )
    preprocessing = {
        "features": {
            "x_00010": {
                "source_column": "x_00010",
                "transform": "identity",
                "mean": 2.5,
                "scale": 1.0,
            }
        },
        "target": {"mean": 7.0, "scale": 1.0},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "site.parquet"
        frame.to_parquet(path, index=False)
        payload = MODULE.site_payload(
            path,
            ["x_00010"],
            preprocessing,
            {
                "target_code": "00400",
                "include_target_history": False,
                "include_do_source_family_indicator": False,
                "target_history_fill_limit_hours": 6,
            },
        )
    assert not any(
        name.startswith("target_history") for name in payload["feature_names"]
    )
    assert payload["features"].shape[1] == 2 + len(MODULE.CALENDAR_FEATURES)


def test_nested_config_inheritance_reaches_data_defaults() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "common1000_lstm_film_ph_sensor_ablation_v1.json"
    )
    config = MODULE.load_config(config_path)
    assert config["random_seed"] == 42
    assert config["horizon_hours"] == 1
    assert config["learning_rate"] == 3e-4
    assert config["include_target_history"] is False
