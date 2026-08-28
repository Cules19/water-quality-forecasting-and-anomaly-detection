from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "run_warmup_adaptation.py"
SPEC = importlib.util.spec_from_file_location("warmup_adaptation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def build_model() -> MODULE.SiteFiLMAffineAdapter:
    specification = {
        "recurrent_type": "lstm",
        "recurrent_layers": 1,
        "conditioning": "film",
    }
    config = {
        "hidden_size": 8,
        "film_hidden_size": 6,
        "head_hidden_size": 4,
        "dropout": 0.0,
    }
    base = MODULE.BASE.SiteAgnosticFiLMRNN(
        specification, dynamic_size=5, metadata_size=3, config=config
    )
    return MODULE.SiteFiLMAffineAdapter(base)


def test_zero_initialized_adapter_is_identity() -> None:
    torch.manual_seed(7)
    model = build_model().eval()
    dynamic = torch.randn(4, 12, 5)
    metadata = torch.randn(4, 3)
    with torch.no_grad():
        expected = model.base_model(dynamic, metadata)
        observed = model(dynamic, metadata)
    assert torch.allclose(expected, observed)


def test_strategy_parameter_counts_are_nested() -> None:
    model = build_model()
    zero = MODULE.configure_strategy(model, "zero_shot")
    low_cost = MODULE.configure_strategy(model, "film_affine")
    full = MODULE.configure_strategy(model, "full_finetune")
    assert zero == 0
    assert low_cost == 2 * 8 + 2
    assert full > low_cost


def test_temporal_split_preserves_order_and_disjointness() -> None:
    indices = np.arange(100, 300)
    train, validation = MODULE.temporal_train_validation_split(
        indices,
        fraction=0.2,
        minimum_validation=24,
        maximum_validation=512,
    )
    assert len(train) == 160
    assert len(validation) == 40
    assert train.max() < validation.min()
    assert len(np.intersect1d(train, validation)) == 0


def test_metric_payload_reports_requested_absolute_error_coverage() -> None:
    metrics = MODULE.metric_payload(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.1, 1.4, 3.2]),
        tolerance=0.5,
    )
    assert metrics["within_tolerance_count"] == 2
    assert np.isclose(metrics["within_tolerance_fraction"], 2 / 3)
    assert metrics["absolute_error_tolerance"] == 0.5
