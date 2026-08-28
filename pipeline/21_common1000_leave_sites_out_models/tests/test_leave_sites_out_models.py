from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_leave_sites_out_models.py"
SPEC = importlib.util.spec_from_file_location("models", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_model_has_no_embedding_and_shapes_match() -> None:
    config = {
        "hidden_size": 8,
        "film_hidden_size": 6,
        "head_hidden_size": 4,
        "dropout": 0.1,
    }
    spec = {"recurrent_type": "gru", "recurrent_layers": 2, "conditioning": "film"}
    model = MODULE.SiteAgnosticFiLMRNN(spec, 5, 7, config)
    assert not any(isinstance(layer, torch.nn.Embedding) for layer in model.modules())
    output = model(torch.randn(3, 24, 5), torch.randn(3, 7))
    assert output.shape == (3,)


def test_lstm_uses_the_same_site_agnostic_film_interface() -> None:
    config = {
        "hidden_size": 8,
        "film_hidden_size": 6,
        "head_hidden_size": 4,
        "dropout": 0.1,
    }
    spec = {"recurrent_type": "lstm", "recurrent_layers": 1, "conditioning": "film"}
    model = MODULE.SiteAgnosticFiLMRNN(spec, 5, 7, config)
    assert not any(isinstance(layer, torch.nn.Embedding) for layer in model.modules())
    assert any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())
    output = model(torch.randn(3, 48, 5), torch.randn(3, 7))
    assert output.shape == (3,)


def test_invalid_recurrent_type_is_rejected() -> None:
    config = {
        "hidden_size": 8,
        "film_hidden_size": 6,
        "head_hidden_size": 4,
        "dropout": 0.1,
    }
    spec = {"recurrent_type": "rnn", "recurrent_layers": 1, "conditioning": "film"}
    try:
        MODULE.SiteAgnosticFiLMRNN(spec, 5, 7, config)
    except ValueError as error:
        assert "Unsupported recurrent_type" in str(error)
    else:
        raise AssertionError("invalid recurrent type was accepted")


def test_identity_film_initialisation() -> None:
    film = MODULE.FiLMGenerator(4, 6, 2, 5)
    gamma, beta = film(torch.randn(3, 4))
    assert torch.allclose(gamma, torch.ones_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))
