"""Configurable FiLM-conditioned stacked LSTM.

Expanded from ``models/demo_lstm_film.py``:
  * arbitrary depth / hidden size,
  * inter-layer dropout,
  * a clean FiLM ablation (``use_film=False`` -> identity modulation) so the
    contribution of site metadata can be measured directly,
  * per-layer FiLM (gamma, beta) initialised at identity so metadata influence
    grows from zero during training.

Author: TSLR
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ExperimentConfig


class FiLMGenerator(nn.Module):
    """Maps static site metadata to per-layer (gamma, beta) modulation."""

    def __init__(self, n_meta: int, hidden_size: int, n_layers: int, film_hidden: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_meta, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * n_layers * hidden_size),
        )
        # Start at identity (gamma=1, beta=0): early training matches the
        # unconditioned model; metadata influence grows from zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta: torch.Tensor):
        params = self.net(meta).view(-1, self.n_layers, 2, self.hidden_size)
        gamma = 1.0 + params[:, :, 0, :]
        beta = params[:, :, 1, :]
        return gamma, beta


class FiLMLSTM(nn.Module):
    """Stacked LSTM whose per-layer outputs are FiLM-modulated by metadata."""

    def __init__(self, cfg: ExperimentConfig, n_sites: int):
        super().__init__()
        self.cfg = cfg
        self.use_film = cfg.use_film
        self.n_layers = cfg.n_layers

        self.site_emb = nn.Embedding(n_sites, cfg.site_emb_dim)
        if cfg.use_film:
            self.film = FiLMGenerator(
                cfg.n_meta, cfg.hidden, cfg.n_layers, cfg.film_hidden
            )
        else:
            self.film = None

        self.layers = nn.ModuleList(
            nn.LSTM(
                input_size=(cfg.n_dynamic + cfg.site_emb_dim) if i == 0 else cfg.hidden,
                hidden_size=cfg.hidden,
                batch_first=True,
            )
            for i in range(cfg.n_layers)
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, site_idx, meta):
        # x_seq: (B, seq, n_dynamic); site_idx: (B,); meta: (B, n_meta)
        emb = self.site_emb(site_idx).unsqueeze(1).expand(-1, x_seq.size(1), -1)
        h = torch.cat([x_seq, emb], dim=-1)

        if self.film is not None:
            gamma, beta = self.film(meta)

        for i, lstm in enumerate(self.layers):
            h, _ = lstm(h)
            if self.film is not None:
                h = gamma[:, i].unsqueeze(1) * h + beta[:, i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = self.drop(h)

        return self.head(h[:, -1, :]).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
