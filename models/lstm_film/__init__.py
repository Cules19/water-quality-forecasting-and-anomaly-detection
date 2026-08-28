"""LSTM + FiLM experiment framework for water-quality (pH) prediction.

Grown from ``models/demo_lstm_film.py``. Provides:
  * a memory-efficient windowed dataset over ``data/processed`` parquet shards,
  * a configurable FiLM-conditioned stacked LSTM,
  * a training loop with early stopping and hydrological metrics (NSE/KGE),
  * a *staged* hyper-parameter sweep (sequence length, hidden size, depth,
    dropout, FiLM ablation) that greedily refines an architecture.

Author: TSLR
"""

from .config import ExperimentConfig, DYNAMIC_PCODES, META_FIELDS, TARGET_PCODE

__all__ = [
    "ExperimentConfig",
    "DYNAMIC_PCODES",
    "META_FIELDS",
    "TARGET_PCODE",
]
