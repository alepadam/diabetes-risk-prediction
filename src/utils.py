"""Shared utilities: config loading, reproducibility, and logging setup.

Every other module in src/ pulls its settings through load_config() rather
than hardcoding paths or hyperparameters, so the whole pipeline can be
re-tuned from config/config.yaml alone.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the project config. Path is resolved relative to project root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix random seeds for numpy and Python's random module for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instead of relying on print() statements."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path from config (given relative to project root) to an absolute Path."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
