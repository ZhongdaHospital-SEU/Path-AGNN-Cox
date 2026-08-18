"""Load dataset manifest from config/datasets.yaml."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_datasets(config_path=None) -> list[dict]:
    path = Path(config_path or ROOT / "config" / "datasets.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["datasets"]


def load_benchmark_config(config_path=None) -> dict:
    path = Path(config_path or ROOT / "config" / "benchmark.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)