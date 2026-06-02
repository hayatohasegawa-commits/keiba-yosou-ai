"""LightGBM 学習スクリプト（骨格）。

3着内に入る確率を予測する二値分類モデル。
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cutoff_date: Optional[str] = None) -> Path:
    """学習を実行し、モデルファイルを保存してパスを返す。

    TODO: build_training_frame() のデータでLightGBM学習を実装。
    """
    import lightgbm as lgb  # noqa: F401（実装時に使用）
    from src.features.build import build_training_frame  # noqa: F401

    cfg = load_config()
    cutoff_date = cutoff_date or datetime.now().strftime("%Y-%m-%d")
    version = datetime.now().strftime("v_%Y%m%d_%H%M%S")
    model_path = PROJECT_ROOT / cfg["paths"]["models_root"] / f"lgbm_{version}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError("学習ロジックは特徴量フレームができてから実装")


if __name__ == "__main__":
    train()
