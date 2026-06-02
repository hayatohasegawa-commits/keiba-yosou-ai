"""LightGBM 推論スクリプト（骨格）。

指定レースの出走馬それぞれについて 3着内確率を出し、SQLite(horse_probabilities)に保存する。
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_top3_probabilities(race_id: str, model_path: Optional[Path] = None) -> pd.DataFrame:
    """指定レースの各馬の3着内確率を返す。

    Returns:
        DataFrame with columns: [horse_id, horse_name, p_top3]
    """
    from src.features.build import build_features_for_race  # noqa: F401

    cfg = load_config()
    if model_path is None:
        models_dir = PROJECT_ROOT / cfg["paths"]["models_root"]
        candidates = sorted(models_dir.glob("lgbm_*.pkl"))
        if not candidates:
            raise FileNotFoundError("学習済みモデルが見つからない。先に src.model.train を実行")
        model_path = candidates[-1]

    raise NotImplementedError("推論ロジックは学習完了後に実装")


def save_probabilities(race_id: str, df: pd.DataFrame, model_version: str) -> None:
    cfg = load_config()
    db_path = PROJECT_ROOT / cfg["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO horse_probabilities
                (race_id, horse_id, p_top3, model_version)
                VALUES (?, ?, ?, ?)
                """,
                (race_id, row["horse_id"], float(row["p_top3"]), model_version),
            )
        conn.commit()
