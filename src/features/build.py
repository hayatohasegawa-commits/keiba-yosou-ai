"""特徴量生成。

DB(results / races / horses / jockeys) から、レース×馬の特徴量テーブルを作る。
LightGBMの入力に使う。

主要特徴量（初期案）:
- recent5_avg_rank: 直近5走の平均着順
- recent5_top3_rate: 直近5走の3着内率
- agari_3f_rel: 上がり3Fの同条件相対値
- jockey_g1_winrate: 騎手のG1勝率
- jockey_horse_synergy: 騎手×馬の組み合わせ過去成績
- dist_match_score: 距離適性スコア（同距離±200mでの複勝率）
- surface_match: 芝/ダート適性
- course_match: コース別複勝率
- days_since_last: 前走からの間隔（日数）
- popularity / odds: 直近の市場評価
"""
from __future__ import annotations

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


def db_connect(cfg: Optional[dict] = None) -> sqlite3.Connection:
    cfg = cfg or load_config()
    db_path = PROJECT_ROOT / cfg["paths"]["db_path"]
    return sqlite3.connect(db_path)


def build_training_frame(cutoff_date: str) -> pd.DataFrame:
    """cutoff_date以前のレースを学習用に整形。

    TODO: 実データ取得後にロジックを埋める。
    """
    raise NotImplementedError("build_training_frame は実データ取り込み後に実装")


def build_features_for_race(race_id: str) -> pd.DataFrame:
    """指定レースの出走馬について特徴量を生成（予測用）。

    TODO: 実データ取得後にロジックを埋める。
    """
    raise NotImplementedError("build_features_for_race は実データ取り込み後に実装")
