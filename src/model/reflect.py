"""リフレクション（振り返り）。

レース結果取り込み後、過去の predictions と実績を突き合わせて、
- 命中率
- 回収率
- 外因分析（どんな特徴量パターンで外したか）
を集計し、Obsidian の PDCAログ に書き出す。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_predictions(since: Optional[str] = None) -> pd.DataFrame:
    """指定日以降の予測を実績と照合。

    TODO: 結果取り込み完了後にロジックを実装。
    """
    raise NotImplementedError("評価ロジックは results 取り込み後に実装")


def write_weekly_reflection(week_label: str) -> Path:
    """週次の振り返りをObsidianに出力。"""
    raise NotImplementedError("週次振り返り出力は評価ロジック実装後に追加")
