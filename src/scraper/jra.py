"""JRA公式サイトからの補助情報取得（骨格）。

主な用途:
- 当日の出馬表（オッズ・馬体重）
- 馬場状態（芝・ダ別の含水率・クッション値）

netkeibaに無い当日情報を補完する。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fetch_yasuda_kinen_info() -> dict:
    """安田記念2026の特設ページから出走馬・枠順・オッズ等を取得。

    TODO: 公式ページの構造を確認して実装。
    """
    raise NotImplementedError("JRA公式ページのパースは後続タスクで実装")
