"""継続学習: 過去の推論ログを次回の Claude推論に注入。

predictions テーブルに保存された rationale を取り出し、
新しい予測時の system/user prompt に「過去の分析パターン」として埋め込む。

これによりプロンプト変えずに「経験を蓄積したエージェント」になる。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "keiba.sqlite"


def get_recent_reasoning(limit: int = 8, race_meta: Optional[dict] = None) -> str:
    """過去の予測根拠を要約して継続学習コンテキストにする。

    優先順位:
    1. 同じレース名・グレード・コース系の過去予測 (race_meta指定時)
    2. 直近の全予測 limit件
    """
    sections = ["# 🧠 累積知見 (過去の自分の推論ログ)"]

    with sqlite3.connect(DB_PATH) as conn:
        relevant = pd.DataFrame()
        if race_meta:
            grade = race_meta.get("grade")
            course = race_meta.get("course")
            surface = race_meta.get("surface")
            distance = race_meta.get("distance")
            try:
                relevant = pd.read_sql_query(
                    """SELECT p.created_at, p.confidence, p.rationale,
                              r.race_name, r.grade, r.course, r.distance, r.surface
                       FROM predictions p LEFT JOIN races r USING(race_id)
                       WHERE p.rationale IS NOT NULL AND length(p.rationale) > 100
                         AND (r.grade=? OR (r.course=? AND r.distance=? AND r.surface=?))
                       ORDER BY p.created_at DESC LIMIT ?""",
                    conn, params=(grade or "", course or "", distance or 0, surface or "", limit),
                )
            except Exception:
                pass

        # フォールバック: 直近全件
        if relevant.empty:
            try:
                relevant = pd.read_sql_query(
                    """SELECT p.created_at, p.confidence, p.rationale,
                              r.race_name, r.grade, r.course, r.distance
                       FROM predictions p LEFT JOIN races r USING(race_id)
                       WHERE p.rationale IS NOT NULL AND length(p.rationale) > 100
                       ORDER BY p.created_at DESC LIMIT ?""",
                    conn, params=(limit,),
                )
            except Exception:
                return ""

        if relevant.empty:
            return ""

        sections.append(f"\n以下は過去{len(relevant)}件の予測根拠。同様のパターンや成功した着眼点を活用してください。\n")
        for i, r in enumerate(relevant.itertuples(), 1):
            label_parts = []
            if getattr(r, "race_name", None): label_parts.append(r.race_name)
            if getattr(r, "grade", None): label_parts.append(f"[{r.grade}]")
            if getattr(r, "course", None) and getattr(r, "distance", None):
                label_parts.append(f"{r.course}{r.distance}m")
            label = " ".join(label_parts) or "レース"

            conf_str = f" 自信度{r.confidence:.2f}" if pd.notna(r.confidence) else ""
            rationale = r.rationale[:400]
            sections.append(f"### 事例{i}: {label}{conf_str}")
            sections.append(rationale + ("..." if len(r.rationale) > 400 else ""))

        # 抽出した「効いた着眼点」のメタサマリ
        sections.append("\n### 反映指針")
        sections.append(
            "- 過去の自信度が高かった予測の論理構造を参考にする\n"
            "- 同条件 (コース・距離・グレード) の予測パターンを優先\n"
            "- 過去に外した要因を考慮 (馬場・展開・人気馬の取りこぼし) \n"
            "- 軸馬の選び方は過去予測の傾向と整合させる"
        )

    return "\n".join(sections)


def get_validated_hypotheses(limit: int = 5) -> str:
    """ai_hypotheses から proven なものを抽出して当てる予想に活用。"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                """SELECT topic, hypothesis, rationale FROM ai_hypotheses
                   WHERE outcome='proven' ORDER BY confidence DESC LIMIT ?""",
                conn, params=(limit,)
            )
    except Exception:
        return ""
    if df.empty:
        return ""
    parts = ["# ✅ 検証済み仮説 (proven)"]
    for _, r in df.iterrows():
        parts.append(f"- **[{r['topic']}]** {r['hypothesis']}")
    return "\n".join(parts)
