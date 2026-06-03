"""Agentic RAG: 予測前に DB照合 + Web検索 でエビデンスを集める。

2エージェント構成 (MVP):
- knowledge_agent: 類似レース・過去予測・仮説をDBから抽出
- websearch_agent: Anthropic web_search で騎手・馬の直近情報

両者の出力を統合して Claude推論の extra_context として渡す。
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .claude_client import get_client, DEFAULT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "db" / "keiba.sqlite"


# ============ Agent 1: Knowledge ============

def knowledge_agent(race_meta: dict, horses: pd.DataFrame) -> str:
    """DB から関連する文脈を抽出。

    1. 同コース・同距離・同馬場の過去レース直近10件
    2. 各出走馬の DB 内戦歴サマリ
    3. 関連仮説 (ai_hypotheses から topic 一致)
    """
    course = race_meta.get("course", "")
    distance = race_meta.get("distance")
    surface = race_meta.get("surface", "")
    grade = race_meta.get("grade", "")

    sections = ["# 🗄️ Knowledge Agent 出力"]
    with sqlite3.connect(DB_PATH) as conn:
        # 1. 同条件の過去レース傾向
        q = """
            SELECT r.date, r.race_name, r.grade,
                   COUNT(DISTINCT res.horse_id) AS starters,
                   (SELECT res2.horse_number FROM results res2
                    WHERE res2.race_id=r.race_id AND res2.rank=1) AS winner_no,
                   (SELECT MAX(res2.popularity) FROM results res2
                    WHERE res2.race_id=r.race_id AND res2.rank<=3) AS top3_max_popularity
            FROM races r JOIN results res ON r.race_id=res.race_id
            WHERE r.course=? AND r.distance=? AND r.surface=?
            GROUP BY r.race_id ORDER BY r.date DESC LIMIT 10
        """
        try:
            similar = pd.read_sql_query(q, conn, params=(course, distance, surface))
            sections.append("\n## 同コース直近10レース")
            sections.append(similar.to_string(index=False))
        except Exception as e:
            sections.append(f"\n## 類似レース取得失敗: {e}")

        # 2. 出走馬の DB 戦歴サマリ
        sections.append("\n## 出走馬の DB戦歴")
        if "horse_id" in horses.columns:
            for _, h in horses.head(10).iterrows():
                hid = h.get("horse_id")
                if not hid:
                    continue
                hist = pd.read_sql_query(
                    """SELECT r.date, r.course, r.distance, res.rank, res.agari_3f, res.popularity
                       FROM results res JOIN races r USING(race_id)
                       WHERE res.horse_id=? ORDER BY r.date DESC LIMIT 5""",
                    conn, params=(hid,),
                )
                line = f"{h.get('horse_number','?')}番 {h.get('horse_name','?')}: "
                if hist.empty:
                    line += "DB戦歴なし"
                else:
                    recent = " / ".join(
                        f"{r['date']}{r['course']}{r['distance']}m {r['rank']}着"
                        for _, r in hist.iterrows()
                    )
                    line += recent
                sections.append(f"- {line}")

        # 3. 関連仮説
        hyp = pd.read_sql_query(
            "SELECT topic, hypothesis, outcome FROM ai_hypotheses WHERE outcome IN ('proven','pending') ORDER BY confidence DESC LIMIT 5",
            conn,
        )
        if not hyp.empty:
            sections.append("\n## 適用可能な仮説 (過去から)")
            for _, r in hyp.iterrows():
                sections.append(f"- [{r['topic']}/{r['outcome']}] {r['hypothesis'][:120]}")

    return "\n".join(sections)


# ============ Agent 2: Web Search ============

def websearch_agent(race_meta: dict, horses: pd.DataFrame, max_results: int = 5) -> str:
    """Anthropic web_search tool で直近情報を取得。

    検索クエリは Claude に決めさせる (race + horse + jockey の組合せ)。
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "# 🌐 WebSearch Agent: APIキー未設定でスキップ"

    horse_list = ""
    if not horses.empty:
        top = horses.head(5)
        horse_list = "\n".join(
            f"- {r.get('horse_number','?')}番 {r.get('horse_name','?')} 騎手:{r.get('jockey_name','?')}"
            for _, r in top.iterrows()
        )

    client = get_client()
    prompt = f"""次の競馬レースについて、Web検索を使って予測に役立つ最新情報を集めてください。

レース: {race_meta.get('race_name','')} {race_meta.get('date','')}
コース: {race_meta.get('course','')} {race_meta.get('surface','')}{race_meta.get('distance','')}m

注目馬と騎手:
{horse_list}

集めてほしい情報:
1. 各馬の直近の調教評価・コンディション情報
2. 騎手の今週の調子・コメント
3. 馬場・天候の予想
4. 過去 G1 同条件レースの傾向 (もし G1 なら)
5. 専門家の予想評価

最後に「## 予測への影響度評価」として、上記情報がどう5点予想に影響するか150字でまとめてください。
"""

    try:
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=3000,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": max_results}],
            messages=[{"role": "user", "content": prompt}],
        )
        # web_search tool結果と text を結合
        parts = ["# 🌐 WebSearch Agent 出力"]
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
            elif getattr(block, "type", None) == "web_search_tool_result":
                parts.append("\n(検索結果取得済)")
        return "\n".join(parts)
    except Exception as e:
        return f"# 🌐 WebSearch Agent: エラー ({e})\n(検索なしで進行)"


# ============ Orchestrator ============

def gather_rag_context(race_meta: dict, horses: pd.DataFrame, enable_web: bool = True) -> str:
    """両エージェントを呼んで extra_context を組み立てる。"""
    parts = []
    parts.append(knowledge_agent(race_meta, horses))
    if enable_web:
        parts.append("\n\n" + websearch_agent(race_meta, horses))
    return "\n\n".join(parts)


def predict_with_rag(race_meta: dict, horses: pd.DataFrame, enable_web: bool = True):
    """Agentic RAG モードで5点予想。"""
    from .trifecta import predict_trifecta
    ctx = gather_rag_context(race_meta, horses, enable_web=enable_web)
    return predict_trifecta(race_meta, horses, extra_context=ctx), ctx
