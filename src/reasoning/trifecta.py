"""3連単5点を組み立てる Claude 推論レイヤー。

LightGBMが出した各馬の3着内確率 (p_top3) と、レース・出走馬の付加情報をClaudeに渡し、
- 本命寄り2点
- 連動型2点
- 穴狙い1点
の計5点と根拠・自信度をJSONで受け取る。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .claude_client import complete

SYSTEM_PROMPT = """あなたは熟練の競馬予想家。
ユーザーから渡される各馬の3着内確率(p_top3)と付加情報を踏まえ、
3連単(1着→2着→3着の順)を5点提案する。最低1点的中を狙う構成:

- 1点目「本命型」: 最高確率の軸馬で1-2-3着の本線
- 2点目「2-3着入替」: 1-3-2着の流し
- 3点目「3着差替」: 1-2-4着 (3着を中穴に差し替え)
- 4点目「2着逆転」: 2-1-3着で2番手馬の1着
- 5点目「穴狙い」: 妙味のある中穴を絡めた1点 (回収率重視)

各馬のp_top3は3着内に入る確率(0.0〜1.0)。馬場状態・天候・脚質も加味する。

必ず以下のJSON形式のみで応答し、前後に説明文を付けない:

{
  "picks": [
    "umaban-umaban-umaban",
    "umaban-umaban-umaban",
    "umaban-umaban-umaban",
    "umaban-umaban-umaban",
    "umaban-umaban-umaban"
  ],
  "rationale": "全体の根拠を300字程度で",
  "confidence": 0.0
}
"""


@dataclass
class TrifectaPrediction:
    picks: list[str]
    rationale: str
    confidence: float


def _format_horse_table(horses: pd.DataFrame) -> str:
    cols = [c for c in [
        "horse_number", "horse_name", "p_top3",
        "popularity", "odds", "jockey_name",
        "handicap", "age", "post_position",
        "recent5_avg_rank", "recent5_top3_rate",
        "prior_top3_rate_course", "prior_top3_rate_distance",
    ] if c in horses.columns]
    return horses[cols].to_csv(index=False)


def build_user_prompt(race_meta: dict, horses: pd.DataFrame) -> str:
    parts = [
        f"レース: {race_meta.get('race_name', '')} ({race_meta.get('grade', '')})",
        f"日付: {race_meta.get('date', '')}",
        f"コース: {race_meta.get('course', '')} {race_meta.get('surface', '')}{race_meta.get('distance', '')}m",
        f"馬場: {race_meta.get('track_cond', '不明')}  天候: {race_meta.get('weather', '不明')}",
        "",
        "出走馬テーブル (CSV):",
        _format_horse_table(horses),
    ]
    return "\n".join(parts)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"JSONが見つからない応答: {text[:200]}")
    return json.loads(m.group(0))


def predict_trifecta(race_meta: dict, horses: pd.DataFrame) -> TrifectaPrediction:
    user = build_user_prompt(race_meta, horses)
    raw = complete(SYSTEM_PROMPT, user, max_tokens=2000)
    obj = _extract_json(raw)
    picks = obj.get("picks") or []
    if not picks and "trifecta_1" in obj:
        # 旧形式対応
        picks = [obj.get(f"trifecta_{i}", "") for i in range(1, 6) if obj.get(f"trifecta_{i}")]
    picks = [p for p in picks if p]
    return TrifectaPrediction(
        picks=picks[:5],
        rationale=obj.get("rationale", ""),
        confidence=float(obj.get("confidence", 0.0)),
    )
