"""3連単3点を組み立てる推論レイヤー。

LightGBMが出した各馬の3着内確率と、レース・出走馬の付加情報をClaudeに渡し、
- 本命型1点
- 連動型1点
- 穴狙い1点
の3点と根拠・自信度をJSONで受け取る。
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
3連単(1着→2着→3着の順)を3点だけ提案する。
最低1点的中を狙う構成：
- 1点目「本命型」: 高確率馬を軸に手堅く
- 2点目「連動型」: 上位2頭固定+3着に中穴
- 3点目「穴狙い」: 妙味のある中穴中心で回収率を狙う

必ず以下のJSON形式のみで応答し、前後に説明文を付けない。

{
  "trifecta_1": "umaban-umaban-umaban",
  "trifecta_2": "umaban-umaban-umaban",
  "trifecta_3": "umaban-umaban-umaban",
  "rationale": "全体の根拠を300字程度で",
  "confidence": 0.0
}
"""


@dataclass
class TrifectaPrediction:
    trifecta_1: str
    trifecta_2: str
    trifecta_3: str
    rationale: str
    confidence: float


def _format_horse_table(horses: pd.DataFrame) -> str:
    cols = [c for c in ["horse_number", "horse_name", "p_top3", "jockey_name",
                        "handicap", "post_position", "odds", "popularity",
                        "recent5_avg_rank"] if c in horses.columns]
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
    raw = complete(SYSTEM_PROMPT, user, temperature=0.3, max_tokens=1500)
    obj = _extract_json(raw)
    return TrifectaPrediction(
        trifecta_1=obj["trifecta_1"],
        trifecta_2=obj["trifecta_2"],
        trifecta_3=obj["trifecta_3"],
        rationale=obj.get("rationale", ""),
        confidence=float(obj.get("confidence", 0.0)),
    )
