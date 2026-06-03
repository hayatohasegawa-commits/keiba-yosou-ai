"""エンリッチ済みレースデータの読み込み・整形。

scripts/enrich_race.py が出力する data/raw/enriched_<race_id>.json を読み込み、
予測タブ・Claude推論で使えるコンテキストに変換する。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
ENRICH_DIR = ROOT / "data" / "raw"


def load_enriched(race_id: str) -> Optional[dict]:
    p = ENRICH_DIR / f"enriched_{race_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def style_map(enriched: dict) -> dict[str, str]:
    """horse_id → 脚質 の辞書を返す。"""
    out = {}
    for h in enriched.get("horses", []):
        if h.get("horse_id"):
            out[h["horse_id"]] = h.get("running_style", "不明")
    return out


def format_for_claude(enriched: dict) -> str:
    """Claude推論プロンプト用のコンテキスト文字列を生成。"""
    if not enriched or "horses" not in enriched:
        return ""
    style_counts = enriched.get("style_counts", {})
    pace = enriched.get("pace_hint", "")
    lines = [
        "【展開予想 (戦歴ベース)】",
        f"脚質分布: " + ", ".join(f"{s}{c}頭" for s, c in style_counts.items()),
        f"想定ペース: {pace}",
        "",
        "【各馬の脚質と直近5走】",
    ]
    for h in enriched["horses"]:
        n = h.get("horse_number") or "?"
        name = h.get("horse_name", "?")
        style = h.get("running_style", "不明")
        recent = h.get("recent_summary", [])
        recent_str = " / ".join(
            f"{r['rank']}着({r['passage']})" for r in recent[:3] if r.get("passage")
        )
        lines.append(f"{n}番 {name}: 脚質={style} 直近: {recent_str or '通過情報なし'}")
    return "\n".join(lines)
