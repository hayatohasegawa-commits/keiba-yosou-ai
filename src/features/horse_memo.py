"""出走馬の軽い説明テキストをDB由来で生成。

DBに溜まったレース結果から、その馬の過去成績を1〜2行に圧縮。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]


def _db_path() -> Path:
    return ROOT / "data" / "db" / "keiba.sqlite"


def build_horse_memo(horse_id: str, current_race_id: Optional[str] = None) -> str:
    """過去レースから簡潔な馬メモを生成。

    例: "DB過去5走 [1-0-2-2] / 東京芝1600m [1-0-0-1] / 直近1着→2着→4着"
    """
    if not horse_id:
        return ""
    with sqlite3.connect(_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        q = """
            SELECT r.race_id, r.date, r.course, r.surface, r.distance,
                   res.rank, res.horse_number, res.agari_3f, res.odds
            FROM results res JOIN races r USING(race_id)
            WHERE res.horse_id = ?
        """
        params: tuple = (horse_id,)
        if current_race_id:
            q += " AND r.race_id != ?"
            params = (horse_id, current_race_id)
        q += " ORDER BY r.date DESC LIMIT 10"
        rows = conn.execute(q, params).fetchall()

    if not rows:
        return "（DBに過去成績なし）"

    n = len(rows)
    finishes = [r["rank"] for r in rows if r["rank"] is not None]
    win = sum(1 for x in finishes if x == 1)
    place2 = sum(1 for x in finishes if x == 2)
    place3 = sum(1 for x in finishes if x == 3)
    others = len(finishes) - win - place2 - place3
    rec = f"DB過去{n}走 [{win}-{place2}-{place3}-{others}]"

    recent = " → ".join(
        f"{r['rank']}着" if r["rank"] else "—" for r in rows[:3]
    )
    parts = [rec]
    if recent:
        parts.append(f"直近 {recent}")
    return " / ".join(parts)


def build_memos_for_entries(entries: list) -> dict[str, str]:
    """出走馬リスト(horse_idを持つ)に対して馬メモ辞書を返す。"""
    out: dict[str, str] = {}
    for e in entries:
        hid = getattr(e, "horse_id", None) or e.get("horse_id") if isinstance(e, dict) else None
        if hid and hid not in out:
            out[hid] = build_horse_memo(hid)
    return out
