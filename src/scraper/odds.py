"""netkeiba 3連単オッズ取得。

API: https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=XXX&type=8
レスポンス: JSON, data.odds["8"]["AABBCC"] = [odds_str, "0.0", num]
  AABBCC: 6桁、1着馬番(2桁)+2着馬番(2桁)+3着馬番(2桁)
"""
from __future__ import annotations

import requests
from typing import Optional

ODDS_API = "https://race.netkeiba.com/api/api_get_jra_odds.html"
USER_AGENT = "KeibaYosouAI/0.1 (research)"


def fetch_trifecta_odds(race_id: str, timeout: int = 15) -> dict[str, float]:
    """指定レースの3連単オッズを取得。

    Returns:
        辞書: "10-1-18" -> 25.3 (倍率)
        取得失敗時は空辞書
    """
    try:
        r = requests.get(
            ODDS_API,
            params={"race_id": race_id, "type": 8},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    odds_dict = data.get("data", {}).get("odds", {}).get("8", {})
    result: dict[str, float] = {}
    for key, vals in odds_dict.items():
        if not isinstance(key, str) or len(key) != 6:
            continue
        try:
            a, b, c = int(key[0:2]), int(key[2:4]), int(key[4:6])
            pick = f"{a}-{b}-{c}"
            odds_val = float(str(vals[0]).replace(",", ""))
            result[pick] = odds_val
        except (ValueError, IndexError, TypeError):
            continue
    return result


def lookup_odds_for_picks(picks: list[str], odds_map: dict[str, float]) -> list[Optional[float]]:
    """予想5点それぞれの3連単オッズを返す。未登録は None。"""
    return [odds_map.get(p) for p in picks]
