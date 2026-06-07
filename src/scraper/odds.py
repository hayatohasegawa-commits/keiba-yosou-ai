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

    data_field = data.get("data")
    if not isinstance(data_field, dict):
        return {}
    odds_dict = data_field.get("odds", {}).get("8", {}) if isinstance(data_field.get("odds"), dict) else {}
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


def _odds_api(race_id: str, type_: int, timeout: int = 15) -> dict:
    try:
        r = requests.get(
            ODDS_API,
            params={"race_id": race_id, "type": type_, "action": "update"},
            headers={"User-Agent": "Mozilla/5.0",
                     "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("data", {}) or {}
    except Exception:
        return {}


def fetch_win_odds(race_id: str, timeout: int = 15) -> dict[int, tuple[float, int]]:
    """単勝オッズと人気を取得。{馬番: (単勝倍率, 人気)}。

    netkeiba odds API type=1 (単勝):
      data.odds["1"]["01"] = [odds_str, "", ninki_str]
    中央の出馬表HTMLには単勝が載らない(JS生成)ため、予測前にこれで補完する。
    """
    data = _odds_api(race_id, 1, timeout)
    block = data.get("odds", {}).get("1", {}) if isinstance(data.get("odds"), dict) else {}
    out: dict[int, tuple[float, int]] = {}
    for k, vals in block.items():
        try:
            num = int(k)
            odds = float(str(vals[0]).replace(",", ""))
            ninki = int(vals[2]) if len(vals) > 2 and str(vals[2]).isdigit() else 99
            out[num] = (odds, ninki)
        except (ValueError, IndexError, TypeError):
            continue
    return out


def fetch_trio_odds(race_id: str, timeout: int = 15) -> dict[str, float]:
    """3連複オッズを取得。キーは "2-4-5"(昇順)。type=7。"""
    data = _odds_api(race_id, 7, timeout)
    block = data.get("odds", {}).get("7", {}) if isinstance(data.get("odds"), dict) else {}
    out: dict[str, float] = {}
    for key, vals in block.items():
        if not isinstance(key, str) or len(key) != 6:
            continue
        try:
            a, b, c = sorted((int(key[0:2]), int(key[2:4]), int(key[4:6])))
            out[f"{a}-{b}-{c}"] = float(str(vals[0]).replace(",", ""))
        except (ValueError, IndexError, TypeError):
            continue
    return out
