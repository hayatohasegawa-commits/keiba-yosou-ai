"""出馬表（pre-race）取得。

NAR/JRA共通: race.netkeiba.com or nar.netkeiba.com の shutuba.html
"""
from __future__ import annotations

import re
import requests
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

USER_AGENT = "KeibaYosouAI/0.1"


@dataclass
class ShutubaHorse:
    horse_id: str
    horse_name: str
    horse_number: Optional[int] = None
    post_position: Optional[int] = None
    sex_age: Optional[str] = None
    handicap: Optional[float] = None
    jockey_id: Optional[str] = None
    jockey_name: Optional[str] = None
    trainer_id: Optional[str] = None
    trainer_name: Optional[str] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None
    body_weight: Optional[float] = None
    body_weight_diff: Optional[float] = None


def fetch_shutuba(race_id: str, nar: bool = True) -> list[ShutubaHorse]:
    """race_id から出馬表を取得。"""
    base = "https://nar.netkeiba.com" if nar else "https://race.netkeiba.com"
    url = f"{base}/race/shutuba.html?race_id={race_id}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.encoding = "EUC-JP"
    soup = BeautifulSoup(r.text, "lxml")

    horses: list[ShutubaHorse] = []
    table = soup.select_one("table.Shutuba_Table") or soup.select_one("table.RaceTable01")
    if not table:
        return horses

    for tr in table.select("tr.HorseList"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        try:
            wakuban_cell = tr.select_one("td.Waku") or tds[0]
            umaban_cell = tr.select_one("td.Umaban") or tds[1]
            horse_link = tr.select_one("span.HorseName a") or tr.select_one("td.HorseInfo a")
            jockey_link = tr.select_one("td.Jockey a")
            trainer_link = tr.select_one("td.Trainer a")
            odds_cell = tr.select_one("td.Popular_Ninki span") or tr.select_one("td.Odds")
            ninki_cell = tr.select_one("td.Popular span") or tr.select_one("span.OddsPeople")

            href = horse_link.get("href", "") if horse_link else ""
            m_h = re.search(r"/horse/(\w+)", href)
            horse_id = m_h.group(1) if m_h else ""
            if not horse_id:
                continue

            h = ShutubaHorse(
                horse_id=horse_id,
                horse_name=horse_link.get_text(strip=True) if horse_link else "",
            )
            try:
                h.post_position = int(wakuban_cell.get_text(strip=True))
            except (ValueError, AttributeError):
                pass
            try:
                h.horse_number = int(umaban_cell.get_text(strip=True))
            except (ValueError, AttributeError):
                pass

            # 性齢・斤量 (3-5番目セル付近)
            for td in tds[3:7]:
                txt = td.get_text(" ", strip=True)
                m_sa = re.match(r"^([牡牝セ騙])(\d+)$", txt)
                if m_sa:
                    h.sex_age = txt
                if re.match(r"^\d{2,3}\.?\d?$", txt) and h.handicap is None and 40 <= float(txt) <= 65:
                    h.handicap = float(txt)

            if jockey_link:
                href_j = jockey_link.get("href", "")
                m_j = re.search(r"/jockey/(?:result/recent/)?(\d+)", href_j)
                if m_j:
                    h.jockey_id = m_j.group(1)
                h.jockey_name = jockey_link.get_text(strip=True)

            if trainer_link:
                href_t = trainer_link.get("href", "")
                m_t = re.search(r"/trainer/(?:result/recent/)?(\d+)", href_t)
                if m_t:
                    h.trainer_id = m_t.group(1)
                h.trainer_name = trainer_link.get_text(strip=True)

            # オッズ・人気
            if odds_cell:
                try:
                    h.odds = float(odds_cell.get_text(strip=True).replace(",", ""))
                except (ValueError, AttributeError):
                    pass
            if ninki_cell:
                try:
                    h.popularity = int(ninki_cell.get_text(strip=True))
                except (ValueError, AttributeError):
                    pass

            horses.append(h)
        except Exception:
            continue
    return horses
