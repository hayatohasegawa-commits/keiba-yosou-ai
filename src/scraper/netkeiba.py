"""netkeiba スクレイパー本実装。

netkeiba race結果ページのテーブル列マッピング:
  0:着順 1:枠番 2:馬番 3:馬名 4:性齢 5:斤量 6:騎手 7:タイム 8:着差
  9-13:プレミアム各種指数 14:通過 15:上り 16:単勝 17:人気 18:馬体重 19-21:premium
  22:調教師 23:馬主 24:賞金

レース情報セクション(div.data_intro):
  h1: レース名 (例: "第20回ヴィクトリアマイル(GI)")
  diary_snap_cut > span: "芝1600m / 天候 : 曇 / 馬場 : 良 / 発走 : 15:40"
  p.smalltxt: "2025年5月18日 2回東京8日目 ..."
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from bs4 import BeautifulSoup

from .cache import RateLimitedCache

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

NETKEIBA_RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
NETKEIBA_HORSE_URL = "https://db.netkeiba.com/horse/{horse_id}/"
NETKEIBA_JOCKEY_URL = "https://db.netkeiba.com/jockey/{jockey_id}/"
NETKEIBA_SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


@dataclass
class RaceMeta:
    race_id: str
    date: str = ""
    course: str = ""
    race_number: Optional[int] = None
    race_name: Optional[str] = None
    grade: Optional[str] = None
    distance: Optional[int] = None
    surface: Optional[str] = None
    direction: Optional[str] = None
    weather: Optional[str] = None
    track_cond: Optional[str] = None
    starters: Optional[int] = None


@dataclass
class ResultRow:
    race_id: str
    horse_id: str
    horse_name: str
    jockey_id: Optional[str] = None
    jockey_name: Optional[str] = None
    rank: Optional[int] = None
    post_position: Optional[int] = None
    horse_number: Optional[int] = None
    sex_age: Optional[str] = None
    handicap: Optional[float] = None
    body_weight: Optional[float] = None
    body_weight_diff: Optional[float] = None
    time: Optional[str] = None
    margin: Optional[str] = None
    agari_3f: Optional[float] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None
    trainer_id: Optional[str] = None
    trainer_name: Optional[str] = None


@dataclass
class RaceData:
    meta: RaceMeta
    results: list[ResultRow] = field(default_factory=list)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cache(cfg: Optional[dict] = None) -> RateLimitedCache:
    cfg = cfg or load_config()
    cache_dir = PROJECT_ROOT / cfg["paths"]["raw_root"] / "netkeiba_html"
    sc = cfg["scraper"]
    return RateLimitedCache(
        cache_dir=cache_dir,
        user_agent=sc["user_agent"],
        rate_limit_seconds=sc["rate_limit_seconds"],
        timeout_seconds=sc["timeout_seconds"],
        ttl_days=sc["cache_ttl_days"],
    )


def fetch_race_html(race_id: str, cache: RateLimitedCache) -> str:
    return cache.get(NETKEIBA_RACE_URL.format(race_id=race_id), encoding="EUC-JP")


def fetch_shutuba_html(race_id: str, cache: RateLimitedCache) -> str:
    return cache.get(NETKEIBA_SHUTUBA_URL.format(race_id=race_id))


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _extract_id(href: Optional[str], pattern: str) -> Optional[str]:
    if not href:
        return None
    m = re.search(pattern, href)
    return m.group(1) if m else None


def parse_race_meta(html: str, race_id: str) -> RaceMeta:
    soup = BeautifulSoup(html, "lxml")
    meta = RaceMeta(race_id=race_id)

    h1 = soup.select_one("div.data_intro h1")
    if h1:
        name = h1.get_text(strip=True)
        meta.race_name = name
        gm = re.search(r"\((G[I1-3]+)\)", name)
        if gm:
            grade_raw = gm.group(1)
            meta.grade = grade_raw.replace("GI", "G1").replace("GII", "G2").replace("GIII", "G3")

    intro_span = soup.select_one("diary_snap_cut span") or soup.select_one("div.data_intro span")
    if intro_span:
        text = intro_span.get_text(" ", strip=True)
        m_dist = re.search(r"(\d{3,4})m", text)
        if m_dist:
            meta.distance = int(m_dist.group(1))
        if "芝" in text:
            meta.surface = "芝"
        elif "ダ" in text or "ダート" in text:
            meta.surface = "ダ"
        if "右" in text:
            meta.direction = "右"
        elif "左" in text:
            meta.direction = "左"
        elif "直線" in text:
            meta.direction = "直線"
        m_w = re.search(r"天候\s*:\s*(\S+)", text)
        if m_w:
            meta.weather = m_w.group(1)
        m_tc = re.search(r"馬場\s*:\s*(\S+)", text)
        if m_tc:
            meta.track_cond = m_tc.group(1)

    smalltxt = soup.select_one("p.smalltxt")
    if smalltxt:
        text = smalltxt.get_text(" ", strip=True)
        m_d = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if m_d:
            meta.date = f"{m_d.group(1)}-{int(m_d.group(2)):02d}-{int(m_d.group(3)):02d}"
        m_c = re.search(r"\d+回(\S+?)\d+日目", text)
        if m_c:
            meta.course = m_c.group(1)

    return meta


_BODY_WEIGHT_RE = re.compile(r"^\s*(\d+)\s*\(\s*([+\-]?\d+)\s*\)\s*$")


def _parse_body_weight(cell: str) -> tuple[Optional[float], Optional[float]]:
    m = _BODY_WEIGHT_RE.match(cell)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


COL = {
    "rank": 0,
    "post_position": 1,
    "horse_number": 2,
    "horse": 3,
    "sex_age": 4,
    "handicap": 5,
    "jockey": 6,
    "time": 7,
    "margin": 8,
    "passage": 14,
    "agari": 15,
    "odds": 16,
    "popularity": 17,
    "body_weight": 18,
    "trainer": 22,
}


def parse_results(html: str, race_id: str) -> list[ResultRow]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.race_table_01")
    if not table:
        logger.warning("results table not found for %s", race_id)
        return []

    rows: list[ResultRow] = []
    for tr in table.select("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 19:
            continue

        def text(idx: int) -> str:
            return tds[idx].get_text(" ", strip=True) if idx < len(tds) else ""

        def link(idx: int) -> Optional[str]:
            if idx >= len(tds):
                return None
            a = tds[idx].find("a")
            return a.get("href") if a and a.has_attr("href") else None

        horse_href = link(COL["horse"])
        jockey_href = link(COL["jockey"])
        trainer_href = link(COL["trainer"])

        horse_id = _extract_id(horse_href, r"/horse/(\d+)/?")
        jockey_id = _extract_id(jockey_href, r"/jockey/(?:result/recent/)?(\d+)/?")
        trainer_id = _extract_id(trainer_href, r"/trainer/(?:result/recent/)?(\d+)/?")
        if not horse_id:
            continue

        bw, bw_diff = _parse_body_weight(text(COL["body_weight"]))

        row = ResultRow(
            race_id=race_id,
            horse_id=horse_id,
            horse_name=text(COL["horse"]),
            jockey_id=jockey_id,
            jockey_name=text(COL["jockey"]),
            rank=_safe_int(text(COL["rank"])),
            post_position=_safe_int(text(COL["post_position"])),
            horse_number=_safe_int(text(COL["horse_number"])),
            sex_age=text(COL["sex_age"]) or None,
            handicap=_safe_float(text(COL["handicap"])),
            time=text(COL["time"]) or None,
            margin=text(COL["margin"]) or None,
            agari_3f=_safe_float(text(COL["agari"])),
            odds=_safe_float(text(COL["odds"])),
            popularity=_safe_int(text(COL["popularity"])),
            body_weight=bw,
            body_weight_diff=bw_diff,
            trainer_id=trainer_id,
            trainer_name=text(COL["trainer"]) or None,
        )
        rows.append(row)
    return rows


def fetch_and_parse_race(race_id: str, cache: Optional[RateLimitedCache] = None) -> RaceData:
    cache = cache or build_cache()
    html = fetch_race_html(race_id, cache)
    meta = parse_race_meta(html, race_id)
    results = parse_results(html, race_id)
    meta.starters = len(results)
    return RaceData(meta=meta, results=results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.scraper.netkeiba <race_id>")
        raise SystemExit(1)
    data = fetch_and_parse_race(sys.argv[1])
    print(data.meta)
    for r in data.results[:5]:
        print(r)
