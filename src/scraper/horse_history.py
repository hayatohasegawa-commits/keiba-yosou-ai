"""馬の戦歴詳細スクレイパー (playwright版)。

通常HTMLでは取れない (AJAX 読込み) ため playwright を使用。
取得項目:
- 過去レースの race_id, 日付, 順位, 馬番, タイム, 上がり, 通過 (脚質判定)
- パドック・調教情報があれば

通過 (passage) 情報の例: "3-3-2-2" → 1コーナー3番手, ... → 先行脚質
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HorseHistoryRow:
    date: str  # YYYY-MM-DD
    race_id: Optional[str] = None
    race_name: Optional[str] = None
    place: Optional[str] = None
    weather: Optional[str] = None
    track_cond: Optional[str] = None
    distance: Optional[int] = None
    surface: Optional[str] = None
    horse_number: Optional[int] = None
    starters: Optional[int] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None
    rank: Optional[int] = None
    jockey: Optional[str] = None
    handicap: Optional[float] = None
    time: Optional[str] = None
    margin: Optional[str] = None
    passage: Optional[str] = None  # 通過位置 "3-3-2-2"
    agari_3f: Optional[float] = None
    body_weight: Optional[float] = None


@dataclass
class HorseHistory:
    horse_id: str
    horse_name: Optional[str] = None
    rows: list[HorseHistoryRow] = field(default_factory=list)


def fetch_horse_history(horse_id: str, *, timeout_ms: int = 15000) -> HorseHistory:
    """playwrightで馬詳細ページから戦歴を取得。"""
    from playwright.sync_api import sync_playwright

    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    history = HorseHistory(horse_id=horse_id)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent="KeibaYosouAI/0.1")
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            # 戦歴テーブルの待機 (networkidle で AJAX完了待ち)
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
                page.wait_for_selector("table.db_h_race_results", timeout=timeout_ms)
            except Exception:
                pass
            time.sleep(1)  # 残りAJAX完了の保険

            # 馬名
            try:
                h1 = page.locator("h1").first.inner_text()
                history.horse_name = h1.strip()
            except Exception:
                pass

            # 戦歴テーブル
            rows = page.locator("table.db_h_race_results tr").all()
            for tr in rows[1:]:  # ヘッダーをスキップ
                tds = tr.locator("td").all()
                if len(tds) < 10:
                    continue
                try:
                    row = HorseHistoryRow(
                        date=tds[0].inner_text().strip() if len(tds) > 0 else "",
                    )
                    # 日付フォーマット: 2025/06/08 → 2025-06-08
                    m_d = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", row.date)
                    if m_d:
                        row.date = f"{m_d.group(1)}-{int(m_d.group(2)):02d}-{int(m_d.group(3)):02d}"

                    if len(tds) > 1: row.place = tds[1].inner_text().strip() or None
                    if len(tds) > 2: row.weather = tds[2].inner_text().strip() or None
                    if len(tds) > 3:
                        rn_text = tds[3].inner_text().strip()
                        if rn_text:
                            # race_idを抽出
                            try:
                                rn_a = tds[3].locator("a").first
                                href = rn_a.get_attribute("href")
                                m = re.search(r"/race/(\d+)", href or "")
                                if m: row.race_id = m.group(1)
                                row.race_name = rn_text
                            except Exception:
                                row.race_name = rn_text
                    # 馬体重・馬番・着順等は列インデックス固定で取得
                    # column mapping (netkeiba 標準):
                    # 0:日付 1:開催 2:天気 3:R 4:レース名 5:映像 6:頭数 7:枠 8:馬番
                    # 9:オッズ 10:人気 11:着順 12:騎手 13:斤量 14:距離 15:馬場 16:馬場指数
                    # 17:タイム 18:着差 19:タイム指数 20:通過 21:ペース 22:上り 23:馬体重 ...
                    def _t(i):
                        return tds[i].inner_text().strip() if i < len(tds) else ""
                    # 正しい列マッピング (33列構成):
                    # 0:日付 1:開催 2:天気 3:R 4:レース名 5:映像 6:頭数 7:枠 8:馬番
                    # 9:オッズ 10:人気 11:着順 12:騎手 13:斤量 14:距離 15:水分量 16:馬場
                    # 17:馬場指数 18:タイム 19:着差 20-24:タイム指数等(プレミアム)
                    # 25:通過 26:ペース 27:上り 28:馬体重 29-32: その他
                    row.starters = _safe_int(_t(6))
                    row.horse_number = _safe_int(_t(8))
                    row.odds = _safe_float(_t(9))
                    row.popularity = _safe_int(_t(10))
                    row.rank = _safe_int(_t(11))
                    row.jockey = _t(12) or None
                    row.handicap = _safe_float(_t(13))
                    dist_txt = _t(14)
                    m_dist = re.search(r"(\d+)", dist_txt)
                    if m_dist:
                        row.distance = int(m_dist.group(1))
                        if "芝" in dist_txt: row.surface = "芝"
                        elif "ダ" in dist_txt: row.surface = "ダ"
                    row.track_cond = _t(16) or None
                    row.time = _t(18) or None
                    row.margin = _t(19) or None
                    row.passage = _t(25) or None  # ★ 通過位置
                    row.agari_3f = _safe_float(_t(27))  # ★ 上がり3F
                    bw_txt = _t(28)
                    m_bw = re.match(r"(\d+)", bw_txt)
                    if m_bw: row.body_weight = float(m_bw.group(1))

                    history.rows.append(row)
                except Exception:
                    continue
        finally:
            browser.close()

    return history


def _safe_int(s):
    if not s: return None
    try: return int(re.sub(r"[^\d]", "", s))
    except (ValueError, TypeError): return None


def _safe_float(s):
    if not s: return None
    try: return float(re.sub(r"[^\d.]", "", s))
    except (ValueError, TypeError): return None


def estimate_running_style(passages: list[str]) -> str:
    """通過位置リストから脚質を推定。

    例: ['3-3-2-2', '2-2-1-1', '4-3-3-3'] → 先行
    平均通過位置と頭数から判定。
    """
    if not passages:
        return "不明"
    positions = []
    for p in passages:
        if not p: continue
        nums = re.findall(r"\d+", p)
        if nums:
            avg = sum(int(n) for n in nums) / len(nums)
            positions.append(avg)
    if not positions:
        return "不明"
    avg_pos = sum(positions) / len(positions)
    if avg_pos <= 2.5: return "逃げ"
    elif avg_pos <= 5: return "先行"
    elif avg_pos <= 8: return "差し"
    else: return "追込"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.scraper.horse_history <horse_id>")
        sys.exit(1)
    h = fetch_horse_history(sys.argv[1])
    print(f"馬名: {h.horse_name}")
    print(f"戦歴: {len(h.rows)}走")
    for r in h.rows[:5]:
        print(f"  {r.date} {r.race_name} {r.rank}着 通過:{r.passage} 上り:{r.agari_3f}")
    passages = [r.passage for r in h.rows[:5] if r.passage]
    print(f"\n推定脚質 (直近5走): {estimate_running_style(passages)}")
