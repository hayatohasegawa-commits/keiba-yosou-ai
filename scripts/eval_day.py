"""指定日・場の結果を取得し、保存済み予測との一致度・各馬券種の的中を集計。

使い方:
    python scripts/eval_day.py --date 2026-06-04 --place 名古屋
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re
import sqlite3
import requests
from bs4 import BeautifulSoup

DB = ROOT / "data" / "db" / "keiba.sqlite"
PLACE_CODE = {"名古屋": "48", "大井": "44", "川崎": "45", "船橋": "43", "笠松": "47"}


def rid_for(date: _dt.date, place: str, rno: int) -> str:
    return f"{date.year:04d}{PLACE_CODE[place]}{date.month:02d}{date.day:02d}{rno:02d}"


def fetch_result(rid: str) -> dict | None:
    """結果ページから着順(馬番)と払戻を取得。未確定なら None。"""
    for base in ("https://nar.netkeiba.com", "https://race.netkeiba.com"):
        try:
            r = requests.get(f"{base}/race/result.html?race_id={rid}",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            html = None
            for enc in ("utf-8", "EUC-JP"):
                h = r.content.decode(enc, "ignore")
                if "着順" in h or "払戻" in h:
                    html = h
                    break
            if html is None:
                continue
            s = BeautifulSoup(html, "lxml")
            t = s.select_one("table.ResultMain") or s.select_one("table.RaceTable01")
            if not t:
                continue
            order = []  # (rank, umaban)
            for tr in t.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if len(tds) < 3:
                    continue
                rk = tds[0].get_text(strip=True)
                num = tds[2].get_text(strip=True)
                if rk.isdigit() and num.isdigit():
                    order.append((int(rk), int(num)))
            if not order:
                continue
            order.sort()
            top3 = [n for _, n in order[:3]]
            if len(top3) < 3:
                continue
            payouts = {}
            for pt in s.select("table.Payout_Detail_Table"):
                for tr in pt.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
                    if len(cells) >= 3:
                        key = cells[0]
                        nums = cells[1]
                        yen = re.findall(r"([\d,]+)円", cells[2])
                        payouts[key] = {"nums": nums, "yen": [int(y.replace(",", "")) for y in yen]}
            name_el = s.select_one("div.RaceName") or s.select_one("h1")
            return {"top3": top3, "payouts": payouts,
                    "race_name": name_el.get_text(strip=True) if name_el else ""}
        except Exception:
            continue
    return None


def get_prediction(rid: str) -> dict | None:
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            "SELECT trifecta_1,trifecta_2,trifecta_3,trifecta_4,trifecta_5,confidence "
            "FROM predictions WHERE race_id=? ORDER BY id DESC LIMIT 1", (rid,)).fetchone()
    if not row:
        return None
    picks = [p for p in row[:5] if p]
    return {"picks": picks, "confidence": row[5]}


def lgbm_topn(rid: str, n: int = 6) -> list[tuple[int, float]]:
    """LightGBM 上位n頭の (馬番, p_top3)。"""
    try:
        from src.model.predict import predict_top3_probabilities
        df = predict_top3_probabilities(rid)
        df = df.dropna(subset=["horse_number"]).head(n)
        return [(int(r.horse_number), float(r.p_top3)) for r in df.itertuples()]
    except Exception:
        return []


def analyze(date: _dt.date, place: str) -> None:
    rows = []
    for rno in range(1, 13):
        rid = rid_for(date, place, rno)
        pred = get_prediction(rid)
        res = fetch_result(rid)
        topn = lgbm_topn(rid, 6)
        rows.append({"rno": rno, "rid": rid, "pred": pred, "res": res, "topn": topn})
        status = "結果あり" if res else "未確定"
        print(f"R{rno:>2} {status} "
              f"pred={pred['picks'][0] if pred else '-'} "
              f"actual={'-'.join(map(str,res['top3'])) if res else '-'}")

    out = ROOT / "data" / "eval" / f"day_{date}_{place}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    print(f"\n保存: {out}")

    # 集計
    finished = [r for r in rows if r["res"] and r["pred"]]
    print(f"\n=== 集計（結果確定 {len(finished)}R） ===")
    san_hit = 0
    box3_hit = 0   # 3連複 上位3頭ボックス(1点)
    box4_hit = 0   # 3連複 上位4頭ボックス(4点)
    box5_hit = 0   # 3連複 上位5頭ボックス(10点)
    umaren_hit = 0  # 馬連 上位 (1-2,1-3,1-4,2-3,...) 5点
    for r in finished:
        actual = r["res"]["top3"]
        actual_set = set(actual)
        actual_tri = "-".join(map(str, actual))
        picks = r["pred"]["picks"]
        nums = [n for n, _ in r["topn"]]  # LightGBM順 上位
        # 3連単5点
        if actual_tri in picks:
            san_hit += 1
        # 3連複ボックス
        if len(nums) >= 3 and actual_set <= set(nums[:3]):
            box3_hit += 1
        if len(nums) >= 4 and actual_set <= set(nums[:4]):
            box4_hit += 1
        if len(nums) >= 5 and actual_set <= set(nums[:5]):
            box5_hit += 1
        # 馬連: 上位2頭が1-2着内 → 上位5頭の全2頭組から(10点)ではなく軸流し4点で近似
        umaren_pairs = set()
        if nums:
            for x in nums[1:5]:
                umaren_pairs.add(frozenset({nums[0], x}))
        top2_set = set(actual[:2])
        if frozenset(top2_set) in umaren_pairs:
            umaren_hit += 1

    n = len(finished) or 1
    print(f"3連単5点(現行)        的中 {san_hit}/{len(finished)}  = {san_hit/n*100:.0f}%")
    print(f"3連複 上位3頭BOX(1点)  的中 {box3_hit}/{len(finished)}  = {box3_hit/n*100:.0f}%")
    print(f"3連複 上位4頭BOX(4点)  的中 {box4_hit}/{len(finished)}  = {box4_hit/n*100:.0f}%")
    print(f"3連複 上位5頭BOX(10点) 的中 {box5_hit}/{len(finished)}  = {box5_hit/n*100:.0f}%")
    print(f"馬連 軸1頭ながし(4点)  的中 {umaren_hit}/{len(finished)} = {umaren_hit/n*100:.0f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None)
    p.add_argument("--place", default="名古屋")
    a = p.parse_args()
    date = _dt.date.fromisoformat(a.date) if a.date else _dt.date.today()
    analyze(date, a.place)


if __name__ == "__main__":
    main()
