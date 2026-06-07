"""中央1開催の結果を取得し、3連単5点・3連複5点の的中/回収率を集計。

予測時と同じ買い目を再生成（DBのエントリー＋オッズから）して、
実結果・実払戻で回収率を出す。

使い方:
    python scripts/eval_central_day.py --base 2026050302 --place 東京 --races 1-12
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.eval_day import fetch_result
from scripts.predict_day import _ranked_horses
from scripts.predict_central_day import sanrentan_5, sanrenpuku_5

DB = ROOT / "data" / "db" / "keiba.sqlite"


def yen(payouts: dict, key: str):
    d = payouts.get(key)
    return d["yen"][0] if d and d.get("yen") else None


def entries_from_db(rid: str) -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        return pd.read_sql_query(
            "SELECT r.*, (SELECT name FROM horses h WHERE h.horse_id=r.horse_id) AS horse_name "
            "FROM results r WHERE race_id=?", conn, params=(rid,))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--place", default="東京")
    ap.add_argument("--races", default="1-12")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.races.split("-"))

    print(f"=== {a.place} 回収率集計（3連単5点 / 3連複5点） ===\n")
    rows = []
    for rno in range(lo, hi + 1):
        rid = f"{a.base}{rno:02d}"
        res = fetch_result(rid)
        if not res:
            print(f"R{rno:>2}: 未確定")
            continue
        actual = res["top3"]
        actual_tri = "-".join(map(str, actual))
        actual_set = frozenset(actual)
        ent = entries_from_db(rid)
        if ent.empty:
            print(f"R{rno:>2}: エントリーなし(予測してない)")
            continue
        ranked, _ = _ranked_horses(rid, ent)
        nums = [int(x) for x in ranked.dropna(subset=["horse_number"]).head(6)["horse_number"].tolist()]
        tan5, puku5 = sanrentan_5(nums), sanrenpuku_5(nums)
        puku_sets = {frozenset(map(int, p.split("-"))) for p in puku5}

        tan_hit = actual_tri in tan5
        puku_hit = actual_set in puku_sets
        p_tan = yen(res["payouts"], "3連単")
        p_puku = yen(res["payouts"], "3連複")
        tan_ret = p_tan if tan_hit else 0
        puku_ret = p_puku if puku_hit else 0
        rows.append((rno, res["race_name"], actual_tri, tan_hit, tan_ret, puku_hit, puku_ret, p_puku))
        mark = lambda b: "○" if b else "×"
        print(f"R{rno:>2} {res['race_name'][:20]:<20} 実{actual_tri:<9} "
              f"3単{mark(tan_hit)}{('+'+str(tan_ret)) if tan_hit else '':>8} | "
              f"3複{mark(puku_hit)}{('+'+str(puku_ret)) if puku_hit else '':>8} (3複配当{p_puku})")

    n = len(rows)
    if not n:
        print("\n確定レースなし")
        return
    tan_hits = sum(1 for r in rows if r[3])
    puku_hits = sum(1 for r in rows if r[5])
    tan_cost, puku_cost = n * 500, n * 500
    tan_ret = sum(r[4] for r in rows)
    puku_ret = sum(r[6] for r in rows)
    print(f"\n=== 集計（確定 {n}R・各5点100円=1レース500円）===")
    print(f"{'馬券':<10}{'的中':>8}{'投資':>9}{'払戻':>10}{'回収率':>9}{'収支':>10}")
    print(f"{'3連単5点':<10}{tan_hits:>3}/{n}{tan_cost:>9}{tan_ret:>10}{tan_ret/tan_cost*100:>8.0f}%{tan_ret-tan_cost:>+10}")
    print(f"{'3連複5点':<10}{puku_hits:>3}/{n}{puku_cost:>9}{puku_ret:>10}{puku_ret/puku_cost*100:>8.0f}%{puku_ret-puku_cost:>+10}")


if __name__ == "__main__":
    main()
