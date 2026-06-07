"""中央1開催の全レースを、単勝オッズ補完つきで予測し、
3連単5点・3連複5点と「推定的中率」を出す本番スクリプト。

的中率は市場の単勝オッズ→勝率に変換し、Plackett-Luceでtop3分布をMC推定。
（モデルは買い目選定、市場オッズは的中率推定に使う＝役割分担）

使い方:
    python scripts/predict_central_day.py --base 2026050302 --place 東京 --date 2026-06-07
    # base = YYYY+場(2)+回(2)+日目(2) の10桁。東京6/7=2026050302
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.predict_day import fetch_shutuba_race, _ranked_horses
from src.scraper.odds import fetch_win_odds
from src.reasoning.bet_builder import build_bets, sanrentan_5, sanrenpuku_5
from src.db import repository as repo

random.seed(42)


def win_probs_from_odds(odds_map: dict[int, tuple[float, int]]) -> dict[int, float]:
    """単勝オッズ→正規化勝率（控除率を除去）。"""
    inv = {n: 1.0 / o for n, (o, _) in odds_map.items() if o and o > 0}
    s = sum(inv.values()) or 1.0
    return {n: v / s for n, v in inv.items()}


def mc_hit_rates(win_p: dict[int, float], tan5: list[str], puku5: list[str],
                 n_sim: int = 30000) -> tuple[float, float]:
    """Plackett-LuceでMCし、3連単5点・3連複5点の的中率を推定。"""
    if not win_p:
        return 0.0, 0.0
    horses = list(win_p.keys())
    weights = [win_p[h] for h in horses]
    tan_set = set(tan5)
    puku_set = set(frozenset(map(int, p.split("-"))) for p in puku5)
    tan_hit = puku_hit = 0
    for _ in range(n_sim):
        # 1着→2着→3着を非復元で重み付き抽選
        pool = horses[:]
        w = weights[:]
        top3 = []
        for _k in range(3):
            tot = sum(w)
            r = random.random() * tot
            acc = 0.0
            idx = 0
            for i, wi in enumerate(w):
                acc += wi
                if r <= acc:
                    idx = i
                    break
            top3.append(pool[idx])
            pool.pop(idx)
            w.pop(idx)
        a, b, c = top3
        if f"{a}-{b}-{c}" in tan_set:
            tan_hit += 1
        if frozenset(top3) in puku_set:
            puku_hit += 1
    return tan_hit / n_sim, puku_hit / n_sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="YYYY+場+回+日目 の10桁 (例 2026050302)")
    ap.add_argument("--place", default="東京")
    ap.add_argument("--date", default=None)
    ap.add_argument("--races", default="1-12")
    ap.add_argument("--save", action="store_true", help="DBに3連複5点を保存")
    a = ap.parse_args()
    date = _dt.date.fromisoformat(a.date) if a.date else _dt.date.today()
    lo, hi = (int(x) for x in a.races.split("-"))

    print(f"=== {date} {a.place} 全力予測（3連単5点／3連複5点／推定的中率） ===\n")
    summary = []
    for rno in range(lo, hi + 1):
        rid = f"{a.base}{rno:02d}"
        data = fetch_shutuba_race(rid, date, a.place)
        if not data or not data.results:
            print(f"R{rno:>2}: 出馬表取得できず")
            continue
        # 単勝オッズで補完（中央HTMLには無いため）
        odds_map = fetch_win_odds(rid)
        for row in data.results:
            if row.horse_number in odds_map:
                o, nin = odds_map[row.horse_number]
                row.odds, row.popularity = o, nin
        repo.save_race_data(data)

        entries = pd.DataFrame([r.__dict__ for r in data.results])
        ranked, source = _ranked_horses(rid, entries)
        top = ranked.dropna(subset=["horse_number"]).head(6)
        nums = [int(x) for x in top["horse_number"].tolist()]
        names = {int(r.horse_number): (getattr(r, "horse_name", "") or "")
                 for r in top.itertuples() if pd.notna(r.horse_number)}
        tan5 = sanrentan_5(nums)
        puku5 = sanrenpuku_5(nums)
        win_p = win_probs_from_odds(odds_map)
        tan_r, puku_r = mc_hit_rates(win_p, tan5, puku5)

        if a.save and puku5:
            repo.insert_prediction(race_id=rid, picks=puku5,
                                   rationale=f"3連複5点 軸{nums[0]} 推定的中率{puku_r*100:.1f}%",
                                   confidence=puku_r, model_version="sanpuku5_oddsfix")
        nm = data.meta.race_name or f"R{rno}"
        axis = nums[0]
        summary.append((rno, nm, axis, names.get(axis, ""), tan5, puku5, tan_r, puku_r))
        print(f"R{rno:>2} {nm[:22]:<22} 軸{axis}番{names.get(axis,'')}")
        print(f"    3連単5点: {' / '.join(tan5)}  （推定的中率 {tan_r*100:4.1f}%）")
        print(f"    3連複5点: {' / '.join(puku5)}  （推定的中率 {puku_r*100:4.1f}%）")

    print("\n=== 推定的中率まとめ ===")
    print(f"{'R':>3} {'レース':<20} {'3連単5点':>9} {'3連複5点':>9}")
    for rno, nm, axis, an, t5, p5, tr, pr in summary:
        print(f"R{rno:>2} {nm[:20]:<20} {tr*100:>7.1f}% {pr*100:>7.1f}%")


if __name__ == "__main__":
    main()
