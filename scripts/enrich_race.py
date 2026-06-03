"""指定レースの全出走馬の戦歴・脚質を取得し、予測用コンテキストを作成。

使い方:
    python scripts/enrich_race.py 202605030211  # 安田記念2026 (出馬表確定後)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
import re
from bs4 import BeautifulSoup

from src.scraper.horse_history import fetch_horse_history, estimate_running_style


def fetch_shutuba_simple(race_id: str, nar: bool = False) -> list[dict]:
    base = "https://nar.netkeiba.com" if nar else "https://race.netkeiba.com"
    url = f"{base}/race/shutuba.html?race_id={race_id}"
    r = requests.get(url, headers={"User-Agent": "KeibaYosouAI/0.1"}, timeout=15)
    r.encoding = "EUC-JP"
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.select_one("table.ShutubaTable") or soup.select_one("table.Shutuba_Table")
    horses = []
    if not table:
        return horses
    for tr in table.select("tr.HorseList"):
        tds = tr.find_all("td")
        if len(tds) < 11: continue
        a = tds[3].find("a", href=re.compile(r"/horse/"))
        if not a: continue
        m = re.search(r"/horse/(\w+)", a.get("href", ""))
        if not m: continue
        horses.append({
            "horse_id": m.group(1),
            "horse_number": int(tds[1].get_text(strip=True)) if tds[1].get_text(strip=True).isdigit() else None,
            "horse_name": a.get_text(strip=True),
            "jockey": tds[6].get_text(strip=True),
        })
    return horses


def enrich_race(race_id: str, nar: bool = False, last_n: int = 5) -> dict:
    print(f"[1/3] 出馬表取得: {race_id}")
    horses = fetch_shutuba_simple(race_id, nar=nar)
    if not horses:
        # 試す: db.netkeiba.com 経由 (過去レース)
        try:
            from src.scraper import netkeiba
            data = netkeiba.fetch_and_parse_race(race_id)
            horses = [{"horse_id": r.horse_id, "horse_number": r.horse_number,
                       "horse_name": r.horse_name, "jockey": r.jockey_name} for r in data.results]
        except Exception:
            pass

    if not horses:
        return {"error": f"出馬表取得失敗: {race_id}"}

    print(f"[2/3] {len(horses)}頭の戦歴をplaywright取得 (約{len(horses)*8}秒)")
    enriched = []
    for i, h in enumerate(horses, 1):
        print(f"  [{i:>2}/{len(horses)}] {h['horse_name']}", end=" ", flush=True)
        try:
            hist = fetch_horse_history(h["horse_id"], timeout_ms=20000)
            recent_rows = [r for r in hist.rows if r.passage and r.rank is not None][:last_n]
            passages = [r.passage for r in recent_rows if r.passage]
            style = estimate_running_style(passages)
            recent_summary = [
                {
                    "date": r.date, "race": r.race_name, "rank": r.rank,
                    "passage": r.passage, "agari": r.agari_3f,
                    "distance": r.distance, "surface": r.surface,
                }
                for r in recent_rows
            ]
            h["running_style"] = style
            h["last_passages"] = passages
            h["recent_n_races"] = len(recent_summary)
            h["recent_summary"] = recent_summary
            print(f"→ 脚質={style}, 戦歴{len(hist.rows)}走")
        except Exception as e:
            h["running_style"] = "不明"
            h["error"] = str(e)[:50]
            print(f"→ ERR: {str(e)[:50]}")
        enriched.append(h)
        time.sleep(0.5)  # サーバ負荷軽減

    # 展開予想サマリ
    style_counts = {}
    for h in enriched:
        s = h.get("running_style", "不明")
        style_counts[s] = style_counts.get(s, 0) + 1
    pace_hint = "ハイペース" if (style_counts.get("逃げ",0)+style_counts.get("先行",0))>= len(enriched)*0.5 else "スローペース"

    print(f"[3/3] 完了")
    print(f"\n=== レース {race_id} 脚質分布 ===")
    for s, c in sorted(style_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}頭")
    print(f"想定ペース: {pace_hint}")

    return {
        "race_id": race_id,
        "horses": enriched,
        "style_counts": style_counts,
        "pace_hint": pace_hint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("race_id")
    parser.add_argument("--nar", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = enrich_race(args.race_id, nar=args.nar)

    out_path = Path(args.output) if args.output else (ROOT / "data" / "raw" / f"enriched_{args.race_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
