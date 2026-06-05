"""指定race_idの出馬表が公開済みか確認。

公開済み: 標準出力に "READY <頭数> <レース名>"
未公開:   "NOTYET"
使い方: python scripts/check_shutuba.py 202605030211
"""
from __future__ import annotations

import sys
import requests
from bs4 import BeautifulSoup


def check(race_id: str) -> tuple[bool, int, str]:
    for base in ("https://race.netkeiba.com", "https://nar.netkeiba.com"):
        try:
            r = requests.get(f"{base}/race/shutuba.html?race_id={race_id}",
                             headers={"User-Agent": "KeibaYosouAI/0.1"}, timeout=15)
            for enc in ("EUC-JP", "utf-8"):
                h = r.content.decode(enc, "ignore")
                if "出馬表" in h or "Shutuba" in h or "RaceName" in h:
                    break
            s = BeautifulSoup(h, "lxml")
            table = s.select_one("table.ShutubaTable") or s.select_one("table.Shutuba_Table")
            rows = table.select("tr.HorseList") if table else []
            if rows:
                nm = s.select_one("div.RaceName") or s.select_one("h1")
                return True, len(rows), (nm.get_text(strip=True) if nm else "")
        except Exception:
            continue
    return False, 0, ""


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "202605030211"
    ok, n, name = check(rid)
    print(f"READY {n} {name}" if ok else "NOTYET")
