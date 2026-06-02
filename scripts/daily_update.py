"""毎日のDB自動更新スクリプト（launchdから呼び出される想定）。

実行内容:
  1. 直近3日間の開催日について netkeiba race list を取得
  2. 新規 race_id を検出して DB に未保存なら fetch & save
  3. 既存予測 (predictions) について実績照合・hit/payout更新（TODO）

ログ: logs/daily_update_YYYY-MM-DD.log
"""
from __future__ import annotations

import logging
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup  # noqa: E402

from src.scraper import netkeiba  # noqa: E402
from src.db import repository as repo  # noqa: E402


def setup_logging() -> logging.Logger:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"daily_update_{date.today():%Y-%m-%d}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    log = logging.getLogger("daily_update")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.addHandler(stream)
    return log


def extract_race_ids_for_date(yyyymmdd: str, cache) -> dict[str, str]:
    """race_list ページから race_id → race_name の辞書を取得。"""
    url = f"https://db.netkeiba.com/race/list/{yyyymmdd}/"
    try:
        html = cache.get(url, encoding="EUC-JP")
    except Exception as e:  # noqa: BLE001
        logging.warning("race list fetch failed for %s: %s", yyyymmdd, e)
        return {}
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, str] = {}
    for a in soup.select('a[href*="/race/"]'):
        m = re.search(r"/race/(\d{12})", a.get("href", ""))
        if m:
            rid = m.group(1)
            out.setdefault(rid, a.get_text(strip=True))
    return out


def main() -> int:
    log = setup_logging()
    cache = netkeiba.build_cache()

    today = date.today()
    target_dates = [today - timedelta(days=d) for d in range(1, 4)]
    log.info("daily update for: %s", [d.isoformat() for d in target_dates])

    existing = set(_existing_race_ids())
    new_count = 0
    err_count = 0

    for d in target_dates:
        ids = extract_race_ids_for_date(d.strftime("%Y%m%d"), cache)
        log.info("%s: %d races found", d, len(ids))
        for rid, name in ids.items():
            if rid in existing:
                continue
            try:
                data = netkeiba.fetch_and_parse_race(rid, cache=cache)
                if data.results:
                    repo.save_race_data(data)
                    new_count += 1
                    log.info("saved %s: %s (%d horses)", rid, data.meta.race_name or name, len(data.results))
            except Exception as e:  # noqa: BLE001
                err_count += 1
                log.warning("fetch failed %s (%s): %s", rid, name, e)

    log.info("done. new=%d, errors=%d, total in DB=%d", new_count, err_count, repo.race_count())
    return 0


def _existing_race_ids() -> list[str]:
    import sqlite3
    with sqlite3.connect(repo.db_path()) as conn:
        return [r[0] for r in conn.execute("SELECT race_id FROM races").fetchall()]


if __name__ == "__main__":
    raise SystemExit(main())
