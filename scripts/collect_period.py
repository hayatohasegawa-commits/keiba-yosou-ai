"""期間指定の大量レース収集スクリプト。

指定の日付範囲（YYYY-MM-DD ～ YYYY-MM-DD）の開催日について、
netkeiba race_list ページから race_id を列挙し、未保存のものを順次収集する。

使い方:
    python scripts/collect_period.py --from 2021-01-01 --to 2026-06-01
    python scripts/collect_period.py --from 2021-01-01 --to 2026-06-01 --weekend-only

カレンダー上の全日付を試すと無駄が多いので、デフォルトは weekend のみ。
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup  # noqa: E402

from src.scraper import netkeiba  # noqa: E402
from src.db import repository as repo  # noqa: E402


def setup_logging(verbose: bool) -> logging.Logger:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"collect_period_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s | %(message)s")
    h_file = logging.FileHandler(log_path, encoding="utf-8")
    h_file.setFormatter(fmt)
    h_stream = logging.StreamHandler()
    h_stream.setFormatter(fmt)
    log = logging.getLogger("collect_period")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers = []
    log.addHandler(h_file)
    log.addHandler(h_stream)
    return log


def existing_race_ids() -> set[str]:
    with sqlite3.connect(repo.db_path()) as conn:
        return {r[0] for r in conn.execute("SELECT race_id FROM races").fetchall()}


def race_ids_for_date(yyyymmdd: str, cache) -> dict[str, str]:
    url = f"https://db.netkeiba.com/race/list/{yyyymmdd}/"
    try:
        html = cache.get(url, encoding="EUC-JP")
    except Exception:
        return {}
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, str] = {}
    for a in soup.select('a[href*="/race/"]'):
        m = re.search(r"/race/(\d{12})", a.get("href", ""))
        if m:
            out.setdefault(m.group(1), a.get_text(strip=True))
    return out


def daterange(d_from: date, d_to: date, weekend_only: bool):
    d = d_from
    while d <= d_to:
        if (not weekend_only) or d.weekday() >= 5:
            yield d
        d += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="d_from", required=True, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", dest="d_to", required=True, help="終了日 YYYY-MM-DD")
    parser.add_argument("--weekend-only", action="store_true", default=True)
    parser.add_argument("--all-days", action="store_true", help="平日も含めて全日対象")
    parser.add_argument("--limit", type=int, default=0, help="最大保存件数(0=制限なし)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    weekend_only = not args.all_days

    log = setup_logging(args.verbose)
    cache = netkeiba.build_cache()

    d_from = datetime.strptime(args.d_from, "%Y-%m-%d").date()
    d_to = datetime.strptime(args.d_to, "%Y-%m-%d").date()

    existing = existing_race_ids()
    saved = 0
    skipped = 0
    errors = 0
    started = time.time()

    log.info("period: %s ~ %s, weekend_only=%s, existing_in_db=%d",
             d_from, d_to, weekend_only, len(existing))

    for d in daterange(d_from, d_to, weekend_only):
        ids = race_ids_for_date(d.strftime("%Y%m%d"), cache)
        if not ids:
            continue
        log.info("%s: %d races on netkeiba", d, len(ids))
        for rid, name in ids.items():
            if rid in existing:
                skipped += 1
                continue
            try:
                data = netkeiba.fetch_and_parse_race(rid, cache=cache)
                if data.results:
                    repo.save_race_data(data)
                    existing.add(rid)
                    saved += 1
                    if saved % 20 == 0:
                        elapsed = time.time() - started
                        log.info("progress: saved=%d skipped=%d errors=%d elapsed=%.0fs",
                                 saved, skipped, errors, elapsed)
                    if args.limit and saved >= args.limit:
                        log.info("limit reached: %d", args.limit)
                        return 0
            except Exception as e:  # noqa: BLE001
                errors += 1
                log.warning("fetch failed %s (%s): %s", rid, name, e)

    elapsed = time.time() - started
    log.info("done. saved=%d, skipped=%d, errors=%d, elapsed=%.0fs, total_in_db=%d",
             saved, skipped, errors, elapsed, repo.race_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
