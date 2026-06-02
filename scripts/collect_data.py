"""過去レースデータ収集スクリプト。

使い方:
    python scripts/collect_data.py --year 2025 --grade G1
    python scripts/collect_data.py --race-ids 202505020811 202405020811

すべてのHTTPはレート制限 + キャッシュ経由。安全に何度でも再実行可能。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scraper import netkeiba, race_id as race_id_mod  # noqa: E402
from src.db import repository as repo  # noqa: E402

log = logging.getLogger("collect")


def collect_one(race_id: str, cache) -> bool:
    try:
        data = netkeiba.fetch_and_parse_race(race_id, cache=cache)
    except Exception as e:  # noqa: BLE001
        log.warning("fetch failed for %s: %s", race_id, e)
        return False
    if not data.results:
        log.info("no results for %s (likely non-existent race_id)", race_id)
        return False
    repo.save_race_data(data)
    log.info("saved %s: %s (%d horses)", race_id, data.meta.race_name, len(data.results))
    return True


def collect_known_g1(name: str, cache) -> int:
    ids = race_id_mod.KNOWN_G1_RACE_IDS.get(name, [])
    n = 0
    for rid in ids:
        if collect_one(rid, cache):
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-ids", nargs="*", help="特定の race_id を指定")
    parser.add_argument("--known-g1", help="KNOWN_G1_RACE_IDSのキー名 (例: 安田記念)")
    parser.add_argument("--year", type=int, help="指定年の主要race_id を網羅探索")
    parser.add_argument("--places", nargs="*", help="場所フィルタ (例: 東京 中山)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cache = netkeiba.build_cache()

    n = 0
    if args.race_ids:
        for rid in args.race_ids:
            if collect_one(rid, cache):
                n += 1
    elif args.known_g1:
        n = collect_known_g1(args.known_g1, cache)
    elif args.year:
        for rid in race_id_mod.iterate_year_races(args.year, args.places):
            if collect_one(rid, cache):
                n += 1
    else:
        parser.print_help()
        sys.exit(1)

    log.info("done. saved %d race(s). total in DB: %d", n, repo.race_count())


if __name__ == "__main__":
    main()
