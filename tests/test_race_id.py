"""race_id ユーティリティの単体テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scraper.race_id import RaceID, build, yasuda_kinen_2026_race_id, PLACE_CODE


def test_build_and_parse_roundtrip():
    rid = build(2025, "東京", 2, 8, 11)
    assert rid == "202505020811"
    parsed = RaceID.parse(rid)
    assert parsed.year == 2025
    assert parsed.place == "東京"
    assert parsed.kaiji == 2
    assert parsed.day == 8
    assert parsed.race_no == 11


def test_all_places_have_unique_codes():
    codes = list(PLACE_CODE.values())
    assert len(codes) == len(set(codes))


def test_yasuda_2026():
    assert yasuda_kinen_2026_race_id() == "202605030211"


if __name__ == "__main__":
    test_build_and_parse_roundtrip()
    test_all_places_have_unique_codes()
    test_yasuda_2026()
    print("OK")
