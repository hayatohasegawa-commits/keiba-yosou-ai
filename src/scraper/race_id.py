"""JRA race_id 生成・パースユーティリティ。

netkeiba/JRA 共通の race_id は以下の12桁:
  YYYY + JJ(場所2桁) + KK(開催回2桁) + DD(日目2桁) + RR(レース番号2桁)

場所コード:
  01 札幌  02 函館  03 福島  04 新潟  05 東京
  06 中山  07 中京  08 京都  09 阪神  10 小倉
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

PLACE_CODE: dict[str, str] = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}
CODE_TO_PLACE: dict[str, str] = {v: k for k, v in PLACE_CODE.items()}


@dataclass(frozen=True)
class RaceID:
    year: int
    place: str
    kaiji: int
    day: int
    race_no: int

    def to_str(self) -> str:
        return f"{self.year:04d}{PLACE_CODE[self.place]}{self.kaiji:02d}{self.day:02d}{self.race_no:02d}"

    @classmethod
    def parse(cls, race_id: str) -> "RaceID":
        if len(race_id) != 12 or not race_id.isdigit():
            raise ValueError(f"invalid race_id: {race_id}")
        return cls(
            year=int(race_id[0:4]),
            place=CODE_TO_PLACE[race_id[4:6]],
            kaiji=int(race_id[6:8]),
            day=int(race_id[8:10]),
            race_no=int(race_id[10:12]),
        )


def build(year: int, place: str, kaiji: int, day: int, race_no: int) -> str:
    return RaceID(year, place, kaiji, day, race_no).to_str()


def iterate_year_races(year: int, places: list[str] | None = None) -> Iterator[str]:
    """指定年の主要レース候補 race_id を網羅生成。

    開催回 1-6, 日目 1-12, R 1-12 の全組み合わせ。
    実在しない race_id も含まれるので、スクレイピング時に存在判定する想定。
    """
    places = places or list(PLACE_CODE.keys())
    for place in places:
        for kaiji in range(1, 7):
            for day in range(1, 13):
                for race_no in range(1, 13):
                    yield build(year, place, kaiji, day, race_no)


KNOWN_G1_RACE_IDS: dict[str, list[str]] = {
    "安田記念": [
        "202505030211",
        "202405030211",
        "202305030211",
        "202205030211",
        "202105030211",
    ],
    "NHKマイルカップ": [
        "202505020611",
        "202405020611",
        "202305020611",
        "202205020611",
        "202105020611",
    ],
    "ヴィクトリアマイル": [
        "202505020811",
        "202405020811",
        "202305020811",
        "202205020811",
        "202105020811",
    ],
}


def yasuda_kinen_2026_race_id() -> str:
    """2026年安田記念の推定 race_id。

    過去5年は 'YYYY05030211' で第3回東京2日目11R固定。
    2026-06-07(日)は3回東京2日目の予想 → 202605030211。
    レース前なのでnetkeibaの確定情報での検証必要。
    """
    return "202605030211"
