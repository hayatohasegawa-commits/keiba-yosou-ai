"""指定日・指定競馬場の全レースをまとめて予測し、DB(races/results/predictions)に保存。

「📅 今日の予測」タブに反映させるためのバッチ。出馬表を取得して
results に出走馬を入れる（rank=NULL）ことで LightGBM が効くようにする。

使い方:
    # 今日の名古屋 全12R
    python scripts/predict_day.py
    # 日付・場・レース範囲を指定
    python scripts/predict_day.py --date 2026-06-04 --place 名古屋 --races 1-12
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re
import requests
from bs4 import BeautifulSoup

from src.scraper.netkeiba import RaceMeta, ResultRow, RaceData
from src.db import repository as repo

log = logging.getLogger("predict_day")

# netkeiba 場所コード（地方競馬 NAR を含む）
PLACE_CODE = {
    "名古屋": "48", "笠松": "47", "園田": "50", "姫路": "51",
    "高知": "54", "佐賀": "55", "金沢": "46", "大井": "44",
    "川崎": "45", "船橋": "43", "浦和": "42", "門別": "30",
    "盛岡": "35", "水沢": "36",
    # 中央
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}


def build_race_id(date: _dt.date, place: str, race_no: int) -> str:
    """YYYY + 場(2) + MM(2) + DD(2) + RR(2)。地方競馬の日付ベースID。"""
    code = PLACE_CODE[place]
    return f"{date.year:04d}{code}{date.month:02d}{date.day:02d}{race_no:02d}"


def _decode(resp: requests.Response) -> str:
    """netkeiba のエンコーディングを推定してテキスト化。"""
    for enc in ("EUC-JP", "utf-8"):
        try:
            txt = resp.content.decode(enc)
            # 文字化けの簡易判定（日本語が壊れていないか）
            if "race" in txt.lower() or "馬" in txt or "出走" in txt:
                return txt
        except UnicodeDecodeError:
            continue
    resp.encoding = resp.apparent_encoding
    return resp.text


def fetch_shutuba_race(race_id: str, date: _dt.date, place: str) -> RaceData | None:
    """出馬表を取得し RaceData を返す（app の実績パーサを踏襲）。"""
    for base in ("https://nar.netkeiba.com", "https://race.netkeiba.com"):
        try:
            r = requests.get(
                f"{base}/race/shutuba.html?race_id={race_id}",
                headers={"User-Agent": "KeibaYosouAI/0.1"}, timeout=15,
            )
            html = _decode(r)
            soup = BeautifulSoup(html, "lxml")
            table = soup.select_one("table.ShutubaTable") or soup.select_one("table.Shutuba_Table")
            if not table:
                continue

            meta = RaceMeta(race_id=race_id, date=date.strftime("%Y-%m-%d"), course=place)
            meta.race_number = int(race_id[-2:])
            name_el = soup.select_one("div.RaceName") or soup.select_one("h1")
            if name_el and name_el.get_text(strip=True):
                meta.race_name = name_el.get_text(strip=True)
            else:
                # 中央はRaceNameがJS生成で空 → <title>から補完
                title = soup.select_one("title")
                if title:
                    meta.race_name = title.get_text(strip=True).split("|")[0].replace("出馬表", "").strip()
            data01 = soup.select_one("div.RaceData01")
            if data01:
                txt = data01.get_text(" ", strip=True)
                m_dist = re.search(r"(\d{3,4})m", txt)
                if m_dist:
                    meta.distance = int(m_dist.group(1))
                if "芝" in txt:
                    meta.surface = "芝"
                elif "ダ" in txt:
                    meta.surface = "ダ"
                m_w = re.search(r"天候\s*:\s*(\S+)", txt)
                if m_w:
                    meta.weather = m_w.group(1)
                m_tc = re.search(r"馬場\s*:\s*(\S+)", txt)
                if m_tc:
                    meta.track_cond = m_tc.group(1)

            results: list[ResultRow] = []
            for tr in table.select("tr.HorseList"):
                tds = tr.find_all("td")
                if len(tds) < 11:
                    continue
                horse_a = tds[3].find("a", href=re.compile(r"/horse/"))
                if not horse_a:
                    continue
                m = re.search(r"/horse/(\w+)", horse_a.get("href", ""))
                horse_id = m.group(1) if m else ""
                if not horse_id:
                    continue
                jockey_a = tds[6].find("a")
                jockey_id = None
                if jockey_a:
                    m_j = re.search(r"/jockey/(?:result/recent/)?(\d+)", jockey_a.get("href", ""))
                    jockey_id = m_j.group(1) if m_j else None
                weight_m = re.match(r"(\d+)\s*\(\s*([+\-]?\d+)\s*\)", tds[8].get_text(strip=True))
                row = ResultRow(
                    race_id=race_id, horse_id=horse_id,
                    horse_name=horse_a.get_text(strip=True),
                    jockey_id=jockey_id,
                    jockey_name=jockey_a.get_text(strip=True) if jockey_a else None,
                    rank=None,
                    horse_number=int(tds[1].get_text(strip=True)) if tds[1].get_text(strip=True).isdigit() else None,
                    post_position=int(tds[0].get_text(strip=True)) if tds[0].get_text(strip=True).isdigit() else None,
                    sex_age=tds[4].get_text(strip=True) or None,
                    handicap=float(tds[5].get_text(strip=True)) if re.match(r"^\d+\.?\d?$", tds[5].get_text(strip=True)) else None,
                    body_weight=float(weight_m.group(1)) if weight_m else None,
                    body_weight_diff=float(weight_m.group(2)) if weight_m else None,
                    odds=float(tds[9].get_text(strip=True).replace(",", "")) if re.match(r"^[\d.,]+$", tds[9].get_text(strip=True)) else None,
                    popularity=int(tds[10].get_text(strip=True)) if tds[10].get_text(strip=True).isdigit() else None,
                )
                results.append(row)
            if results:
                meta.starters = len(results)
                return RaceData(meta=meta, results=results)
        except Exception as e:  # noqa: BLE001
            log.debug("fetch error %s: %s", base, e)
            continue
    return None


def _simple_top3(entries_df):
    """人気・オッズベースの簡易3着内確率（LightGBM フォールバック）。"""
    df = entries_df.copy()
    if "popularity" in df.columns and df["popularity"].notna().any():
        rank = df["popularity"].fillna(99).rank(method="first")
        raw = 1.0 / rank
    elif "odds" in df.columns and df["odds"].notna().any():
        raw = 1.0 / df["odds"].fillna(999)
    else:
        raw = 1.0
    df["p_top3"] = (raw / float(raw.sum() or 1) * 3).clip(0, 1)
    return df.sort_values("p_top3", ascending=False)


def _build_5(ranked) -> tuple[list[str], str, float]:
    """確率上位5頭で「1着固定流し型」5点 + 根拠を組む。"""
    top = ranked.head(5).reset_index(drop=True)
    if len(top) < 4:
        return [], "出走馬不足", 0.0
    n = top["horse_number"].astype("Int64").tolist()
    names = top["horse_name"].tolist() if "horse_name" in top.columns else ["?"] * len(top)
    probs = top["p_top3"].tolist() if "p_top3" in top.columns else [0.0] * len(top)
    odds = top["odds"].tolist() if "odds" in top.columns else [0.0] * len(top)
    while len(n) < 5: n.append(0)
    while len(names) < 5: names.append("?")
    while len(probs) < 5: probs.append(0.0)
    while len(odds) < 5: odds.append(0.0)
    picks = [
        f"{n[0]}-{n[1]}-{n[2]}",
        f"{n[0]}-{n[2]}-{n[1]}",
        f"{n[0]}-{n[1]}-{n[3]}",
        f"{n[0]}-{n[3]}-{n[1]}",
        f"{n[0]}-{n[2]}-{n[3]}",
    ]
    od = lambda v: f"{v:.1f}倍" if v else "—"
    rationale = (
        f"【全体観】LightGBM(AUC0.83・約7.7万着分)で各馬の3着内確率を算出し、上位5頭で"
        f"「1着固定流し型」5点を構築。バックテストで3点版より命中率が高い構成。\n\n"
        f"【軸馬: {n[0]}番 {names[0]}】p_top3={probs[0]:.1%}・単勝{od(odds[0])}。確率が明確に高く軸。\n\n"
        f"【相手2-3着】{n[1]}番 {names[1]}(p={probs[1]:.1%})と{n[2]}番 {names[2]}(p={probs[2]:.1%})が筆頭。"
        f"2-3着の入替を含め4点に厚く張る。\n\n"
        f"【穴狙い】5点目は{n[0]}-{n[2]}-{n[3]}で4番手 {names[3]}(p={probs[3]:.1%})を3着に絡め回収率も狙う。"
    )
    conf = float(probs[0]) if probs[0] else 0.40
    return picks, rationale, conf


def _ranked_horses(race_id: str, entries_df):
    """LightGBM(失敗時は人気)で確率降順の ranked DF と source を返す。"""
    source = "LightGBM"
    ranked = None
    try:
        from src.model.predict import predict_top3_probabilities
        lgbm = predict_top3_probabilities(race_id)
        if lgbm is not None and not lgbm.empty and lgbm["p_top3"].notna().any():
            ranked = entries_df.merge(lgbm[["horse_id", "p_top3"]], on="horse_id", how="left")
            ranked = ranked.sort_values("p_top3", ascending=False)
    except Exception as e:  # noqa: BLE001
        log.warning("LightGBM 失敗 %s: %s", race_id, e)

    if ranked is None or "p_top3" not in ranked.columns or ranked["p_top3"].isna().all():
        source = "人気・オッズベース"
        ranked = _simple_top3(entries_df)
    return ranked, source


def build_trifecta(race_id: str, entries_df) -> tuple[list[str], str, float, str]:
    """LightGBM 確率で 5点（1着固定流し）を組む。失敗時は人気・オッズで。"""
    ranked, source = _ranked_horses(race_id, entries_df)
    picks, rationale, conf = _build_5(ranked)
    return picks, rationale, conf, source


def build_betplan(race_id: str, entries_df):
    """bet_builder で 3連単/3連複 をまとめて組む。返り値 (BetPlan, source, ranked_df)。"""
    from src.reasoning.bet_builder import build_bets
    ranked, source = _ranked_horses(race_id, entries_df)
    top = ranked.dropna(subset=["horse_number"]).head(6)
    pairs = [(int(r.horse_number), float(getattr(r, "p_top3", 0) or 0)) for r in top.itertuples()]
    return build_bets(pairs), source, ranked


def process_race(rid: str, date: _dt.date, place: str, bet: str) -> bool:
    """1レースを予測してDB保存。成功でTrue。"""
    import pandas as pd
    from src.reasoning.bet_builder import rationale as bet_rationale
    data = fetch_shutuba_race(rid, date, place)
    if not data or not data.results:
        print("出馬表取得できず（未確定/非開催）")
        return False

    # DBへ出走馬を保存（rank=NULL）→ LightGBM が使えるように
    repo.save_race_data(data)

    entries_df = pd.DataFrame([r.__dict__ for r in data.results])
    plan, source, ranked = build_betplan(rid, entries_df)
    if plan.axis is None:
        print(f"予測組成できず（{len(data.results)}頭・{source}）")
        return False

    names, probs = {}, {}
    for r in ranked.head(6).itertuples():
        if pd.notna(getattr(r, "horse_number", None)):
            names[int(r.horse_number)] = getattr(r, "horse_name", "") or ""
            probs[int(r.horse_number)] = float(getattr(r, "p_top3", 0) or 0)

    if bet == "sanrenpuku":
        save_picks = plan.sanrenpuku_box4
        mv = "sanpuku_box4_lgbm" if source == "LightGBM" else "sanpuku_box4_pop"
    else:
        save_picks = plan.sanrentan_main + plan.sanrentan_nagashi[:3]
        mv = "lgbm_latest" if source == "LightGBM" else "popularity_v1"

    repo.insert_prediction(
        race_id=rid, picks=save_picks,
        rationale=bet_rationale(plan, names, probs),
        confidence=probs.get(plan.axis, 0.4),
        model_version=mv,
    )
    print(f"✓ {data.meta.race_name or ''} {len(data.results)}頭 / {source} / "
          f"軸{plan.axis}")
    print(f"   3連複BOX4: {' / '.join(plan.sanrenpuku_box4)}")
    print(f"   3連複軸流し6: {' / '.join(plan.sanrenpuku_axis)}")
    print(f"   3連単本線: {' / '.join(plan.sanrentan_main)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (省略時は今日)")
    parser.add_argument("--place", default="名古屋")
    parser.add_argument("--races", default="1-12", help="例: 1-12 や 8-12")
    parser.add_argument("--race-id", default=None, help="race_id直指定（中央単発レース用）")
    parser.add_argument("--bet", default="sanrenpuku", choices=["sanrentan", "sanrenpuku"],
                        help="DB保存する馬券種（既定:3連複 / 検証でROIプラス）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
    )

    date = _dt.date.fromisoformat(args.date) if args.date else _dt.date.today()

    # race_id直指定モード（中央G1など）
    if args.race_id:
        print(f"=== {args.race_id} 単発予測 ({args.place}) ===\n")
        ok = process_race(args.race_id, date, args.place, args.bet)
        print(f"\n完了: {'成功' if ok else '失敗'}")
        return

    lo, hi = (int(x) for x in args.races.split("-")) if "-" in args.races else (int(args.races), int(args.races))

    print(f"=== {date} {args.place} R{lo}-R{hi} 予測バッチ ===\n")
    ok, ng = 0, 0
    for rno in range(lo, hi + 1):
        rid = build_race_id(date, args.place, rno)
        print(f"[R{rno:>2}] {rid} ... ", end="", flush=True)
        if process_race(rid, date, args.place, args.bet):
            ok += 1
        else:
            ng += 1
        time.sleep(0.6)

    print(f"\n完了: 成功 {ok}R / 失敗 {ng}R")
    print("→ アプリ「📅 今日の予測」タブで日付を選んで確認してください。")


if __name__ == "__main__":
    main()
