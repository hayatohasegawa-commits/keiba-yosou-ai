"""バックテスト評価スクリプト。

DBに保存済みのレース全件について、3つの予測戦略を試して命中率を比較する:

戦略:
  1. baseline_popularity: 1番人気→2番人気→3番人気の単純予想
  2. simple_oddsbased: オッズ反比例の上位3頭 (本命/連動/穴の3点)
  3. claude (オプション): Claude推論で3連単3点

評価指標:
  - top1_hit: 1着馬を予想1点目で当てた率
  - any3_top3: 3点のうち1点でも3連単的中した率
  - any3_top3_box: 3点のうち1点でも3連複（順不同）の意味で当たった率

使い方:
    python scripts/evaluate_predictions.py
    python scripts/evaluate_predictions.py --strategies baseline_popularity simple_oddsbased
    python scripts/evaluate_predictions.py --strategies claude --max-races 10
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402

from src.db import repository as repo  # noqa: E402


def fetch_race_entries(race_id: str) -> pd.DataFrame:
    with sqlite3.connect(repo.db_path()) as conn:
        df = pd.read_sql_query(
            """SELECT horse_id, horse_number, rank, odds, popularity,
                      (SELECT name FROM horses h WHERE h.horse_id=res.horse_id) AS horse_name,
                      (SELECT name FROM jockeys j WHERE j.jockey_id=res.jockey_id) AS jockey_name
               FROM results res WHERE race_id=?""",
            conn, params=(race_id,),
        )
    return df


def predict_baseline_popularity(entries: pd.DataFrame) -> list[str]:
    df = entries.dropna(subset=["popularity", "horse_number"]).sort_values("popularity")
    if len(df) < 3:
        return []
    n = df["horse_number"].astype(int).tolist()
    return [
        f"{n[0]}-{n[1]}-{n[2]}",
        f"{n[0]}-{n[2]}-{n[1]}",
        f"{n[1]}-{n[0]}-{n[2]}",
    ]


def predict_simple_oddsbased(entries: pd.DataFrame) -> list[str]:
    df = entries.copy()
    if df["odds"].notna().any():
        df = df.sort_values("odds")
    else:
        df = df.sort_values("popularity")
    df = df.dropna(subset=["horse_number"])
    if len(df) < 4:
        return predict_baseline_popularity(entries)
    n = df["horse_number"].astype(int).tolist()
    return [
        f"{n[0]}-{n[1]}-{n[2]}",
        f"{n[0]}-{n[2]}-{n[3]}",
        f"{n[1]}-{n[0]}-{n[4]}" if len(n) >= 5 else f"{n[1]}-{n[0]}-{n[3]}",
    ]


def predict_claude(entries: pd.DataFrame, race_meta: dict) -> list[str]:
    from src.reasoning.trifecta import predict_trifecta
    df = entries.copy()
    if "p_top3" not in df.columns:
        df["p_top3"] = 0.0
    pred = predict_trifecta(race_meta, df)
    return [pred.trifecta_1, pred.trifecta_2, pred.trifecta_3]


def actual_top3_trifecta(entries: pd.DataFrame) -> Optional[str]:
    top3 = entries.dropna(subset=["rank"]).sort_values("rank").head(3)
    if len(top3) < 3:
        return None
    nums = top3["horse_number"].astype(int).tolist()
    return f"{nums[0]}-{nums[1]}-{nums[2]}"


def actual_top3_set(entries: pd.DataFrame) -> Optional[set[int]]:
    top3 = entries.dropna(subset=["rank"]).sort_values("rank").head(3)
    if len(top3) < 3:
        return None
    return set(top3["horse_number"].astype(int).tolist())


def picks_to_set(pick: str) -> set[int]:
    return set(int(x) for x in pick.split("-"))


def evaluate(strategies: list[str], max_races: Optional[int] = None) -> pd.DataFrame:
    with sqlite3.connect(repo.db_path()) as conn:
        race_ids = [
            r[0] for r in conn.execute(
                "SELECT race_id FROM races ORDER BY date DESC"
            ).fetchall()
        ]
        race_meta_map = {r[0]: dict(zip(["race_name","grade","date","course","surface","distance","weather","track_cond"], r[1:]))
                         for r in conn.execute(
                            "SELECT race_id, race_name, grade, date, course, surface, distance, weather, track_cond FROM races"
                         ).fetchall()}

    if max_races:
        race_ids = race_ids[:max_races]

    rows = []
    for rid in race_ids:
        entries = fetch_race_entries(rid)
        actual_tri = actual_top3_trifecta(entries)
        actual_set = actual_top3_set(entries)
        if not actual_tri or not actual_set:
            continue

        for strat in strategies:
            try:
                if strat == "baseline_popularity":
                    picks = predict_baseline_popularity(entries)
                elif strat == "simple_oddsbased":
                    picks = predict_simple_oddsbased(entries)
                elif strat == "claude":
                    picks = predict_claude(entries, race_meta_map.get(rid, {}))
                else:
                    continue
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {rid} {strat}: {e}")
                continue

            if not picks:
                continue

            trifecta_hit = any(p == actual_tri for p in picks)
            top1_hit = any(p.startswith(actual_tri.split("-")[0] + "-") for p in picks)
            box_hit = any(picks_to_set(p) == actual_set for p in picks)

            rows.append({
                "race_id": rid,
                "race_name": race_meta_map.get(rid, {}).get("race_name", ""),
                "date": race_meta_map.get(rid, {}).get("date", ""),
                "strategy": strat,
                "pick_1": picks[0],
                "pick_2": picks[1] if len(picks) > 1 else "",
                "pick_3": picks[2] if len(picks) > 2 else "",
                "actual": actual_tri,
                "trifecta_hit": int(trifecta_hit),
                "top1_hit": int(top1_hit),
                "box_hit": int(box_hit),
            })

    df = pd.DataFrame(rows)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.groupby("strategy").agg(
        n_races=("race_id", "count"),
        trifecta_hit_rate=("trifecta_hit", "mean"),
        top1_hit_rate=("top1_hit", "mean"),
        box_hit_rate=("box_hit", "mean"),
    ).reset_index()
    agg[["trifecta_hit_rate", "top1_hit_rate", "box_hit_rate"]] = (
        agg[["trifecta_hit_rate", "top1_hit_rate", "box_hit_rate"]] * 100
    ).round(1)
    return agg


def save_results(df: pd.DataFrame, summary_df: pd.DataFrame) -> dict:
    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"detail_{ts}.csv"
    summary_path = out_dir / f"summary_{ts}.csv"
    df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    latest = out_dir / "latest_summary.csv"
    summary_df.to_csv(latest, index=False)
    return {"detail": str(detail_path), "summary": str(summary_path), "latest": str(latest)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", nargs="+",
        default=["baseline_popularity", "simple_oddsbased"],
        choices=["baseline_popularity", "simple_oddsbased", "claude"],
    )
    parser.add_argument("--max-races", type=int, default=None)
    args = parser.parse_args()

    if "claude" in args.strategies and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️ claude strategy needs ANTHROPIC_API_KEY in .env")
        args.strategies.remove("claude")

    df = evaluate(args.strategies, args.max_races)
    if df.empty:
        print("評価可能なレースがありませんでした (results が空 / rank 未取得 等)")
        return 1

    summary_df = summary(df)
    print("\n=== 評価サマリ ===")
    print(summary_df.to_string(index=False))

    paths = save_results(df, summary_df)
    print(f"\n保存: {json.dumps(paths, indent=2, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
