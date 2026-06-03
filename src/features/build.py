"""特徴量エンジニアリング: LightGBM学習データを作る。

各 (race, horse) サンプルについて、そのレース以前の情報のみから特徴量を計算する。
データリーク防止に注意。

主特徴量:
  事前情報（出馬表に書いてある）:
    popularity, odds, handicap, post_position, age (sex_age解析)
  履歴系（DB過去レースから集計、当該レース以前のみ）:
    recent_n_avg_rank, recent_n_top3_rate, recent_n_agari_avg
    days_since_last_race
    same_course_top3_rate, same_distance_top3_rate, same_surface_top3_rate
  騎手・調教師:
    jockey_top3_rate, jockey_n_races
    trainer_top3_rate
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "db" / "keiba.sqlite"


_SEX_AGE_RE = re.compile(r"([牡牝セ騙])\s*(\d+)")


def parse_age(s) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = _SEX_AGE_RE.search(s)
    return int(m.group(2)) if m else None


def parse_sex(s) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = _SEX_AGE_RE.search(s)
    return m.group(1) if m else None


def load_all_results() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT r.race_id, r.date, r.course, r.surface, r.distance,
                   r.track_cond, r.weather, r.grade, r.starters,
                   res.horse_id, res.jockey_id, res.trainer_id,
                   res.horse_number, res.post_position, res.rank,
                   res.sex_age, res.handicap, res.body_weight, res.body_weight_diff,
                   res.agari_3f, res.odds, res.popularity
            FROM results res JOIN races r USING(race_id)
            """,
            conn,
            parse_dates=["date"],
        )
    df["age"] = df["sex_age"].apply(parse_age)
    df["sex"] = df["sex_age"].apply(parse_sex)
    df["is_top3"] = (df["rank"].fillna(99) <= 3).astype(int)
    df = df.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)
    return df


def _rolling_features_for_horse(grp: pd.DataFrame) -> pd.DataFrame:
    """馬ごと(grp)の時系列で、各行の「自身を含まない過去」を集計。"""
    g = grp.sort_values("date")
    rank = g["rank"]
    agari = g["agari_3f"]
    top3 = (rank <= 3).astype(float)

    # 自分を含めないシフトしてから rolling/expanding
    shifted_rank = rank.shift(1)
    shifted_agari = agari.shift(1)
    shifted_top3 = top3.shift(1)

    out = pd.DataFrame(index=g.index)
    out["recent5_avg_rank"] = shifted_rank.rolling(5, min_periods=1).mean()
    out["recent5_top3_rate"] = shifted_top3.rolling(5, min_periods=1).mean()
    out["recent5_agari_avg"] = shifted_agari.rolling(5, min_periods=1).mean()
    out["career_n"] = shifted_rank.expanding().count()
    out["career_top3_rate"] = shifted_top3.expanding().mean()

    days_since = g["date"].diff().dt.days
    out["days_since_last_race"] = days_since
    return out


def _agg_top3_rate_by(df: pd.DataFrame, group_keys: list[str], suffix: str) -> pd.DataFrame:
    """指定キーで「自身を含まない過去」のtop3率を計算。

    expand集計+shiftの軽量版。
    """
    df = df.sort_values(["date"]).copy()
    top3 = (df["rank"] <= 3).astype(float)
    # キー単位での累積カウント
    grp = df.groupby(group_keys)
    cumcount = grp.cumcount()  # 0,1,2...
    cumsum_top3 = top3.groupby([df[k] for k in group_keys]).transform("cumsum")
    # 自身を除く
    prior_n = cumcount  # 自身までの累計から1引いた = 自身前のn件
    prior_top3 = cumsum_top3 - top3
    rate = prior_top3 / prior_n.replace(0, np.nan)
    out = pd.DataFrame({
        f"prior_n_{suffix}": prior_n,
        f"prior_top3_rate_{suffix}": rate,
    }, index=df.index)
    return out


def build_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """全レースの (race, horse) 行に特徴量を付与したフレームを返す。"""
    if df is None:
        df = load_all_results()

    # 馬ごとの履歴特徴量
    hist = (
        df.groupby("horse_id", group_keys=False)
        .apply(_rolling_features_for_horse)
        .sort_index()
    )
    df = df.join(hist)

    # 騎手の過去top3率
    df = df.join(_agg_top3_rate_by(df, ["jockey_id"], "jockey"))
    # 調教師の過去top3率
    df = df.join(_agg_top3_rate_by(df, ["trainer_id"], "trainer"))
    # 馬×コース
    df = df.join(_agg_top3_rate_by(df, ["horse_id", "course"], "course"))
    # 馬×距離
    df = df.join(_agg_top3_rate_by(df, ["horse_id", "distance"], "distance"))
    # 馬×馬場種別
    df = df.join(_agg_top3_rate_by(df, ["horse_id", "surface"], "surface"))

    # 🆕 thisweek系: 騎手の過去7日/3日の勝率・複勝率
    jw7 = _rolling_window_rates(df, "jockey_id", 7)
    df["jockey_last7days_n"] = jw7["_n_7"]
    df["jockey_last7days_winrate"] = jw7["_win_7"]
    df["jockey_last7days_top3_rate"] = jw7["_top3_7"]
    jw3 = _rolling_window_rates(df, "jockey_id", 3)
    df["jockey_last3days_n"] = jw3["_n_3"]
    df["jockey_last3days_winrate"] = jw3["_win_3"]
    # 調教師
    tw7 = _rolling_window_rates(df, "trainer_id", 7)
    df["trainer_last7days_n"] = tw7["_n_7"]
    df["trainer_last7days_winrate"] = tw7["_win_7"]
    df["trainer_last7days_top3_rate"] = tw7["_top3_7"]

    return df


FEATURE_COLS = [
    # 出馬表系
    "popularity", "odds", "handicap", "post_position", "age",
    "body_weight", "body_weight_diff",
    # 履歴
    "recent5_avg_rank", "recent5_top3_rate", "recent5_agari_avg",
    "career_n", "career_top3_rate", "days_since_last_race",
    # 騎手・調教師 (生涯)
    "prior_n_jockey", "prior_top3_rate_jockey",
    "prior_n_trainer", "prior_top3_rate_trainer",
    # 適性
    "prior_n_course", "prior_top3_rate_course",
    "prior_n_distance", "prior_top3_rate_distance",
    "prior_n_surface", "prior_top3_rate_surface",
    # 🆕 thisweek系（直近の勢い）
    "jockey_last7days_n", "jockey_last7days_winrate", "jockey_last7days_top3_rate",
    "jockey_last3days_n", "jockey_last3days_winrate",
    "trainer_last7days_n", "trainer_last7days_winrate", "trainer_last7days_top3_rate",
]


def _rolling_window_rates(df: pd.DataFrame, group_key: str, days: int) -> pd.DataFrame:
    """指定キーごとに「過去N日間の勝率・複勝率」を計算 (自身を含まない)。

    各 (date, group_key) について、その日の以前 days 日間の集計。
    """
    df = df.sort_values("date").copy()
    df = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    df["is_win"] = (df["rank"] == 1).astype(float)
    df["is_top3"] = (df["rank"] <= 3).astype(float)

    out = pd.DataFrame(index=df["_orig_idx"])
    out[f"_n_{days}"] = 0
    out[f"_win_{days}"] = 0.0
    out[f"_top3_{days}"] = 0.0

    # group_key単位で時系列にwindow集計
    for gid, g in df.groupby(group_key):
        if pd.isna(gid):
            continue
        g = g.sort_values("date").reset_index(drop=True)
        # 各行で「自身より前」かつ「days日以内」のレースを集計
        dates = g["date"].values
        wins = g["is_win"].values
        top3s = g["is_top3"].values
        n_arr = []
        w_arr = []
        t_arr = []
        for i, d in enumerate(dates):
            # 過去days日以内、自身を除く
            lo = d - pd.Timedelta(days=days)
            mask = (dates >= lo) & (dates < d)
            n = int(mask.sum())
            n_arr.append(n)
            if n > 0:
                w_arr.append(float(wins[mask].sum()) / n)
                t_arr.append(float(top3s[mask].sum()) / n)
            else:
                w_arr.append(0.0)
                t_arr.append(0.0)
        out.loc[g["_orig_idx"], f"_n_{days}"] = n_arr
        out.loc[g["_orig_idx"], f"_win_{days}"] = w_arr
        out.loc[g["_orig_idx"], f"_top3_{days}"] = t_arr
    return out


def build_training_frame(cutoff_date: Optional[str] = None) -> tuple[pd.DataFrame, pd.Series]:
    """指定cutoff以前のレースで学習用 (X, y) を作る。"""
    df = build_features()
    if cutoff_date:
        df = df[df["date"] < pd.Timestamp(cutoff_date)]
    df = df.dropna(subset=["rank"])  # 棄権等を除外
    X = df[FEATURE_COLS].astype(float)
    y = df["is_top3"]
    return X, y, df


def build_features_for_race(race_id: str) -> pd.DataFrame:
    """指定レースの出走馬について特徴量を生成（予測用）。"""
    df = build_features()
    race = df[df["race_id"] == race_id].copy()
    return race
