"""LightGBM 推論。

学習済みモデルを使って指定レースの各馬の3着内確率を返す。
"""
from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.features.build import build_features_for_race  # noqa: E402


def _model_path(custom: Optional[Path] = None) -> Path:
    if custom:
        return custom
    p = PROJECT_ROOT / "models" / "lgbm_latest.pkl"
    if not p.exists():
        candidates = sorted((PROJECT_ROOT / "models").glob("lgbm_*.pkl"))
        if not candidates:
            raise FileNotFoundError("学習済みモデルが見つからない。`python -m src.model.train` を実行")
        p = candidates[-1]
    return p


def load_model(custom: Optional[Path] = None) -> dict:
    p = _model_path(custom)
    with p.open("rb") as f:
        bundle = pickle.load(f)
    bundle["path"] = p
    return bundle


_CALIBRATOR_CACHE: dict = {}


def load_calibrator() -> Optional[object]:
    """較正器(Isotonic)があれば返す。無ければNone。"""
    if "iso" in _CALIBRATOR_CACHE:
        return _CALIBRATOR_CACHE["iso"]
    p = PROJECT_ROOT / "models" / "calibrator_latest.pkl"
    iso = None
    if p.exists():
        try:
            with p.open("rb") as f:
                iso = pickle.load(f).get("isotonic")
        except Exception:
            iso = None
    _CALIBRATOR_CACHE["iso"] = iso
    return iso


def predict_top3_probabilities(race_id: str, model: Optional[dict] = None) -> pd.DataFrame:
    """指定レースの各馬の3着内確率(p_top3)を返す。

    Returns: DataFrame with columns [horse_id, horse_number, horse_name, p_top3, ...feature_cols]
    """
    model = model or load_model()
    feats = build_features_for_race(race_id)
    if feats.empty:
        raise ValueError(f"出走馬が見つからない: {race_id}")

    X = feats[model["features"]].astype(float).fillna(feats[model["features"]].mean(numeric_only=True))
    # 平均でうまく埋まらない場合は0
    X = X.fillna(0)
    p = model["booster"].predict(X)
    feats = feats.copy()
    feats["p_top3_raw"] = p
    # 較正器があれば実測3着内率に補正（単調なので順位は不変、値が信頼できる）
    iso = load_calibrator()
    feats["p_top3"] = iso.predict(p) if iso is not None else p

    # 馬名を付与
    horse_ids = feats["horse_id"].tolist()
    if horse_ids:
        with sqlite3.connect(PROJECT_ROOT / "data" / "db" / "keiba.sqlite") as conn:
            placeholders = ",".join("?" * len(horse_ids))
            names = pd.read_sql_query(
                f"SELECT horse_id, name AS horse_name FROM horses WHERE horse_id IN ({placeholders})",
                conn, params=horse_ids,
            )
        feats = feats.merge(names, on="horse_id", how="left")
    return feats.sort_values("p_top3", ascending=False).reset_index(drop=True)


def trifecta_5_from_probs(df: pd.DataFrame) -> list[str]:
    """確率上位を使って3連単5点を構築（1着固定流し型）。

    1着 = 確率1位
    2-3着 = 確率2-5位の組み合わせから5点

    実証で 1着固定流しが好成績。
    """
    top = df.dropna(subset=["horse_number"]).head(5)
    if len(top) < 4:
        return []
    n = top["horse_number"].astype(int).tolist()
    if len(n) < 4:
        return []
    if len(n) < 5:
        # 4頭でフォールバック
        return [
            f"{n[0]}-{n[1]}-{n[2]}",
            f"{n[0]}-{n[2]}-{n[1]}",
            f"{n[0]}-{n[1]}-{n[3]}",
            f"{n[0]}-{n[3]}-{n[1]}",
            f"{n[0]}-{n[2]}-{n[3]}",
        ]
    return [
        f"{n[0]}-{n[1]}-{n[2]}",
        f"{n[0]}-{n[2]}-{n[1]}",
        f"{n[0]}-{n[1]}-{n[3]}",
        f"{n[0]}-{n[3]}-{n[1]}",
        f"{n[0]}-{n[2]}-{n[3]}",
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("race_id")
    args = parser.parse_args()
    df = predict_top3_probabilities(args.race_id)
    print(df[["horse_number", "horse_name", "p_top3"]].head(10).to_string(index=False))
    print()
    print("3連単5点:")
    for i, t in enumerate(trifecta_5_from_probs(df), 1):
        print(f"  {i}. {t}")
