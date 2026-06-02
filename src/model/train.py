"""LightGBM 学習スクリプト。

各 (race, horse) について「3着内に入る確率」を予測する二値分類モデル。
時系列split: 学習データはcutoff以前、評価は以降。
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from src.features.build import build_features, FEATURE_COLS  # noqa: E402


def train(
    cutoff_date: str = "2025-01-01",
    out_dir: Optional[Path] = None,
    params: Optional[dict] = None,
) -> Path:
    """cutoff_date を境に時系列split で LightGBM 学習し、モデルを保存。"""
    out_dir = out_dir or PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] building features ...")
    df = build_features()
    df = df.dropna(subset=["rank"])

    train_df = df[df["date"] < pd.Timestamp(cutoff_date)]
    valid_df = df[df["date"] >= pd.Timestamp(cutoff_date)]
    print(f"[train] train={len(train_df)}, valid={len(valid_df)}")

    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df["is_top3"].values
    X_valid = valid_df[FEATURE_COLS].astype(float)
    y_valid = valid_df["is_top3"].values

    params = params or {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    print(f"[train] training ...")
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    pred_valid = booster.predict(X_valid)
    ll = log_loss(y_valid, np.clip(pred_valid, 1e-7, 1-1e-7))
    auc = roc_auc_score(y_valid, pred_valid)
    print(f"[train] valid logloss={ll:.4f}, AUC={auc:.4f}")

    # feature importance
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    print("[train] feature importance (top 10):")
    print(imp.head(10).to_string(index=False))

    version = datetime.now().strftime("v_%Y%m%d_%H%M%S")
    model_path = out_dir / f"lgbm_{version}.pkl"
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "booster": booster,
                "features": FEATURE_COLS,
                "version": version,
                "cutoff_date": cutoff_date,
                "metrics": {"logloss": ll, "auc": auc},
            },
            f,
        )
    latest = out_dir / "lgbm_latest.pkl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(model_path.name)
    print(f"[train] saved: {model_path}")
    return model_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2025-01-01", help="学習データの上限日付")
    args = parser.parse_args()
    train(args.cutoff)
