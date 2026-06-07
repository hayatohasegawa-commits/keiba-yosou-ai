"""LightGBMの生スコアを実測3着内率に較正する(Isotonic回帰)。

cutoff以降(=学習に使っていないホールドアウト)で
raw_score → is_top3 の単調写像を学習し保存する。リークなし。

使い方: python scripts/calibrate_model.py
出力: models/calibrator_latest.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features.build import build_features, FEATURE_COLS
from src.model.predict import load_model


def main() -> None:
    bundle = load_model()
    booster = bundle["booster"]
    cutoff = bundle.get("cutoff_date", "2025-01-01")
    print(f"[calib] model={bundle.get('version')} cutoff={cutoff}")

    df = build_features().dropna(subset=["rank"])
    valid = df[df["date"] >= pd.Timestamp(cutoff)]
    print(f"[calib] holdout rows = {len(valid)}")

    X = valid[bundle["features"]].astype(float)
    X = X.fillna(X.mean(numeric_only=True)).fillna(0)
    raw = booster.predict(X)
    y = valid["is_top3"].values

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y)
    cal = iso.predict(raw)

    # 較正前後の指標
    def ece(p, y, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1])
            if m.sum():
                e += abs(p[m].mean() - y[m].mean()) * m.sum()
        return e / len(p)

    print(f"[calib] raw : mean_pred={raw.mean():.3f} actual={y.mean():.3f} ECE={ece(raw,y):.4f}")
    print(f"[calib] cal : mean_pred={cal.mean():.3f} actual={y.mean():.3f} ECE={ece(cal,y):.4f}")
    # 代表点の写像
    for s in [0.3, 0.5, 0.7, 0.85, 0.9]:
        print(f"   raw {s:.2f} -> cal {float(iso.predict([s])[0]):.3f}")

    out = ROOT / "models" / "calibrator_latest.pkl"
    with out.open("wb") as f:
        pickle.dump({"isotonic": iso, "model_version": bundle.get("version"),
                     "cutoff": cutoff, "n": len(valid)}, f)
    print(f"[calib] saved: {out}")


if __name__ == "__main__":
    main()
