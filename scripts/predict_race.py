"""指定レースの3連単3点予測を実行し、Obsidianに書き出す。

使い方:
    python scripts/predict_race.py --race-id 202605020811
    python scripts/predict_race.py --yasuda-kinen-2026

流れ:
  1. 出馬表をスクレイピング
  2. 出走馬の特徴量を生成
  3. LightGBMで3着内確率予測
  4. Claudeに渡して3連単3点を組ませる
  5. Obsidianノートに出力 + DB(predictions)に記録
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db import repository as repo  # noqa: E402
from src.scraper import race_id as race_id_mod  # noqa: E402

log = logging.getLogger("predict")


def predict(race_id: str) -> None:
    from src.features.build import build_features_for_race
    from src.model.predict import predict_top3_probabilities, save_probabilities
    from src.reasoning.trifecta import predict_trifecta
    from src.output.obsidian_writer import write_prediction_note

    feats = build_features_for_race(race_id)
    probs = predict_top3_probabilities(race_id)
    probs_df = feats.merge(probs[["horse_id", "p_top3"]], on="horse_id", how="left")

    model_version = "lgbm_latest"
    save_probabilities(race_id, probs[["horse_id", "p_top3"]], model_version)

    race_meta = {
        "date": feats.attrs.get("date"),
        "race_name": feats.attrs.get("race_name"),
        "grade": feats.attrs.get("grade"),
        "course": feats.attrs.get("course"),
        "distance": feats.attrs.get("distance"),
        "surface": feats.attrs.get("surface"),
        "weather": feats.attrs.get("weather"),
        "track_cond": feats.attrs.get("track_cond"),
    }
    tri = predict_trifecta(race_meta, probs_df)
    note_path = write_prediction_note(
        race_meta=race_meta,
        horses=probs_df,
        trifecta=tri.__dict__,
        model_version=model_version,
    )
    repo.insert_prediction(
        race_id=race_id,
        trifecta_1=tri.trifecta_1,
        trifecta_2=tri.trifecta_2,
        trifecta_3=tri.trifecta_3,
        rationale=tri.rationale,
        confidence=tri.confidence,
        model_version=model_version,
    )
    log.info("prediction saved: %s", note_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-id")
    parser.add_argument("--yasuda-kinen-2026", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    rid = args.race_id
    if args.yasuda_kinen_2026:
        rid = race_id_mod.yasuda_kinen_2026_race_id()

    if not rid:
        parser.print_help()
        sys.exit(1)

    predict(rid)


if __name__ == "__main__":
    main()
