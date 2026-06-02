"""Obsidian Vault に予測結果ノートを出力する。"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def vault_project_dir() -> Path:
    vault = os.environ.get(
        "OBSIDIAN_VAULT_PATH",
        "/Users/hasegawahayato/Documents/Obsidian Vault",
    )
    sub = os.environ.get("OBSIDIAN_PROJECT_DIR", "01_プロジェクト/競馬予想AI")
    return Path(vault) / sub


def write_prediction_note(
    race_meta: dict,
    horses: pd.DataFrame,
    trifecta: dict,
    model_version: str,
) -> Path:
    """予測結果ノートを Obsidian に出力。

    Args:
        race_meta: {date, race_name, grade, course, distance, surface, ...}
        horses: 出走馬一覧 + p_top3 を含むDataFrame
        trifecta: {trifecta_1, trifecta_2, trifecta_3, rationale, confidence}
        model_version: 学習モデルのバージョン文字列
    """
    out_dir = vault_project_dir() / "予測結果"
    out_dir.mkdir(parents=True, exist_ok=True)

    date = race_meta.get("date", datetime.now().strftime("%Y-%m-%d"))
    race_name = race_meta.get("race_name", "レース")
    filename = f"{date}_{race_name}.md"
    out_path = out_dir / filename

    horses_md = horses.to_markdown(index=False) if not horses.empty else "(出走馬データなし)"

    content = f"""---
tags: [競馬予想AI, 予測結果, {race_meta.get('grade', '')}]
race_date: {date}
race_name: {race_name}
model_version: {model_version}
created: {datetime.now().isoformat(timespec='seconds')}
---

# {date} {race_name} 予測

## レース情報
- **グレード**: {race_meta.get('grade', '')}
- **コース**: {race_meta.get('course', '')} {race_meta.get('surface', '')}{race_meta.get('distance', '')}m
- **馬場**: {race_meta.get('track_cond', '不明')}　**天候**: {race_meta.get('weather', '不明')}

## 3連単 3点予想

| 種別 | 買い目 |
|---|---|
| 本命型 | **{trifecta.get('trifecta_1', '')}** |
| 連動型 | **{trifecta.get('trifecta_2', '')}** |
| 穴狙い | **{trifecta.get('trifecta_3', '')}** |

**自信度**: {trifecta.get('confidence', 0):.2f}

## 根拠

{trifecta.get('rationale', '')}

## 出走馬と3着内確率

{horses_md}

## メタ
- モデルバージョン: `{model_version}`
- 予測作成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
[[00_設計]] | [[週次振り返り]]
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_verification_note(
    race_meta: dict,
    trifecta: dict,
    actual_top3: list[str],
    hit: int,
    payout: Optional[int] = None,
) -> Path:
    out_dir = vault_project_dir() / "結果検証"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = race_meta.get("date", datetime.now().strftime("%Y-%m-%d"))
    race_name = race_meta.get("race_name", "レース")
    filename = f"{date}_{race_name}_検証.md"
    out_path = out_dir / filename

    hit_label = "**的中** ✅" if hit else "**不的中** ❌"
    content = f"""---
tags: [競馬予想AI, 結果検証, {race_meta.get('grade', '')}]
race_date: {date}
race_name: {race_name}
hit: {bool(hit)}
created: {datetime.now().isoformat(timespec='seconds')}
---

# {date} {race_name} 検証

## 結果
- 着順 (1-2-3着): **{' - '.join(actual_top3)}**
- 判定: {hit_label}
- 払戻: {payout if payout is not None else '未入力'}円

## 予測との照合
| 種別 | 買い目 |
|---|---|
| 本命型 | {trifecta.get('trifecta_1', '')} |
| 連動型 | {trifecta.get('trifecta_2', '')} |
| 穴狙い | {trifecta.get('trifecta_3', '')} |

## 振り返りメモ
- なぜ当たった/外したか:
- 次回への改善点:

---
[[00_設計]] | [[週次振り返り]] | [[{date}_{race_name}]]
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path
