"""SQLite DB初期化スクリプト。

使い方:
    python -m src.db.init_db
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    print(f"[init_db] schema applied: {db_path}")


def main() -> None:
    cfg = load_config()
    db_path = PROJECT_ROOT / cfg["paths"]["db_path"]
    init_db(db_path)


if __name__ == "__main__":
    main()
