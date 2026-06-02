"""DB保存/取得のレポジトリ層。

スクレイパー → DB の橋渡しはここに集約する。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def db_path() -> Path:
    return PROJECT_ROOT / _load_config()["paths"]["db_path"]


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def upsert_race(conn: sqlite3.Connection, meta) -> None:
    conn.execute(
        """
        INSERT INTO races (race_id, date, course, race_number, race_name, grade,
                           distance, surface, direction, weather, track_cond, starters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id) DO UPDATE SET
            date=excluded.date, course=excluded.course, race_number=excluded.race_number,
            race_name=excluded.race_name, grade=excluded.grade, distance=excluded.distance,
            surface=excluded.surface, direction=excluded.direction,
            weather=excluded.weather, track_cond=excluded.track_cond,
            starters=excluded.starters
        """,
        (
            meta.race_id, meta.date, meta.course, getattr(meta, "race_number", None),
            meta.race_name, meta.grade, meta.distance, meta.surface,
            getattr(meta, "direction", None), meta.weather, meta.track_cond, meta.starters,
        ),
    )


def upsert_horse(conn: sqlite3.Connection, horse_id: str, name: str,
                 sex: Optional[str] = None, birth_year: Optional[int] = None,
                 sire: Optional[str] = None, dam: Optional[str] = None,
                 trainer: Optional[str] = None, owner: Optional[str] = None) -> None:
    conn.execute(
        """
        INSERT INTO horses (horse_id, name, sex, birth_year, sire, dam, trainer, owner)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(horse_id) DO UPDATE SET
            name=excluded.name,
            sex=COALESCE(excluded.sex, horses.sex),
            birth_year=COALESCE(excluded.birth_year, horses.birth_year),
            sire=COALESCE(excluded.sire, horses.sire),
            dam=COALESCE(excluded.dam, horses.dam),
            trainer=COALESCE(excluded.trainer, horses.trainer),
            owner=COALESCE(excluded.owner, horses.owner)
        """,
        (horse_id, name, sex, birth_year, sire, dam, trainer, owner),
    )


def upsert_jockey(conn: sqlite3.Connection, jockey_id: str, name: str) -> None:
    conn.execute(
        """
        INSERT INTO jockeys (jockey_id, name)
        VALUES (?, ?)
        ON CONFLICT(jockey_id) DO UPDATE SET name=excluded.name
        """,
        (jockey_id, name),
    )


def upsert_result(conn: sqlite3.Connection, row) -> None:
    conn.execute(
        """
        INSERT INTO results (
            race_id, horse_id, jockey_id, rank, post_position, horse_number,
            sex_age, handicap, body_weight, body_weight_diff, time, margin,
            agari_3f, odds, popularity, trainer_id, trainer_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id, horse_id) DO UPDATE SET
            jockey_id=excluded.jockey_id, rank=excluded.rank,
            post_position=excluded.post_position, horse_number=excluded.horse_number,
            sex_age=excluded.sex_age, handicap=excluded.handicap,
            body_weight=excluded.body_weight, body_weight_diff=excluded.body_weight_diff,
            time=excluded.time, margin=excluded.margin, agari_3f=excluded.agari_3f,
            odds=excluded.odds, popularity=excluded.popularity,
            trainer_id=excluded.trainer_id, trainer_name=excluded.trainer_name
        """,
        (
            row.race_id, row.horse_id, row.jockey_id, row.rank, row.post_position,
            row.horse_number, getattr(row, "sex_age", None), row.handicap,
            row.body_weight, row.body_weight_diff, row.time, row.margin,
            row.agari_3f, row.odds, row.popularity,
            getattr(row, "trainer_id", None), getattr(row, "trainer_name", None),
        ),
    )


def save_race_data(race_data) -> None:
    """RaceData (meta + results) を一括保存。"""
    with connect() as conn:
        upsert_race(conn, race_data.meta)
        for row in race_data.results:
            if row.horse_id:
                upsert_horse(conn, row.horse_id, row.horse_name)
            if row.jockey_id and row.jockey_name:
                upsert_jockey(conn, row.jockey_id, row.jockey_name)
            upsert_result(conn, row)
        conn.commit()


def list_race_ids_by_grade(grade: str) -> list[str]:
    with connect() as conn:
        cur = conn.execute("SELECT race_id FROM races WHERE grade=? ORDER BY date", (grade,))
        return [r[0] for r in cur.fetchall()]


def race_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]


def insert_prediction(
    race_id: str,
    trifecta_1: str,
    trifecta_2: str,
    trifecta_3: str,
    rationale: str,
    confidence: float,
    model_version: str,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                race_id, trifecta_1, trifecta_2, trifecta_3,
                rationale, confidence, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (race_id, trifecta_1, trifecta_2, trifecta_3, rationale, confidence, model_version),
        )
        conn.commit()
        return cur.lastrowid


def mark_prediction_result(prediction_id: int, hit: int, payout: Optional[int] = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE predictions SET hit=?, payout=? WHERE id=?",
            (hit, payout, prediction_id),
        )
        conn.commit()


def save_chat_turn(
    session_id: str,
    turn_index: int,
    role: str,
    content: str,
    tab_context: Optional[str] = None,
    race_id: Optional[str] = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_conversations
                (session_id, turn_index, role, content, tab_context, race_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_index, role, content, tab_context, race_id),
        )
        conn.commit()
        return cur.lastrowid


def get_chat_history(session_id: str, limit: int = 200) -> list[tuple]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT role, content, created_at FROM chat_conversations "
            "WHERE session_id=? ORDER BY turn_index ASC LIMIT ?",
            (session_id, limit),
        )
        return cur.fetchall()


def list_chat_sessions(limit: int = 50) -> list[tuple]:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT session_id,
                   MIN(created_at) AS started,
                   MAX(created_at) AS ended,
                   COUNT(*) AS turns
            FROM chat_conversations
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def save_chat_reflection(
    period_from: str,
    period_to: str,
    insights: str,
    prompt_diff: Optional[str] = None,
    derived_n: int = 0,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_reflections
                (period_from, period_to, insights, prompt_diff, derived_n)
            VALUES (?, ?, ?, ?, ?)
            """,
            (period_from, period_to, insights, prompt_diff, derived_n),
        )
        conn.commit()
        return cur.lastrowid
