"""チャット会話のリフレクション。

直近N日間の会話を読み込み、Claudeに以下を分析させる:
  - ユーザーの関心傾向
  - チャットbotで答えきれていない領域
  - システムプロンプト改善案（prompt_diff）
  - 新機能の示唆

結果は chat_reflections テーブルに保存。

使い方:
    python scripts/reflect_chat.py --days 7
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from src.db import repository as repo  # noqa: E402

REFLECT_SYSTEM = """あなたは会話分析の専門家。
ユーザーと競馬予想チャットbotの会話ログを読み、以下4点を箇条書きで簡潔に出力してください:

1. **ユーザーの関心傾向**（よく聞かれる話題・レース・馬・データ種別）
2. **不足している知識領域**（答えきれていない・はぐらかしているもの）
3. **システムプロンプト改善案**（より良い回答のための具体的な追記/修正）
4. **新機能の示唆**（アプリに足すと喜ばれそうな機能）

合計1000字以内。マークダウン可。
"""


def fetch_recent_chats(days: int) -> tuple[str, str, list[tuple]]:
    import sqlite3
    period_to = datetime.now().isoformat(timespec="seconds")
    period_from = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with sqlite3.connect(repo.db_path()) as conn:
        cur = conn.execute(
            """SELECT session_id, turn_index, role, content, created_at
               FROM chat_conversations
               WHERE created_at >= ?
               ORDER BY session_id, turn_index""",
            (period_from,),
        )
        rows = cur.fetchall()
    return period_from, period_to, rows


def format_chats_for_llm(rows: list[tuple]) -> str:
    if not rows:
        return ""
    chunks = []
    current_sess = None
    for sess_id, turn, role, content, ts in rows:
        if sess_id != current_sess:
            chunks.append(f"\n--- session: {sess_id} ({ts}) ---")
            current_sess = sess_id
        chunks.append(f"[{role}] {content}")
    return "\n".join(chunks)


def reflect(days: int) -> None:
    period_from, period_to, rows = fetch_recent_chats(days)
    if not rows:
        print(f"会話なし (since {period_from})")
        return

    convo_text = format_chats_for_llm(rows)
    print(f"対象: {len(rows)} turns, {len({r[0] for r in rows})} sessions")

    from anthropic import Anthropic
    client = Anthropic()
    msg = client.messages.create(
        model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"),
        max_tokens=2000,
        system=REFLECT_SYSTEM,
        messages=[{"role": "user", "content": convo_text[:80000]}],
    )
    insights = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    print("--- INSIGHTS ---")
    print(insights)

    reflection_id = repo.save_chat_reflection(
        period_from=period_from,
        period_to=period_to,
        insights=insights,
        derived_n=len(rows),
    )
    print(f"\n保存しました: chat_reflections.id={reflection_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="直近何日分の会話を分析")
    args = parser.parse_args()
    reflect(args.days)


if __name__ == "__main__":
    main()
