"""Claude Code セッション (.jsonl) を keiba.sqlite の code_sessions に取り込む。

ソース: ~/.claude/projects/-Users-hasegawahayato-Documents-Obsidian-Vault/*.jsonl

毎日 launchd で実行すれば、会話履歴が DB から検索可能になる。

使い方:
    python scripts/ingest_sessions.py             # 全セッション取込
    python scripts/ingest_sessions.py --recent 7   # 直近7日分
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLAUDE_PROJECT_DIR = Path("/Users/hasegawahayato/.claude/projects/-Users-hasegawahayato-Documents-Obsidian-Vault")
DB_PATH = ROOT / "data" / "db" / "keiba.sqlite"


def parse_jsonl(jsonl_path: Path) -> dict:
    """jsonlファイルからセッション要約を抽出。"""
    session_id = jsonl_path.stem
    started, ended = None, None
    n_turns = 0
    n_tool_calls = 0
    user_messages: list[str] = []
    assistant_messages: list[str] = []
    files_modified: set[str] = set()
    keiba_related = False

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get("timestamp")
            if ts:
                if started is None: started = ts
                ended = ts
            mtype = obj.get("type", "")
            if mtype == "user":
                msg = obj.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_messages.append(content[:200])
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            user_messages.append(c.get("text", "")[:200])
                n_turns += 1
                if any(k in str(content) for k in ["競馬", "keiba", "race", "予測", "オッズ"]):
                    keiba_related = True
            elif mtype == "assistant":
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "tool_use":
                                n_tool_calls += 1
                                tool = c.get("name", "")
                                inp = c.get("input", {})
                                # ファイル編集の検出
                                if tool in ("Edit", "Write") and "file_path" in inp:
                                    files_modified.add(inp["file_path"])
                            elif c.get("type") == "text":
                                assistant_messages.append(c.get("text", "")[:200])

    topic = _detect_topic(user_messages + assistant_messages, keiba_related)
    return {
        "session_id": session_id,
        "started": started,
        "ended": ended,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "topic_summary": topic,
        "files_modified": json.dumps(sorted(files_modified)[:20], ensure_ascii=False),
        "jsonl_path": str(jsonl_path),
    }


def _detect_topic(messages: list[str], keiba: bool) -> str:
    """簡易トピック検出。"""
    text = " ".join(messages)[:5000]
    topics = []
    if keiba: topics.append("競馬予想AI")
    if "デプロイ" in text or "Streamlit Cloud" in text: topics.append("デプロイ")
    if "LightGBM" in text or "再学習" in text: topics.append("モデル学習")
    if "Claude推論" in text or "プロンプト" in text: topics.append("推論改善")
    if "playwright" in text.lower(): topics.append("スクレイピング拡張")
    if "Obsidian" in text or "Vault" in text: topics.append("Vault整備")
    if "Notion" in text: topics.append("Notion連携")
    if "デザイン" in text or "CSS" in text: topics.append("UI改善")
    return " / ".join(topics) if topics else "(未分類)"


def ingest(recent_days: Optional[int] = None) -> int:
    if not CLAUDE_PROJECT_DIR.exists():
        print(f"ERROR: {CLAUDE_PROJECT_DIR} が見つかりません")
        return 0
    files = sorted(CLAUDE_PROJECT_DIR.glob("*.jsonl"))
    print(f"jsonl ファイル: {len(files)}")
    if recent_days:
        cutoff = datetime.now() - timedelta(days=recent_days)
        files = [f for f in files if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff]
        print(f"直近{recent_days}日内: {len(files)}件")

    conn = sqlite3.connect(DB_PATH)
    saved = 0
    for f in files:
        try:
            data = parse_jsonl(f)
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            continue
        conn.execute("""
            INSERT INTO code_sessions
            (session_id, started, ended, n_turns, n_tool_calls,
             topic_summary, files_modified, jsonl_path)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                ended=excluded.ended,
                n_turns=excluded.n_turns,
                n_tool_calls=excluded.n_tool_calls,
                topic_summary=excluded.topic_summary,
                files_modified=excluded.files_modified
        """, (
            data["session_id"], data["started"], data["ended"],
            data["n_turns"], data["n_tool_calls"],
            data["topic_summary"], data["files_modified"], data["jsonl_path"],
        ))
        saved += 1
        print(f"  ✓ {f.name} turns={data['n_turns']} tools={data['n_tool_calls']} topic={data['topic_summary']}")
    conn.commit()
    conn.close()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent", type=int, default=None, help="直近N日内のみ")
    args = parser.parse_args()
    n = ingest(recent_days=args.recent)
    print(f"\n取込完了: {n}件")


if __name__ == "__main__":
    main()
