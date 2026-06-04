#!/bin/bash
# 毎日のDB自動同期 (daily_update.py 後に launchd で実行)
# 1. データ収集
# 2. Claude Codeセッション取込
# 3. git add → commit → push (変更があれば)

set -u  # -e は外す: pushエラーで他の処理続行

PROJECT_DIR="/Users/hasegawahayato/Desktop/競馬予想AI"
LOG_FILE="$PROJECT_DIR/logs/daily_sync_$(date +%Y-%m-%d).log"

cd "$PROJECT_DIR" || exit 1

{
    echo "=== $(date) daily_sync 開始 ==="

    # PATH 整備 (launchd 経由でも git/python が見つかるよう)
    export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

    # 1. データ収集
    echo "--- daily_update.py ---"
    "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/daily_update.py" 2>&1

    # 2. Claude Codeセッション取込
    echo "--- ingest_sessions.py ---"
    "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/ingest_sessions.py" --recent 7 2>&1

    # 3. git add 対象を限定
    echo "--- git status ---"
    git add data/db/keiba.sqlite 2>&1 || true
    git add predictions/ 2>&1 || true
    git add data/raw/enriched_*.json 2>&1 || true

    if git diff --staged --quiet; then
        echo "変更なし、pushスキップ"
    else
        msg="Auto-sync $(date '+%Y-%m-%d %H:%M')"
        echo "--- commit ---"
        git commit -m "$msg" 2>&1
        echo "--- push ---"
        git push 2>&1
    fi

    echo "=== $(date) daily_sync 完了 ==="
} >> "$LOG_FILE" 2>&1
