# 公開デプロイ手順 (Streamlit Community Cloud)

「パスワード入れたら誰でも見れる」アプリ化のための手順書。

## ⚠️ 公開前チェック

- [ ] `.env` は git に含まれていない (.gitignore済み)
- [ ] `.streamlit/secrets.toml` は git に含まれていない (.gitignore済み)
- [ ] APIキーをコードに直接書いていない
- [ ] チャットbotは公開時オフ推奨 (`disable_chat = true`)

## 手順

### Step 1: ローカルでパスワードを設定

```bash
cd ~/Desktop/競馬予想AI
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
open -e .streamlit/secrets.toml
```

`app_password` を好きな文字列に変更、`disable_chat = true` のまま保存。

ローカル起動して動作確認:
```bash
bash scripts/run_app.sh
```
パスワードを入れないと中身が見えなければOK。

### Step 2: GitHubリポジトリ作成

```bash
cd ~/Desktop/競馬予想AI
git init
git add .
git status   # .env と secrets.toml が **含まれていないこと** を確認
git commit -m "Initial commit"
```

GitHub.com で **Private** リポジトリを作成 → ローカルを push:
```bash
git remote add origin https://github.com/<your-username>/keiba-yosou-ai.git
git branch -M main
git push -u origin main
```

### Step 3: Streamlit Community Cloud にデプロイ

1. https://share.streamlit.io にアクセス → GitHub でサインアップ
2. **"New app"** → リポジトリ・ブランチ・ファイルを指定:
   - Repository: `your-username/keiba-yosou-ai`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
3. **"Advanced settings" → Secrets** に `.streamlit/secrets.toml` の中身をコピペ
4. **Deploy** ボタン

完了するとURL（例: `https://keiba-yosou-ai.streamlit.app`）が発行されます。

### Step 4: 共有

そのURLをパスワードと一緒に見せたい人に渡せばOK。

## 既知の制約

- **DB(SQLite)はStreamlit Cloud上ではephemeral**: アプリ再起動でデータが消えます。
  - 解決策A: SQLiteファイルをgitにコミット（小さい、データ静的）
  - 解決策B: SupabaseやTurso等の外部DBに移行（要追加実装）
- **スクレイピングの可否**: netkeibaへの自動アクセスがStreamlit Cloud側で制限される可能性
  - 公開版は閲覧専用にし、データ更新はローカルで実施 → DBコミットしてpushするのが現実的
- **同時アクセス**: 無料プランは多人数同時アクセスに弱い

## DBスナップショットを公開版に含める方法

```bash
# .gitignore で除外されているDBを公開用に許可
echo '!data/db/keiba.sqlite' >> .gitignore   # 競合するので注意
git add -f data/db/keiba.sqlite
git commit -m "Include DB snapshot for public viewing"
git push
```

定期的にローカルで `python scripts/collect_period.py` を回し、git push で公開版にデータ反映。

## チャットbotを公開版でも有効にしたい場合

API課金リスクを承知の上で:
1. `.streamlit/secrets.toml` で `disable_chat = false`
2. `ANTHROPIC_API_KEY = "sk-ant-..."` を追加
3. Streamlit Cloud Settings → Secrets に同じ内容を貼る
4. **Anthropicコンソールで使用量上限（Usage Limits）を設定**しておく
