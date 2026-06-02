# 競馬予想AI

netkeiba・JRA等の公開データを自動収集し、**LightGBM + Claude Opus 4.7** のハイブリッドで3連単3点を提案する個人用予測システム。レース後の結果を取り込んで毎週リフレクション（PDCA）し、予測精度を継続的に育てていく。

- **設計書**: `Obsidian Vault/01_プロジェクト/競馬予想AI/00_設計.md`
- **初回ターゲット**: 2026-06-07 安田記念 (G1, 東京 芝1600m)

## ディレクトリ

```
競馬予想AI/
├── data/         # データ（スクレイピング生 + SQLite + 特徴量）
├── src/
│   ├── scraper/  # netkeiba/JRA 収集
│   ├── db/       # SQLite スキーマ・初期化
│   ├── features/ # 特徴量生成
│   ├── model/    # LightGBM 学習・推論・振り返り
│   ├── reasoning/# Claude 推論レイヤー（3連単組み立て）
│   └── output/   # Obsidianノート書き出し
├── models/       # 学習済みモデル
├── notebooks/    # EDA・検証
└── config.yaml
```

## セットアップ

```bash
cd ~/Desktop/競馬予想AI

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY を設定

python -m src.db.init_db
```

## 5日間ロードマップ (2026-06-02 → 06-07)

| 日 | タスク | 状態 |
|---|---|---|
| 06-02 火 | プロジェクト骨格・DBスキーマ | ✅ |
| 06-03 水 | netkeibaスクレイパー実装 / 過去5年G1収集 | ⏳ |
| 06-04 木 | 特徴量生成 / LightGBM学習 | ⏳ |
| 06-05 金 | Claude推論層 / 安田記念で初回予測 | ⏳ |
| 06-06 土 | 前日オッズ取り込み / 予測確定 | ⏳ |
| 06-07 日 | レース観戦 / 翌日に検証・PDCA | ⏳ |

## 使い方（完成時）

```bash
python -m src.db.init_db
python -m src.scraper.netkeiba 202605020811
python -m src.model.train
python -m src.model.predict --race-id <race_id>

python scripts/predict_race.py --race-id <race_id>

python -m src.model.reflect --since 2026-06-01
```

## 設計のポイント

1. **DBとノートの分離**: 構造化データはSQLite、人間が読む成果物はObsidian
2. **キャッシュ前提のスクレイピング**: パース調整中にnetkeibaへ何度もアクセスしない
3. **数値モデル + LLM のハイブリッド**: LightGBMで確率、Claudeで文脈・展開を加味
4. **3点で最低1点的中を狙う**: 本命型/連動型/穴狙いの3パターン構成
5. **PDCA前提の構造**: 予測ログ・結果・振り返りを全部DBとVaultに残す

## 注意

- スクレイピングは利用規約・rate_limit厳守。netkeibaに過剰負荷をかけない
- ギャンブルは余剰資金の範囲で。本AIは予測精度を育てる研究プロジェクト
- API利用料は1レースあたり数十円〜数百円の想定
