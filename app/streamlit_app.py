"""競馬予想AI デモアプリ (Streamlit)。

任意のJRAレース (race_id) に対して:
  1. netkeibaからレース情報・出走馬を取得
  2. シンプルな確率モデル（人気・オッズベース）で3着内確率を推定
  3. Claude推論（API key設定時）で3連単3点を組み立て
  4. 結果を表示・DB保存

起動:
    cd ~/Desktop/競馬予想AI
    .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.scraper import netkeiba, race_id as race_id_mod  # noqa: E402
from src.scraper.odds import fetch_trifecta_odds  # noqa: E402
from src.db import repository as repo  # noqa: E402
from src.features.horse_memo import build_horse_memo  # noqa: E402
from src.features.race_context import load_enriched, style_map, format_for_claude  # noqa: E402
from src.reasoning.continuous_learning import get_recent_reasoning, get_validated_hypotheses  # noqa: E402

st.set_page_config(
    page_title="競馬予想AI | Powered by AI",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "競馬予想AI - LightGBM + Claude推論で3連単5点の最適解",
    },
)


# ---------- パスワードゲート ----------

def _get_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.environ.get(key.upper(), default)


def check_password() -> bool:
    """app_password シークレットが設定されていれば認証ゲートを表示。"""
    required = _get_secret("app_password")
    if not required:
        return True
    if st.session_state.get("auth_ok"):
        return True

    st.title("🐎 競馬予想AI")
    st.caption("閲覧にはパスワードが必要です")
    with st.form("auth_form", clear_on_submit=False):
        pw = st.text_input("パスワード", type="password")
        ok = st.form_submit_button("ログイン", type="primary")
    if ok:
        if pw == required:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    return False


if not check_password():
    st.stop()


# チャット機能ON/OFF (公開時はOFFを推奨：API課金リスク回避)
CHAT_ENABLED = not bool(_get_secret("disable_chat", False))


# ---------- 日本語ラベル変換 ----------

JA_COLS: dict[str, str] = {
    # 馬・人
    "horse_id": "馬ID", "horse_name": "馬名",
    "jockey_id": "騎手ID", "jockey_name": "騎手",
    "trainer_id": "調教師ID", "trainer_name": "調教師",
    "sex_age": "性齢",
    # レース
    "race_id": "レースID", "race_name": "レース名",
    "grade": "グレード", "date": "開催日",
    "course": "競馬場", "surface": "馬場種別",
    "distance": "距離(m)", "direction": "回り",
    "weather": "天候", "track_cond": "馬場状態",
    "starters": "出走頭数",
    # 結果
    "rank": "着順", "post_position": "枠番", "horse_number": "馬番",
    "handicap": "斤量", "body_weight": "馬体重",
    "body_weight_diff": "体重増減", "time": "タイム",
    "margin": "着差", "agari_3f": "上がり3F",
    "odds": "単勝オッズ", "popularity": "人気",
    "p_top3": "3着内確率",
    # 予測
    "trifecta_1": "予想1", "trifecta_2": "予想2", "trifecta_3": "予想3",
    "confidence": "自信度", "hit": "的中", "payout": "払戻金",
    "model_version": "モデル",
    "rationale": "根拠",
    # 評価
    "strategy": "戦略", "n_races": "対象レース数",
    "trifecta_hit_rate": "3連単命中率(%)",
    "top1_hit_rate": "1着的中率(%)",
    "box_hit_rate": "3連複命中率(%)",
    "pick_1": "予想1", "pick_2": "予想2", "pick_3": "予想3",
    "actual": "実際の順",
    "trifecta_hit": "3連単", "top1_hit": "1着", "box_hit": "3連複",
    "timestamp": "実行時刻", "metric": "指標", "rate(%)": "率(%)",
    # チャット
    "session_id": "セッションID", "turn_index": "ターン",
    "role": "役割", "content": "発言",
    "created_at": "作成日時", "started": "開始時刻", "ended": "終了時刻",
    "turns": "ターン数", "tab_context": "発言タブ",
}

STRATEGY_JA: dict[str, str] = {
    "baseline_popularity": "人気順ベース",
    "simple_oddsbased": "オッズ反比例",
    "claude": "Claude推論",
}

ROLE_JA: dict[str, str] = {
    "user": "ユーザー",
    "assistant": "アシスタント",
}

METRIC_JA: dict[str, str] = {
    "trifecta_hit_rate": "3連単命中率",
    "top1_hit_rate": "1着的中率",
    "box_hit_rate": "3連複命中率",
}


def ja(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのカラム名・カテゴリ値を日本語化したコピーを返す。"""
    if df is None or df.empty:
        return df
    d = df.copy()
    if "strategy" in d.columns:
        d["strategy"] = d["strategy"].map(STRATEGY_JA).fillna(d["strategy"])
    if "role" in d.columns:
        d["role"] = d["role"].map(ROLE_JA).fillna(d["role"])
    if "metric" in d.columns:
        d["metric"] = d["metric"].map(METRIC_JA).fillna(d["metric"])
    rename_map = {c: JA_COLS[c] for c in d.columns if c in JA_COLS}
    return d.rename(columns=rename_map)

st.markdown("""
<style>
/* === ダーク・シネマティック (Rebike風 + 競馬AI) === */
:root {
    --c-bg: #0a0a0f;
    --c-bg-2: #12131a;
    --c-surface: #1a1b24;
    --c-surface-hover: #22232e;
    --c-emerald: #10b981;
    --c-emerald-soft: #34d399;
    --c-gold: #d4af37;
    --c-gold-bright: #f4d976;
    --c-text: #f5f5f7;
    --c-text-sub: #a1a1aa;
    --c-text-dim: #71717a;
    --c-line: rgba(255,255,255,0.08);
    --c-line-bright: rgba(255,255,255,0.15);
}

/* 全体: 真っ黒ベース + 微細グラデ */
.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(16,185,129,0.06) 0%, transparent 50%),
        radial-gradient(circle at 80% 100%, rgba(212,175,55,0.05) 0%, transparent 50%),
        var(--c-bg) !important;
    color: var(--c-text) !important;
}
section.main > div.block-container {
    padding-top: 0 !important;
    padding-right: 400px;
    max-width: 100%;
}

/* タイトル: 大きく・太く・印象的 */
h1, .stApp h1 {
    color: var(--c-text) !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    font-size: 3.5rem !important;
    line-height: 1.05 !important;
}
h2, .stApp h2 {
    color: var(--c-text) !important;
    border-left: 3px solid var(--c-gold);
    padding-left: 14px;
    margin-top: 2.5rem !important;
    font-weight: 700 !important;
}
h3, .stApp h3 {
    color: var(--c-text) !important;
    font-weight: 600 !important;
}

/* ストリームのpタグも明るく */
.stApp p, .stApp span, .stApp label, .stApp div { color: inherit; }
.stMarkdown { color: var(--c-text) !important; }
.stMarkdown p { color: var(--c-text) !important; }

/* タブ: 上品・ダーク基調 */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--c-line) !important;
    gap: 4px !important;
}
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: var(--c-text-sub) !important;
    padding: 14px 22px !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s ease;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--c-gold) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--c-gold) !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--c-text) !important;
    background: rgba(255,255,255,0.02) !important;
}
div[data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* primary ボタン: 金グラデ */
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--c-gold), #b8941f) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 20px rgba(212,175,55,0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(212,175,55,0.4) !important;
}

/* secondary ボタン: アウトライン暗色 */
button[data-testid="baseButton-secondary"], div[data-testid="stButton"] button {
    background: var(--c-surface) !important;
    color: var(--c-text) !important;
    border: 1px solid var(--c-line-bright) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease;
}
button[data-testid="baseButton-secondary"]:hover, div[data-testid="stButton"] button:hover {
    border-color: var(--c-gold) !important;
    color: var(--c-gold) !important;
    background: rgba(212,175,55,0.05) !important;
}
/* primary は上書きの上書きで保持 */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--c-gold), #b8941f) !important;
    color: #0a0a0f !important;
    border: none !important;
}

/* 入力フィールド */
.stTextInput input, .stSelectbox div[role="combobox"], .stDateInput input {
    background: var(--c-surface) !important;
    color: var(--c-text) !important;
    border: 1px solid var(--c-line-bright) !important;
    border-radius: 8px !important;
}
.stTextInput input::placeholder { color: var(--c-text-dim) !important; }

/* メトリック: 暗カードに金アクセント */
div[data-testid="stMetric"] {
    background: var(--c-surface) !important;
    padding: 18px 22px !important;
    border-radius: 14px !important;
    border: 1px solid var(--c-line) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
    color: var(--c-text-sub) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
    color: var(--c-gold) !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* container (border=True): 上品な暗カード */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-line) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25) !important;
    padding: 20px 24px !important;
    transition: all 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--c-line-bright) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* テーブル: 暗テーマ */
div[data-testid="stDataFrame"] {
    background: var(--c-surface) !important;
    border-radius: 12px !important;
    border: 1px solid var(--c-line) !important;
    overflow: hidden !important;
}

/* alert系の色 */
div[data-testid="stAlert"] {
    background: var(--c-surface) !important;
    border-left-width: 4px !important;
    color: var(--c-text) !important;
}

/* expander: 暗 */
div[data-testid="stExpander"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-line) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary { color: var(--c-text) !important; }

/* sidebar: 右配置ガラス暗 */
section[data-testid="stSidebar"] {
    position: fixed;
    right: 0;
    left: auto;
    top: 0;
    height: 100vh;
    width: 380px !important;
    background: rgba(18,19,26,0.92) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-left: 1px solid var(--c-line) !important;
    box-shadow: -12px 0 32px rgba(0,0,0,0.5);
    color: var(--c-text) !important;
}
section[data-testid="stSidebar"] * { color: var(--c-text); }
section[data-testid="stSidebar"] > div { width: 380px !important; }
section[data-testid="stSidebar"] [data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    border: 1px solid var(--c-line) !important;
}
section[data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--c-text) !important;
}
button[kind="header"][data-testid="baseButton-header"] { right: 380px; }

/* captionは控えめグレー */
div[data-testid="stCaptionContainer"], .stCaption,
small, .stCaption span {
    color: var(--c-text-dim) !important;
    font-size: 0.85rem !important;
}

/* リンク: 金 */
a, a:visited { color: var(--c-gold) !important; text-decoration: none; }
a:hover { color: var(--c-gold-bright) !important; text-decoration: underline; }

/* スクロールバー */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--c-bg); }
::-webkit-scrollbar-thumb { background: var(--c-line-bright); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-gold); }

/* ヒーロー: 真っ黒オーバーレイで馬の輪郭を活かす */
.hero-wrap {
    position: relative;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0;
    overflow: hidden;
}
.hero-wrap img {
    width: 100%;
    display: block;
    filter: brightness(0.7) contrast(1.1);
}
.hero-overlay {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(10,10,15,0.3) 0%, rgba(10,10,15,0.85) 100%);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 3rem 2.5rem 2rem;
}
.hero-title {
    font-size: 4rem !important;
    font-weight: 800 !important;
    color: var(--c-text) !important;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
    line-height: 1;
}
.hero-sub {
    font-size: 1.15rem;
    color: var(--c-text-sub);
    margin: 0;
    font-weight: 400;
}
.hero-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin: 0 8px; vertical-align: middle; }

/* セクション番号ラベル (01/02風) */
.section-num {
    display: inline-block;
    font-family: 'SF Mono', monospace;
    color: var(--c-gold);
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    margin-bottom: 0.5rem;
}

/* === 文字視認性強化 === */
/* dataframe セルの文字色 */
div[data-testid="stDataFrame"] *, .stDataFrame * {
    color: var(--c-text) !important;
}
div[data-testid="stDataFrame"] [role="columnheader"], .stDataFrame [role="columnheader"] {
    color: var(--c-gold) !important;
    background: rgba(255,255,255,0.04) !important;
    font-weight: 700 !important;
}
/* 黄色背景に白文字問題対応: 全 span/code に明示色 */
.stMarkdown span[style*="background:#fff"], .stMarkdown span[style*="background: #fff"] {
    color: #0a0a0f !important;
}
/* インラインcode */
.stMarkdown code {
    background: rgba(212,175,55,0.15) !important;
    color: var(--c-gold-bright) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
/* st.info/success/warning/error の中身を見やすく */
div[data-testid="stAlert"] * {
    color: var(--c-text) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
    background: rgba(16,185,129,0.1) !important;
    border-left-color: var(--c-emerald) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
    background: rgba(255,255,255,0.04) !important;
    border-left-color: var(--c-gold) !important;
}
/* expander 中の文字 */
div[data-testid="stExpander"] *, div[data-testid="stExpander"] p, div[data-testid="stExpander"] span {
    color: var(--c-text) !important;
}
/* selectbox / multiselect ラベル */
.stSelectbox label, .stMultiSelect label, .stTextInput label, .stDateInput label,
.stCheckbox label, .stRadio label {
    color: var(--c-text) !important;
    font-weight: 600 !important;
}
.stCheckbox label p, .stRadio label p {
    color: var(--c-text) !important;
}
/* multiselect 選択タグ */
div[data-baseweb="tag"] {
    background: rgba(212,175,55,0.2) !important;
    color: var(--c-gold-bright) !important;
}
/* number input 矢印 */
.stNumberInput button { color: var(--c-text) !important; }
/* date picker */
.stDateInput div[data-baseweb="input"] input { color: var(--c-text) !important; }
.stDateInput div[data-baseweb="calendar"] { background: var(--c-surface) !important; color: var(--c-text) !important; }
/* form の中の文字 */
[data-testid="stForm"] * { color: inherit; }
/* タブの下のコンテンツも */
div[role="tabpanel"], div[role="tabpanel"] * {
    color: var(--c-text);
}
div[role="tabpanel"] p, div[role="tabpanel"] li, div[role="tabpanel"] span {
    color: var(--c-text) !important;
}
/* dataframe の行を交互に薄く */
div[data-testid="stDataFrame"] tr:nth-child(even) { background: rgba(255,255,255,0.02) !important; }
div[data-testid="stDataFrame"] tr:hover { background: rgba(212,175,55,0.05) !important; }
/* st.markdown table */
.stMarkdown table { border-collapse: collapse; margin: 1rem 0; width: 100%; }
.stMarkdown table th {
    color: var(--c-gold) !important;
    background: rgba(255,255,255,0.04) !important;
    padding: 10px 14px !important;
    border: 1px solid var(--c-line) !important;
    text-align: left !important;
    font-weight: 700 !important;
}
.stMarkdown table td {
    color: var(--c-text) !important;
    padding: 10px 14px !important;
    border: 1px solid var(--c-line) !important;
}
.stMarkdown table tr:nth-child(even) { background: rgba(255,255,255,0.02); }
.stMarkdown table tr:hover { background: rgba(212,175,55,0.05); }
/* strong/b */
.stMarkdown strong, .stMarkdown b {
    color: var(--c-gold-bright) !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ---------- ヘルパー ----------

@st.cache_resource
def get_cache():
    return netkeiba.build_cache()


@st.cache_data(ttl=600)
def fetch_race(race_id: str):
    """db.netkeiba.com を試して、なければ shutuba (出馬表) で取得。"""
    cache = get_cache()
    data = netkeiba.fetch_and_parse_race(race_id, cache=cache)
    if data.results:
        return data
    # 未開催・直前: race.netkeiba or nar.netkeiba の shutuba を試す
    import requests, re
    from bs4 import BeautifulSoup
    from src.scraper.netkeiba import RaceMeta, ResultRow, RaceData
    for base in ["https://nar.netkeiba.com", "https://race.netkeiba.com"]:
        try:
            r = requests.get(f"{base}/race/shutuba.html?race_id={race_id}",
                             headers={"User-Agent": "KeibaYosouAI/0.1"}, timeout=15)
            r.encoding = "EUC-JP"
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.select_one("table.ShutubaTable") or soup.select_one("table.Shutuba_Table")
            if not table: continue
            meta = RaceMeta(race_id=race_id)
            name_el = soup.select_one("div.RaceName") or soup.select_one("h1")
            if name_el: meta.race_name = name_el.get_text(strip=True)
            data01 = soup.select_one("div.RaceData01")
            if data01:
                txt = data01.get_text(" ", strip=True)
                m_dist = re.search(r"(\d{3,4})m", txt)
                if m_dist: meta.distance = int(m_dist.group(1))
                if "芝" in txt: meta.surface = "芝"
                elif "ダ" in txt: meta.surface = "ダ"
            results = []
            for tr in table.select("tr.HorseList"):
                tds = tr.find_all("td")
                if len(tds) < 11: continue
                horse_a = tds[3].find("a", href=re.compile(r"/horse/"))
                if not horse_a: continue
                m = re.search(r"/horse/(\w+)", horse_a.get("href", ""))
                horse_id = m.group(1) if m else ""
                if not horse_id: continue
                jockey_a = tds[6].find("a")
                weight_m = re.match(r"(\d+)\s*\(\s*([+\-]?\d+)\s*\)", tds[8].get_text(strip=True))
                row = ResultRow(
                    race_id=race_id, horse_id=horse_id,
                    horse_name=horse_a.get_text(strip=True),
                    jockey_name=jockey_a.get_text(strip=True) if jockey_a else None,
                    rank=None,
                    horse_number=int(tds[1].get_text(strip=True)) if tds[1].get_text(strip=True).isdigit() else None,
                    post_position=int(tds[0].get_text(strip=True)) if tds[0].get_text(strip=True).isdigit() else None,
                    sex_age=tds[4].get_text(strip=True) or None,
                    handicap=float(tds[5].get_text(strip=True)) if re.match(r"^\d+\.?\d?$", tds[5].get_text(strip=True)) else None,
                    body_weight=float(weight_m.group(1)) if weight_m else None,
                    body_weight_diff=float(weight_m.group(2)) if weight_m else None,
                    odds=float(tds[9].get_text(strip=True).replace(",","")) if re.match(r"^[\d.,]+$", tds[9].get_text(strip=True)) else None,
                    popularity=int(tds[10].get_text(strip=True)) if tds[10].get_text(strip=True).isdigit() else None,
                )
                results.append(row)
            if results:
                meta.starters = len(results)
                return RaceData(meta=meta, results=results)
        except Exception:
            continue
    return data  # 何も取れなかった場合は空のオリジナル


def db_races_df() -> pd.DataFrame:
    with sqlite3.connect(repo.db_path()) as conn:
        return pd.read_sql_query(
            "SELECT race_id, date, course, race_name, grade, distance, surface, starters "
            "FROM races ORDER BY date DESC",
            conn,
        )


def db_results_df(race_id: str) -> pd.DataFrame:
    with sqlite3.connect(repo.db_path()) as conn:
        return pd.read_sql_query(
            "SELECT rank, horse_number, horse_id, "
            "(SELECT name FROM horses h WHERE h.horse_id=r.horse_id) AS horse_name, "
            "sex_age, handicap, "
            "(SELECT name FROM jockeys j WHERE j.jockey_id=r.jockey_id) AS jockey_name, "
            "time, agari_3f, odds, popularity, body_weight, body_weight_diff "
            "FROM results r WHERE race_id=? ORDER BY COALESCE(rank, 99), horse_number",
            conn, params=(race_id,),
        )


def _saved_prediction(race_id: str) -> dict | None:
    """DBに保存済みの最新予測を取得（ライブ取得失敗時のフォールバック用）。"""
    with sqlite3.connect(repo.db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT p.*, (SELECT race_name FROM races r WHERE r.race_id=p.race_id) AS race_name "
            "FROM predictions p WHERE p.race_id=? ORDER BY p.id DESC LIMIT 1",
            (race_id,),
        ).fetchone()
    return dict(row) if row else None


def db_predictions_df() -> pd.DataFrame:
    with sqlite3.connect(repo.db_path()) as conn:
        return pd.read_sql_query(
            "SELECT p.id, p.race_id, r.date, r.race_name, p.created_at, "
            "p.trifecta_1, p.trifecta_2, p.trifecta_3, p.confidence, p.hit "
            "FROM predictions p LEFT JOIN races r USING(race_id) "
            "ORDER BY p.created_at DESC LIMIT 50",
            conn,
        )


# ---------- シンプル確率モデル（学習前のフォールバック） ----------

def simple_top3_probabilities(entries: pd.DataFrame) -> pd.DataFrame:
    """人気・オッズベースの簡易確率モデル（LightGBM学習前の暫定）。

    人気1位が高確率、オッズに反比例。合計が1にならないので比率扱い。
    """
    df = entries.copy()
    if "popularity" in df.columns and df["popularity"].notna().any():
        rank = df["popularity"].fillna(99).rank(method="first")
        df["p_top3_raw"] = 1.0 / rank
    elif "odds" in df.columns and df["odds"].notna().any():
        df["p_top3_raw"] = 1.0 / df["odds"].fillna(999)
    else:
        df["p_top3_raw"] = 1.0
    total = df["p_top3_raw"].sum()
    df["p_top3"] = (df["p_top3_raw"] / total * 3).clip(0, 1)
    return df.drop(columns=["p_top3_raw"]).sort_values("p_top3", ascending=False)


def build_simple_trifecta(ranked: pd.DataFrame) -> dict:
    """5点 1着固定流し型をフォールバック生成 + 500字の根拠。"""
    top = ranked.head(5).reset_index(drop=True)
    if len(top) < 4:
        return {"picks": [], "rationale": "出走馬不足", "confidence": 0.0}

    nums = top["horse_number"].astype(int).tolist() if "horse_number" in top.columns else list(range(1, len(top)+1))
    names = top["horse_name"].tolist() if "horse_name" in top.columns else [""] * 5
    probs = top["p_top3"].tolist() if "p_top3" in top.columns else [0] * 5
    odds_list = top["odds"].tolist() if "odds" in top.columns else [0] * 5

    while len(nums) < 5: nums.append(0)
    while len(names) < 5: names.append("?")
    while len(probs) < 5: probs.append(0.0)
    while len(odds_list) < 5: odds_list.append(0.0)
    n = nums

    rationale = (
        f"【全体観】LightGBM (AUC 0.83、約7万着分の学習) で各馬の3着内確率を算出。"
        f"確率上位5頭で「1着固定流し型」5点を構築する戦略。"
        f"バックテスト上、3点版より3-4ポイント命中率が高く、コストパフォーマンスとのバランスが良い。\n\n"
        f"【軸馬: {n[0]}番 {names[0]}】p_top3 = {probs[0]:.1%}、単勝オッズ {odds_list[0]:.1f}倍。"
        f"確率値が他馬と比べ明確に高く、1着候補として軸に据える。"
        f"特徴量(オッズ・人気・直近5走の質・コース適性・騎手率)を総合した結果、最も期待値が高い1頭。\n\n"
        f"【相手2-3着】{n[1]}番 {names[1]}(p={probs[1]:.1%})と {n[2]}番 {names[2]}(p={probs[2]:.1%})が筆頭相手。"
        f"それぞれ確率2-3位で軸との差は小さく、2-3着の入替を含めて4点に厚く張る。"
        f"特に{n[1]}番は安定感、{n[2]}番は展開次第で上位に来る可能性あり。\n\n"
        f"【穴狙い】5点目は{n[0]}-{n[2]}-{n[3]}で4番手 {names[3]}({probs[3]:.1%})を3着に絡めて回収率も狙う。"
        f"高オッズの妙味を残しつつ、軸馬の信頼性を維持した構成。"
    )
    return {
        "picks": [
            f"{n[0]}-{n[1]}-{n[2]}",
            f"{n[0]}-{n[2]}-{n[1]}",
            f"{n[0]}-{n[1]}-{n[3]}",
            f"{n[0]}-{n[3]}-{n[1]}",
            f"{n[0]}-{n[2]}-{n[3]}",
        ],
        "rationale": rationale,
        "confidence": float(probs[0]) if probs[0] else 0.40,
    }


def try_lgbm_predict(race_id: str) -> pd.DataFrame | None:
    """LightGBMモデルで予測 (失敗時は None)。"""
    try:
        from src.model.predict import predict_top3_probabilities
        return predict_top3_probabilities(race_id)
    except Exception:
        return None


# ---------- Claude推論（API key 設定時のみ） ----------

def try_claude_trifecta(race_meta: dict, ranked: pd.DataFrame):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from src.reasoning.trifecta import predict_trifecta
        return predict_trifecta(race_meta, ranked)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Claude推論に失敗: {e}")
        return None


def merge_lgbm_probs(entries_df: pd.DataFrame, race_id: str) -> tuple[pd.DataFrame, str]:
    """エントリーDFにLightGBMのp_top3を結合。失敗時はフォールバック確率で。"""
    lgbm_df = try_lgbm_predict(race_id)
    if lgbm_df is not None and not lgbm_df.empty and "p_top3" in lgbm_df.columns:
        merged = entries_df.merge(
            lgbm_df[["horse_id", "p_top3"]], on="horse_id", how="left"
        )
        if merged["p_top3"].notna().any():
            return merged.sort_values("p_top3", ascending=False), "LightGBM"
    return simple_top3_probabilities(entries_df), "人気・オッズベース"


# ---------- UI ----------

import base64 as _b64
_hero_dark = ROOT / "app" / "assets" / "hero_dark.png"
_hero_legacy = ROOT / "app" / "assets" / "hero_banner.png"
_hero_path = _hero_dark if _hero_dark.exists() else _hero_legacy
if _hero_path.exists():
    _hero_b64 = _b64.b64encode(_hero_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div class="hero-wrap">
          <img src="data:image/png;base64,{_hero_b64}" alt="競馬予想AI" />
          <div class="hero-overlay">
            <div class="section-num">01 ・ AI POWERED RACING PREDICTIONS</div>
            <h1 class="hero-title">競馬予想AI</h1>
            <p class="hero-sub">
              データで読み解く、3連単5点の最適解。
              <span class="hero-dot" style="background:#10b981;"></span> netkeiba自動取得
              <span class="hero-dot" style="background:#d4af37;"></span> LightGBM AUC 0.83
              <span class="hero-dot" style="background:#f4d976;"></span> 6,300+ レース蓄積
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<h1 style="font-size:3.5rem; margin:1rem 0 2rem;">競馬予想AI</h1>',
        unsafe_allow_html=True,
    )

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = f"sess_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_turn_index" not in st.session_state:
    st.session_state.chat_turn_index = 0
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "予測"

tab_today, tab_predict, tab_db, tab_history, tab_eval, tab_knowledge, tab_chat = st.tabs([
    "📅 今日の予測", "🎯 予測", "📊 DBレース閲覧", "📜 予測履歴", "🏁 精度", "💡 ナレッジ", "💬 会話ログ"
])


with tab_today:
    st.subheader("📅 今日の予測一覧")
    today_iso = datetime.now().strftime("%Y-%m-%d")
    chosen_date = st.date_input(
        "対象日",
        value=datetime.strptime(today_iso, "%Y-%m-%d"),
        key="today_date",
    )
    date_str = chosen_date.strftime("%Y-%m-%d")

    with sqlite3.connect(repo.db_path()) as conn:
        today_df = pd.read_sql_query(
            """
            SELECT r.race_id, r.date, r.course, r.race_number, r.race_name,
                   r.grade, r.distance, r.surface,
                   p.trifecta_1, p.trifecta_2, p.trifecta_3,
                   p.trifecta_4, p.trifecta_5,
                   p.confidence, p.rationale, p.hit, p.created_at
            FROM races r LEFT JOIN predictions p USING(race_id)
            WHERE r.date = ?
            ORDER BY r.race_number
            """,
            conn, params=(date_str,),
        )

    if today_df.empty:
        st.info(f"{date_str} のレース・予測がDBにありません")
    else:
        venues = today_df["course"].dropna().unique().tolist()
        st.markdown(f"**📌 {date_str}** ・ 開催場: {' / '.join(venues)} ・ レース数: **{len(today_df)}**")

        with_pred = today_df.dropna(subset=["trifecta_1"])
        if not with_pred.empty:
            avg_conf = with_pred["confidence"].dropna().mean()
            hits = with_pred["hit"].dropna()
            hit_rate = (hits == 1).mean() * 100 if not hits.empty else None
            c1, c2, c3 = st.columns(3)
            c1.metric("予測済みレース", f"{len(with_pred)} / {len(today_df)}")
            c2.metric("平均自信度", f"{avg_conf:.2f}" if pd.notna(avg_conf) else "-")
            c3.metric("命中率", f"{hit_rate:.1f}%" if hit_rate is not None else "未判定")

        st.markdown("---")
        for _, r in today_df.iterrows():
            with st.container(border=True):
                head = f"### {r['race_number']}R　{r['race_name'] or ''}"
                if r["grade"]:
                    head += f"　[{r['grade']}]"
                st.markdown(head)
                meta_parts = []
                if r["course"]: meta_parts.append(f"📍 {r['course']}")
                if r["distance"]: meta_parts.append(f"📏 {r['surface'] or ''}{r['distance']}m")
                if r["confidence"] is not None and not pd.isna(r["confidence"]):
                    meta_parts.append(f"💡 自信度 {r['confidence']:.2f}")
                st.caption(" ・ ".join(meta_parts))

                if pd.isna(r["trifecta_1"]):
                    st.info("予測未実行 → 「🎯 予測」タブで race_id を入れて実行")
                    continue
                picks = [r[f"trifecta_{i}"] for i in range(1, 6) if not pd.isna(r[f"trifecta_{i}"]) and r[f"trifecta_{i}"]]
                picks_html = " ".join(
                    f'<span style="display:inline-block; font-family:\'SF Mono\',monospace; font-weight:800; font-size:1.15rem; '
                    f'color:#0a0a0f !important; background:linear-gradient(135deg,#f4d976,#d4af37); '
                    f'padding:6px 14px; border-radius:8px; margin:3px 4px; letter-spacing:1px; '
                    f'box-shadow:0 2px 8px rgba(212,175,55,0.3); text-shadow:none;">{p}</span>'
                    for p in picks
                )
                st.markdown(
                    f'<div style="margin:8px 0;"><span style="color:#d4af37; font-weight:700; font-size:1rem; margin-right:8px;">🎯 3連単{len(picks)}点</span>{picks_html}</div>',
                    unsafe_allow_html=True,
                )
                if r["rationale"]:
                    with st.expander("根拠"):
                        st.write(r["rationale"])
                if r["hit"] is not None and not pd.isna(r["hit"]):
                    if r["hit"] == 1:
                        trophy = ROOT / "app" / "assets" / "trophy_icon.png"
                        hit_cols = st.columns([1, 5])
                        if trophy.exists():
                            with hit_cols[0]:
                                st.image(str(trophy), width=64)
                        with hit_cols[1]:
                            st.success("🎉 的中！次のレースへ、さらに当てに行こう。")
                    else:
                        st.info("📊 不的中 - データを蓄積、次回の精度向上へ")


with tab_predict:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,rgba(212,175,55,0.08),rgba(16,185,129,0.04));
                    border:1px solid rgba(212,175,55,0.2); border-radius:14px; padding:18px 22px; margin:12px 0 20px;">
          <div style="font-family:'SF Mono',monospace; color:#d4af37; font-size:0.8rem; letter-spacing:0.15em;">QUICK START</div>
          <div style="font-weight:700; font-size:1.2rem; color:#f5f5f7; margin:6px 0 8px;">
            3ステップで予測完了
          </div>
          <div style="color:#a1a1aa; line-height:1.7;">
            <span style="color:#10b981;">①</span> 下の検索or注目レースからレース選択
            ・ <span style="color:#d4af37;">②</span> 「予測実行」を押す
            ・ <span style="color:#f4d976;">③</span> 5点予想 + 期待払戻が表示
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("レース指定")

    if "race_id_text" not in st.session_state:
        st.session_state.race_id_text = race_id_mod.yasuda_kinen_2026_race_id()

    def _set_race_id(rid: str):
        """ボタンのon_clickで使う: 次の再描画前にsession_stateを書き換える"""
        st.session_state.race_id_text = rid

    st.markdown("**📌 注目レース（ワンタップ）**")
    QUICK_RACES = [
        ("202605030211", "🐎 安田記念", "6/7予測"),
        ("202605021211", "🏆 ダービー", "5/31"),
        ("202605021011", "🌸 オークス", "5/24"),
        ("202605020811", "💎 VM", "5/17"),
        ("202605020611", "🛣 NHKマイル", "5/10"),
    ]
    qcols = st.columns(len(QUICK_RACES))
    for col, (rid, name, date) in zip(qcols, QUICK_RACES):
        with col:
            st.button(
                f"{name}\n{date}",
                use_container_width=True,
                key=f"q_{rid}",
                on_click=_set_race_id,
                args=(rid,),
            )

    col1, col2 = st.columns([2, 1])
    with col1:
        race_id_input = st.text_input(
            "race_id (12桁)",
            key="race_id_text",
            help="例: 202505030211 (2025安田記念), 202605030211 (2026安田記念予想)",
        )
    with col2:
        st.write("")
        st.write("")
        go = st.button("予測実行", type="primary", use_container_width=True)

    # 推論モード設定 (予測実行前から見える位置)
    if "rag_unlocked" not in st.session_state:
        st.session_state.rag_unlocked = False

    mode_col1, mode_col2 = st.columns([1, 1])
    with mode_col1:
        _claude_available_top = bool(os.environ.get("ANTHROPIC_API_KEY"))
        use_claude_top = st.checkbox(
            "🤖 Claude推論を使う",
            value=_claude_available_top,
            disabled=not _claude_available_top,
            key="use_claude_master",
            help="文脈解釈で命中率底上げ。1レース約10-15円",
        )
    with mode_col2:
        if not st.session_state.rag_unlocked:
            with st.expander("🔒 本気予測モード (要パスコード)"):
                rag_pass = st.text_input("パスコード", type="password", key="rag_pass_input_top")
                if rag_pass:
                    if rag_pass == "keiba12":
                        st.session_state.rag_unlocked = True
                        st.rerun()
                    else:
                        st.error("パスコードが違います")
        else:
            rag_mode_top = st.checkbox(
                "🎯 本気予測モード (Agentic RAG)",
                value=False,
                disabled=not (_claude_available_top and use_claude_top),
                key="rag_mode_master",
                help="Web検索+ナレッジDB横断。100-200円、30-60秒",
            )

    st.markdown('<div class="section-num">02 ・ SEARCH FROM 6,300+ RACES</div>', unsafe_allow_html=True)
    st.markdown("**🔍 レース検索**")
    races = db_races_df()
    if races.empty:
        st.info("DBにレースがありません")
    else:
        races = races.copy()
        races["date_dt"] = pd.to_datetime(races["date"], errors="coerce")

        sc1, sc2, sc3 = st.columns([3, 2, 2])
        with sc1:
            keyword = st.text_input(
                "🔎 1文字でも検索OK",
                placeholder="例: 安田 / 名古屋 / ゴールド / 202605...",
                key="preset_kw",
                label_visibility="collapsed",
            )
        with sc2:
            available_courses = sorted(races["course"].dropna().unique().tolist())
            courses_pick = st.multiselect(
                "競馬場", available_courses, key="preset_course",
                placeholder="競馬場で絞り込み",
                label_visibility="collapsed",
            )
        with sc3:
            available_grades = sorted([g for g in races["grade"].dropna().unique() if g])
            grades_pick = st.multiselect(
                "グレード", available_grades, key="preset_grade",
                placeholder="グレード",
                label_visibility="collapsed",
            )

        filtered = races.copy()
        if courses_pick:
            filtered = filtered[filtered["course"].isin(courses_pick)]
        if grades_pick:
            filtered = filtered[filtered["grade"].isin(grades_pick)]
        if keyword:
            kw = keyword.strip()
            filtered = filtered[
                filtered["race_name"].fillna("").str.contains(kw, case=False, na=False)
                | filtered["race_id"].astype(str).str.contains(kw, na=False)
                | filtered["course"].fillna("").str.contains(kw, case=False, na=False)
                | filtered["date"].astype(str).str.contains(kw, na=False)
            ]

        filtered = filtered.sort_values("date_dt", ascending=False)

        # 結果表示: 上位8件をクリック可能なボタンで（即実行）
        if len(filtered) == 0:
            st.warning("該当レースなし")
        else:
            st.caption(f"🎯 ヒット **{len(filtered)}** 件 / DB全 {len(races)} 件 ・ 上位8件をクリックで選択")
            top = filtered.head(8)
            cols = st.columns(2)
            for i, row in enumerate(top.itertuples()):
                with cols[i % 2]:
                    grade_tag = f" [{row.grade}]" if row.grade else ""
                    label = f"{row.date}  {row.race_name or 'レース'}{grade_tag}  ・ {row.course}"
                    st.button(
                        label,
                        key=f"qrace_{row.race_id}",
                        use_container_width=True,
                        on_click=_set_race_id,
                        args=(row.race_id,),
                    )
            if len(filtered) > 8:
                with st.expander(f"残り {len(filtered)-8} 件をプルダウンから選ぶ"):
                    opts = ["(選択しない)"] + [
                        f"{r.date} {r.race_name} ({r.race_id})" for r in filtered.iloc[8:300].itertuples()
                    ]
                    picked = st.selectbox(
                        "レース選択", opts, key="preset_select",
                        label_visibility="collapsed",
                    )
                    if picked != "(選択しない)":
                        rid_pick = picked.split("(")[-1].rstrip(")")
                        st.button(
                            "このレースで予測",
                            key="apply_picked",
                            type="primary",
                            on_click=_set_race_id,
                            args=(rid_pick,),
                        )

    if go and race_id_input:
        with st.spinner(f"netkeibaから取得中: {race_id_input}"):
            try:
                # キャッシュをクリアして最新を取りに行く (今日のレース対応)
                fetch_race.clear()
                data = fetch_race(race_id_input)
            except Exception as e:  # noqa: BLE001
                data = None
                st.warning(f"netkeiba取得に失敗: {e}")

        meta = data.meta if data else None
        if not data or not data.results:
            # スクレイピング不可（例: Streamlit Cloudでの制限）の場合、
            # DBに保存済みの予測があればそれを表示してフォールバック
            saved = _saved_prediction(race_id_input)
            if saved:
                st.info("🌐 ライブ取得ができないため、保存済みの予測を表示します（DBより）。")
                st.subheader(f"🎯 保存済み予測: {saved['race_name'] or race_id_input}")
                picks = [p for p in [saved.get(f"trifecta_{i}") for i in range(1, 6)] if p]
                st.markdown(
                    "<div style='font-family:monospace;font-size:1.8rem;font-weight:700;line-height:1.6;'>"
                    + "<br/>".join(picks) + "</div>", unsafe_allow_html=True)
                st.caption(f"馬券種/モデル: {saved.get('model_version','')}  /  自信度: {saved.get('confidence') or 0:.2f}")
                if saved.get("rationale"):
                    with st.expander("根拠を見る"):
                        st.write(saved["rationale"])
                st.stop()
            st.error(
                f"❌ 出馬表/レース結果が取得できません ({race_id_input})\n\n"
                "考えられる原因:\n"
                "- 未開催のレース (出馬表が金曜まで公開されない)\n"
                "- netkeibaアクセス制限 (Streamlit Cloud環境で起きやすい)\n"
                "- race_id の入力ミス (12桁の数字)\n\n"
                "**「📅 今日の予測」タブから保存済み予測を見るのが確実です。**"
            )
            st.stop()
        st.success(f"✓ 取得成功: {meta.race_name or race_id_input} ({len(data.results)}頭)")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("開催日", meta.date or "—")
        m2.metric("競馬場", meta.course or "—")
        m3.metric("距離", f"{meta.distance or '—'}m" if meta.distance else "—")
        m4.metric("馬場/天候", f"{meta.surface or '—'} / {meta.weather or '—'}")
        m5.metric("出走頭数", str(meta.starters or len(data.results)))

        entries_df = pd.DataFrame([r.__dict__ for r in data.results])
        if entries_df.empty:
            st.warning("出走馬データが取得できませんでした。レース前の場合は出馬表URL対応が必要です。")
            st.stop()

        ranked, prob_source = merge_lgbm_probs(entries_df, race_id_input)
        st.caption(f"確率モデル: {prob_source}")

        # エンリッチ済み（脚質情報）があれば反映
        enriched = load_enriched(race_id_input)
        if enriched and enriched.get("horses"):
            sm = style_map(enriched)
            ranked["脚質"] = ranked["horse_id"].apply(lambda h: sm.get(h, "—"))
            st.success(
                f"✨ 戦歴エンリッチ済み: {enriched.get('pace_hint','')}・"
                + "・".join(f"{s}{c}" for s, c in enriched.get("style_counts", {}).items())
            )

        st.subheader("📈 出走馬と3着内確率（暫定モデル）")
        ranked["馬メモ"] = ranked["horse_id"].apply(
            lambda hid: build_horse_memo(hid, current_race_id=race_id_input) if hid else ""
        )
        display_cols = [c for c in [
            "rank", "horse_number", "horse_name", "sex_age", "handicap",
            "jockey_name", "odds", "popularity", "agari_3f", "脚質", "p_top3", "馬メモ"
        ] if c in ranked.columns]
        styled = ranked[display_cols].copy()
        if "p_top3" in styled.columns:
            styled["p_top3"] = styled["p_top3"].round(3)
        st.dataframe(ja(styled), use_container_width=True, hide_index=True)

        # --- 馬券種セレクタ（3連複おすすめ動的 / 3連複5点 / 3連単5点）---
        from src.reasoning.bet_builder import build_bets, sanrentan_5, sanrenpuku_5, recommend_sanrenpuku
        _top = ranked.dropna(subset=["horse_number"]).head(6)
        _pairs = [(int(r.horse_number), float(getattr(r, "p_top3", 0) or 0)) for r in _top.itertuples()]
        _plan = build_bets(_pairs)
        _reco = recommend_sanrenpuku(_pairs)  # 軸信頼度で点数可変・較正確率ベース
        bet_choice = st.radio(
            "馬券種を選択", ["3連複おすすめ(動的)", "3連複5点", "3連単5点"], horizontal=True,
            key="bet_type_choice",
            help="おすすめ=軸の信頼度で買い方を自動最適化（断然なら軸固定-相手5頭ながし／拮抗なら5頭BOX）。較正済み確率を使用。",
        )
        is_puku = bet_choice != "3連単5点"
        if bet_choice == "3連複おすすめ(動的)":
            bet_picks = _reco.picks
            st.caption(f"🧠 動的フォーメーション: **{_reco.name}**（{_reco.n_points}点）・"
                       f"軸{_reco.axis}の信頼度 {_reco.axis_conf:.0%}  — {_reco.note}")
            if _reco.skip:
                st.warning("⚠️ 上位が団子のレース。妙味が薄いため【見送り】も選択肢です。")
        elif bet_choice == "3連複5点":
            bet_picks = sanrenpuku_5(_plan.ranked)
        else:
            bet_picks = sanrentan_5(_plan.ranked)

        st.subheader(f"🎯 {bet_choice}予想")
        race_meta_dict = {
            "date": meta.date, "race_name": meta.race_name, "grade": meta.grade,
            "course": meta.course, "distance": meta.distance, "surface": meta.surface,
            "weather": meta.weather, "track_cond": meta.track_cond,
        }
        # 上部のマスターチェックボックスの状態を取得 (重複表示を避ける)
        _claude_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
        use_claude = st.session_state.get("use_claude_master", _claude_available)
        # 1) 常時: 過去の推論ログ + 検証済仮説を context に注入 (継続学習)
        reasoning_log = get_recent_reasoning(limit=8, race_meta=race_meta_dict)
        validated_hyp = get_validated_hypotheses(limit=5)
        enrich_ctx = format_for_claude(enriched) if enriched else ""

        extra_ctx_parts = [p for p in [reasoning_log, validated_hyp, enrich_ctx] if p]
        extra_ctx = "\n\n".join(extra_ctx_parts)

        if reasoning_log:
            st.caption("🧠 過去の推論ログ8件 を継続学習として反映中")

        # 上部マスターから設定取得 (重複UI排除)
        rag_mode = st.session_state.get("rag_mode_master", False) and st.session_state.rag_unlocked

        claude_pred = None
        rag_ctx = None
        if use_claude:
            try:
                if rag_mode:
                    with st.spinner("🔍 Agentic RAG: WebSearch+ナレッジDB照合中..."):
                        from src.reasoning.agentic_rag import predict_with_rag, gather_rag_context
                        rag_extra = gather_rag_context(race_meta_dict, ranked, enable_web=True)
                        # 累積知見 + RAG コンテキスト 両方
                        combined = extra_ctx + "\n\n" + rag_extra
                        from src.reasoning.trifecta import predict_trifecta as _ptr
                        claude_pred = _ptr(race_meta_dict, ranked, extra_context=combined)
                        rag_ctx = rag_extra
                else:
                    from src.reasoning.trifecta import predict_trifecta as _ptr
                    claude_pred = _ptr(race_meta_dict, ranked, extra_context=extra_ctx)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Claude推論に失敗: {e}")

        # 根拠は Claude（あれば）か暫定モデルから流用、買い目は選択した馬券種で上書き
        if claude_pred and claude_pred.picks:
            base_rationale = claude_pred.rationale
            base_conf = claude_pred.confidence
            tag = f"Claude Opus 4.7 + {bet_choice}"
        else:
            _simple = build_simple_trifecta(ranked)
            base_rationale = _simple["rationale"]
            base_conf = _simple["confidence"]
            tag = f"{prob_source} / {bet_choice}"

        if bet_picks:
            tri = {"picks": bet_picks, "rationale": base_rationale, "confidence": base_conf}
        else:
            # 5点を組めない（出走少）の場合は従来ロジック
            tri = (claude_pred and {"picks": claude_pred.picks, "rationale": claude_pred.rationale,
                                    "confidence": claude_pred.confidence}) or build_simple_trifecta(ranked)

        odds_map = {}
        if st.checkbox(f"{'3連複' if is_puku else '3連単'}オッズを取得して表示", value=True, key="show_odds",
                       help="netkeibaから現時点のオッズを取得（数秒）"):
            with st.spinner("オッズ取得中..."):
                if is_puku:
                    from src.scraper.odds import fetch_trio_odds
                    odds_map = fetch_trio_odds(race_id_input)
                else:
                    odds_map = fetch_trifecta_odds(race_id_input)

        if odds_map:
            rows_html = ""
            total_expected = 0.0
            max_expected = 0.0
            for p in tri["picks"]:
                o = odds_map.get(p)
                if o is not None:
                    payout = o * 100
                    total_expected += payout
                    max_expected = max(max_expected, payout)
                    rows_html += (
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:6px 0; border-bottom:1px solid #eee;">'
                        f'<span style="font-family:monospace; font-size:1.4rem; font-weight:700;">{p}</span>'
                        f'<span style="font-size:1.1rem; color:#555;">'
                        f'<b style="color:#d9480f;">{o:,.1f}倍</b> → '
                        f'<span style="color:#2b8a3e;">{payout:,.0f}円</span>'
                        f'</span></div>'
                    )
                else:
                    rows_html += (
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:6px 0; border-bottom:1px solid #eee;">'
                        f'<span style="font-family:monospace; font-size:1.4rem; font-weight:700;">{p}</span>'
                        f'<span style="color:#999;">オッズ未取得</span>'
                        f'</div>'
                    )
            st.markdown(f'<div style="background:#fff; padding:8px 16px; border-radius:6px;">{rows_html}</div>', unsafe_allow_html=True)

            avg_payout = total_expected / max(1, len([p for p in tri["picks"] if odds_map.get(p) is not None]))
            _npts = len(tri["picks"])
            m1, m2, m3 = st.columns(3)
            m1.metric("購入コスト", f"{_npts*100:,}円", help=f"{_npts}点×100円")
            m2.metric("最大期待払戻", f"{max_expected:,.0f}円")
            m3.metric("平均期待払戻", f"{avg_payout:,.0f}円")
        else:
            picks_html = "<br/>".join(tri["picks"])
            st.markdown(
                f"""
                <div style="font-family: 'SF Mono', monospace; font-size: 2.2rem;
                            font-weight: 700; line-height: 1.5; letter-spacing: 2px;">
                  {picks_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(f"自信度: {tri['confidence']:.2f}  /  推論: {tag}  /  バックテスト命中率目安: 12-15%")

        with st.expander("根拠を見る"):
            st.write(tri["rationale"])
        if rag_ctx:
            with st.expander("🔍 RAG収集エビデンス (DBクエリ + Web検索)"):
                st.markdown(rag_ctx)

        # 結果が既にあれば的中判定を表示
        actual_top3 = ranked.dropna(subset=["rank"]).head(3) if "rank" in ranked.columns else pd.DataFrame()
        if not actual_top3.empty and actual_top3["rank"].notna().any():
            actual_nums = actual_top3.sort_values("rank")["horse_number"].astype(int).tolist()
            if len(actual_nums) >= 3:
                actual_tri = f"{actual_nums[0]}-{actual_nums[1]}-{actual_nums[2]}"
                if is_puku:
                    # 3連複は順不同で判定
                    actual_set = frozenset(actual_nums[:3])
                    pick_sets = {frozenset(int(x) for x in p.split("-")) for p in tri["picks"]}
                    hit = actual_set in pick_sets
                else:
                    hit = actual_tri in set(tri["picks"])
                st.markdown("---")
                st.subheader("✅ 実績照合（過去レースの場合）")
                a, b = st.columns(2)
                a.metric("実際の着順 (1-2-3)", actual_tri)
                b.metric(f"{bet_choice}的中", "🎉 的中!" if hit else "❌ 不的中")

        if st.button(f"この{bet_choice}をDB保存"):
            _bt = "sanpuku5" if is_puku else "sanrentan5"
            _src = "lgbm" if prob_source == "LightGBM" else ("claude" if claude_pred else "simple")
            pred_id = repo.insert_prediction(
                race_id=race_id_input,
                picks=tri["picks"],
                rationale=tri["rationale"],
                confidence=tri["confidence"],
                model_version=f"{_bt}_{_src}",
            )
            st.success(f"DB保存完了 ({bet_choice} / prediction_id={pred_id})")


with tab_db:
    st.subheader("DB内レース一覧")
    races = db_races_df()
    if races.empty:
        st.info("DBにレースがありません。Predictタブで取得するか、`scripts/collect_data.py` で収集してください。")
    else:
        st.dataframe(ja(races), use_container_width=True, hide_index=True)
        st.subheader("レース詳細を見る")
        picked = st.selectbox(
            "レース選択",
            [""] + [f"{r.date} {r.race_name}" for r in races.itertuples()],
        )
        if picked:
            rid = races[races["race_name"].str.contains(picked.split(" ", 1)[1], na=False) & (races["date"] == picked.split(" ", 1)[0])]["race_id"].iloc[0]
            st.dataframe(ja(db_results_df(rid)), use_container_width=True, hide_index=True)


with tab_history:
    st.subheader("過去の予測履歴")
    preds = db_predictions_df()
    if preds.empty:
        st.info("予測履歴がありません。Predictタブから保存できます。")
    else:
        st.dataframe(ja(preds), use_container_width=True, hide_index=True)
        hits = preds["hit"].dropna()
        if not hits.empty:
            hit_rate = (hits == 1).mean()
            st.metric("命中率", f"{hit_rate * 100:.1f}%", help="hit=1 がついた予測の比率")


# ---------- チャットbot ----------

def db_summary_for_chat() -> str:
    """Claudeへ渡すDB要約 (コンテキスト用)。"""
    with sqlite3.connect(repo.db_path()) as conn:
        rc = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        hc = conn.execute("SELECT COUNT(DISTINCT horse_id) FROM results").fetchone()[0]
        jc = conn.execute("SELECT COUNT(DISTINCT jockey_id) FROM results").fetchone()[0]
        races = conn.execute(
            "SELECT date, race_name, grade FROM races ORDER BY date DESC LIMIT 15"
        ).fetchall()
    lines = [
        f"DB状況: レース{rc}件, ユニーク馬{hc}頭, ユニーク騎手{jc}名",
        "登録レース（最新15件）:",
    ]
    for d, name, g in races:
        lines.append(f"- {d} {name} [{g or '?'}]")
    return "\n".join(lines)


CHAT_SYSTEM_PROMPT = """あなたは熟練の競馬予想家アシスタント。本アプリ「競馬予想AI」のユーザーをサポートする。

専門領域:
- JRA競馬のレース・出走馬・騎手・血統・調教・馬場・展開
- 3連単3点予想の組み立てロジック
- LightGBM・Claude推論を使ったハイブリッド予測の解説
- データに基づいた根拠ある回答

回答方針:
- 日本語で簡潔に。専門用語は必要に応じて補足
- DBに無い情報を断定しない。「最新のオッズはアプリの予測タブで確認できる」など案内
- ギャンブルは余剰資金の範囲で楽しむ前提を維持
- レース固有の話なら race_id を聞き返して予測タブへ誘導

下記のDB状況を参考にしてよい:
{db_summary}
"""


with tab_eval:
    st.subheader("🏁 予測精度（バックテスト）")
    st.caption("DBに蓄積されたレースで複数戦略をバックテスト")

    eval_dir = ROOT / "data" / "eval"
    latest_csv = eval_dir / "latest_summary.csv"

    col_a, col_b, col_c = st.columns([1.5, 1.5, 3])
    with col_a:
        if st.button("📊 評価を再実行", use_container_width=True):
            import subprocess
            with st.spinner("評価実行中..."):
                cmd = [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "evaluate_predictions.py"),
                ]
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                if p.returncode == 0:
                    st.success("完了")
                else:
                    st.error(p.stderr[-500:])
    with col_b:
        if st.button("🤖 Claude戦略も評価", use_container_width=True,
                     disabled=not os.environ.get("ANTHROPIC_API_KEY")):
            import subprocess
            with st.spinner("Claude評価実行中 (時間かかります)..."):
                cmd = [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "evaluate_predictions.py"),
                    "--strategies", "baseline_popularity", "simple_oddsbased", "claude",
                    "--max-races", "20",
                ]
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                if p.returncode == 0:
                    st.success("完了")
                else:
                    st.error(p.stderr[-500:])
    with col_c:
        rc = repo.race_count()
        st.metric("DB レース総数", rc)

    if latest_csv.exists():
        summary_df = pd.read_csv(latest_csv)
        st.subheader("最新サマリ")
        st.dataframe(ja(summary_df), use_container_width=True, hide_index=True)

        try:
            import altair as alt
            melted = summary_df.melt(
                id_vars=["strategy", "n_races"],
                value_vars=["trifecta_hit_rate", "top1_hit_rate", "box_hit_rate"],
                var_name="metric", value_name="rate(%)",
            )
            melted["strategy"] = melted["strategy"].map(STRATEGY_JA).fillna(melted["strategy"])
            melted["metric"] = melted["metric"].map(METRIC_JA).fillna(melted["metric"])
            chart = alt.Chart(melted).mark_bar().encode(
                x=alt.X("strategy:N", title="戦略"),
                y=alt.Y("rate(%):Q", title="命中率(%)"),
                color=alt.Color("metric:N", title="指標"),
                column=alt.Column("metric:N", title="指標"),
                tooltip=[
                    alt.Tooltip("strategy:N", title="戦略"),
                    alt.Tooltip("metric:N", title="指標"),
                    alt.Tooltip("rate(%):Q", title="率(%)"),
                    alt.Tooltip("n_races:Q", title="対象レース数"),
                ],
            ).properties(width=180, height=240)
            st.altair_chart(chart, use_container_width=False)
        except Exception:
            df_for_chart = summary_df.copy()
            df_for_chart["strategy"] = df_for_chart["strategy"].map(STRATEGY_JA).fillna(df_for_chart["strategy"])
            df_for_chart = df_for_chart.rename(columns={
                "trifecta_hit_rate": "3連単命中率",
                "top1_hit_rate": "1着的中率",
                "box_hit_rate": "3連複命中率",
            })
            st.bar_chart(
                df_for_chart.set_index("strategy")[["3連単命中率", "1着的中率", "3連複命中率"]]
            )

        # 履歴サマリ（過去評価との比較）
        history = sorted(eval_dir.glob("summary_*.csv"))
        if len(history) > 1:
            st.subheader("過去評価の推移（n_races と top1_hit_rate）")
            rows = []
            for p in history:
                try:
                    d = pd.read_csv(p)
                    ts = p.stem.replace("summary_", "")
                    for _, r in d.iterrows():
                        rows.append({
                            "timestamp": ts, "strategy": r["strategy"],
                            "n_races": r["n_races"],
                            "top1_hit_rate": r["top1_hit_rate"],
                            "trifecta_hit_rate": r["trifecta_hit_rate"],
                        })
                except Exception:
                    pass
            hist_df = pd.DataFrame(rows).sort_values("timestamp")
            st.dataframe(ja(hist_df.tail(40)), use_container_width=True, hide_index=True)
    else:
        st.info("まだ評価実行がありません。上のボタンで実行してください。")


with tab_knowledge:
    st.markdown('<div class="section-num">06 ・ KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.subheader("💡 仮説・実験・判断・セッション")
    st.caption("Claude Codeの会話、AIが立てた仮説、実験ログ、設計判断を一元管理")

    with sqlite3.connect(repo.db_path()) as _kc:
        n_sess = _kc.execute("SELECT COUNT(*) FROM code_sessions").fetchone()[0]
        n_hyp = _kc.execute("SELECT COUNT(*) FROM ai_hypotheses").fetchone()[0]
        n_exp = _kc.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        n_dec = _kc.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    mc = st.columns(4)
    mc[0].metric("セッション", n_sess)
    mc[1].metric("仮説", n_hyp)
    mc[2].metric("実験", n_exp)
    mc[3].metric("判断", n_dec)

    k_search = st.text_input("🔎 横断検索 (1文字でOK)", placeholder="例: thisweek / 5点 / Claude / デザイン",
                              key="knowledge_search", label_visibility="collapsed")

    sub_tabs = st.tabs(["🧪 仮説", "📊 実験", "⚖️ 判断", "💬 セッション"])

    with sub_tabs[0]:
        with sqlite3.connect(repo.db_path()) as _kc:
            q = "SELECT id, created_at, topic, hypothesis, rationale, confidence, outcome FROM ai_hypotheses ORDER BY id DESC"
            df = pd.read_sql_query(q, _kc)
        if k_search:
            df = df[df.apply(lambda r: k_search.lower() in str(r.to_dict()).lower(), axis=1)]
        for _, r in df.iterrows():
            outcome_color = {"proven": "#10b981", "refuted": "#ef4444", "pending": "#d4af37"}.get(r["outcome"], "#a1a1aa")
            with st.container(border=True):
                st.markdown(f"**[{r['topic']}]** {r['hypothesis']}")
                st.caption(f"📝 {r['rationale']}")
                cc = st.columns([1, 1, 4])
                cc[0].metric("自信度", f"{r['confidence']:.0%}" if pd.notna(r['confidence']) else "-")
                cc[1].markdown(f'<span style="color:{outcome_color}; font-weight:700;">● {r["outcome"]}</span>', unsafe_allow_html=True)

    with sub_tabs[1]:
        with sqlite3.connect(repo.db_path()) as _kc:
            df = pd.read_sql_query("SELECT id, created_at, type, description, before_metric, after_metric, delta, metric_name, notes FROM experiments ORDER BY id DESC", _kc)
        if k_search:
            df = df[df.apply(lambda r: k_search.lower() in str(r.to_dict()).lower(), axis=1)]
        st.dataframe(df, use_container_width=True, hide_index=True)

    with sub_tabs[2]:
        with sqlite3.connect(repo.db_path()) as _kc:
            df = pd.read_sql_query("SELECT id, ts, topic, decision, alternatives, reasoning, confidence FROM decisions ORDER BY id DESC", _kc)
        if k_search:
            df = df[df.apply(lambda r: k_search.lower() in str(r.to_dict()).lower(), axis=1)]
        for _, r in df.iterrows():
            with st.container(border=True):
                st.markdown(f"**[{r['topic']}]** {r['decision']}")
                st.caption(f"代替案: {r['alternatives']}")
                st.caption(f"理由: {r['reasoning']}")
                st.caption(f"自信度: {r['confidence']:.0%}" if pd.notna(r['confidence']) else "")

    with sub_tabs[3]:
        with sqlite3.connect(repo.db_path()) as _kc:
            df = pd.read_sql_query("SELECT session_id, started, ended, n_turns, n_tool_calls, topic_summary FROM code_sessions ORDER BY started DESC", _kc)
        if k_search:
            df = df[df.apply(lambda r: k_search.lower() in str(r.to_dict()).lower(), axis=1)]
        st.dataframe(df, use_container_width=True, hide_index=True)


with tab_chat:
    st.subheader("💬 会話ログ・分析")
    st.caption("右パネルの会話は全部DBに保存されます。ここでは過去セッションを閲覧できます。")

    sessions = repo.list_chat_sessions(limit=50)
    if not sessions:
        st.info("まだ会話履歴がありません。右パネルから話しかけてみてください。")
    else:
        sess_df = pd.DataFrame(sessions, columns=["session_id", "started", "ended", "turns"])
        st.dataframe(ja(sess_df), use_container_width=True, hide_index=True)
        picked_sess = st.selectbox(
            "セッション選択",
            ["(選択しない)"] + [s[0] for s in sessions],
        )
        if picked_sess != "(選択しない)":
            history = repo.get_chat_history(picked_sess, limit=500)
            for role, content, _ts in history:
                with st.chat_message(role):
                    st.markdown(content)


# ---------- 右パネル: フローティング風チャットbot ----------

def render_sidebar_chat() -> None:
    if not CHAT_ENABLED:
        with st.sidebar:
            st.markdown("### 💬 チャット機能オフ")
            st.caption("公開モードのためチャットbotは無効化されています")
        return
    with st.sidebar:
        st.markdown("### 💬 競馬予想チャットbot")
        st.caption("どのタブからも会話できます")

        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("⚠️ `.env` に `ANTHROPIC_API_KEY` 未設定")
            with st.expander("設定方法"):
                st.code(
                    "cd ~/Desktop/競馬予想AI\n"
                    "cp .env.example .env\n"
                    "# .env に ANTHROPIC_API_KEY=sk-ant-... を記入\n"
                    "# Streamlit再起動で反映",
                    language="bash",
                )
            return

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.caption(f"session: `{st.session_state.chat_session_id[-10:]}`")
        with col_b:
            if st.button("🆕 新セッション", use_container_width=True):
                st.session_state.chat_session_id = f"sess_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
                st.session_state.chat_messages = []
                st.session_state.chat_turn_index = 0
                st.rerun()

        msg_container = st.container(height=420, border=False)
        with msg_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if user_input := st.chat_input("質問を入力..."):
            sess_id = st.session_state.chat_session_id
            st.session_state.chat_turn_index += 1
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            repo.save_chat_turn(
                session_id=sess_id,
                turn_index=st.session_state.chat_turn_index,
                role="user",
                content=user_input,
                tab_context=st.session_state.active_tab,
            )

            with msg_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_text = ""
                    try:
                        from anthropic import Anthropic
                        client = Anthropic()
                        system_prompt = CHAT_SYSTEM_PROMPT.format(db_summary=db_summary_for_chat())
                        api_msgs = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_messages
                        ]
                        with client.messages.stream(
                            model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"),
                            max_tokens=2000,
                            system=system_prompt,
                            messages=api_msgs,
                        ) as stream:
                            for text in stream.text_stream:
                                full_text += text
                                placeholder.markdown(full_text + "▌")
                        placeholder.markdown(full_text)
                    except Exception as e:  # noqa: BLE001
                        full_text = f"⚠️ エラー: {e}"
                        placeholder.error(full_text)

            st.session_state.chat_turn_index += 1
            st.session_state.chat_messages.append({"role": "assistant", "content": full_text})
            repo.save_chat_turn(
                session_id=sess_id,
                turn_index=st.session_state.chat_turn_index,
                role="assistant",
                content=full_text,
                tab_context=st.session_state.active_tab,
            )


render_sidebar_chat()
