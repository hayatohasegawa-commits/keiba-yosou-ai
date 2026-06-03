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

st.set_page_config(
    page_title="競馬予想AI",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded",
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
/* sidebarを右寄せフローティング風に */
section[data-testid="stSidebar"] {
    position: fixed;
    right: 0;
    left: auto;
    top: 0;
    height: 100vh;
    width: 380px !important;
    border-left: 1px solid rgba(128,128,128,0.25);
    box-shadow: -4px 0 12px rgba(0,0,0,0.08);
}
section[data-testid="stSidebar"] > div { width: 380px !important; }
/* メインエリアの右にsidebar分の余白 */
section.main > div.block-container { padding-right: 400px; max-width: 100%; }
/* sidebar内のチャット入力を見やすく */
section[data-testid="stSidebar"] [data-testid="stChatInput"] { background: rgba(0,0,0,0.02); }
/* 折りたたみボタンを右寄せに合わせる */
button[kind="header"][data-testid="baseButton-header"] { right: 380px; }
</style>
""", unsafe_allow_html=True)


# ---------- ヘルパー ----------

@st.cache_resource
def get_cache():
    return netkeiba.build_cache()


@st.cache_data(ttl=3600)
def fetch_race(race_id: str):
    cache = get_cache()
    data = netkeiba.fetch_and_parse_race(race_id, cache=cache)
    return data


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
    """5点 1着固定流し型をフォールバック生成。"""
    top = ranked.head(5).reset_index(drop=True)
    if len(top) < 4:
        return {"picks": [], "rationale": "出走馬不足", "confidence": 0.0}

    nums = top["horse_number"].astype(int).tolist() if "horse_number" in top.columns else list(range(1, len(top)+1))
    n = nums + [0]*5
    return {
        "picks": [
            f"{n[0]}-{n[1]}-{n[2]}",
            f"{n[0]}-{n[2]}-{n[1]}",
            f"{n[0]}-{n[1]}-{n[3]}",
            f"{n[0]}-{n[3]}-{n[1]}",
            f"{n[0]}-{n[2]}-{n[3]}",
        ],
        "rationale": (
            "暫定モデル: 確率上位5頭で1着固定流し型5点。"
            "1着 = 最高確率馬、2-3着は2-5位の組合せ。"
            "バックテストで12.7% (3点版より優位)。"
        ),
        "confidence": 0.40,
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

st.title("🐎 競馬予想AI デモ")
st.caption("任意のJRAレースに対して3連単3点を提案 / netkeibaから自動取得 / DB蓄積でPDCA")

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = f"sess_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_turn_index" not in st.session_state:
    st.session_state.chat_turn_index = 0
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "予測"

tab_predict, tab_db, tab_history, tab_eval, tab_chat = st.tabs([
    "🎯 予測", "📊 DBレース閲覧", "📜 予測履歴", "🏁 精度", "💬 会話ログ"
])


with tab_predict:
    st.subheader("レース指定")
    col1, col2 = st.columns([2, 1])
    with col1:
        race_id_input = st.text_input(
            "race_id (12桁)",
            value=race_id_mod.yasuda_kinen_2026_race_id(),
            help="例: 202505030211 (2025安田記念), 202605030211 (2026安田記念予想)",
        )
    with col2:
        st.write("")
        st.write("")
        go = st.button("予測実行", type="primary", use_container_width=True)

    with st.expander("🔍 プリセット検索 (DBから選ぶ)", expanded=False):
        races = db_races_df()
        if races.empty:
            st.info("DBにレースがありません。予測タブで取得すると蓄積されます。")
        else:
            races = races.copy()
            races["date_dt"] = pd.to_datetime(races["date"], errors="coerce")
            min_date = races["date_dt"].min().date()
            max_date = races["date_dt"].max().date()

            f1, f2 = st.columns([2, 3])
            with f1:
                default_from = max(min_date, max_date - timedelta(days=60))
                date_range = st.date_input(
                    "期間",
                    value=(default_from, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="preset_date",
                )
            with f2:
                keyword = st.text_input(
                    "🔎 検索 (レース名 / race_id)",
                    placeholder="例: 安田記念 / 202605030211",
                    key="preset_kw",
                )

            f3, f4 = st.columns(2)
            with f3:
                available_courses = sorted(races["course"].dropna().unique().tolist())
                courses_pick = st.multiselect("競馬場", available_courses, key="preset_course")
            with f4:
                available_grades = sorted([g for g in races["grade"].dropna().unique() if g])
                grades_pick = st.multiselect("グレード", available_grades, key="preset_grade")

            filtered = races.copy()
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                d_from, d_to = date_range
                filtered = filtered[
                    (filtered["date_dt"] >= pd.Timestamp(d_from))
                    & (filtered["date_dt"] <= pd.Timestamp(d_to))
                ]
            if courses_pick:
                filtered = filtered[filtered["course"].isin(courses_pick)]
            if grades_pick:
                filtered = filtered[filtered["grade"].isin(grades_pick)]
            if keyword:
                kw = keyword.strip()
                filtered = filtered[
                    filtered["race_name"].fillna("").str.contains(kw, case=False, na=False)
                    | filtered["race_id"].astype(str).str.contains(kw, na=False)
                ]

            filtered = filtered.sort_values("date_dt", ascending=False)
            st.caption(f"🎯 ヒット {len(filtered)} 件 / DB全 {len(races)} 件")

            if len(filtered) == 0:
                st.info("該当レースなし。条件を緩めてください")
            else:
                MAX_OPTIONS = 300
                display = filtered.head(MAX_OPTIONS)
                truncated = len(filtered) > MAX_OPTIONS
                opts = ["(選択しない)"] + [
                    f"{r.date} {r.race_name} ({r.race_id})" for r in display.itertuples()
                ]
                picked = st.selectbox(
                    f"レース選択（直近{len(display)}件{'・以下省略' if truncated else ''}）",
                    opts,
                    key="preset_select",
                )
                if picked != "(選択しない)":
                    race_id_input = picked.split("(")[-1].rstrip(")")
                    st.success(f"✓ 選択: **{race_id_input}** → このまま「予測実行」を押すとこのレースで予測します")

    if go and race_id_input:
        with st.spinner(f"netkeibaから取得中: {race_id_input}"):
            try:
                data = fetch_race(race_id_input)
            except Exception as e:  # noqa: BLE001
                st.error(f"取得失敗: {e}")
                st.stop()

        meta = data.meta
        st.success(f"取得成功: {meta.race_name or race_id_input}")

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

        st.subheader("📈 出走馬と3着内確率（暫定モデル）")
        ranked["馬メモ"] = ranked["horse_id"].apply(
            lambda hid: build_horse_memo(hid, current_race_id=race_id_input) if hid else ""
        )
        display_cols = [c for c in [
            "rank", "horse_number", "horse_name", "sex_age", "handicap",
            "jockey_name", "odds", "popularity", "agari_3f", "p_top3", "馬メモ"
        ] if c in ranked.columns]
        styled = ranked[display_cols].copy()
        if "p_top3" in styled.columns:
            styled["p_top3"] = styled["p_top3"].round(3)
        st.dataframe(ja(styled), use_container_width=True, hide_index=True)

        st.subheader("🎯 3連単5点予想")
        race_meta_dict = {
            "date": meta.date, "race_name": meta.race_name, "grade": meta.grade,
            "course": meta.course, "distance": meta.distance, "surface": meta.surface,
            "weather": meta.weather, "track_cond": meta.track_cond,
        }
        use_claude = st.checkbox(
            "Claude推論を使う (.envにANTHROPIC_API_KEY必要)",
            value=False,
            disabled=not CHAT_ENABLED or not os.environ.get("ANTHROPIC_API_KEY"),
            help="チェック時はClaudeに5点の組立を委任 (API料金あり)",
        )
        claude_pred = try_claude_trifecta(race_meta_dict, ranked) if use_claude else None

        if claude_pred and claude_pred.picks:
            tri = {
                "picks": claude_pred.picks,
                "rationale": claude_pred.rationale,
                "confidence": claude_pred.confidence,
            }
            tag = "Claude Opus 4.7 (5点)"
        else:
            tri = build_simple_trifecta(ranked)
            tag = f"暫定モデル / {prob_source}"

        odds_map = {}
        if st.checkbox("3連単オッズを取得して表示", value=True, key="show_odds",
                       help="netkeibaから現時点のオッズを取得（数秒）"):
            with st.spinner("オッズ取得中..."):
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
            m1, m2, m3 = st.columns(3)
            m1.metric("購入コスト", "500円", help="5点×100円")
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

        # 結果が既にあれば的中判定を表示
        actual_top3 = ranked.dropna(subset=["rank"]).head(3) if "rank" in ranked.columns else pd.DataFrame()
        if not actual_top3.empty and actual_top3["rank"].notna().any():
            actual_nums = actual_top3.sort_values("rank")["horse_number"].astype(int).tolist()
            if len(actual_nums) >= 3:
                actual_tri = f"{actual_nums[0]}-{actual_nums[1]}-{actual_nums[2]}"
                hit = actual_tri in set(tri["picks"])
                st.markdown("---")
                st.subheader("✅ 実績照合（過去レースの場合）")
                a, b = st.columns(2)
                a.metric("実際の着順 (1-2-3)", actual_tri)
                b.metric("5点的中", "🎉 的中!" if hit else "❌ 不的中")

        if st.button("予測をDB保存"):
            pred_id = repo.insert_prediction(
                race_id=race_id_input,
                picks=tri["picks"],
                rationale=tri["rationale"],
                confidence=tri["confidence"],
                model_version="lgbm_v1" if prob_source == "LightGBM" else ("claude_v1" if claude_pred else "simple_v1"),
            )
            st.success(f"DB保存完了 (prediction_id={pred_id})")


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
