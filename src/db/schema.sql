-- 競馬予想AI SQLite schema

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS races (
    race_id      TEXT PRIMARY KEY,
    date         TEXT NOT NULL,
    course       TEXT NOT NULL,
    race_number  INTEGER,
    race_name    TEXT,
    grade        TEXT,
    distance     INTEGER,
    surface      TEXT,
    direction    TEXT,
    weather      TEXT,
    track_cond   TEXT,
    starters     INTEGER,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS horses (
    horse_id     TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    sex          TEXT,
    birth_year   INTEGER,
    sire         TEXT,
    dam          TEXT,
    trainer      TEXT,
    owner        TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS results (
    race_id        TEXT NOT NULL,
    horse_id       TEXT NOT NULL,
    jockey_id      TEXT,
    rank           INTEGER,
    post_position  INTEGER,
    horse_number   INTEGER,
    sex_age        TEXT,
    handicap       REAL,
    body_weight    REAL,
    body_weight_diff REAL,
    time           TEXT,
    margin         TEXT,
    agari_3f       REAL,
    odds           REAL,
    popularity     INTEGER,
    trainer_id     TEXT,
    trainer_name   TEXT,
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
    FOREIGN KEY (jockey_id) REFERENCES jockeys(jockey_id)
);

CREATE INDEX IF NOT EXISTS idx_results_horse ON results(horse_id);
CREATE INDEX IF NOT EXISTS idx_results_jockey ON results(jockey_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_course ON races(course);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    trifecta_1      TEXT,
    trifecta_2      TEXT,
    trifecta_3      TEXT,
    trifecta_4      TEXT,
    trifecta_5      TEXT,
    rationale       TEXT,
    confidence      REAL,
    model_version   TEXT,
    hit             INTEGER,
    payout          INTEGER,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id);

CREATE TABLE IF NOT EXISTS horse_probabilities (
    race_id        TEXT NOT NULL,
    horse_id       TEXT NOT NULL,
    p_top3         REAL,
    p_win          REAL,
    model_version  TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (race_id, horse_id, model_version)
);

-- チャットbotの会話履歴（リフレクションループ用）
CREATE TABLE IF NOT EXISTS chat_conversations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    turn_index   INTEGER NOT NULL,
    role         TEXT NOT NULL,    -- user / assistant
    content      TEXT NOT NULL,
    tab_context  TEXT,             -- どのタブで発言したか
    race_id      TEXT,             -- 関連レース（あれば）
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_conversations(created_at);

-- 会話からの洞察（週次バッチで生成）
CREATE TABLE IF NOT EXISTS chat_reflections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT DEFAULT (datetime('now')),
    period_from  TEXT,
    period_to    TEXT,
    insights     TEXT,             -- ユーザーの関心傾向・改善案
    prompt_diff  TEXT,             -- 推論プロンプト改善差分
    derived_n    INTEGER           -- 元になった会話turn数
);

-- ========== 統合ナレッジDB: 仮説・実験・セッション・判断 ==========

-- Claude Code セッション（私とユーザーの会話セッション）
CREATE TABLE IF NOT EXISTS code_sessions (
    session_id     TEXT PRIMARY KEY,
    started        TEXT,
    ended          TEXT,
    n_turns        INTEGER,
    n_tool_calls   INTEGER,
    topic_summary  TEXT,
    key_decisions  TEXT,   -- JSON array
    files_modified TEXT,   -- JSON array
    jsonl_path     TEXT,
    ingested_at    TEXT DEFAULT (datetime('now'))
);

-- AI が立てた仮説（予測戦略・特徴量・UX 等）
CREATE TABLE IF NOT EXISTS ai_hypotheses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT DEFAULT (datetime('now')),
    source       TEXT,        -- claude_code / vault / streamlit
    topic        TEXT,        -- prediction-strategy / feature / ux / data-collection
    hypothesis   TEXT NOT NULL,
    rationale    TEXT,
    confidence   REAL,
    validated_by INTEGER,     -- experiments.id
    outcome      TEXT,        -- proven / refuted / pending
    session_id   TEXT,
    FOREIGN KEY (session_id) REFERENCES code_sessions(session_id)
);

-- 実験記録（再学習・特徴量追加・プロンプト改善 等）
CREATE TABLE IF NOT EXISTS experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT DEFAULT (datetime('now')),
    type         TEXT,        -- retrain / feature / prompt / scrape
    description  TEXT,
    before_metric REAL,
    after_metric  REAL,
    delta         REAL,
    metric_name   TEXT,       -- auc / hit_rate_5 / logloss
    config_json   TEXT,
    notes         TEXT,
    session_id    TEXT
);

-- 設計判断ログ（なぜそう決めたか）
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT DEFAULT (datetime('now')),
    topic        TEXT,
    decision     TEXT NOT NULL,
    alternatives TEXT,
    reasoning    TEXT,
    confidence   REAL,
    revisited    INTEGER DEFAULT 0,
    session_id   TEXT,
    FOREIGN KEY (session_id) REFERENCES code_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_topic ON ai_hypotheses(topic);
CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(type);
CREATE INDEX IF NOT EXISTS idx_decisions_topic ON decisions(topic);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON code_sessions(started);
