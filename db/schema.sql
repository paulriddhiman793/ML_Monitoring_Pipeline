CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_type TEXT,
    hyperparameters TEXT,
    training_metrics TEXT,
    feature_importance TEXT,
    data_statistics TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_features TEXT NOT NULL,
    prediction_value REAL,
    prediction_time_ms REAL,
    metadata TEXT,
    quality_flags TEXT
);

CREATE TABLE IF NOT EXISTS ground_truth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    actual_value REAL NOT NULL,
    observation_timestamp TEXT NOT NULL,
    absolute_error REAL,
    squared_error REAL,
    percentage_error REAL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
);

CREATE TABLE IF NOT EXISTS data_quality_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prediction_id TEXT,
    severity TEXT,
    issues TEXT,
    warnings TEXT,
    is_valid INTEGER
);

CREATE TABLE IF NOT EXISTS drift_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    drift_report TEXT NOT NULL,
    drift_severity TEXT,
    features_drifting TEXT
);

CREATE TABLE IF NOT EXISTS hourly_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predictions_count INTEGER,
    data_quality_rate REAL,
    avg_latency_ms REAL
);

CREATE TABLE IF NOT EXISTS daily_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    performance_metrics TEXT,
    drift_report TEXT,
    alerts TEXT,
    error_segments TEXT
);

CREATE TABLE IF NOT EXISTS weekly_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_ending TEXT NOT NULL,
    importance_analysis TEXT,
    shap_analysis TEXT,
    drift_summary TEXT,
    retrain_decision TEXT
);

CREATE TABLE IF NOT EXISTS retraining_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    scheduled_at TEXT NOT NULL,
    trigger_reasons TEXT,
    confidence TEXT,
    status TEXT DEFAULT 'scheduled'
);

CREATE TABLE IF NOT EXISTS alerts_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    severity TEXT,
    alert_type TEXT,
    message TEXT,
    details TEXT,
    acknowledged INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS baseline_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    statistics TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);