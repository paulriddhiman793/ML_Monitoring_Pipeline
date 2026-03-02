import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


class Database:
    """Synchronous SQLite database wrapper for ML monitoring."""

    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema = f.read()
        with self._conn() as conn:
            conn.executescript(schema)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._conn() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.fetchall()

    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()

    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._conn() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()

    # ── Training Runs ──────────────────────────────────────────────────────

    def insert_training_run(self, data: Dict[str, Any]):
        self.execute(
            """INSERT OR REPLACE INTO training_runs
               (training_run_id, timestamp, model_version, model_type,
                hyperparameters, training_metrics, feature_importance, data_statistics)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                data["training_run_id"], data["timestamp"], data["model_version"],
                data.get("model_type"), json.dumps(data.get("hyperparameters", {})),
                json.dumps(data.get("training_metrics", {})),
                json.dumps(data.get("feature_importance", {})),
                json.dumps(data.get("data_statistics", {})),
            ),
        )

    def get_latest_training_run(self, model_version: str) -> Optional[Dict]:
        row = self.fetchone(
            "SELECT * FROM training_runs WHERE model_version=? ORDER BY timestamp DESC LIMIT 1",
            (model_version,),
        )
        return dict(row) if row else None

    # ── Predictions ────────────────────────────────────────────────────────

    def insert_prediction(self, data: Dict[str, Any]) -> str:
        self.execute(
            """INSERT INTO predictions
               (prediction_id, timestamp, model_version, input_features,
                prediction_value, prediction_time_ms, metadata, quality_flags)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                data["prediction_id"], data["timestamp"], data["model_version"],
                json.dumps(data["input_features"]),
                data.get("prediction_value"),
                data.get("prediction_time_ms"),
                json.dumps(data.get("metadata", {})),
                json.dumps(data.get("quality_flags", {})),
            ),
        )
        return data["prediction_id"]

    def get_predictions_with_gt(self, model_version: str, hours: int = 24) -> List[Dict]:
        rows = self.fetchall(
            """SELECT p.prediction_value, gt.actual_value,
                      gt.absolute_error, gt.squared_error, gt.percentage_error
               FROM predictions p
               JOIN ground_truth gt ON p.prediction_id = gt.prediction_id
               WHERE p.model_version=?
               AND datetime(gt.observation_timestamp) > datetime('now', ? || ' hours')""",
            (model_version, f"-{hours}"),
        )
        return [dict(r) for r in rows]

    def count_pending_ground_truth(self, model_version: str, hours: int = 24) -> int:
        row = self.fetchone(
            """SELECT COUNT(*) as cnt FROM predictions p
               LEFT JOIN ground_truth gt ON p.prediction_id = gt.prediction_id
               WHERE p.model_version=?
               AND datetime(p.timestamp) > datetime('now', ? || ' hours')
               AND gt.prediction_id IS NULL""",
            (model_version, f"-{hours}"),
        )
        return row["cnt"] if row else 0

    def get_recent_predictions(self, hours: int = 24) -> List[Dict]:
        rows = self.fetchall(
            """SELECT * FROM predictions
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')
               ORDER BY timestamp DESC""",
            (f"-{hours}",),
        )
        return [dict(r) for r in rows]

    # ── Ground Truth ───────────────────────────────────────────────────────

    def insert_ground_truth(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO ground_truth
               (prediction_id, actual_value, observation_timestamp,
                absolute_error, squared_error, percentage_error)
               VALUES (?,?,?,?,?,?)""",
            (
                data["prediction_id"], data["actual_value"],
                data["observation_timestamp"],
                data.get("absolute_error"), data.get("squared_error"),
                data.get("percentage_error"),
            ),
        )

    def get_prediction_for_gt(self, prediction_id: str) -> Optional[Dict]:
        row = self.fetchone(
            "SELECT prediction_value, timestamp FROM predictions WHERE prediction_id=?",
            (prediction_id,),
        )
        return dict(row) if row else None

    # ── Quality Logs ───────────────────────────────────────────────────────

    def insert_quality_log(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO data_quality_logs
               (timestamp, prediction_id, severity, issues, warnings, is_valid)
               VALUES (?,?,?,?,?,?)""",
            (
                data["timestamp"], data.get("prediction_id"),
                data.get("severity"), json.dumps(data.get("issues", [])),
                json.dumps(data.get("warnings", [])),
                1 if data.get("valid") else 0,
            ),
        )

    def get_quality_metrics(self, hours: int = 24) -> List[Dict]:
        rows = self.fetchall(
            """SELECT * FROM data_quality_logs
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')""",
            (f"-{hours}",),
        )
        return [dict(r) for r in rows]

    # ── Drift Logs ─────────────────────────────────────────────────────────

    def insert_drift_log(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO drift_logs
               (timestamp, drift_report, drift_severity, features_drifting)
               VALUES (?,?,?,?)""",
            (
                data["timestamp"], json.dumps(data["drift_report"]),
                data.get("drift_severity"),
                json.dumps(data.get("features_with_drift", [])),
            ),
        )

    def get_drift_logs(self, days: int = 7) -> List[Dict]:
        rows = self.fetchall(
            """SELECT * FROM drift_logs
               WHERE datetime(timestamp) > datetime('now', ? || ' days')
               ORDER BY timestamp""",
            (f"-{days}",),
        )
        return [dict(r) for r in rows]

    # ── Baseline Stats ─────────────────────────────────────────────────────

    def upsert_baseline_stats(self, model_version: str, feature_name: str, stats: Dict):
        self.execute(
            """INSERT OR REPLACE INTO baseline_statistics
               (model_version, feature_name, statistics)
               VALUES (?,?,?)""",
            (model_version, feature_name, json.dumps(stats)),
        )

    def get_baseline_stats(self, model_version: str) -> Dict[str, Any]:
        rows = self.fetchall(
            "SELECT feature_name, statistics FROM baseline_statistics WHERE model_version=?",
            (model_version,),
        )
        return {r["feature_name"]: json.loads(r["statistics"]) for r in rows}

    # ── Alerts ─────────────────────────────────────────────────────────────

    def insert_alert(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO alerts_log
               (timestamp, severity, alert_type, message, details)
               VALUES (?,?,?,?,?)""",
            (
                data["timestamp"], data.get("severity"),
                data.get("type"), data.get("message"),
                json.dumps(data.get("details", {})),
            ),
        )

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        rows = self.fetchall(
            """SELECT * FROM alerts_log
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')
               ORDER BY timestamp DESC""",
            (f"-{hours}",),
        )
        return [dict(r) for r in rows]

    # ── Misc ───────────────────────────────────────────────────────────────

    def insert_hourly_metric(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO hourly_metrics
               (timestamp, predictions_count, data_quality_rate, avg_latency_ms)
               VALUES (?,?,?,?)""",
            (data["timestamp"], data.get("predictions_count"),
             data.get("data_quality_rate"), data.get("avg_latency_ms")),
        )

    def insert_daily_evaluation(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO daily_evaluations
               (date, performance_metrics, drift_report, alerts, error_segments)
               VALUES (?,?,?,?,?)""",
            (
                data["date"], json.dumps(data.get("performance_metrics", {})),
                json.dumps(data.get("drift_report", {})),
                json.dumps(data.get("alerts", [])),
                json.dumps(data.get("error_segments", {})),
            ),
        )

    def insert_weekly_analysis(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO weekly_analyses
               (week_ending, importance_analysis, shap_analysis, drift_summary, retrain_decision)
               VALUES (?,?,?,?,?)""",
            (
                data["week_ending"],
                json.dumps(data.get("importance_analysis", {})),
                json.dumps(data.get("shap_analysis", {})),
                json.dumps(data.get("drift_summary", {})),
                json.dumps(data.get("retrain_decision", {})),
            ),
        )

    def insert_retraining_job(self, data: Dict[str, Any]):
        self.execute(
            """INSERT INTO retraining_jobs
               (job_id, scheduled_at, trigger_reasons, confidence, status)
               VALUES (?,?,?,?,?)""",
            (
                data["job_id"], data["scheduled_at"],
                json.dumps(data.get("trigger_reasons", [])),
                data.get("confidence"), data.get("status", "scheduled"),
            ),
        )