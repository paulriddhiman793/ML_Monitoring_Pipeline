"""
Main Pipeline Orchestrator — Dog Health Monitoring
"""
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from rich.console import Console
from rich.table import Table

from api.client import MLBackendClient
from api.proxy import PredictionProxy
from core.alerts import AlertSystem
from core.data_quality import DataQualityMonitor
from core.drift_detector import DataDriftDetector
from core.explainability import ExplainabilityMonitor
from core.logger import MLMonitoringLogger
from core.performance import ModelPerformanceMonitor
from core.retraining import RetrainingTrigger
from db.database import Database

console = Console()


# ── Dog backend baseline stats (used when backend is unreachable) ──────────
_DOG_FALLBACK_BASELINE = {
    "weight_kg": {
        "type": "numerical", "mean": 25.0, "std": 12.0,
        "min": 2.0, "max": 60.0, "missing_rate": 0.0,
    },
    "heart_rate_bpm": {
        "type": "numerical", "mean": 100.0, "std": 25.0,
        "min": 50.0, "max": 200.0, "missing_rate": 0.0,
    },
    "temperature_celsius": {
        "type": "numerical", "mean": 22.0, "std": 8.0,
        "min": -10.0, "max": 50.0, "missing_rate": 0.0,
    },
    "humidity_pct": {
        "type": "numerical", "mean": 60.0, "std": 20.0,
        "min": 20.0, "max": 100.0, "missing_rate": 0.0,
    },
    "speed_kmh": {
        "type": "numerical", "mean": 5.0, "std": 4.0,
        "min": 0.0, "max": 25.0, "missing_rate": 0.0,
    },
    "breed": {
        "type": "categorical",
        "categories": [
            "Labrador Retriever", "Bulldog", "Pug", "German Shepherd",
            "Golden Retriever", "Beagle", "Poodle", "Rottweiler",
            "Yorkshire Terrier", "Boxer",
        ],
        "category_distribution": {},
        "missing_rate": 0.0,
    },
}

_DOG_VALIDATION_RULES = [
    {
        "name": "weight_positive",
        "description": "Dog weight must be > 0",
        "condition": lambda f: f.get("weight_kg", 1) > 0,
    },
    {
        "name": "heart_rate_range",
        "description": "Heart rate must be 30–300 bpm",
        "condition": lambda f: 30 <= f.get("heart_rate_bpm", 100) <= 300,
    },
    {
        "name": "humidity_range",
        "description": "Humidity must be 0–100%",
        "condition": lambda f: 0 <= f.get("humidity_pct", 50) <= 100,
    },
    {
        "name": "speed_non_negative",
        "description": "Speed cannot be negative",
        "condition": lambda f: f.get("speed_kmh", 0) >= 0,
    },
]


class MLMonitoringPipeline:
    """
    Complete ML monitoring pipeline for the Dog Health backend.

    Usage:
        pipeline = MLMonitoringPipeline({"ml_backend_url": "https://..."})
        pipeline.start()
        result = pipeline.predict({
            "breed": "Labrador Retriever",
            "weight_kg": 30.0,
            "heart_rate_bpm": 85.0,
            "temperature_celsius": 25.0,
            "humidity_pct": 60.0,
            "speed_kmh": 5.5
        })
        pipeline.submit_ground_truth(result["prediction_id"], 0.35)
        pipeline.stop()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False

        # ── DB ────────────────────────────────────────────────────────────
        db_path = (config.get("database_url", "monitoring.db")
                   .replace("sqlite+aiosqlite:///./", "")
                   .replace("sqlite:///./", ""))
        self.db = Database(db_path)

        # ── Backend client ─────────────────────────────────────────────────
        self.backend = MLBackendClient(
            base_url=config["ml_backend_url"],
            api_key=config.get("ml_api_key"),
            predict_endpoint=config.get("predict_endpoint", "/predict"),
            model_info_endpoint=config.get("model_info_endpoint", "/model-info"),
        )

        # ── Fetch model metadata from backend ──────────────────────────────
        console.print(f"[cyan]Connecting to backend: {config['ml_backend_url']}[/cyan]")
        model_info = self.backend.get_model_info()
        self.model_version = (
            config.get("model_version")
            or model_info.get("model_version", "catboost_dog_v3.0.0")
        )
        console.print(f"[green]✓ Model: {self.model_version}[/green]")
        if model_info.get("val_r2"):
            console.print(f"[green]  Validation R²={model_info['val_r2']:.4f}  "
                          f"RMSE={model_info['val_rmse']:.4f}[/green]")

        # ── Build baseline stats ───────────────────────────────────────────
        # Priority: DB > config > auto-fetched from backend > fallback
        baseline_stats = self.db.get_baseline_stats(self.model_version)

        if not baseline_stats:
            baseline_stats = config.get("baseline_stats") or {}

        if not baseline_stats:
            # Try to infer from /features endpoint
            feat_info = self.backend.get_features()
            if feat_info.get("status") == "success":
                console.print("[cyan]Building baseline from /features endpoint…[/cyan]")
                baseline_stats = _DOG_FALLBACK_BASELINE.copy()
                # Verify required features match what backend declared
                required = feat_info.get("required_base_features", [])
                for feat in required:
                    if feat not in baseline_stats:
                        baseline_stats[feat] = {
                            "type": "numerical", "mean": 0, "std": 1,
                            "min": -1e6, "max": 1e6, "missing_rate": 0.0,
                        }
            else:
                console.print("[yellow]Using built-in dog baseline stats.[/yellow]")
                baseline_stats = _DOG_FALLBACK_BASELINE.copy()

            # Persist to DB
            for feat, stats in baseline_stats.items():
                self.db.upsert_baseline_stats(self.model_version, feat, stats)

        # Store baseline training metrics from model info if not yet in DB
        if not self.db.get_latest_training_run(self.model_version) and model_info.get("val_r2"):
            self.db.insert_training_run({
                "training_run_id": f"init_{self.model_version}",
                "timestamp": datetime.now().isoformat(),
                "model_version": self.model_version,
                "model_type": model_info.get("model_type", "CatBoost Regressor"),
                "hyperparameters": {},
                "training_metrics": {
                    "r2": model_info["val_r2"],
                    "rmse": model_info["val_rmse"],
                },
                "feature_importance": {},
                "data_statistics": baseline_stats,
            })
            console.print("[green]✓ Baseline metrics seeded from backend.[/green]")

        thresholds = config.get("thresholds", {})

        # ── Components ─────────────────────────────────────────────────────
        self.logger = MLMonitoringLogger(self.db, self.model_version)

        self.quality_monitor = DataQualityMonitor(
            baseline_stats=baseline_stats,
            validation_rules=config.get("validation_rules", _DOG_VALIDATION_RULES),
        )

        self.performance_monitor = ModelPerformanceMonitor(self.db, self.model_version)

        self.drift_detector = DataDriftDetector(
            baseline_stats=baseline_stats,
            thresholds=thresholds.get("drift"),
        )

        self.explainability_monitor = ExplainabilityMonitor(
            model=config.get("model_object"),
            baseline_importance=config.get("baseline_feature_importance", {}),
        )

        self.retraining_trigger = RetrainingTrigger(
            db=self.db,
            performance=self.performance_monitor,
            drift=self.drift_detector,
            thresholds=thresholds.get("retraining"),
        )

        self.alert_system = AlertSystem(
            db=self.db,
            thresholds=thresholds.get("alerts"),
            channels=config.get("alert_channels", ["console"]),
        )

        self.proxy = PredictionProxy(self.backend, self.logger, self.quality_monitor)

        # ── Scheduler ──────────────────────────────────────────────────────
        self.scheduler = BackgroundScheduler()

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self):
        self.running = True
        interval = self.config.get("monitoring_interval_seconds", 3600)
        self.scheduler.add_job(self.run_hourly_checks, "interval",
                               seconds=interval, id="hourly")
        self.scheduler.add_job(self.run_daily_evaluation, "cron",
                               hour=self.config.get("daily_eval_hour", 2), id="daily")
        self.scheduler.add_job(self.run_weekly_analysis, "cron",
                               day_of_week=self.config.get("weekly_eval_day", 0),
                               hour=3, id="weekly")
        self.scheduler.start()
        console.print("[bold green]✓ ML Monitoring Pipeline started[/bold green]")
        self._print_status()

    def stop(self):
        self.running = False
        self.scheduler.shutdown(wait=False)
        console.print("[yellow]Monitoring pipeline stopped.[/yellow]")

    def predict(self, features: Dict[str, Any],
                metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a dog health prediction through the full monitoring pipeline."""
        return self.proxy.predict(features, metadata)

    def submit_ground_truth(self, prediction_id: str,
                            actual_risk_score: float) -> Dict[str, Any]:
        """Submit the observed risk score for a past prediction."""
        return self.proxy.submit_ground_truth(prediction_id, actual_risk_score)

    def register_training_run(self, metadata: Dict[str, Any]):
        self.logger.log_training_run(metadata)
        if "data_stats" in metadata:
            for feat, stats in metadata["data_stats"].items():
                self.db.upsert_baseline_stats(self.model_version, feat, stats)
            new_baseline = self.db.get_baseline_stats(self.model_version)
            self.quality_monitor.baseline_stats = new_baseline
            self.drift_detector.baseline_stats = new_baseline
        console.print("[green]Training run registered and baseline updated.[/green]")

    # ── Scheduled Jobs ──────────────────────────────────────────────────────

    def run_hourly_checks(self):
        console.print(f"[dim]{datetime.now():%H:%M} — Hourly check…[/dim]")
        preds = self.db.get_recent_predictions(hours=1)
        quality = self.quality_monitor.get_quality_summary(self.db, hours=1)
        avg_latency = (
            sum(p.get("prediction_time_ms") or 0 for p in preds) / len(preds)
            if preds else 0
        )
        self.db.insert_hourly_metric({
            "timestamp": datetime.now().isoformat(),
            "predictions_count": len(preds),
            "data_quality_rate": quality.get("data_quality_rate", 1.0),
            "avg_latency_ms": avg_latency,
        })
        self.alert_system.check_and_fire({
            "data_quality_rate": quality.get("data_quality_rate", 1.0),
            "avg_latency_ms": avg_latency,
        })

    def run_daily_evaluation(self):
        console.print(f"[cyan]{datetime.now():%Y-%m-%d} — Daily evaluation…[/cyan]")
        perf = self.performance_monitor.calculate_metrics(hours=24)
        recent_preds = self.db.get_recent_predictions(hours=24)
        drift_report: Dict[str, Any] = {}
        if recent_preds:
            df = self._preds_to_df(recent_preds)
            if df is not None:
                drift_report = self.drift_detector.detect_drift(df)
                self.db.insert_drift_log({
                    "timestamp": datetime.now().isoformat(),
                    "drift_report": drift_report,
                    "drift_severity": drift_report.get("drift_severity", "none"),
                    "features_with_drift": drift_report.get("features_with_drift", []),
                })
        segments = self.performance_monitor.analyze_error_segments(hours=24)
        combined = {**perf, **drift_report}
        alerts = self.alert_system.check_and_fire(combined)
        self.db.insert_daily_evaluation({
            "date": datetime.now().date().isoformat(),
            "performance_metrics": perf,
            "drift_report": drift_report,
            "alerts": alerts,
            "error_segments": segments,
        })
        rmse = perf.get("rmse", "N/A")
        drift_sev = drift_report.get("drift_severity", "N/A")
        console.print(f"[cyan]  RMSE={rmse}  Drift={drift_sev}  Alerts={len(alerts)}[/cyan]")

    def run_weekly_analysis(self):
        console.print(f"[magenta]{datetime.now():%Y-%m-%d} — Weekly analysis…[/magenta]")
        recent_preds = self.db.get_recent_predictions(hours=168)
        imp: Dict[str, Any] = {}
        shap: Dict[str, Any] = {}
        if recent_preds:
            df = self._preds_to_df(recent_preds)
            if df is not None:
                imp = self.explainability_monitor.calculate_importance(df)
                shap = self.explainability_monitor.explain_sample(df)
        drift_summary = self.drift_detector.get_drift_summary(self.db, days=7)
        retrain = self.retraining_trigger.should_retrain()
        if retrain["should_retrain"]:
            job = self.retraining_trigger.schedule_retraining(retrain)
            console.print(f"[bold red]⚠ Retraining scheduled: {job.get('job_id')}[/bold red]")
        self.db.insert_weekly_analysis({
            "week_ending": datetime.now().date().isoformat(),
            "importance_analysis": imp,
            "shap_analysis": shap,
            "drift_summary": drift_summary,
            "retrain_decision": retrain,
        })

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _preds_to_df(self, preds: list) -> Optional[pd.DataFrame]:
        rows = []
        for p in preds:
            try:
                feats = json.loads(p.get("input_features") or "{}")
                rows.append(feats)
            except Exception:
                pass
        return pd.DataFrame(rows) if rows else None

    def get_dashboard_data(self) -> Dict[str, Any]:
        perf = self.performance_monitor.calculate_metrics(hours=24)
        quality = self.quality_monitor.get_quality_summary(self.db, hours=24)
        alerts = self.db.get_recent_alerts(hours=24)
        drift_logs = self.db.get_drift_logs(days=1)
        last_drift: Dict[str, Any] = {}
        if drift_logs:
            try:
                last_drift = json.loads(drift_logs[-1]["drift_report"])
            except Exception:
                pass
        recent_preds = self.db.get_recent_predictions(hours=24)
        return {
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "performance": perf,
            "data_quality": quality,
            "drift": last_drift,
            "recent_alerts": alerts,
            "predictions_last_24h": len(recent_preds),
            "pipeline_running": self.running,
        }

    def _print_status(self):
        table = Table(title="🐕 Dog Health ML Monitoring Pipeline")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        for comp in ["Logging", "Data Quality Monitor", "Performance Monitor",
                     "Drift Detector", "Explainability Monitor",
                     "Retraining Trigger", "Alert System"]:
            table.add_row(comp, "✓ Active")
        table.add_row("Model Version", self.model_version)
        table.add_row("Backend", self.config["ml_backend_url"])
        console.print(table)