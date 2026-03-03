"""
pipeline/orchestrator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generic ML Model Monitoring Pipeline Orchestrator

Completely domain-agnostic — works with any ML backend regardless of:
  • Response format  (handled by adapter)
  • Feature types    (inferred automatically or supplied in config)
  • Model task       (regression, classification, scoring, ranking)
  • Prediction field (configured via adapter)

Configuration priority (highest → lowest):
  1. Explicit config dict passed to __init__
  2. Environment variables
  3. Defaults baked into this file

Usage (any backend):
    from pipeline.orchestrator import MLMonitoringPipeline

    pipeline = MLMonitoringPipeline({
        "ml_backend_url": "https://your-model-api.com",
        "adapter": {"type": "flat", "prediction_field": "score"},
    })
    pipeline.start()

    result = pipeline.predict({"feature_a": 1.2, "feature_b": "cat"})
    pipeline.submit_ground_truth(result["prediction_id"], actual_value=1.5)

    pipeline.stop()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from rich.console import Console
from rich.table import Table

from api.adapters import BaseAdapter, get_adapter
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
log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _infer_feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Automatically infer baseline statistics from a DataFrame of feature rows.
    Called the first time we have enough predictions to build a baseline.
    Handles both numerical and categorical columns.
    """
    stats: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            stats[col] = {
                "type": "numerical",
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 1.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "missing_rate": float(df[col].isna().mean()),
            }
        else:
            vc = series.value_counts(normalize=True)
            stats[col] = {
                "type": "categorical",
                "categories": vc.index.tolist(),
                "category_distribution": vc.to_dict(),
                "missing_rate": float(df[col].isna().mean()),
            }
    return stats


def _preds_to_df(predictions: List[Dict]) -> Optional[pd.DataFrame]:
    """Parse a list of prediction DB rows into a feature DataFrame."""
    rows = []
    for p in predictions:
        try:
            feats = json.loads(p.get("input_features") or "{}")
            if feats:
                rows.append(feats)
        except (json.JSONDecodeError, TypeError):
            continue
    return pd.DataFrame(rows) if rows else None


def _resolve_db_path(database_url: str) -> str:
    """Strip SQLAlchemy prefixes to get a plain file path for our DB wrapper."""
    return (
        database_url
        .replace("sqlite+aiosqlite:///./", "")
        .replace("sqlite:///./", "")
        .replace("sqlite:///", "")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MLMonitoringPipeline:
    """
    Generic ML model monitoring pipeline.

    Accepts any backend URL + adapter combination.
    No domain-specific logic lives here — all backend-specific
    knowledge belongs in the adapter (api/adapters.py).

    Lifecycle:
        __init__  → connect, load/build baseline, wire components
        start()   → launch background scheduler
        predict() → monitored prediction (quality check → backend → log)
        stop()    → graceful shutdown
    """

    # ── Construction ───────────────────────────────────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.running = False
        self._callbacks: List[Callable[[Dict], None]] = []

        self._setup_database()
        self._setup_backend()
        self._setup_baseline()
        self._setup_components()
        self._setup_scheduler()

    # ── Setup phases (called once during __init__) ─────────────────────────

    def _setup_database(self) -> None:
        db_url = self.config.get("database_url", "monitoring.db")
        self.db = Database(_resolve_db_path(db_url))
        console.print(f"[dim]Database: {_resolve_db_path(db_url)}[/dim]")

    def _setup_backend(self) -> None:
        """
        Build the adapter and backend client.
        The adapter is the only piece that knows about backend response shapes.
        """
        adapter_config: Dict[str, Any] = self.config.get("adapter", {"type": "flat"})
        self.adapter: BaseAdapter = get_adapter(adapter_config)

        self.backend = MLBackendClient(
            base_url=self.config["ml_backend_url"],
            adapter=self.adapter,
            api_key=self.config.get("ml_api_key"),
            predict_endpoint=self.config.get("predict_endpoint", "/predict"),
            model_info_endpoint=self.config.get("model_info_endpoint", "/model/info"),
            health_endpoint=self.config.get("health_endpoint", "/health"),
        )

        console.print(f"[cyan]Connecting → {self.config['ml_backend_url']}[/cyan]")

        # Probe the backend for model metadata
        model_info = self.backend.get_model_info()
        self.model_version: str = (
            self.config.get("model_version")
            or model_info.get("model_version", "v1")
        )
        self._model_info = model_info
        console.print(f"[green]✓ Model version: {self.model_version}[/green]")

        baseline_metrics = model_info.get("baseline_metrics", {})
        if baseline_metrics:
            readable = "  ".join(f"{k}={v:.4f}" for k, v in baseline_metrics.items() if v)
            if readable:
                console.print(f"[green]  {readable}[/green]")

    def _setup_baseline(self) -> None:
        """
        Resolve baseline feature statistics from the highest-priority source:
          1. Already in DB  (persisted from a prior run)
          2. Supplied in config as baseline_stats dict
          3. Fetched from the backend's /features or equivalent endpoint
          4. Deferred — will be built automatically after MIN_SAMPLES_FOR_BASELINE
             predictions have been logged (lazy bootstrap)

        Also seeds the training_runs table with backend metrics if first run.
        """
        MIN_SAMPLES_FOR_BASELINE = self.config.get("min_samples_for_baseline", 50)

        # Source 1: DB
        baseline_stats = self.db.get_baseline_stats(self.model_version)
        if baseline_stats:
            console.print(f"[green]✓ Baseline loaded from DB ({len(baseline_stats)} features)[/green]")
            self.baseline_stats = baseline_stats
            self._baseline_ready = True
            self._min_samples_for_baseline = MIN_SAMPLES_FOR_BASELINE
            self._seed_training_run_if_missing()
            return

        # Source 2: Config
        if self.config.get("baseline_stats"):
            baseline_stats = self.config["baseline_stats"]
            console.print(f"[green]✓ Baseline from config ({len(baseline_stats)} features)[/green]")
            self._persist_baseline(baseline_stats)
            self.baseline_stats = baseline_stats
            self._baseline_ready = True
            self._min_samples_for_baseline = MIN_SAMPLES_FOR_BASELINE
            self._seed_training_run_if_missing()
            return

        # Source 3: Backend /features endpoint (if adapter supports it)
        backend_baseline = self._fetch_baseline_from_backend()
        if backend_baseline:
            console.print(f"[green]✓ Baseline from backend ({len(backend_baseline)} features)[/green]")
            self._persist_baseline(backend_baseline)
            self.baseline_stats = backend_baseline
            self._baseline_ready = True
            self._min_samples_for_baseline = MIN_SAMPLES_FOR_BASELINE
            self._seed_training_run_if_missing()
            return

        # Source 4: Deferred — will be inferred from first N predictions
        console.print(
            f"[yellow]⚠ No baseline stats found. Will auto-infer after "
            f"{MIN_SAMPLES_FOR_BASELINE} predictions.[/yellow]"
        )
        self.baseline_stats = {}
        self._baseline_ready = False
        self._min_samples_for_baseline = MIN_SAMPLES_FOR_BASELINE

    def _setup_components(self) -> None:
        """Wire all monitoring components together."""
        thresholds = self.config.get("thresholds", {})

        self.logger = MLMonitoringLogger(self.db, self.model_version)

        self.quality_monitor = DataQualityMonitor(
            baseline_stats=self.baseline_stats,
            validation_rules=self.config.get("validation_rules", []),
        )

        self.performance_monitor = ModelPerformanceMonitor(
            self.db, self.model_version
        )

        self.drift_detector = DataDriftDetector(
            baseline_stats=self.baseline_stats,
            thresholds=thresholds.get("drift"),
        )

        self.explainability_monitor = ExplainabilityMonitor(
            model=self.config.get("model_object"),
            baseline_importance=self.config.get("baseline_feature_importance", {}),
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
            channels=self.config.get("alert_channels", ["console"]),
        )

        self.proxy = PredictionProxy(
            backend_client=self.backend,
            logger=self.logger,
            quality_monitor=self.quality_monitor,
        )

    def _setup_scheduler(self) -> None:
        self.scheduler = BackgroundScheduler(
            job_defaults={"coalesce": True, "max_instances": 1}
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background monitoring jobs and print status."""
        if self.running:
            console.print("[yellow]Pipeline already running.[/yellow]")
            return

        self.running = True
        interval_s = self.config.get("monitoring_interval_seconds", 3600)

        self.scheduler.add_job(
            self.run_hourly_checks, "interval",
            seconds=interval_s, id="hourly",
        )
        self.scheduler.add_job(
            self.run_daily_evaluation, "cron",
            hour=self.config.get("daily_eval_hour", 2), id="daily",
        )
        self.scheduler.add_job(
            self.run_weekly_analysis, "cron",
            day_of_week=self.config.get("weekly_eval_day", 0),
            hour=3, id="weekly",
        )
        self.scheduler.start()
        console.print("[bold green]✓ Monitoring pipeline started[/bold green]")
        self._print_status()

    def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if not self.running:
            return
        self.running = False
        self.scheduler.shutdown(wait=False)
        console.print("[yellow]Pipeline stopped.[/yellow]")

    def add_prediction_callback(self, fn: Callable[[Dict], None]) -> None:
        """
        Register a function called after every successful prediction.
        Useful for WebSocket feeds, custom loggers, or downstream triggers.
        fn receives the full prediction response dict.
        """
        self._callbacks.append(fn)
        self.proxy.add_prediction_callback(fn)

    # ── Public prediction API ──────────────────────────────────────────────

    def predict(
        self,
        features: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route a prediction through the full monitoring pipeline:
          data quality check → backend call → log → lazy baseline bootstrap
        Returns the backend response enriched with prediction_id and quality info.
        """
        result = self.proxy.predict(features, metadata)

        # Lazy baseline: once enough predictions exist, auto-build baseline
        if not self._baseline_ready:
            self._try_bootstrap_baseline()

        return result

    def submit_ground_truth(
        self, prediction_id: str, actual_value: float
    ) -> Dict[str, Any]:
        """
        Log the real observed outcome for a past prediction.
        This enables performance metrics (RMSE, MAE, R²) to be computed.
        actual_value must be on the same scale as the predicted value.
        """
        return self.proxy.submit_ground_truth(prediction_id, actual_value)

    def register_training_run(self, metadata: Dict[str, Any]) -> None:
        """
        Register a new training run and update baseline statistics.
        Call this whenever the model is retrained so the pipeline
        recalibrates its drift and performance baselines.

        metadata keys:
            run_id          (str)   unique identifier for this run
            model_type      (str)   e.g. "CatBoost", "XGBoost", "RandomForest"
            params          (dict)  hyperparameters used
            metrics         (dict)  training/validation metrics {"rmse": 0.1, "r2": 0.9}
            feature_importance (dict) {feature_name: importance_score}
            data_stats      (dict)  baseline stats per feature (same schema as baseline_stats)
        """
        self.logger.log_training_run(metadata)

        if "data_stats" in metadata:
            self._persist_baseline(metadata["data_stats"])
            new_baseline = self.db.get_baseline_stats(self.model_version)
            self._update_baseline(new_baseline)

        console.print("[green]✓ Training run registered — baseline updated.[/green]")

    # ── Scheduled evaluation jobs ──────────────────────────────────────────

    def run_hourly_checks(self) -> Dict[str, Any]:
        """
        Lightweight checks — run every hour (or configured interval).
        • Count predictions
        • Compute data quality rate
        • Track average latency
        • Fire latency / quality alerts
        """
        ts = datetime.now()
        console.print(f"[dim]{ts:%H:%M} — Hourly check[/dim]")

        preds = self.db.get_recent_predictions(hours=1)
        quality = self.quality_monitor.get_quality_summary(self.db, hours=1)

        avg_latency = (
            sum(p.get("prediction_time_ms") or 0 for p in preds) / len(preds)
            if preds else 0.0
        )

        hourly_snapshot = {
            "timestamp": ts.isoformat(),
            "predictions_count": len(preds),
            "data_quality_rate": quality.get("data_quality_rate", 1.0),
            "avg_latency_ms": avg_latency,
        }
        self.db.insert_hourly_metric(hourly_snapshot)
        self.alert_system.check_and_fire(hourly_snapshot)

        return hourly_snapshot

    def run_daily_evaluation(self) -> Dict[str, Any]:
        """
        Full daily evaluation — runs at 2 AM by default.
        • Performance metrics vs baseline (RMSE, MAE, R², bias)
        • Data drift detection per feature (PSI, KS test)
        • Error segment analysis
        • Alert generation
        """
        ts = datetime.now()
        console.print(f"[cyan]{ts:%Y-%m-%d} — Daily evaluation[/cyan]")

        # Performance (requires ground truth to have been submitted)
        perf = self.performance_monitor.calculate_metrics(hours=24)

        # Drift detection (requires predictions to exist)
        drift_report: Dict[str, Any] = {}
        if self._baseline_ready:
            recent_preds = self.db.get_recent_predictions(hours=24)
            feature_df = _preds_to_df(recent_preds)
            if feature_df is not None and not feature_df.empty:
                drift_report = self.drift_detector.detect_drift(feature_df)
                self.db.insert_drift_log({
                    "timestamp": ts.isoformat(),
                    "drift_report": drift_report,
                    "drift_severity": drift_report.get("drift_severity", "none"),
                    "features_with_drift": drift_report.get("features_with_drift", []),
                })
        else:
            console.print("[yellow]  Drift skipped — baseline not ready yet[/yellow]")

        # Error segment analysis
        segments = self.performance_monitor.analyze_error_segments(hours=24)

        # Alerts
        combined = {**perf, **drift_report}
        alerts = self.alert_system.check_and_fire(combined)

        # Persist
        self.db.insert_daily_evaluation({
            "date": ts.date().isoformat(),
            "performance_metrics": perf,
            "drift_report": drift_report,
            "alerts": alerts,
            "error_segments": segments,
        })

        self._log_daily_summary(perf, drift_report, alerts)

        return {
            "performance": perf,
            "drift": drift_report,
            "segments": segments,
            "alerts": alerts,
        }

    def run_weekly_analysis(self) -> Dict[str, Any]:
        """
        Deep weekly analysis — runs Monday 3 AM by default.
        • Feature importance shift (requires model_object in config)
        • SHAP value analysis (requires model_object + shap installed)
        • Drift trend over 7 days
        • Retraining decision (multi-signal)
        """
        ts = datetime.now()
        console.print(f"[magenta]{ts:%Y-%m-%d} — Weekly analysis[/magenta]")

        importance_analysis: Dict[str, Any] = {}
        shap_analysis: Dict[str, Any] = {}

        recent_preds = self.db.get_recent_predictions(hours=168)
        feature_df = _preds_to_df(recent_preds)
        if feature_df is not None and not feature_df.empty:
            importance_analysis = self.explainability_monitor.calculate_importance(feature_df)
            shap_analysis = self.explainability_monitor.explain_sample(feature_df)

        drift_summary = self.drift_detector.get_drift_summary(self.db, days=7)
        retrain_decision = self.retraining_trigger.should_retrain()

        if retrain_decision.get("should_retrain"):
            job = self.retraining_trigger.schedule_retraining(retrain_decision)
            console.print(
                f"[bold red]⚠ Retraining scheduled — "
                f"job_id={job.get('job_id')}  "
                f"confidence={retrain_decision.get('confidence')}[/bold red]"
            )

        self.db.insert_weekly_analysis({
            "week_ending": ts.date().isoformat(),
            "importance_analysis": importance_analysis,
            "shap_analysis": shap_analysis,
            "drift_summary": drift_summary,
            "retrain_decision": retrain_decision,
        })

        return {
            "importance_analysis": importance_analysis,
            "shap_analysis": shap_analysis,
            "drift_summary": drift_summary,
            "retrain_decision": retrain_decision,
        }

    # ── Dashboard snapshot ─────────────────────────────────────────────────

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Single call that returns everything the dashboard needs.
        Consumed by GET /dashboard in the FastAPI layer.
        """
        perf = self.performance_monitor.calculate_metrics(hours=24)
        quality = self.quality_monitor.get_quality_summary(self.db, hours=24)
        alerts = self.db.get_recent_alerts(hours=24)
        recent_preds = self.db.get_recent_predictions(hours=24)

        # Last drift report
        last_drift: Dict[str, Any] = {}
        drift_logs = self.db.get_drift_logs(days=1)
        if drift_logs:
            try:
                last_drift = json.loads(drift_logs[-1]["drift_report"])
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "pipeline_running": self.running,
            "baseline_ready": self._baseline_ready,
            "performance": perf,
            "data_quality": quality,
            "drift": last_drift,
            "recent_alerts": alerts,
            "predictions_last_24h": len(recent_preds),
            "backend_url": self.config["ml_backend_url"],
            "adapter_type": self.config.get("adapter", {}).get("type", "flat"),
        }

    # ── Baseline management ────────────────────────────────────────────────

    def update_baseline(self, new_baseline: Dict[str, Any]) -> None:
        """
        Manually replace baseline statistics (e.g. after a data pipeline change).
        Updates all components that depend on baseline in-place — no restart needed.
        """
        self._persist_baseline(new_baseline)
        self._update_baseline(new_baseline)
        console.print(f"[green]✓ Baseline updated ({len(new_baseline)} features)[/green]")

    def get_baseline(self) -> Dict[str, Any]:
        """Return current baseline statistics."""
        return self.baseline_stats.copy()

    # ── Private helpers ────────────────────────────────────────────────────

    def _persist_baseline(self, stats: Dict[str, Any]) -> None:
        """Write baseline stats to DB."""
        for feature_name, feature_stats in stats.items():
            self.db.upsert_baseline_stats(self.model_version, feature_name, feature_stats)

    def _update_baseline(self, stats: Dict[str, Any]) -> None:
        """Push new baseline into all live components without restarting."""
        self.baseline_stats = stats
        self.quality_monitor.baseline_stats = stats
        self.drift_detector.baseline_stats = stats
        self._baseline_ready = True

    def _seed_training_run_if_missing(self) -> None:
        """
        If no training run exists for this model version yet,
        seed one using metrics returned by the backend's model-info endpoint.
        This gives performance monitoring a baseline to compare against.
        """
        if self.db.get_latest_training_run(self.model_version):
            return

        baseline_metrics = self._model_info.get("baseline_metrics", {})
        if not baseline_metrics:
            return

        self.db.insert_training_run({
            "training_run_id": f"init_{self.model_version}",
            "timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
            "model_type": self._model_info.get("raw", {}).get("model_type", "unknown"),
            "hyperparameters": {},
            "training_metrics": baseline_metrics,
            "feature_importance": {},
            "data_statistics": self.baseline_stats,
        })
        console.print("[green]✓ Training run seeded from backend model-info[/green]")

    def _fetch_baseline_from_backend(self) -> Dict[str, Any]:
        """
        Ask the adapter if it can provide baseline stats from the backend.
        Adapters may optionally implement get_baseline_stats() by calling
        a /features or /schema endpoint. Returns empty dict if not supported.
        """
        try:
            if hasattr(self.adapter, "get_baseline_stats"):
                stats = self.adapter.get_baseline_stats(self.backend)
                if stats:
                    return stats
        except Exception as exc:
            console.print(f"[dim]Could not fetch baseline from backend: {exc}[/dim]")
        return {}

    def _try_bootstrap_baseline(self) -> None:
        """
        Lazy baseline bootstrap — called after each prediction until
        MIN_SAMPLES_FOR_BASELINE predictions have been collected.
        Once enough data exists, infer baseline stats from real traffic.
        """
        all_preds = self.db.get_recent_predictions(hours=24 * 30)  # up to 30 days
        if len(all_preds) < self._min_samples_for_baseline:
            return

        feature_df = _preds_to_df(all_preds)
        if feature_df is None or feature_df.empty:
            return

        inferred = _infer_feature_stats(feature_df)
        if not inferred:
            return

        self._persist_baseline(inferred)
        self._update_baseline(inferred)
        self._seed_training_run_if_missing()

        console.print(
            f"[bold green]✓ Baseline auto-built from {len(all_preds)} predictions "
            f"({len(inferred)} features)[/bold green]"
        )

    def _log_daily_summary(
        self,
        perf: Dict[str, Any],
        drift: Dict[str, Any],
        alerts: List[Dict],
    ) -> None:
        parts = []
        if "rmse" in perf:
            parts.append(f"RMSE={perf['rmse']:.4f}")
        if "r2" in perf:
            parts.append(f"R²={perf['r2']:.4f}")
        if "vs_baseline" in perf:
            chg = perf["vs_baseline"].get("rmse_change", 0)
            parts.append(f"RMSE_Δ={chg:+.1f}%")
        if drift:
            parts.append(f"Drift={drift.get('drift_severity', 'none')}")
            n = len(drift.get("features_with_drift", []))
            if n:
                parts.append(f"({n} features)")
        parts.append(f"Alerts={len(alerts)}")
        console.print(f"[cyan]  {' · '.join(parts)}[/cyan]")

    def _print_status(self) -> None:
        table = Table(
            title="ML Monitoring Pipeline",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Component", style="cyan", min_width=28)
        table.add_column("Status", style="green")
        table.add_column("Detail", style="dim")

        rows = [
            ("Logging",                "✓ Active",  "predictions · ground truth · quality"),
            ("Data Quality Monitor",   "✓ Active",  "missing · out-of-range · business rules"),
            ("Performance Monitor",    "✓ Active",  "RMSE · MAE · R² · bias · segments"),
            ("Drift Detector",         "✓ Active",  "PSI · KS test · chi² (categorical)"),
            ("Explainability Monitor", "✓ Active",  "feature importance · SHAP values"),
            ("Retraining Trigger",     "✓ Active",  "multi-signal · confidence scoring"),
            ("Alert System",           "✓ Active",  "console" + (" · slack" if self.config.get("slack_webhook") else "")),
        ]
        for name, status, detail in rows:
            table.add_row(name, status, detail)

        table.add_section()
        table.add_row("Model Version",   self.model_version,                           "")
        table.add_row("Backend",         self.config["ml_backend_url"],                "")
        table.add_row("Adapter",         self.config.get("adapter", {}).get("type", "flat"), "")
        table.add_row("Baseline",        "✓ Ready" if self._baseline_ready
                      else f"⏳ Waiting for {self._min_samples_for_baseline} predictions", "")
        table.add_row("DB",              _resolve_db_path(self.config.get("database_url", "monitoring.db")), "")

        console.print(table)