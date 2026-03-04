"""
pipeline/orchestrator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generic ML Model Monitoring Pipeline Orchestrator

Three operating modes:
  Mode 1 — Auto-simulation  : background thread fires synthetic predictions
  Mode 2 — Continuous feed  : browser UI drives a prediction loop
  Mode 3 — Production       : waits for POST /predict from your real app

Domain-agnostic — all backend-specific knowledge lives in the adapter.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from rich.console import Console
from rich.table import Table

from api.adapters import ADAPTER_REGISTRY, BaseAdapter
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


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _infer_feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Auto-build baseline statistics from a DataFrame of feature rows."""
    stats: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            stats[col] = {
                "type":         "numerical",
                "mean":         float(series.mean()),
                "std":          float(series.std()) if len(series) > 1 else 1.0,
                "min":          float(series.min()),
                "max":          float(series.max()),
                "missing_rate": float(df[col].isna().mean()),
            }
        else:
            vc = series.value_counts(normalize=True)
            stats[col] = {
                "type":                  "categorical",
                "categories":            vc.index.tolist(),
                "category_distribution": vc.to_dict(),
                "missing_rate":          float(df[col].isna().mean()),
            }
    return stats


def _preds_to_df(predictions: List[Dict]) -> Optional[pd.DataFrame]:
    rows = []
    for p in predictions:
        try:
            feats = json.loads(p.get("input_features") or "{}")
            if feats:
                rows.append(feats)
        except (json.JSONDecodeError, TypeError):
            continue
    return pd.DataFrame(rows) if rows else None


def _resolve_db_path(url: str) -> str:
    return (
        url
        .replace("sqlite+aiosqlite:///./", "")
        .replace("sqlite:///./", "")
        .replace("sqlite:///", "")
    )


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

class MLMonitoringPipeline:
    """
    Generic ML monitoring pipeline — works with any backend + adapter.

    Mode 1 — Auto-simulation
        pipeline.start_simulation(feature_generator, interval_seconds=5)
        Background thread fires predictions on a timer.

    Mode 2 — Continuous frontend feed
        pipeline.start_continuous_feed_mode()
        The dashboard UI's JavaScript loop calls POST /feed/predict repeatedly.
        pipeline.record_feed_prediction() increments the counter each time.

    Mode 3 — Production (default)
        pipeline.set_production_mode()
        Just start() and wait — your app calls POST /predict.
    """

    # ── Init ──────────────────────────────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config      = config
        self.running     = False
        self.active_mode: Optional[int] = None

        # Simulation state (Mode 1)
        self._sim_running: bool = False
        self._sim_count:   int  = 0
        self._sim_thread:  Optional[threading.Thread] = None

        # Continuous feed state (Mode 2)
        self._feed_armed:   bool = False
        self._feed_count:   int  = 0

        self._callbacks: List[Callable[[Dict], None]] = []

        self._setup_database()
        self._setup_backend()
        self._setup_baseline()
        self._setup_components()
        self._setup_scheduler()

    # ── Setup phases ──────────────────────────────────────────

    def _setup_database(self) -> None:
        db_path  = _resolve_db_path(self.config.get("database_url", "monitoring.db"))
        self.db  = Database(db_path)
        console.print(f"[dim]Database  : {db_path}[/dim]")

    def _setup_backend(self) -> None:
        adapter_cfg  = self.config.get("adapter", {"type": "flat"})
        adapter_type = adapter_cfg.get("type", "flat")

        cls = ADAPTER_REGISTRY.get(adapter_type)
        if cls is None:
            raise ValueError(
                f"Unknown adapter '{adapter_type}'. "
                f"Available: {list(ADAPTER_REGISTRY.keys())}"
            )
        # Only forward kwargs that the adapter's __init__ actually declares
        valid  = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in adapter_cfg.items() if k != "type" and k in valid}
        self.adapter: BaseAdapter = cls(**kwargs)

        self.backend = MLBackendClient(
            base_url=self.config["ml_backend_url"],
            adapter=self.adapter,
            api_key=self.config.get("ml_api_key"),
            predict_endpoint=self.config.get("predict_endpoint",    "/predict"),
            model_info_endpoint=self.config.get("model_info_endpoint", "/model/info"),
            health_endpoint=self.config.get("health_endpoint",      "/health"),
        )

        console.print(f"[cyan]Connecting → {self.config['ml_backend_url']}[/cyan]")
        self._model_info     = self.backend.get_model_info()
        self.model_version   = (
            self.config.get("model_version")
            or self._model_info.get("model_version", "v1")
        )
        console.print(f"[green]✓ Model    : {self.model_version}[/green]")
        bm = self._model_info.get("baseline_metrics", {})
        if bm:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in bm.items() if v)
            if parts:
                console.print(f"[green]  {parts}[/green]")

    def _setup_baseline(self) -> None:
        MIN = self.config.get("min_samples_for_baseline", 50)
        self._min_samples = MIN

        # Source 1: DB (fastest — used on every subsequent run)
        bs = self.db.get_baseline_stats(self.model_version)
        if bs:
            console.print(f"[green]✓ Baseline : DB ({len(bs)} features)[/green]")
            self.baseline_stats  = bs
            self._baseline_ready = True
            self._seed_training_run_if_missing()
            return

        # Source 2: Explicit config
        if self.config.get("baseline_stats"):
            bs = self.config["baseline_stats"]
            console.print(f"[green]✓ Baseline : config ({len(bs)} features)[/green]")
            self._persist_baseline(bs)
            self.baseline_stats  = bs
            self._baseline_ready = True
            self._seed_training_run_if_missing()
            return

        # Source 3: Adapter's get_baseline_stats() hook (optional)
        bs = self._fetch_baseline_from_backend()
        if bs:
            console.print(f"[green]✓ Baseline : backend ({len(bs)} features)[/green]")
            self._persist_baseline(bs)
            self.baseline_stats  = bs
            self._baseline_ready = True
            self._seed_training_run_if_missing()
            return

        # Source 4: Deferred — auto-built after MIN predictions
        console.print(
            f"[yellow]⚠ Baseline : deferred "
            f"(auto-builds after {MIN} predictions)[/yellow]"
        )
        self.baseline_stats  = {}
        self._baseline_ready = False

    def _setup_components(self) -> None:
        T = self.config.get("thresholds", {})
        self.logger               = MLMonitoringLogger(self.db, self.model_version)
        self.quality_monitor      = DataQualityMonitor(
            baseline_stats=self.baseline_stats,
            validation_rules=self.config.get("validation_rules", []),
        )
        self.performance_monitor  = ModelPerformanceMonitor(self.db, self.model_version)
        self.drift_detector       = DataDriftDetector(
            baseline_stats=self.baseline_stats,
            thresholds=T.get("drift"),
        )
        self.explainability_monitor = ExplainabilityMonitor(
            model=self.config.get("model_object"),
            baseline_importance=self.config.get("baseline_feature_importance", {}),
        )
        self.retraining_trigger   = RetrainingTrigger(
            db=self.db,
            performance=self.performance_monitor,
            drift=self.drift_detector,
            thresholds=T.get("retraining"),
        )
        self.alert_system         = AlertSystem(
            db=self.db,
            thresholds=T.get("alerts"),
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

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Start background monitoring scheduler."""
        if self.running:
            return
        self.running = True
        s = self.config.get("monitoring_interval_seconds", 3600)
        self.scheduler.add_job(self.run_hourly_checks,    "interval", seconds=s,     id="hourly")
        self.scheduler.add_job(self.run_daily_evaluation, "cron",     hour=self.config.get("daily_eval_hour", 2), id="daily")
        self.scheduler.add_job(self.run_weekly_analysis,  "cron",     day_of_week=self.config.get("weekly_eval_day", 0), hour=3, id="weekly")
        self.scheduler.start()
        console.print("[bold green]✓ Pipeline : started[/bold green]")
        self._print_status()

    def stop(self) -> None:
        """Gracefully stop the pipeline."""
        if not self.running:
            return
        self.stop_simulation()
        self.running = False
        self.scheduler.shutdown(wait=False)
        console.print("[yellow]Pipeline stopped.[/yellow]")

    def add_prediction_callback(self, fn: Callable[[Dict], None]) -> None:
        """Register a callback fired after every successful prediction."""
        self._callbacks.append(fn)
        self.proxy.add_prediction_callback(fn)

    # ── Prediction API ────────────────────────────────────────

    def predict(
        self,
        features: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route one prediction through quality check → backend → log."""
        result = self.proxy.predict(features, metadata)
        if not self._baseline_ready:
            self._try_bootstrap_baseline()
        return result

    def submit_ground_truth(self, prediction_id: str, actual_value: float) -> Dict[str, Any]:
        """Log the real outcome for a past prediction (enables RMSE/R²)."""
        return self.proxy.submit_ground_truth(prediction_id, actual_value)

    def register_training_run(self, metadata: Dict[str, Any]) -> None:
        """Register a new training run and refresh baselines."""
        self.logger.log_training_run(metadata)
        if "data_stats" in metadata:
            self._persist_baseline(metadata["data_stats"])
            self._update_baseline(self.db.get_baseline_stats(self.model_version))
        console.print("[green]✓ Training run registered — baseline updated.[/green]")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODE 1 — Auto-simulation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def start_simulation(
        self,
        feature_generator:  Callable[[], Dict[str, Any]],
        interval_seconds:   float = 5.0,
        max_predictions:    int   = 0,
        auto_ground_truth:  bool  = False,
        ground_truth_fn:    Optional[Callable[[Dict], float]] = None,
    ) -> None:
        """
        Start Mode 1 — auto-simulation.

        Args:
            feature_generator  : callable that returns a feature dict on each call
            interval_seconds   : seconds between each automatic prediction
            max_predictions    : stop after N predictions (0 = run forever)
            auto_ground_truth  : immediately submit simulated ground truth
            ground_truth_fn    : callable(result) → float (required if auto_ground_truth)
        """
        if self._sim_running:
            console.print("[yellow]Simulation already running.[/yellow]")
            return

        self.active_mode   = 1
        self._sim_running  = True
        self._sim_count    = 0

        def _loop() -> None:
            while self._sim_running:
                if max_predictions > 0 and self._sim_count >= max_predictions:
                    self._sim_running = False
                    console.print(f"[green]▶ Simulation complete — {self._sim_count} predictions sent.[/green]")
                    break
                try:
                    features = feature_generator()
                    result   = self.predict(features)
                    self._sim_count += 1

                    if auto_ground_truth and ground_truth_fn and result.get("prediction_id"):
                        actual = ground_truth_fn(result)
                        self.submit_ground_truth(result["prediction_id"], actual)

                    console.print(
                        f"[dim]  [sim {self._sim_count:04d}] "
                        f"pred={result.get('prediction')}  "
                        f"id=...{str(result.get('prediction_id',''))[-8:]}[/dim]"
                    )
                except Exception as exc:
                    console.print(f"[red]Simulation error: {exc}[/red]")

                time.sleep(interval_seconds)

        self._sim_thread = threading.Thread(target=_loop, daemon=True, name="ml-sim-loop")
        self._sim_thread.start()

    def stop_simulation(self) -> Dict[str, Any]:
        """Stop Mode 1 simulation."""
        if not self._sim_running:
            return {"stopped": False, "reason": "not running"}
        self._sim_running = False
        count = self._sim_count
        console.print(f"[yellow]■ Simulation stopped — {count} predictions sent.[/yellow]")
        return {"stopped": True, "total_predictions_sent": count}

    def simulation_status(self) -> Dict[str, Any]:
        return {
            "running":          self._sim_running,
            "predictions_sent": self._sim_count,
            "mode":             self.active_mode,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODE 2 — Continuous frontend feed
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def start_continuous_feed_mode(self) -> Dict[str, Any]:
        """
        Arm Mode 2. The browser UI will call POST /feed/predict in a loop.
        This method just flips the flag and resets the counter — no thread needed.
        """
        self.active_mode  = 2
        self._feed_armed  = True
        self._feed_count  = 0
        console.print("[bold yellow]▶ Mode 2 — Continuous feed mode armed.[/bold yellow]")
        return {"mode": 2, "armed": True}

    def stop_continuous_feed_mode(self) -> Dict[str, Any]:
        self._feed_armed = False
        console.print("[yellow]■ Continuous feed stopped.[/yellow]")
        return {"mode": 2, "armed": False, "predictions_sent": self._feed_count}

    def record_feed_prediction(self) -> None:
        """Called by the API layer after each /feed/predict request."""
        if self._feed_armed:
            self._feed_count += 1

    def continuous_feed_status(self) -> Dict[str, Any]:
        return {
            "armed":            self._feed_armed,
            "predictions_sent": self._feed_count,
            "mode":             self.active_mode,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODE 3 — Production
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def set_production_mode(self) -> Dict[str, Any]:
        """Arm Mode 3 — just sets the flag, no extra work needed."""
        self.active_mode = 3
        console.print("[bold magenta]▶ Mode 3 — Production mode active.[/bold magenta]")
        return {"mode": 3}

    # ── Scheduled jobs ────────────────────────────────────────

    def run_hourly_checks(self) -> Dict[str, Any]:
        ts   = datetime.now()
        console.print(f"[dim]{ts:%H:%M} — Hourly check[/dim]")
        preds   = self.db.get_recent_predictions(hours=1)
        quality = self.quality_monitor.get_quality_summary(self.db, hours=1)
        avg_lat = (
            sum(p.get("prediction_time_ms") or 0 for p in preds) / len(preds)
            if preds else 0.0
        )
        snap = {
            "timestamp":         ts.isoformat(),
            "predictions_count": len(preds),
            "data_quality_rate": quality.get("data_quality_rate", 1.0),
            "avg_latency_ms":    avg_lat,
        }
        self.db.insert_hourly_metric(snap)
        self.alert_system.check_and_fire(snap)
        return snap

    def run_daily_evaluation(self) -> Dict[str, Any]:
        ts   = datetime.now()
        console.print(f"[cyan]{ts:%Y-%m-%d} — Daily evaluation[/cyan]")
        perf = self.performance_monitor.calculate_metrics(hours=24)

        drift_report: Dict[str, Any] = {}
        if self._baseline_ready:
            df = _preds_to_df(self.db.get_recent_predictions(hours=24))
            if df is not None and not df.empty:
                drift_report = self.drift_detector.detect_drift(df)
                self.db.insert_drift_log({
                    "timestamp":           ts.isoformat(),
                    "drift_report":        drift_report,
                    "drift_severity":      drift_report.get("drift_severity", "none"),
                    "features_with_drift": drift_report.get("features_with_drift", []),
                })
        else:
            console.print("[yellow]  Drift skipped — baseline not ready[/yellow]")

        segments = self.performance_monitor.analyze_error_segments(hours=24)
        alerts   = self.alert_system.check_and_fire({**perf, **drift_report})
        self.db.insert_daily_evaluation({
            "date":                ts.date().isoformat(),
            "performance_metrics": perf,
            "drift_report":        drift_report,
            "alerts":              alerts,
            "error_segments":      segments,
        })
        self._log_daily_summary(perf, drift_report, alerts)
        return {"performance": perf, "drift": drift_report, "segments": segments, "alerts": alerts}

    def run_weekly_analysis(self) -> Dict[str, Any]:
        ts   = datetime.now()
        console.print(f"[magenta]{ts:%Y-%m-%d} — Weekly analysis[/magenta]")
        imp = shap = {}
        df  = _preds_to_df(self.db.get_recent_predictions(hours=168))
        if df is not None and not df.empty:
            imp  = self.explainability_monitor.calculate_importance(df)
            shap = self.explainability_monitor.explain_sample(df)

        drift_summary    = self.drift_detector.get_drift_summary(self.db, days=7)
        retrain_decision = self.retraining_trigger.should_retrain()

        if retrain_decision.get("should_retrain"):
            job = self.retraining_trigger.schedule_retraining(retrain_decision)
            console.print(
                f"[bold red]⚠ Retraining scheduled: {job.get('job_id')}  "
                f"confidence={retrain_decision.get('confidence')}[/bold red]"
            )

        self.db.insert_weekly_analysis({
            "week_ending":         ts.date().isoformat(),
            "importance_analysis": imp,
            "shap_analysis":       shap,
            "drift_summary":       drift_summary,
            "retrain_decision":    retrain_decision,
        })
        return {
            "importance_analysis": imp,
            "shap_analysis":       shap,
            "drift_summary":       drift_summary,
            "retrain_decision":    retrain_decision,
        }

    # ── Dashboard snapshot ────────────────────────────────────

    def get_dashboard_data(self) -> Dict[str, Any]:
        perf    = self.performance_monitor.calculate_metrics(hours=24)
        quality = self.quality_monitor.get_quality_summary(self.db, hours=24)
        alerts  = self.db.get_recent_alerts(hours=24)
        recent  = self.db.get_recent_predictions(hours=24)

        last_drift: Dict[str, Any] = {}
        logs = self.db.get_drift_logs(days=1)
        if logs:
            try:
                last_drift = json.loads(logs[-1]["drift_report"])
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            "model_version":        self.model_version,
            "timestamp":            datetime.now().isoformat(),
            "pipeline_running":     self.running,
            "baseline_ready":       self._baseline_ready,
            "active_mode":          self.active_mode,
            "mode_label":           {1: "Auto-Simulation", 2: "Continuous Feed", 3: "Production"}.get(self.active_mode, "Not set"),
            "simulation_status":    self.simulation_status(),
            "feed_status":          self.continuous_feed_status(),
            "performance":          perf,
            "data_quality":         quality,
            "drift":                last_drift,
            "recent_alerts":        alerts,
            "predictions_last_24h": len(recent),
            "backend_url":          self.config["ml_backend_url"],
            "adapter_type":         self.config.get("adapter", {}).get("type", "flat"),
        }

    # ── Baseline management ───────────────────────────────────

    def update_baseline(self, new_baseline: Dict[str, Any]) -> None:
        """Hot-swap baseline without restarting."""
        self._persist_baseline(new_baseline)
        self._update_baseline(new_baseline)
        console.print(f"[green]✓ Baseline updated ({len(new_baseline)} features)[/green]")

    def get_baseline(self) -> Dict[str, Any]:
        return self.baseline_stats.copy()

    def _persist_baseline(self, stats: Dict[str, Any]) -> None:
        for name, feat_stats in stats.items():
            self.db.upsert_baseline_stats(self.model_version, name, feat_stats)

    def _update_baseline(self, stats: Dict[str, Any]) -> None:
        self.baseline_stats                 = stats
        self.quality_monitor.baseline_stats = stats
        self.drift_detector.baseline_stats  = stats
        self._baseline_ready                = True

    def _seed_training_run_if_missing(self) -> None:
        if self.db.get_latest_training_run(self.model_version):
            return
        bm = self._model_info.get("baseline_metrics", {})
        if not bm:
            return
        self.db.insert_training_run({
            "training_run_id":    f"init_{self.model_version}",
            "timestamp":          datetime.now().isoformat(),
            "model_version":      self.model_version,
            "model_type":         self._model_info.get("raw", {}).get("model_type", "unknown"),
            "hyperparameters":    {},
            "training_metrics":   bm,
            "feature_importance": {},
            "data_statistics":    self.baseline_stats,
        })
        console.print("[green]✓ Training run seeded from backend[/green]")

    def _fetch_baseline_from_backend(self) -> Dict[str, Any]:
        try:
            if hasattr(self.adapter, "get_baseline_stats"):
                return self.adapter.get_baseline_stats(self.backend) or {}
        except Exception as exc:
            console.print(f"[dim]Could not fetch baseline from backend: {exc}[/dim]")
        return {}

    def _try_bootstrap_baseline(self) -> None:
        """Lazy bootstrap: build baseline from first N predictions."""
        all_preds = self.db.get_recent_predictions(hours=24 * 30)
        if len(all_preds) < self._min_samples:
            return
        df = _preds_to_df(all_preds)
        if df is None or df.empty:
            return
        inferred = _infer_feature_stats(df)
        if not inferred:
            return
        self._persist_baseline(inferred)
        self._update_baseline(inferred)
        self._seed_training_run_if_missing()
        console.print(
            f"[bold green]✓ Baseline auto-built from {len(all_preds)} predictions "
            f"({len(inferred)} features)[/bold green]"
        )

    # ── Logging ───────────────────────────────────────────────

    def _log_daily_summary(self, perf, drift, alerts) -> None:
        parts = []
        if "rmse" in perf:      parts.append(f"RMSE={perf['rmse']:.4f}")
        if "r2" in perf:        parts.append(f"R²={perf['r2']:.4f}")
        if "vs_baseline" in perf:
            chg = perf["vs_baseline"].get("rmse_change", 0)
            parts.append(f"RMSE_Δ={chg:+.1f}%")
        if drift:
            parts.append(f"Drift={drift.get('drift_severity','none')}")
            n = len(drift.get("features_with_drift", []))
            if n: parts.append(f"({n} features)")
        parts.append(f"Alerts={len(alerts)}")
        console.print(f"[cyan]  {' · '.join(parts)}[/cyan]")

    def _print_status(self) -> None:
        tbl = Table(
            title="ML Monitoring Pipeline",
            show_header=True,
            header_style="bold cyan",
            box=None,
        )
        tbl.add_column("Component",  style="cyan",  min_width=26)
        tbl.add_column("Status",     style="green", min_width=10)
        tbl.add_column("Detail",     style="dim")
        for name, detail in [
            ("Logging",                "predictions · ground truth · quality"),
            ("Data Quality Monitor",   "missing · out-of-range · business rules"),
            ("Performance Monitor",    "RMSE · MAE · R² · bias · segments"),
            ("Drift Detector",         "PSI · KS · chi² (categorical)"),
            ("Explainability Monitor", "feature importance · SHAP"),
            ("Retraining Trigger",     "multi-signal · confidence scoring"),
            ("Alert System",           "console"),
        ]:
            tbl.add_row(name, "✓ Active", detail)
        tbl.add_section()
        tbl.add_row("Model",    self.model_version, "")
        tbl.add_row("Backend",  self.config["ml_backend_url"], "")
        tbl.add_row("Adapter",  self.config.get("adapter", {}).get("type", "flat"), "")
        tbl.add_row(
            "Baseline",
            "✓ Ready" if self._baseline_ready
            else f"⏳ needs {self._min_samples} predictions",
            "",
        )
        console.print(tbl)