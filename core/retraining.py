"""Component 6: Retraining Trigger System"""
from datetime import datetime
from typing import Any, Dict

from db.database import Database
from core.performance import ModelPerformanceMonitor
from core.drift_detector import DataDriftDetector


class RetrainingTrigger:

    def __init__(self, db: Database, performance: ModelPerformanceMonitor,
                 drift: DataDriftDetector, thresholds: Dict = None):
        self.db = db
        self.performance = performance
        self.drift = drift
        self.thresholds = thresholds or {
            "performance_degradation_pct": 10,
            "min_days_since_training": 30,
            "min_new_samples": 10000,
        }

    def should_retrain(self) -> Dict[str, Any]:
        decision: Dict[str, Any] = {
            "should_retrain": False,
            "confidence": "low",
            "reasons": [],
            "signals": {},
            "blocked_reason": None,
        }

        # Signal 1: Performance degradation
        perf = self.performance.calculate_metrics(hours=168)
        if "vs_baseline" in perf:
            deg = perf["vs_baseline"]["rmse_change"]
            decision["signals"]["rmse_change_pct"] = deg
            if deg > self.thresholds["performance_degradation_pct"]:
                decision["reasons"].append({
                    "type": "performance_degradation",
                    "severity": "high",
                    "detail": f"RMSE increased {deg:.1f}%",
                })
                decision["should_retrain"] = True

        # Signal 2: Data drift
        drift_summary = self.drift.get_drift_summary(self.db, days=7)
        if "timeline" in drift_summary and drift_summary["timeline"]:
            last = drift_summary["timeline"][-1]
            sev = last.get("severity", "none")
            decision["signals"]["drift_severity"] = sev
            if sev in ("critical", "high"):
                decision["reasons"].append({
                    "type": "data_drift", "severity": sev,
                    "detail": f"Drift severity: {sev}",
                })
                decision["should_retrain"] = True

        # Signal 3: Time since last training
        last_run = self.db.fetchone(
            "SELECT timestamp FROM training_runs ORDER BY timestamp DESC LIMIT 1"
        )
        if last_run:
            try:
                last_ts = datetime.fromisoformat(last_run["timestamp"])
                days_since = (datetime.now() - last_ts).days
                decision["signals"]["days_since_training"] = days_since
                if days_since > self.thresholds["min_days_since_training"]:
                    decision["reasons"].append({
                        "type": "time_based", "severity": "medium",
                        "detail": f"{days_since} days since last training",
                    })
            except Exception:
                pass

        # Signal 4: Sufficient new data
        if last_run:
            row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM ground_truth WHERE observation_timestamp > ?",
                (last_run["timestamp"],),
            )
            new_samples = row["cnt"] if row else 0
            decision["signals"]["new_samples"] = new_samples
            if new_samples < self.thresholds["min_new_samples"] and decision["should_retrain"]:
                decision["should_retrain"] = False
                decision["blocked_reason"] = f"Insufficient new data: only {new_samples} samples"

        # Signal 5: Error segment analysis
        seg = self.performance.analyze_error_segments(hours=168)
        n_bad = len(seg.get("high_error_segments", []))
        if n_bad > 0:
            decision["signals"]["high_error_segments"] = n_bad
            decision["reasons"].append({
                "type": "segment_performance", "severity": "medium",
                "detail": f"Poor performance on {n_bad} segments",
            })

        # Confidence
        high_reasons = [r for r in decision["reasons"] if r["severity"] == "high"]
        med_reasons = [r for r in decision["reasons"] if r["severity"] in ("high", "medium")]
        if len(high_reasons) >= 2:
            decision["confidence"] = "high"
        elif len(med_reasons) >= 2:
            decision["confidence"] = "medium"

        return decision

    def schedule_retraining(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if not decision["should_retrain"] or decision["confidence"] not in ("high", "medium"):
            return {"scheduled": False}
        job = {
            "job_id": f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "scheduled_at": datetime.now().isoformat(),
            "trigger_reasons": decision["reasons"],
            "confidence": decision["confidence"],
            "status": "scheduled",
        }
        self.db.insert_retraining_job(job)
        return job