"""Component 7: Alert System"""
import json
from datetime import datetime
from typing import Any, Dict, List

from db.database import Database
from rich.console import Console

console = Console()


class AlertSystem:

    def __init__(self, db: Database, thresholds: Dict = None, channels: List[str] = None):
        self.db = db
        self.channels = channels or ["console"]
        self.thresholds = thresholds or {
            "rmse_increase_pct": 15,
            "mae_increase_pct": 15,
            "r2_decrease": 0.05,
            "drift_severities": ["high", "critical"],
            "data_quality_rate": 0.90,
            "prediction_latency_ms": 100,
        }

    def check_and_fire(self, metrics: Dict[str, Any]) -> List[Dict]:
        alerts: List[Dict] = []

        # Performance
        if "vs_baseline" in metrics:
            if metrics["vs_baseline"].get("rmse_change", 0) > self.thresholds["rmse_increase_pct"]:
                alerts.append(self._alert("critical", "performance_degradation",
                    f"RMSE up {metrics['vs_baseline']['rmse_change']:.1f}%"))
            if metrics["vs_baseline"].get("r2_change", 0) < -self.thresholds["r2_decrease"]:
                alerts.append(self._alert("warning", "performance_degradation",
                    f"R2 dropped {abs(metrics['vs_baseline']['r2_change']):.3f}"))

        # Drift
        if metrics.get("drift_severity") in self.thresholds["drift_severities"]:
            sev = "critical" if metrics["drift_severity"] == "critical" else "warning"
            alerts.append(self._alert(sev, "data_drift",
                f"Drift detected: {metrics['drift_severity'].upper()}",
                {"features": metrics.get("features_with_drift", [])}))

        # Data quality
        if metrics.get("data_quality_rate", 1.0) < self.thresholds["data_quality_rate"]:
            alerts.append(self._alert("warning", "data_quality",
                f"Quality rate: {metrics['data_quality_rate']:.1%}"))

        # Latency
        if metrics.get("avg_latency_ms", 0) > self.thresholds["prediction_latency_ms"]:
            alerts.append(self._alert("warning", "latency",
                f"Avg latency {metrics['avg_latency_ms']:.0f}ms"))

        for alert in alerts:
            self.db.insert_alert(alert)
            self._dispatch(alert)

        return alerts

    def _alert(self, severity: str, alert_type: str, message: str,
               details: Dict = None) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "type": alert_type,
            "message": message,
            "details": details or {},
        }

    def _dispatch(self, alert: Dict):
        if "console" in self.channels:
            color = "red" if alert["severity"] == "critical" else "yellow"
            console.print(f"[{color}][ALERT][/{color}] [{alert['severity'].upper()}] "
                          f"{alert['type']}: {alert['message']}")
        # Extend here: email, Slack, PagerDuty, etc.