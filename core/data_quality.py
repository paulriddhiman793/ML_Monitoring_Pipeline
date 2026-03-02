"""Component 2: Data Quality Monitor"""
import math
from typing import Any, Dict

from db.database import Database


class DataQualityMonitor:

    def __init__(self, baseline_stats: Dict[str, Any], validation_rules: list = None):
        self.baseline_stats = baseline_stats
        self.rules = validation_rules or []

    def validate_input(self, features: Dict[str, Any]) -> Dict[str, Any]:
        report = {"valid": True, "issues": [], "warnings": [], "severity": "none"}

        # Missing features
        missing = set(self.baseline_stats.keys()) - set(features.keys())
        if missing:
            report["valid"] = False
            report["issues"].append({"type": "missing_features", "features": list(missing), "severity": "critical"})
            report["severity"] = "critical"

        # Null / NaN values
        for k, v in features.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                training_null_rate = self.baseline_stats.get(k, {}).get("missing_rate", 0)
                if training_null_rate < 0.05:
                    report["issues"].append({"type": "unexpected_null", "feature": k, "severity": "high"})
                    if report["severity"] not in ("critical",):
                        report["severity"] = "high"

        # Out-of-range / extreme values
        for fname, value in features.items():
            if fname not in self.baseline_stats:
                continue
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            stats = self.baseline_stats[fname]

            if stats.get("type") == "numerical":
                lo, hi = stats.get("min", float("-inf")), stats.get("max", float("inf"))
                buf = (hi - lo) * 0.1
                if value < lo - buf or value > hi + buf:
                    report["warnings"].append({
                        "type": "out_of_range", "feature": fname,
                        "value": value, "training_range": [lo, hi], "severity": "medium",
                    })
                    if report["severity"] == "none":
                        report["severity"] = "medium"

                mean, std = stats.get("mean", 0), stats.get("std", 1)
                z = abs((value - mean) / std) if std > 0 else 0
                if z > 3:
                    report["warnings"].append({
                        "type": "extreme_value", "feature": fname,
                        "value": value, "z_score": round(z, 2), "severity": "medium",
                    })

            elif stats.get("type") == "categorical":
                cats = set(stats.get("categories", []))
                if cats and value not in cats:
                    report["warnings"].append({
                        "type": "unseen_category", "feature": fname,
                        "value": value, "severity": "medium",
                    })
                    if report["severity"] == "none":
                        report["severity"] = "medium"

        # Business rules
        for rule in self.rules:
            try:
                if not rule["condition"](features):
                    report["valid"] = False
                    report["issues"].append({
                        "type": "business_rule_violation",
                        "rule": rule["name"],
                        "description": rule.get("description", ""),
                        "severity": "critical",
                    })
                    report["severity"] = "critical"
            except Exception:
                pass

        return report

    def get_quality_summary(self, db: Database, hours: int = 24) -> Dict[str, Any]:
        logs = db.get_quality_metrics(hours)
        if not logs:
            return {"error": "No data in time window"}
        total = len(logs)
        import json
        critical = sum(1 for v in logs if v["severity"] == "critical")
        high = sum(1 for v in logs if v["severity"] == "high")
        medium = sum(1 for v in logs if v["severity"] == "medium")
        issue_counts: Dict[str, int] = {}
        for log in logs:
            for issue in json.loads(log.get("issues") or "[]") + json.loads(log.get("warnings") or "[]"):
                t = issue.get("type", "unknown")
                issue_counts[t] = issue_counts.get(t, 0) + 1
        return {
            "time_window_hours": hours,
            "total_validations": total,
            "data_quality_rate": round((total - critical) / total, 4),
            "critical_issues": critical,
            "high_issues": high,
            "medium_issues": medium,
            "issue_breakdown": issue_counts,
            "top_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        }