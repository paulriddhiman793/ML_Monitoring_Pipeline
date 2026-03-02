"""Component 4: Data Drift Detector"""
import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from db.database import Database


class DataDriftDetector:

    def __init__(self, baseline_stats: Dict[str, Any], thresholds: Dict = None):
        self.baseline_stats = baseline_stats
        self.thresholds = thresholds or {
            "psi_threshold": 0.2,
            "ks_statistic_threshold": 0.1,
            "chi2_pvalue_threshold": 0.05,
            "mean_shift_threshold": 0.15,
            "std_shift_threshold": 0.20,
        }

    def detect_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(current_df),
            "features_with_drift": [],
            "drift_severity": "none",
            "detailed_analysis": {},
        }

        for feat in current_df.columns:
            if feat not in self.baseline_stats:
                continue
            baseline = self.baseline_stats[feat]
            vals = current_df[feat].dropna()
            if len(vals) == 0:
                continue

            feat_drift: Dict[str, Any] = {"feature": feat, "drift_detected": False, "metrics": {}}

            if baseline.get("type") == "numerical":
                metrics = self._numerical_drift(vals, baseline)
                feat_drift["metrics"] = metrics
                if (metrics["psi"] > self.thresholds["psi_threshold"] or
                        (metrics.get("ks_statistic") or 0) > self.thresholds["ks_statistic_threshold"] or
                        metrics["mean_shift_pct"] > self.thresholds["mean_shift_threshold"] * 100 or
                        metrics["std_shift_pct"] > self.thresholds["std_shift_threshold"] * 100):
                    feat_drift["drift_detected"] = True
                    feat_drift["drift_type"] = self._classify_drift(metrics)
                    report["features_with_drift"].append(feat)

            elif baseline.get("type") == "categorical":
                metrics = self._categorical_drift(vals, baseline)
                feat_drift["metrics"] = metrics
                if (metrics["psi"] > self.thresholds["psi_threshold"] or
                        (metrics.get("chi2_pvalue") or 1.0) < self.thresholds["chi2_pvalue_threshold"]):
                    feat_drift["drift_detected"] = True
                    feat_drift["new_categories"] = metrics.get("new_categories", [])
                    feat_drift["missing_categories"] = metrics.get("missing_categories", [])
                    report["features_with_drift"].append(feat)

            report["detailed_analysis"][feat] = feat_drift

        n_drift = len(report["features_with_drift"])
        n_total = len(current_df.columns)
        rate = n_drift / n_total if n_total else 0
        if rate > 0.30:
            report["drift_severity"] = "critical"
        elif rate > 0.15:
            report["drift_severity"] = "high"
        elif rate > 0.05:
            report["drift_severity"] = "medium"
        elif n_drift > 0:
            report["drift_severity"] = "low"

        return report

    def _numerical_drift(self, vals: pd.Series, baseline: Dict) -> Dict:
        arr = np.array(vals)
        cur_mean, cur_std = float(np.mean(arr)), float(np.std(arr))
        b_mean, b_std = baseline.get("mean", 0), baseline.get("std", 1)
        mean_shift = abs(cur_mean - b_mean) / b_mean * 100 if b_mean != 0 else 0
        std_shift = abs(cur_std - b_std) / b_std * 100 if b_std != 0 else 0
        psi = self._psi(arr, baseline)
        ks = self._ks(arr, baseline)
        return {
            "current_mean": cur_mean, "baseline_mean": b_mean, "mean_shift_pct": round(mean_shift, 2),
            "current_std": cur_std, "baseline_std": b_std, "std_shift_pct": round(std_shift, 2),
            "psi": round(psi, 4), "ks_statistic": round(ks, 4) if ks else None,
        }

    def _psi(self, arr: np.ndarray, baseline: Dict, buckets: int = 10) -> float:
        lo, hi = baseline.get("min", arr.min()), baseline.get("max", arr.max())
        if lo == hi:
            return 0.0
        bins = np.linspace(lo, hi, buckets + 1)
        b_counts = np.ones(buckets) / buckets
        c_hist, _ = np.histogram(arr, bins=bins)
        c_counts = c_hist / max(len(arr), 1)
        eps = 1e-4
        c_counts = np.where(c_counts == 0, eps, c_counts)
        b_counts = np.where(b_counts == 0, eps, b_counts)
        return float(np.sum((c_counts - b_counts) * np.log(c_counts / b_counts)))

    def _ks(self, arr: np.ndarray, baseline: Dict) -> float:
        try:
            from scipy import stats
            sample = np.random.normal(baseline.get("mean", 0), max(baseline.get("std", 1), 1e-6), len(arr))
            stat, _ = stats.ks_2samp(arr, sample)
            return float(stat)
        except Exception:
            return 0.0

    def _categorical_drift(self, vals: pd.Series, baseline: Dict) -> Dict:
        b_cats = set(baseline.get("categories", []))
        c_cats = set(vals.unique())
        c_dist = vals.value_counts(normalize=True).to_dict()
        b_dist = baseline.get("category_distribution", {})
        eps = 1e-4
        psi = sum(
            (c_dist.get(c, eps) - b_dist.get(c, eps)) * np.log(c_dist.get(c, eps) / b_dist.get(c, eps))
            for c in b_cats | c_cats
        )
        chi2_p = None
        try:
            from scipy import stats
            obs = [sum(vals == c) for c in b_cats]
            exp = [b_dist.get(c, 0) * len(vals) for c in b_cats]
            if len(obs) > 1 and sum(exp) > 0:
                _, chi2_p = stats.chisquare(obs, exp)
        except Exception:
            pass
        return {
            "psi": round(psi, 4),
            "chi2_pvalue": chi2_p,
            "new_categories": list(c_cats - b_cats),
            "missing_categories": list(b_cats - c_cats),
            "current_distribution": c_dist,
            "baseline_distribution": b_dist,
        }

    def _classify_drift(self, m: Dict) -> str:
        if m["mean_shift_pct"] > self.thresholds["mean_shift_threshold"] * 100:
            return "mean_shift_upward" if m["current_mean"] > m["baseline_mean"] else "mean_shift_downward"
        if m["std_shift_pct"] > self.thresholds["std_shift_threshold"] * 100:
            return "increased_variance" if m["current_std"] > m["baseline_std"] else "decreased_variance"
        return "distribution_shift"

    def get_drift_summary(self, db: Database, days: int = 7) -> Dict[str, Any]:
        logs = db.get_drift_logs(days)
        if not logs:
            return {"error": "No drift data available"}
        freq: Dict[str, int] = {}
        timeline = []
        for log in logs:
            report = json.loads(log["drift_report"])
            for f in report.get("features_with_drift", []):
                freq[f] = freq.get(f, 0) + 1
            timeline.append({
                "timestamp": log["timestamp"],
                "severity": report.get("drift_severity"),
                "n_drifting": len(report.get("features_with_drift", [])),
            })
        return {
            "analysis_period_days": days,
            "total_checks": len(logs),
            "top_drifting_features": sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10],
            "timeline": timeline,
        }