"""Component 3: Model Performance Monitor"""
import json
import math
from typing import Any, Dict

import numpy as np

from db.database import Database


class ModelPerformanceMonitor:

    def __init__(self, db: Database, model_version: str):
        self.db = db
        self.model_version = model_version
        self._baseline = self._load_baseline()

    def _load_baseline(self) -> Dict:
        row = self.db.get_latest_training_run(self.model_version)
        if row:
            try:
                return json.loads(row.get("training_metrics") or "{}")
            except Exception:
                return {}
        return {}

    def calculate_metrics(self, hours: int = 24) -> Dict[str, Any]:
        results = self.db.get_predictions_with_gt(self.model_version, hours)
        if not results:
            return {
                "error": "No ground truth data available",
                "pending_ground_truth": self.db.count_pending_ground_truth(self.model_version, hours),
            }

        preds = np.array([r["prediction_value"] for r in results])
        actuals = np.array([r["actual_value"] for r in results])
        errors = preds - actuals

        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics: Dict[str, Any] = {
            "time_window_hours": hours,
            "sample_size": len(results),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "mape": float(np.mean(np.abs(errors / actuals * 100))),
            "r2": r2,
            "bias": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "max_error": float(np.max(np.abs(errors))),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "percentiles": {
                "p10": float(np.percentile(np.abs(errors), 10)),
                "p50": float(np.percentile(np.abs(errors), 50)),
                "p90": float(np.percentile(np.abs(errors), 90)),
                "p95": float(np.percentile(np.abs(errors), 95)),
                "p99": float(np.percentile(np.abs(errors), 99)),
            },
        }

        if self._baseline:
            base_rmse = self._baseline.get("rmse", 1)
            base_mae = self._baseline.get("mae", 1)
            metrics["vs_baseline"] = {
                "rmse_change": round((metrics["rmse"] - base_rmse) / base_rmse * 100, 2),
                "mae_change": round((metrics["mae"] - base_mae) / base_mae * 100, 2),
                "r2_change": round(metrics["r2"] - self._baseline.get("r2", 0), 4),
            }

        return metrics

    def analyze_error_segments(self, hours: int = 24) -> Dict[str, Any]:
        """Error analysis broken down by feature segments."""
        import pandas as pd

        rows = self.db.fetchall(
            """SELECT p.input_features, p.prediction_value, gt.actual_value, gt.absolute_error
               FROM predictions p
               JOIN ground_truth gt ON p.prediction_id = gt.prediction_id
               WHERE p.model_version=?
               AND datetime(gt.observation_timestamp) > datetime('now', ? || ' hours')""",
            (self.model_version, f"-{hours}"),
        )
        if not rows:
            return {"error": "No data available"}

        data = []
        for r in rows:
            feats = json.loads(r["input_features"])
            feats["absolute_error"] = r["absolute_error"]
            feats["actual_value"] = r["actual_value"]
            data.append(feats)

        df = pd.DataFrame(data)
        segments: Dict[str, Any] = {}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ("absolute_error", "actual_value")]

        for feat in num_cols:
            try:
                df[f"{feat}_q"] = pd.qcut(df[feat], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                grp = df.groupby(f"{feat}_q")["absolute_error"].agg(["mean", "median", "count"])
                segments[feat] = {
                    "type": "numerical",
                    "quartile_performance": grp.to_dict(),
                    "worst_quartile": str(grp["mean"].idxmax()),
                    "best_quartile": str(grp["mean"].idxmin()),
                }
            except Exception:
                continue

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if not c.endswith("_q")]
        for feat in cat_cols:
            grp = df.groupby(feat)["absolute_error"].agg(["mean", "median", "count"])
            grp = grp[grp["count"] >= 5]
            if len(grp):
                segments[feat] = {
                    "type": "categorical",
                    "category_performance": grp.to_dict(),
                    "worst_category": str(grp["mean"].idxmax()),
                    "best_category": str(grp["mean"].idxmin()),
                }

        high_error = self._identify_high_error_segments(segments)
        return {
            "time_window_hours": hours,
            "total_samples": len(df),
            "segment_analysis": segments,
            "high_error_segments": high_error,
        }

    def _identify_high_error_segments(self, segments: Dict, multiplier: float = 1.5) -> list:
        high = []
        for feat, analysis in segments.items():
            if analysis["type"] == "numerical":
                perf = analysis["quartile_performance"]["mean"]
                avg = np.mean(list(perf.values()))
                for q, err in perf.items():
                    if err > avg * multiplier:
                        high.append({"feature": feat, "segment": q, "avg_error": err,
                                     "overall_avg": avg, "ratio": round(err / avg, 2)})
            elif analysis["type"] == "categorical":
                perf = analysis["category_performance"]["mean"]
                avg = np.mean(list(perf.values()))
                for cat, err in perf.items():
                    if err > avg * multiplier:
                        high.append({"feature": feat, "segment": cat, "avg_error": err,
                                     "overall_avg": avg, "ratio": round(err / avg, 2)})
        return sorted(high, key=lambda x: x["ratio"], reverse=True)