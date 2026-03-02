"""Component 5: Feature Importance & Explainability Monitor"""
from typing import Any, Dict

import numpy as np
import pandas as pd


class ExplainabilityMonitor:

    def __init__(self, model=None, baseline_importance: Dict[str, float] = None):
        self.model = model
        self.baseline_importance = baseline_importance or {}

    def calculate_importance(self, recent_df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "No model loaded"}
        if len(recent_df) < 100:
            return {"error": "Insufficient data (need >= 100 samples)"}
        try:
            feature_names = recent_df.columns.tolist()
            raw = self.model.get_feature_importance()
            current = dict(zip(feature_names, raw))
        except Exception as e:
            return {"error": f"Cannot get feature importance: {e}"}

        changes: Dict[str, Any] = {}
        for feat, cur in current.items():
            base = self.baseline_importance.get(feat, 0)
            change_pct = ((cur - base) / base * 100) if base > 0 else (100 if cur > 0 else 0)
            changes[feat] = {"current": cur, "baseline": base, "change_pct": round(change_pct, 2)}

        significant = {k: v for k, v in changes.items() if abs(v["change_pct"]) > 20}
        return {
            "current_importance": current,
            "importance_changes": changes,
            "significant_changes": significant,
            "top_current": sorted(current.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_baseline": sorted(self.baseline_importance.items(), key=lambda x: x[1], reverse=True)[:10],
        }

    def explain_sample(self, df: pd.DataFrame, n: int = 100) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "No model loaded"}
        try:
            import shap
            sample = df.sample(min(n, len(df)))
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(sample)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            shap_imp = dict(zip(sample.columns.tolist(), mean_abs.tolist()))
            totals = np.abs(shap_vals).sum(axis=1)
            threshold = float(np.percentile(totals, 95))
            anomalous = int(np.sum(totals > threshold))
            return {
                "shap_importance": shap_imp,
                "n_samples": len(sample),
                "anomalous_predictions": anomalous,
                "anomaly_threshold": threshold,
            }
        except ImportError:
            return {"error": "shap not installed"}
        except Exception as e:
            return {"error": str(e)}