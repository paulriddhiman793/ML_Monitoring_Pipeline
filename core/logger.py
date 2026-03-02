"""Component 1: Comprehensive Logging Layer"""
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from db.database import Database


class MLMonitoringLogger:

    def __init__(self, db: Database, model_version: str):
        self.db = db
        self.model_version = model_version

    def log_training_run(self, metadata: Dict[str, Any]):
        self.db.insert_training_run({
            "training_run_id": metadata.get("run_id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
            "model_type": metadata.get("model_type", "unknown"),
            "hyperparameters": metadata.get("params", {}),
            "training_metrics": metadata.get("metrics", {}),
            "feature_importance": metadata.get("feature_importance", {}),
            "data_statistics": metadata.get("data_stats", {}),
        })
        # Store baseline statistics for drift detection
        for feat, stats in metadata.get("data_stats", {}).items():
            self.db.upsert_baseline_stats(self.model_version, feat, stats)

    def log_prediction(self, features: Dict[str, Any], prediction_value: Any,
                       inference_time_ms: float, metadata: Dict = None,
                       quality_flags: Dict = None) -> str:
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.db.insert_prediction({
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
            "input_features": features,
            "prediction_value": float(prediction_value) if prediction_value is not None else None,
            "prediction_time_ms": inference_time_ms,
            "metadata": metadata or {},
            "quality_flags": quality_flags or {},
        })
        return prediction_id

    def log_ground_truth(self, prediction_id: str, actual_value: float) -> Dict[str, Any]:
        pred = self.db.get_prediction_for_gt(prediction_id)
        if not pred:
            return {"error": f"Prediction {prediction_id} not found"}

        error = actual_value - pred["prediction_value"]
        self.db.insert_ground_truth({
            "prediction_id": prediction_id,
            "actual_value": actual_value,
            "observation_timestamp": datetime.now().isoformat(),
            "absolute_error": abs(error),
            "squared_error": error ** 2,
            "percentage_error": abs(error / actual_value * 100) if actual_value != 0 else None,
        })
        return {"success": True, "absolute_error": abs(error)}

    def log_quality(self, quality_report: Dict[str, Any], prediction_id: Optional[str] = None):
        self.db.insert_quality_log({
            "timestamp": datetime.now().isoformat(),
            "prediction_id": prediction_id,
            "severity": quality_report.get("severity", "none"),
            "issues": quality_report.get("issues", []),
            "warnings": quality_report.get("warnings", []),
            "valid": quality_report.get("valid", True),
        })