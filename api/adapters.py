"""
Backend Adapters
Each adapter knows how to:
  1. Build the request body for that backend
  2. Extract the prediction value from the response
  3. Extract any extra metadata worth logging

To add a new backend: subclass BaseAdapter and register it.
"""
from typing import Any, Dict, Optional


class BaseAdapter:
    """
    Standard contract every adapter must fulfill.
    Internal prediction format:
    {
        "prediction": <float>,          # the main numeric value being monitored
        "extra": { ...any other fields } # optional, gets stored in metadata
    }
    """

    def build_request_body(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform flat feature dict into whatever body the backend expects.
        Default: send features flat (most REST APIs).
        """
        return features

    def extract_prediction(self, response: Dict[str, Any]) -> Optional[float]:
        """
        Pull the main numeric prediction out of the backend response.
        Must be overridden.
        """
        raise NotImplementedError

    def extract_extra(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pull any additional fields worth logging (labels, scores, etc.)
        Default: return everything except the prediction field.
        """
        return {}

    def get_model_version(self, model_info: Dict[str, Any]) -> str:
        return model_info.get("model_version", "v1")

    def get_baseline_metrics(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training metrics (rmse, r2, etc.) from /model-info response."""
        return {}


# ── Built-in adapters ────────────────────────────────────────────────────

class FlatPredictionAdapter(BaseAdapter):
    """
    Backend returns: { "prediction": 42.3 }
    or:              { "output": 42.3 }
    or:              { "score": 0.87 }
    Configure which field to read via prediction_field.
    """
    def __init__(self, prediction_field: str = "prediction"):
        self.prediction_field = prediction_field

    def extract_prediction(self, response: Dict[str, Any]) -> Optional[float]:
        val = response.get(self.prediction_field)
        return float(val) if val is not None else None

    def extract_extra(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in response.items() if k != self.prediction_field}


class NestedPredictionAdapter(BaseAdapter):
    """
    Backend returns: { "status": "success", "data": { "score": 0.42, ... } }
    Configure the nesting path and field name.
    """
    def __init__(self, data_key: str = "data", prediction_field: str = "prediction"):
        self.data_key = data_key
        self.prediction_field = prediction_field

    def build_request_body(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return features  # still sends flat features

    def extract_prediction(self, response: Dict[str, Any]) -> Optional[float]:
        payload = response.get(self.data_key, response)
        val = payload.get(self.prediction_field)
        return float(val) if val is not None else None

    def extract_extra(self, response: Dict[str, Any]) -> Dict[str, Any]:
        payload = response.get(self.data_key, {})
        return {k: v for k, v in payload.items() if k != self.prediction_field}


class WrappedFeaturesAdapter(BaseAdapter):
    """
    Backend expects: { "features": { ...feature dict... } }
    instead of flat body.
    """
    def __init__(self, prediction_field: str = "prediction",
                 features_key: str = "features"):
        self.prediction_field = prediction_field
        self.features_key = features_key

    def build_request_body(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return {self.features_key: features}  # wrap features

    def extract_prediction(self, response: Dict[str, Any]) -> Optional[float]:
        val = response.get(self.prediction_field)
        return float(val) if val is not None else None


class ClassificationAdapter(BaseAdapter):
    """
    Classification backend returns:
    { "label": "cat", "confidence": 0.92, "probabilities": {"cat": 0.92, "dog": 0.08} }
    Monitors the confidence score as the numeric prediction value.
    """
    def __init__(self, confidence_field: str = "confidence",
                 label_field: str = "label"):
        self.confidence_field = confidence_field
        self.label_field = label_field

    def extract_prediction(self, response: Dict[str, Any]) -> Optional[float]:
        val = response.get(self.confidence_field)
        return float(val) if val is not None else None

    def extract_extra(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in response.items()
                if k != self.confidence_field}


class ListOutputAdapter(BaseAdapter):
    """
    Backend returns a list: [0.92] or [[0.1, 0.9]]
    Monitors the first element (or specified index).
    """
    def __init__(self, output_field: str = "predictions", index: int = 0):
        self.output_field = output_field
        self.index = index

    def extract_prediction(self, response: Dict[str, Any]) -> Optional[float]:
        outputs = response.get(self.output_field, response)
        if isinstance(outputs, list):
            val = outputs[self.index]
            if isinstance(val, list):
                val = val[self.index]
            return float(val)
        return None


class DogHealthAdapter(NestedPredictionAdapter):
    """
    Adapter for https://rppooo-dog-backend.hf.space
    Response: { "status": "success", "data": { "risk_score": 0.116, ... } }
    """
    def __init__(self):
        super().__init__(data_key="data", prediction_field="risk_score")

    def get_model_version(self, model_info: Dict[str, Any]) -> str:
        return f"catboost_dog_v{model_info.get('version', '3.0.0')}"

    def get_baseline_metrics(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        cfg = model_info.get("configuration", {})
        perf = cfg.get("performance", {})
        return {
            "r2": perf.get("val_r2"),
            "rmse": perf.get("val_rmse"),
        }


# ── Registry ─────────────────────────────────────────────────────────────

ADAPTER_REGISTRY: Dict[str, type] = {
    "flat":           FlatPredictionAdapter,
    "nested":         NestedPredictionAdapter,
    "wrapped":        WrappedFeaturesAdapter,
    "classification": ClassificationAdapter,
    "list":           ListOutputAdapter,
    "dog_health":     DogHealthAdapter,
}


def get_adapter(adapter_config: Dict[str, Any]) -> BaseAdapter:
    """
    Build an adapter from config dict. Example configs:

    # Dog health backend (current)
    {"type": "dog_health"}

    # Generic flat output
    {"type": "flat", "prediction_field": "score"}

    # Nested response
    {"type": "nested", "data_key": "result", "prediction_field": "value"}

    # Wrapped features
    {"type": "wrapped", "features_key": "inputs", "prediction_field": "output"}

    # Classification
    {"type": "classification", "confidence_field": "probability", "label_field": "class"}

    # Custom adapter passed directly
    {"type": "custom", "instance": MyAdapter()}
    """
    if adapter_config.get("type") == "custom":
        return adapter_config["instance"]

    adapter_type = adapter_config.get("type", "flat")
    cls = ADAPTER_REGISTRY.get(adapter_type)
    if not cls:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    # Pass all config keys except 'type' as constructor kwargs
    kwargs = {k: v for k, v in adapter_config.items() if k != "type"}
    return cls(**kwargs)