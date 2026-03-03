"""
Entry Point
Run: python main.py --backend-url http://localhost:8000
"""
import os
import signal
import sys
import time
from typing import Optional

import typer
import uvicorn
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()
console = Console()
app_cli = typer.Typer()


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Merge env vars
    cfg["ml_backend_url"] = os.getenv("ML_BACKEND_URL", "http://localhost:8000")
    cfg["ml_api_key"] = os.getenv("ML_API_KEY")
    cfg["predict_endpoint"] = os.getenv("PREDICT_ENDPOINT", "/predict")
    cfg["model_info_endpoint"] = os.getenv("MODEL_INFO_ENDPOINT", "/model/info")
    cfg["database_url"] = os.getenv("DATABASE_URL", "monitoring.db")
    cfg["alert_channels"] = ["console"]
    return cfg


@app_cli.command()
def start(
    backend_url: str = typer.Option(None, "--backend-url", "-b"),
    api_key: str = typer.Option(None, "--api-key", "-k"),
    adapter: str = typer.Option("flat", "--adapter", "-a",
        help="Adapter type: flat | nested | wrapped | classification | list | dog_health"),
    prediction_field: str = typer.Option("prediction", "--prediction-field",
        help="Field name containing the prediction in the response"),
    data_key: str = typer.Option("data", "--data-key",
        help="For nested adapter: outer key containing prediction data"),
    predict_endpoint: str = typer.Option("/predict", "--predict-endpoint"),
    model_info_endpoint: str = typer.Option("/model/info", "--model-info-endpoint"),
    config: str = typer.Option("config/config.yaml", "--config", "-c"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port"),
):
    from pipeline.orchestrator import MLMonitoringPipeline
    from dashboard.app import app as fastapi_app, set_pipeline

    cfg = load_config(config)
    if backend_url:
        cfg["ml_backend_url"] = backend_url
    if api_key:
        cfg["ml_api_key"] = api_key

    cfg["predict_endpoint"] = predict_endpoint
    cfg["model_info_endpoint"] = model_info_endpoint

    # Build adapter config from CLI flags
    cfg["adapter"] = {
        "type": adapter,
        "prediction_field": prediction_field,
        **({"data_key": data_key} if adapter == "nested" else {}),
    }

    pipeline = MLMonitoringPipeline(cfg)
    pipeline.start()
    set_pipeline(pipeline)

    def shutdown(sig, frame):
        pipeline.stop(); sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    console.print(f"[green]Dashboard → http://{host}:{port}/docs[/green]")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


@app_cli.command()
def demo(
    backend_url: str = typer.Option("http://localhost:8000", "--backend-url"),
):
    """
    Run a quick demo: simulates predictions + ground truth against a mock backend.
    Useful for testing without a real model backend.
    """
    import random
    from pipeline.orchestrator import MLMonitoringPipeline

    cfg = load_config()
    cfg["ml_backend_url"] = backend_url
    # Inject fake baseline stats so demo works without a real backend
    cfg["baseline_stats"] = {
        "feature_1": {"type": "numerical", "mean": 10.0, "std": 2.0, "min": 0.0, "max": 20.0, "missing_rate": 0.0},
        "feature_2": {"type": "numerical", "mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0, "missing_rate": 0.0},
        "category": {"type": "categorical", "categories": ["A", "B", "C"], "category_distribution": {"A": 0.5, "B": 0.3, "C": 0.2}},
    }

    pipeline = MLMonitoringPipeline(cfg)

    console.print("[bold green]Demo: sending 20 synthetic predictions…[/bold green]")
    pred_ids = []
    for i in range(20):
        features = {
            "feature_1": random.gauss(10, 2),
            "feature_2": random.uniform(0, 1),
            "category": random.choice(["A", "B", "C"]),
        }
        result = pipeline.predict(features)
        if result.get("prediction_id"):
            pred_ids.append(result["prediction_id"])
        console.print(f"  [{i+1}] pred={result.get('prediction')}  id={result.get('prediction_id')}")

    console.print("\n[bold]Submitting ground truth for first 10 predictions…[/bold]")
    for pid in pred_ids[:10]:
        pipeline.submit_ground_truth(pid, random.gauss(130, 15))

    console.print("\n[bold]Running daily evaluation…[/bold]")
    pipeline.run_daily_evaluation()

    console.print("\n[bold]Dashboard snapshot:[/bold]")
    import json
    snap = pipeline.get_dashboard_data()
    console.print_json(json.dumps(snap, indent=2, default=str))


if __name__ == "__main__":
    app_cli()