# ML Monitoring Pipeline

An ML model monitoring pipeline with real-time drift detection, performance tracking, and an interactive dashboard.

---

## Project Structure

```
├── api/           # API endpoints for prediction submission and monitoring
├── config/        # Configuration files (config.yaml)
├── core/          # Core monitoring logic and algorithms
├── dashboard/     # FastAPI-based web dashboard
├── db/            # Database models and persistence layer
├── pipeline/      # Main orchestration logic
├── main.py        # CLI entry point
├── requirements.txt  # Python dependencies
└── .env           # Environment configuration
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/paulriddhiman793/ML_Monitoring_Pipeline.git
cd ML_Monitoring_Pipeline
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `ML_BACKEND_URL` | URL of your ML model backend | `http://localhost:8000` |
| `ML_API_KEY` | API key for authentication (optional) | — |
| `DATABASE_URL` | Path to SQLite database | `monitoring.db` |

---

## Usage

### Quick Start with Demo

Launch the pipeline with synthetic demo data (no real backend required):

```bash
python main.py demo --backend-url http://localhost:8000 --adapter flat
```

This command:
- Generates 10 synthetic predictions
- Submits ground truth for 5 predictions
- Runs daily evaluation
- Displays dashboard data

### Start the Monitoring Pipeline

To start the full pipeline connected to your ML model backend:

```bash
python main.py start --backend-url http://localhost:8000 --api-key YOUR_API_KEY --adapter flat --port 8080
```

### CLI Options

| Option | Description | Default |
|---|---|---|
| `--backend-url`, `-b` | URL of your ML model backend | — |
| `--api-key`, `-k` | API key for authentication | — |
| `--adapter`, `-a` | Response format adapter | `flat` |
| `--prediction-field` | Field name containing predictions | `prediction` |
| `--data-key` | Key for nested data | `data` |
| `--predict-endpoint` | Model prediction endpoint | `/predict` |
| `--model-info-endpoint` | Model info endpoint | `/model/info` |
| `--health-endpoint` | Health check endpoint | `/health` |
| `--config`, `-c` | Path to config file | `config/config.yaml` |
| `--host` | Server host | `0.0.0.0` |
| `--port` | Server port | `8080` |

**Available adapters:**

- `flat` — Simple key-value response
- `nested` — Nested response with data key
- `wrapped` — Wrapped response structure
- `classification` — Classification model responses
- `list` — List-based responses
- `dog_health` — Health check format

---

## Dashboard

Once the pipeline is running, access the interactive dashboard at:

```
http://localhost:8080/docs
```

The dashboard provides:
- Real-time prediction metrics
- Data drift visualizations
- Model performance charts
- Ground truth submission interface
- Alert and anomaly displays

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework for API and dashboard |
| `uvicorn` | ASGI server |
| `httpx` | HTTP client for backend communication |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation and analysis |
| `scipy` | Statistical functions |
| `shap` | Model explainability |
| `catboost` | Gradient boosting for ML |
| `scikit-learn` | Machine learning algorithms |
| `sqlalchemy` | ORM for database operations |
| `aiosqlite` | Async SQLite driver |
| `pyyaml` | YAML configuration parsing |
| `python-dotenv` | Environment variable management |
| `apscheduler` | Job scheduling for automated evaluations |
| `rich` | Rich terminal output |
| `typer` | CLI framework |
| `websockets` | WebSocket support for real-time updates |

---

## Configuration

### config.yaml

Create a `config/config.yaml` file to customize the monitoring pipeline:

```yaml
# Model backend configuration
ml_backend_url: "http://localhost:8000"
ml_api_key: null

# Database configuration
database_url: "monitoring.db"

# Monitoring configuration
baseline_stats:
  feature_1:
    type: "numerical"
    mean: 10.0
    std: 2.0
    min: 0.0
    max: 20.0
    missing_rate: 0.0

# Alert channels
alert_channels:
  - console
```

---

## API Integration

### Making Predictions

```python
from main import MLMonitoringPipeline

pipeline = MLMonitoringPipeline(config)
pipeline.start()

# Submit a prediction
result = pipeline.predict({
    "feature_1": 10.5,
    "feature_2": 0.7,
    "category": "A"
})
```

### Submitting Ground Truth

```python
# prediction_id obtained from pipeline.predict()
pipeline.submit_ground_truth(prediction_id, actual_value)
```

### Running Evaluation

```python
pipeline.run_daily_evaluation()

# Get dashboard data
dashboard_data = pipeline.get_dashboard_data()
```

---

## Development

### Project Structure

- `api/` — API endpoint definitions and request/response handlers
- `config/` — Configuration management and defaults
- `core/` — Core monitoring algorithms (drift detection, performance metrics)
- `dashboard/` — FastAPI application and visualization endpoints
- `db/` — SQLAlchemy models and database operations
- `pipeline/` — Main orchestrator coordinating all components
- `main.py` — CLI entry point with Typer commands

### Adding Custom Adapters

To support additional response formats, implement a new adapter in the `api/` directory following the existing adapter patterns.

---

## Troubleshooting

**Backend Connection Issues**
- Verify `ML_BACKEND_URL` is correct and the backend is running
- Check `ML_API_KEY` if required by your backend
- Ensure network connectivity between the pipeline and backend

**Database Errors**
- Delete `monitoring.db` to reset the database
- Check file permissions in the project directory
- Ensure sufficient disk space for the database

**Dashboard Not Loading**
- Verify the pipeline is running on the correct host and port
- Check browser console for JavaScript errors
- Ensure no firewall blocks access to the dashboard port

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository:
[GitHub Issues](https://github.com/paulriddhiman793/ML_Monitoring_Pipeline/issues)

---

*Last Updated: March 2026 — Maintained by [paulriddhiman793](https://github.com/paulriddhiman793)*


