
from pathlib import Path

import mlflow


# backend/ directory
BACKEND_DIR = Path(__file__).resolve().parents[1]

# Local MLflow storage
MLFLOW_DIR = BACKEND_DIR / "mlruns"
MLFLOW_DB = BACKEND_DIR / "mlflow.db"

# SQLite tracking database
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

EXPERIMENT_NAME = "Portfolio_Forecasting_Benchmark"


def setup_mlflow():
    """Configure MLflow and retrieve or create the forecasting experiment."""

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=MLFLOW_DIR.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment_id
