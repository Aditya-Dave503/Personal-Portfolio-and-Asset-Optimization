
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow

from ml.tracking import setup_mlflow


def main():
    # Initialize our MLflow tracking configuration
    setup_mlflow()

    # Keep infrastructure testing separate from real experiments
    mlflow.set_experiment("MLflow_Setup_Verification")

    with mlflow.start_run(run_name="tracking_smoke_test"):

        # 1. Log parameters
        mlflow.log_params({
            "model": "test_model",
            "forecast_horizon": 21,
            "asset": "NIFTY_EQUITIES",
        })

        # 2. Log illustrative test metrics (not model results)
        mlflow.log_metrics({
            "test_mae": 0.025,
            "test_rmse": 0.035,
        })

        # 3. Log tags
        mlflow.set_tags({
            "purpose": "mlflow_setup_verification",
            "status": "test_only",
        })

        # 4. Log a small artifact
        with TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "test_report.txt"
            artifact_path.write_text(
                "MLflow tracking verification successful.",
                encoding="utf-8",
            )

            mlflow.log_artifact(str(artifact_path))

        print("MLflow tracking test completed successfully.")


if __name__ == "__main__":
    main()
