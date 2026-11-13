
"""
Multi-Asset Model Comparison Experiment.

Run from backend:
    python -m validation.compare_models

Compares six regression models across four assets using identical
features, forward targets and walk-forward validation folds.

Does not modify production models.
"""

from pathlib import Path
import json
import tempfile
import time

import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

from ml.indian_market_data import get_synchronized_market_data
from ml.ml_engine import extract_asset_features, extract_forward_targets
from ml.tracking import setup_mlflow


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BACKEND = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BACKEND / "outputs" / "model_comparison"

EXPERIMENT_NAME = "Portfolio_Model_Comparison"

HORIZON = 21
N_SPLITS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------

def build_models(fold):
    """
    Return fresh model instances for each fold.

    Linear models use StandardScaler fitted exclusively
    on the training data through sklearn Pipeline.
    """

    return {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),

        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),

        "lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=0.001,
                max_iter=20000,
            )),
        ]),

        "elastic_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=20000,
            )),
        ]),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE + fold,
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=RANDOM_STATE + fold,
        ),
    }


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
        "prediction_bias": float(np.mean(predicted - actual)),
    }


# ---------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------

def generate_pdf(summary_df, fold_df, output_path, market):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=21,
        alignment=1,
        spaceAfter=15,
    ))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    story = []

    story.append(Paragraph(
        "Multi-Asset ML Model Comparison Report",
        styles["ReportTitle"],
    ))

    story.append(Paragraph(
        "Personal Portfolio and Asset Optimization Platform",
        styles["Heading2"],
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"Market period: {market.index.min().date()} to "
        f"{market.index.max().date()}",
        styles["BodyText"],
    ))

    story.append(Paragraph(
        f"Forecast horizon: {HORIZON} observations | "
        f"Validation: {N_SPLITS}-fold expanding walk-forward | "
        f"Gap: {HORIZON} observations",
        styles["BodyText"],
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "1. Executive Summary",
        styles["Heading1"],
    ))

    for asset in summary_df["asset"].unique():
        asset_df = summary_df[summary_df["asset"] == asset]
        best = asset_df.loc[asset_df["pooled_mae"].idxmin()]

        story.append(Paragraph(
            f"<b>{asset}:</b> Lowest pooled OOF MAE was achieved by "
            f"{best['model']} ({best['pooled_mae']:.5f}).",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "2. Aggregate Model Comparison",
        styles["Heading1"],
    ))

    table_data = [[
        "Asset",
        "Model",
        "Fold MAE",
        "Pooled MAE",
        "Fold RMSE",
        "Pooled RMSE",
        "Directional Accuracy",
    ]]

    for _, row in summary_df.iterrows():
        table_data.append([
            row["asset"],
            row["model"],
            f"{row['fold_mae_mean']:.5f}",
            f"{row['pooled_mae']:.5f}",
            f"{row['fold_rmse_mean']:.5f}",
            f"{row['pooled_rmse']:.5f}",
            f"{row['directional_accuracy_mean']:.2%}",
        ])

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[105, 120, 75, 75, 75, 75, 105],
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24364B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F0F3F6"),
        ]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph(
        "3. Pooled MAE Comparison",
        styles["Heading1"],
    ))

    for asset in summary_df["asset"].unique():
        asset_df = summary_df[
            summary_df["asset"] == asset
        ].sort_values("pooled_mae")

        fig, ax = plt.subplots(figsize=(9, 3.5))

        ax.bar(
            asset_df["model"],
            asset_df["pooled_mae"],
        )

        ax.set_title(f"{asset} - Pooled Out-of-Fold MAE")
        ax.set_ylabel("MAE (return units)")
        ax.tick_params(axis="x", rotation=25)

        fig.tight_layout()

        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as temp:
            chart_path = Path(temp.name)

        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        story.append(Image(
            str(chart_path),
            width=7.5 * inch,
            height=2.9 * inch,
        ))

        story.append(Spacer(1, 10))

    story.append(PageBreak())

    story.append(Paragraph(
        "4. Fold-Level Stability",
        styles["Heading1"],
    ))

    fold_table = [[
        "Asset",
        "Model",
        "Fold",
        "MAE",
        "RMSE",
        "Validation Start",
        "Validation End",
    ]]

    for _, row in fold_df.iterrows():
        fold_table.append([
            row["asset"],
            row["model"],
            str(row["fold"]),
            f"{row['mae']:.5f}",
            f"{row['rmse']:.5f}",
            row["validation_start"],
            row["validation_end"],
        ])

    fold_report = Table(
        fold_table,
        repeatRows=1,
        colWidths=[105, 115, 40, 70, 70, 105, 105],
    )

    fold_report.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24364B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F0F3F6"),
        ]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))

    story.append(fold_report)
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "5. Experimental Methodology and Limitations",
        styles["Heading1"],
    ))

    notes = [
        "All models use identical synchronized market observations.",
        "All models use the same technical feature set and forward-return targets.",
        "The target is a compounded 21-observation forward return.",
        "Validation uses five expanding-window TimeSeriesSplit folds with a 21-observation gap.",
        "Linear models use StandardScaler fitted separately within each training fold.",
        "Model hyperparameters are fixed for this comparison and are not optimized on the outer validation folds.",
        "Fold averages and pooled out-of-fold metrics are reported separately.",
        "Adjacent forward-return targets overlap, so errors may be serially dependent.",
        "The results are forecasting validation results, not evidence of future investment performance.",
    ]

    for note in notes:
        story.append(Paragraph(
            f"• {note}",
            styles["BodyText"],
        ))
        story.append(Spacer(1, 5))

    doc.build(story)


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    market = get_synchronized_market_data().sort_index()

    if market.index.has_duplicates:
        raise ValueError("Duplicate market dates detected.")

    if not market.index.is_monotonic_increasing:
        raise ValueError("Market dates must be sorted.")

    if market.isna().any().any():
        raise ValueError("Missing market returns detected.")

    if not np.isfinite(market.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite market returns detected.")

    all_fold_rows = []
    all_prediction_rows = []
    all_summary_rows = []

    print("\nMULTI-ASSET MODEL COMPARISON")
    print(f"Assets: {list(market.columns)}")
    print(f"Observations: {len(market)}")
    print(f"Forecast horizon: {HORIZON}")
    print(f"Folds: {N_SPLITS}")
    print(f"Gap: {HORIZON}")

    with mlflow.start_run(
        run_name="six_model_four_asset_comparison"
    ) as parent_run:

        mlflow.log_params({
            "forecast_horizon": HORIZON,
            "n_splits": N_SPLITS,
            "validation_gap": HORIZON,
            "feature_set": "technical_only",
            "validation_method": "expanding_time_series_split",
            "model_count": 6,
            "asset_count": len(market.columns),
        })

        for asset in market.columns:

            print(f"\nEvaluating {asset}")

            features = extract_asset_features(market[asset])
            target = extract_forward_targets(
                market[asset],
                HORIZON,
            )

            labeled = pd.concat(
                [features, target],
                axis=1,
            ).dropna()

            feature_cols = list(features.columns)

            X = labeled[feature_cols]
            y = labeled[target.name]

            if not np.isfinite(X.to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid features: {asset}")

            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid targets: {asset}")

            splitter = TimeSeriesSplit(
                n_splits=N_SPLITS,
                gap=HORIZON,
            )

            # Materialize once so every model receives identical folds.
            splits = list(splitter.split(X))

            model_names = list(build_models(1).keys())

            predictions_by_model = {
                name: [] for name in model_names
            }

            actual_by_model = {
                name: [] for name in model_names
            }

            for fold, (train_idx, valid_idx) in enumerate(
                splits, start=1
            ):

                X_train = X.iloc[train_idx]
                X_valid = X.iloc[valid_idx]

                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]

                models = build_models(fold)

                for model_name, model in models.items():

                    start_time = time.perf_counter()

                    model.fit(X_train, y_train)

                    training_seconds = (
                        time.perf_counter() - start_time
                    )

                    predictions = model.predict(X_valid)

                    metrics = calculate_metrics(
                        y_valid,
                        predictions,
                    )

                    predictions_by_model[model_name].extend(
                        predictions.tolist()
                    )

                    actual_by_model[model_name].extend(
                        y_valid.tolist()
                    )

                    all_fold_rows.append({
                        "asset": asset,
                        "model": model_name,
                        "fold": fold,
                        "train_observations": len(train_idx),
                        "validation_observations": len(valid_idx),
                        "train_end": str(
                            X.index[train_idx[-1]].date()
                        ),
                        "validation_start": str(
                            X.index[valid_idx[0]].date()
                        ),
                        "validation_end": str(
                            X.index[valid_idx[-1]].date()
                        ),
                        "row_separation": int(
                            valid_idx[0] - train_idx[-1]
                        ),
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "directional_accuracy":
                            metrics["directional_accuracy"],
                        "prediction_bias": metrics["prediction_bias"],
                        "training_seconds": training_seconds,
                    })

                    for date, actual, predicted in zip(
                        X.index[valid_idx],
                        y_valid.to_numpy(),
                        predictions,
                    ):
                        all_prediction_rows.append({
                            "asset": asset,
                            "model": model_name,
                            "fold": fold,
                            "date": str(date.date()),
                            "actual_return": float(actual),
                            "predicted_return": float(predicted),
                            "residual": float(actual - predicted),
                        })

                    print(
                        f"  Fold {fold} | {model_name}: "
                        f"MAE={metrics['mae']:.5f} "
                        f"RMSE={metrics['rmse']:.5f}"
                    )

            # ---------------------------------------------
            # ONE MLflow RUN PER ASSET AND MODEL
            # ---------------------------------------------

            for model_name in model_names:

                actual = np.asarray(
                    actual_by_model[model_name],
                    dtype=float,
                )

                predicted = np.asarray(
                    predictions_by_model[model_name],
                    dtype=float,
                )

                pooled = calculate_metrics(actual, predicted)

                model_fold_df = pd.DataFrame([
                    row for row in all_fold_rows
                    if row["asset"] == asset
                    and row["model"] == model_name
                ])

                fold_mae_mean = float(
                    model_fold_df["mae"].mean()
                )

                fold_rmse_mean = float(
                    model_fold_df["rmse"].mean()
                )

                fold_direction_mean = float(
                    model_fold_df["directional_accuracy"].mean()
                )

                fold_bias_mean = float(
                    model_fold_df["prediction_bias"].mean()
                )

                training_time_mean = float(
                    model_fold_df["training_seconds"].mean()
                )

                summary = {
                    "asset": asset,
                    "model": model_name,
                    "fold_mae_mean": fold_mae_mean,
                    "fold_rmse_mean": fold_rmse_mean,
                    "pooled_mae": pooled["mae"],
                    "pooled_rmse": pooled["rmse"],
                    "directional_accuracy_mean": fold_direction_mean,
                    "pooled_directional_accuracy":
                        pooled["directional_accuracy"],
                    "prediction_bias_mean": fold_bias_mean,
                    "mean_training_seconds": training_time_mean,
                    "validation_observations": len(actual),
                }

                all_summary_rows.append(summary)

                with mlflow.start_run(
                    run_name=f"{asset}_{model_name}",
                    nested=True,
                ):

                    mlflow.log_params({
                        "asset": asset,
                        "model": model_name,
                        "forecast_horizon": HORIZON,
                        "n_splits": N_SPLITS,
                        "validation_gap": HORIZON,
                        "feature_count": len(feature_cols),
                    })

                    mlflow.log_metrics({
                        "fold_mae_mean": fold_mae_mean,
                        "fold_rmse_mean": fold_rmse_mean,
                        "pooled_mae": pooled["mae"],
                        "pooled_rmse": pooled["rmse"],
                        "directional_accuracy_mean": fold_direction_mean,
                        "pooled_directional_accuracy":
                            pooled["directional_accuracy"],
                        "prediction_bias_mean": fold_bias_mean,
                        "mean_training_seconds": training_time_mean,
                    })

                    for _, row in model_fold_df.iterrows():
                        fold = int(row["fold"])

                        mlflow.log_metrics({
                            f"fold_{fold}_mae": row["mae"],
                            f"fold_{fold}_rmse": row["rmse"],
                            f"fold_{fold}_directional_accuracy":
                                row["directional_accuracy"],
                        })

                    prediction_df = pd.DataFrame([
                        row for row in all_prediction_rows
                        if row["asset"] == asset
                        and row["model"] == model_name
                    ])

                    with tempfile.TemporaryDirectory() as temp:
                        temp = Path(temp)

                        fold_file = temp / "fold_metrics.csv"
                        prediction_file = temp / "oof_predictions.csv"

                        model_fold_df.to_csv(
                            fold_file,
                            index=False,
                        )

                        prediction_df.to_csv(
                            prediction_file,
                            index=False,
                        )

                        mlflow.log_artifact(str(fold_file))
                        mlflow.log_artifact(str(prediction_file))

                print(
                    f"  {model_name} pooled MAE: "
                    f"{pooled['mae']:.5f}"
                )

        # -------------------------------------------------
        # SAVE CONSOLIDATED OUTPUTS
        # -------------------------------------------------

        summary_df = pd.DataFrame(all_summary_rows)
        fold_df = pd.DataFrame(all_fold_rows)
        prediction_df = pd.DataFrame(all_prediction_rows)

        summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
        fold_path = OUTPUT_DIR / "model_comparison_folds.csv"
        prediction_path = OUTPUT_DIR / "model_comparison_oof.csv"
        metadata_path = OUTPUT_DIR / "model_comparison_metadata.json"
        pdf_path = OUTPUT_DIR / "model_comparison_report.pdf"

        summary_df.to_csv(summary_path, index=False)
        fold_df.to_csv(fold_path, index=False)
        prediction_df.to_csv(prediction_path, index=False)

        metadata = {
            "experiment": EXPERIMENT_NAME,
            "parent_run_id": parent_run.info.run_id,
            "assets": list(market.columns),
            "models": model_names,
            "forecast_horizon": HORIZON,
            "n_splits": N_SPLITS,
            "validation_gap": HORIZON,
            "validation": "Expanding TimeSeriesSplit",
            "features": "Technical only",
            "market_start": str(market.index.min().date()),
            "market_end": str(market.index.max().date()),
            "production_models_modified": False,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        generate_pdf(
            summary_df,
            fold_df,
            pdf_path,
            market,
        )

        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(fold_path))
        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(pdf_path))

        mlflow.log_metric(
            "assets_evaluated",
            len(market.columns),
        )

        mlflow.log_metric(
            "models_evaluated",
            len(model_names),
        )

        mlflow.log_metric(
            "folds_per_model",
            N_SPLITS,
        )

    print("\nMODEL COMPARISON COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print(f"PDF: {pdf_path}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
