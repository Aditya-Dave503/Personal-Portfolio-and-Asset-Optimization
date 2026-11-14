
"""
Four-asset forecasting benchmark with MLflow tracking and PDF reporting.

Run from backend:
    python -m validation.benchmark_forecasting
"""

from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)

from ml.indian_market_data import (
    get_synchronized_market_data,
)

from ml.ml_engine import (
    extract_asset_features,
    extract_forward_targets,
)

from ml.tracking import setup_mlflow


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BACKEND = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BACKEND / "outputs" / "forecasting_benchmark"

HORIZON = 21
N_SPLITS = 5
ROLLING_WINDOW = 63
RANDOM_STATE = 42

EXPERIMENT_NAME = "Portfolio_Forecasting_Benchmark"

METHODS = [
    "zero_return",
    "historical_mean",
    "historical_median",
    "rolling_mean",
    "hist_gradient_boosting",
]


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def build_model(fold):
    return HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=RANDOM_STATE + fold,
    )


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
    }


# ---------------------------------------------------------
# BASELINE PREDICTIONS
# ---------------------------------------------------------

def rolling_mean_predictions(y, valid_idx, horizon):
    """
    At validation position i, use only forward-return labels whose
    complete 21-observation outcome was observable by position i.

    The latest eligible historical target is y[i - horizon].
    """

    predictions = []

    for position in valid_idx:
        last_matured_position = position - horizon

        if last_matured_position < 0:
            raise ValueError("Insufficient matured history for rolling mean.")

        historical_values = y.iloc[
            max(0, last_matured_position - ROLLING_WINDOW + 1):
            last_matured_position + 1
        ]

        if historical_values.empty:
            raise ValueError("No matured observations for rolling mean.")

        predictions.append(float(historical_values.mean()))

    return np.asarray(predictions, dtype=float)


# ---------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------

def generate_pdf(summary_df, fold_df, output_path, date_range):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=21,
        leading=26,
        spaceAfter=16,
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
        "Multi-Asset Forecasting Benchmark Report",
        styles["ReportTitle"],
    ))

    story.append(Paragraph(
        "Personal Portfolio and Asset Optimization Platform",
        styles["Heading2"],
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Market data:</b> {date_range[0]} to {date_range[1]}",
        styles["BodyText"],
    ))

    story.append(Paragraph(
        f"<b>Forecast horizon:</b> {HORIZON} observations "
        f"&nbsp;&nbsp; <b>Validation:</b> {N_SPLITS}-fold expanding "
        f"walk-forward &nbsp;&nbsp; <b>Gap:</b> {HORIZON} observations",
        styles["BodyText"],
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))

    for asset in summary_df["asset"].unique():
        asset_df = summary_df[summary_df["asset"] == asset]
        best = asset_df.loc[asset_df["mae_mean"].idxmin()]

        story.append(Paragraph(
            f"<b>{asset}:</b> Lowest average fold MAE was achieved by "
            f"{best['method']} ({best['mae_mean']:.5f}).",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 15))
    story.append(Paragraph("2. Aggregate Benchmark Results", styles["Heading1"]))

    table_data = [[
        "Asset", "Method", "Mean MAE", "Mean RMSE",
        "Directional Accuracy"
    ]]

    for _, row in summary_df.iterrows():
        direction = (
            "N/A"
            if pd.isna(row["directional_accuracy_mean"])
            else f"{row['directional_accuracy_mean']:.2%}"
        )

        table_data.append([
            row["asset"],
            row["method"],
            f"{row['mae_mean']:.5f}",
            f"{row['rmse_mean']:.5f}",
            direction,
        ])

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[145, 145, 90, 90, 125],
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24364B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F3F6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("3. MAE Comparison", styles["Heading1"]))

    for asset in summary_df["asset"].unique():
        asset_df = summary_df[summary_df["asset"] == asset]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(asset_df["method"], asset_df["mae_mean"])
        ax.set_title(f"{asset} - Average Fold MAE")
        ax.set_ylabel("MAE (return units)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            chart_path = Path(temp.name)

        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        story.append(Image(str(chart_path), width=7.5 * inch, height=2.9 * inch))
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    story.append(Paragraph("4. Fold-Level Results", styles["Heading1"]))

    fold_table = [[
        "Asset", "Method", "Fold", "MAE", "RMSE",
        "Validation Start", "Validation End"
    ]]

    for _, row in fold_df.iterrows():
        fold_table.append([
            row["asset"],
            row["method"],
            str(row["fold"]),
            f"{row['mae']:.5f}",
            f"{row['rmse']:.5f}",
            row["validation_start"],
            row["validation_end"],
        ])

    fold_report_table = Table(
        fold_table,
        repeatRows=1,
        colWidths=[105, 120, 40, 75, 75, 100, 100],
    )

    fold_report_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24364B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F3F6")]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))

    story.append(fold_report_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("5. Methodology and Limitations", styles["Heading1"]))

    methodology = [
        "All assets use the same synchronized market observations.",
        "Technical features are generated using the existing feature extraction function.",
        "Targets are compounded forward returns over 21 observations.",
        "Five expanding-window TimeSeriesSplit folds are used with a 21-observation gap.",
        "The rolling-mean baseline uses only matured historical forward-return labels.",
        "HistGradientBoosting uses the controlled hyperparameters from the previous macro ablation experiment.",
        "Results are validation results and do not establish future investment performance.",
        "Adjacent 21-observation forward-return targets overlap; prediction errors may therefore be serially dependent.",
    ]

    for item in methodology:
        story.append(Paragraph(f"• {item}", styles["BodyText"]))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "The lowest MAE method is not automatically the best portfolio strategy. "
        "Portfolio-level backtesting, transaction costs, risk and drawdown "
        "must also be evaluated.",
        styles["Italic"],
    ))

    doc.build(story)


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    market = get_synchronized_market_data().sort_index()

    if market.index.has_duplicates or not market.index.is_monotonic_increasing:
        raise ValueError("Market dates must be unique and sorted.")

    if market.isna().any().any():
        raise ValueError("Market data contains missing values.")

    if not np.isfinite(market.to_numpy(dtype=float)).all():
        raise ValueError("Market data contains non-finite values.")

    print("\nMULTI-ASSET FORECASTING BENCHMARK")
    print(f"Market observations: {len(market)}")
    print(f"Assets: {list(market.columns)}")
    print(f"Date range: {market.index.min().date()} to {market.index.max().date()}")
    print(f"Folds: {N_SPLITS} | Gap: {HORIZON} observations")

    all_fold_rows = []
    all_prediction_rows = []
    all_summary_rows = []

    with mlflow.start_run(run_name="four_asset_forecasting_benchmark") as parent_run:

        mlflow.log_params({
            "forecast_horizon": HORIZON,
            "n_splits": N_SPLITS,
            "validation_gap": HORIZON,
            "rolling_window": ROLLING_WINDOW,
            "validation_method": "expanding_time_series_split",
            "feature_set": "technical_only",
            "assets": ",".join(market.columns),
        })

        for asset in market.columns:
            print(f"\nProcessing {asset} ...")

            features = extract_asset_features(market[asset])
            target = extract_forward_targets(market[asset], HORIZON)

            labeled = pd.concat([features, target], axis=1).dropna()

            feature_cols = list(features.columns)
            X = labeled[feature_cols]
            y = labeled[target.name]

            if not np.isfinite(X.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite features found for {asset}.")

            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite targets found for {asset}.")

            splitter = TimeSeriesSplit(
                n_splits=N_SPLITS,
                gap=HORIZON,
            )

            method_predictions = {method: [] for method in METHODS}
            method_actuals = {method: [] for method in METHODS}
            method_fold_metrics = {method: [] for method in METHODS}

            for fold, (train_idx, valid_idx) in enumerate(
                splitter.split(X), start=1
            ):
                X_train = X.iloc[train_idx]
                X_valid = X.iloc[valid_idx]

                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]

                predictions = {}

                predictions["zero_return"] = np.zeros(
                    len(valid_idx), dtype=float
                )

                predictions["historical_mean"] = np.full(
                    len(valid_idx), float(y_train.mean())
                )

                predictions["historical_median"] = np.full(
                    len(valid_idx), float(y_train.median())
                )

                predictions["rolling_mean"] = rolling_mean_predictions(
                    y, valid_idx, HORIZON
                )

                model = build_model(fold)
                model.fit(X_train, y_train)

                predictions["hist_gradient_boosting"] = model.predict(X_valid)

                for method in METHODS:
                    pred = np.asarray(predictions[method], dtype=float)
                    actual = y_valid.to_numpy(dtype=float)

                    metrics = calculate_metrics(actual, pred)

                    if method == "zero_return":
                        metrics["directional_accuracy"] = np.nan

                    method_predictions[method].extend(pred.tolist())
                    method_actuals[method].extend(actual.tolist())
                    method_fold_metrics[method].append(metrics)

                    all_fold_rows.append({
                        "asset": asset,
                        "method": method,
                        "fold": fold,
                        "train_observations": len(train_idx),
                        "validation_observations": len(valid_idx),
                        "train_end": str(X.index[train_idx[-1]].date()),
                        "validation_start": str(X.index[valid_idx[0]].date()),
                        "validation_end": str(X.index[valid_idx[-1]].date()),
                        "row_separation": int(valid_idx[0] - train_idx[-1]),
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "directional_accuracy": metrics["directional_accuracy"],
                    })

                    for date, actual_value, predicted_value in zip(
                        X.index[valid_idx], actual, pred
                    ):
                        all_prediction_rows.append({
                            "asset": asset,
                            "method": method,
                            "fold": fold,
                            "date": str(date.date()),
                            "actual_return": float(actual_value),
                            "predicted_return": float(predicted_value),
                            "residual": float(actual_value - predicted_value),
                        })

                print(
                    f"  Fold {fold}/{N_SPLITS}: "
                    f"train={len(train_idx)}, validation={len(valid_idx)}"
                )

            # Log one MLflow child run per asset and method.
            for method in METHODS:
                fold_metrics = method_fold_metrics[method]

                mae_mean = float(np.mean([m["mae"] for m in fold_metrics]))
                rmse_mean = float(np.mean([m["rmse"] for m in fold_metrics]))

                directional_values = [
                    m["directional_accuracy"]
                    for m in fold_metrics
                    if np.isfinite(m["directional_accuracy"])
                ]

                directional_mean = (
                    float(np.mean(directional_values))
                    if directional_values else np.nan
                )

                summary_row = {
                    "asset": asset,
                    "method": method,
                    "mae_mean": mae_mean,
                    "rmse_mean": rmse_mean,
                    "directional_accuracy_mean": directional_mean,
                    "folds": N_SPLITS,
                    "validation_observations": len(method_actuals[method]),
                }

                all_summary_rows.append(summary_row)

                with mlflow.start_run(
                    run_name=f"{asset}_{method}",
                    nested=True,
                ):
                    mlflow.log_params({
                        "asset": asset,
                        "method": method,
                        "forecast_horizon": HORIZON,
                        "validation_gap": HORIZON,
                        "n_splits": N_SPLITS,
                        "rolling_window": ROLLING_WINDOW,
                        "feature_count": len(feature_cols),
                    })

                    mlflow.log_metrics({
                        "mae_mean": mae_mean,
                        "rmse_mean": rmse_mean,
                    })

                    if np.isfinite(directional_mean):
                        mlflow.log_metric(
                            "directional_accuracy_mean",
                            directional_mean,
                        )

                    for fold_number, metrics in enumerate(fold_metrics, 1):
                        mlflow.log_metrics({
                            f"fold_{fold_number}_mae": metrics["mae"],
                            f"fold_{fold_number}_rmse": metrics["rmse"],
                        })

                        if np.isfinite(metrics["directional_accuracy"]):
                            mlflow.log_metric(
                                f"fold_{fold_number}_directional_accuracy",
                                metrics["directional_accuracy"],
                            )

                    prediction_df = pd.DataFrame([
                        row for row in all_prediction_rows
                        if row["asset"] == asset and row["method"] == method
                    ])

                    fold_df = pd.DataFrame([
                        row for row in all_fold_rows
                        if row["asset"] == asset and row["method"] == method
                    ])

                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir = Path(temp_dir)

                        prediction_path = temp_dir / "oof_predictions.csv"
                        fold_path = temp_dir / "fold_metrics.csv"

                        prediction_df.to_csv(prediction_path, index=False)
                        fold_df.to_csv(fold_path, index=False)

                        mlflow.log_artifact(str(prediction_path))
                        mlflow.log_artifact(str(fold_path))

                print(
                    f"  {method}: MAE={mae_mean:.6f}, RMSE={rmse_mean:.6f}"
                )

        # -------------------------------------------------
        # SAVE CONSOLIDATED OUTPUTS
        # -------------------------------------------------

        fold_df = pd.DataFrame(all_fold_rows)
        prediction_df = pd.DataFrame(all_prediction_rows)
        summary_df = pd.DataFrame(all_summary_rows)

        summary_path = OUTPUT_DIR / "benchmark_summary.csv"
        fold_path = OUTPUT_DIR / "fold_metrics.csv"
        prediction_path = OUTPUT_DIR / "oof_predictions.csv"
        metadata_path = OUTPUT_DIR / "benchmark_metadata.json"
        pdf_path = OUTPUT_DIR / "forecasting_benchmark_report.pdf"

        summary_df.to_csv(summary_path, index=False)
        fold_df.to_csv(fold_path, index=False)
        prediction_df.to_csv(prediction_path, index=False)

        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "mlflow_parent_run_id": parent_run.info.run_id,
            "forecast_horizon": HORIZON,
            "n_splits": N_SPLITS,
            "gap_observations": HORIZON,
            "rolling_window": ROLLING_WINDOW,
            "validation_method": "Expanding TimeSeriesSplit",
            "feature_set": "technical_only",
            "assets": list(market.columns),
            "methods": METHODS,
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
            (
                str(market.index.min().date()),
                str(market.index.max().date()),
            ),
        )

        # Log consolidated artifacts to parent run.
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(fold_path))
        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(pdf_path))

        mlflow.log_metric("assets_evaluated", len(market.columns))
        mlflow.log_metric("methods_evaluated", len(METHODS))
        mlflow.log_metric("folds_per_asset", N_SPLITS)

    print("\nBENCHMARK COMPLETE")
    print(f"Results directory: {OUTPUT_DIR}")
    print(f"PDF report: {pdf_path}")
    print(f"MLflow experiment ID: {experiment_id}")
    print(f"Parent run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
