"""
VAR Research + PDF Report
Run from backend:
    python -m validation.investigate_var

Outputs:
    outputs/var_research/var_research_summary.csv
    outputs/var_research/var_research_folds.csv
    outputs/var_research/var_research_predictions.csv
    outputs/var_research/var_research_report.pdf

MLflow experiment: VAR_Research
"""

from pathlib import Path
import json
import time
import warnings
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)

from ml.indian_market_data import get_synchronized_market_data
from ml.ml_engine import extract_asset_features, extract_forward_targets
from ml.tracking import setup_mlflow


# ---------------- CONFIG ----------------

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "var_research"
EXPERIMENT = "VAR_Research"

HORIZON = 21
N_SPLITS = 5
GAP = 21
VAR_LAGS = [1, 2, 3, 5, 10]
AR_LAGS = 2
SEED = 42


# ---------------- HELPERS ----------------

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
        "bias": float(np.mean(predicted - actual)),
    }


def prepare_common_data(market):
    feature_map = {}
    target_map = {}

    for asset in market.columns:
        feature_map[asset] = extract_asset_features(market[asset])
        target_map[asset] = extract_forward_targets(
            market[asset], horizon_days=HORIZON
        ).rename(asset)

    target_df = pd.concat(target_map.values(), axis=1)
    common = target_df.dropna().index

    for asset in market.columns:
        common = common.intersection(feature_map[asset].dropna().index)

    common = common.sort_values()
    target_df = target_df.loc[common]
    feature_map = {a: feature_map[a].loc[common] for a in market.columns}

    for asset in market.columns:
        if not np.isfinite(feature_map[asset].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite features for {asset}")
        if not np.isfinite(target_df[asset].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite targets for {asset}")

    return feature_map, target_df, common


def compound_var_forecast(fitted, observed_returns):
    daily = fitted.forecast(
        y=np.asarray(observed_returns, dtype=float)[-fitted.k_ar:],
        steps=HORIZON,
    )
    return np.prod(1.0 + daily, axis=0) - 1.0


def compound_ar_forecast(params, observed_returns):
    history = list(np.asarray(observed_returns, dtype=float))
    intercept = float(params[0])
    coefficients = np.asarray(params[1:], dtype=float)
    daily_forecasts = []

    for _ in range(HORIZON):
        recent = np.asarray(history[-AR_LAGS:][::-1], dtype=float)
        nxt = intercept + float(np.dot(coefficients, recent))
        daily_forecasts.append(nxt)
        history.append(nxt)

    return float(np.prod(1.0 + np.asarray(daily_forecasts)) - 1.0)


def save_csv_artifact(df, filename):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        df.to_csv(p, index=False)
        mlflow.log_artifact(str(p))


# ---------------- PDF ----------------

def make_asset_chart(summary, asset, path):
    part = summary[summary.asset == asset].sort_values("pooled_mae")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(part.model, part.pooled_mae)
    ax.set_title(f"{asset}: pooled out-of-fold MAE")
    ax.set_ylabel("MAE (return units)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_pdf(summary, folds, metadata, pdf_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=landscape(A4),
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    story = [
        Paragraph("VAR Research and Model Comparison", styles["Title"]),
        Paragraph("Personal Portfolio and Asset Optimization Platform",
                  styles["Heading2"]),
        Spacer(1, 10),
        Paragraph(
            f"Market: {metadata['market_start']} to {metadata['market_end']} | "
            f"Common forecast origins: {metadata['n_common_dates']} | "
            f"Horizon: {HORIZON} observations | "
            f"{N_SPLITS} expanding folds | Gap: {GAP}",
            styles["BodyText"],
        ),
        Spacer(1, 14),
        Paragraph("1. Executive Summary", styles["Heading1"]),
    ]

    for asset in summary.asset.unique():
        part = summary[summary.asset == asset].sort_values("pooled_mae")
        best = part.iloc[0]
        story.append(Paragraph(
            f"<b>{asset}:</b> Lowest pooled MAE: {best.model} "
            f"({best.pooled_mae:.5f}); directional accuracy "
            f"{best.pooled_directional_accuracy:.2%}.",
            styles["BodyText"],
        ))

    story += [
        Spacer(1, 12),
        Paragraph("2. Pooled Out-of-Fold Results", styles["Heading1"]),
    ]

    rows = [[
        "Asset", "Model", "Pooled MAE", "Pooled RMSE",
        "Directional Acc.", "Mean Fold MAE", "Train Sec."
    ]]
    for _, r in summary.sort_values(["asset", "pooled_mae"]).iterrows():
        rows.append([
            r.asset, r.model, f"{r.pooled_mae:.5f}",
            f"{r.pooled_rmse:.5f}",
            f"{r.pooled_directional_accuracy:.2%}",
            f"{r.fold_mae_mean:.5f}",
            f"{r.mean_training_seconds:.3f}",
        ])

    tbl = Table(rows, repeatRows=1,
                colWidths=[100, 125, 75, 75, 90, 85, 65])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26384A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F3F6")]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(PageBreak())
    story.append(Paragraph("3. Model Performance by Asset", styles["Heading1"]))

    chart_paths = []
    for i, asset in enumerate(summary.asset.unique()):
        chart = pdf_path.parent / f"_var_research_chart_{i}.png"
        make_asset_chart(summary, asset, chart)
        chart_paths.append(chart)
        story.append(Paragraph(asset, styles["Heading2"]))
        story.append(Image(str(chart), width=7.8 * inch, height=3.1 * inch))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("4. Fold-Level Performance", styles["Heading1"]))

    fold_rows = [[
        "Asset", "Model", "Fold", "MAE", "RMSE",
        "Direction", "Validation Start", "Validation End"
    ]]
    for _, r in folds.sort_values(["asset", "model", "fold"]).iterrows():
        fold_rows.append([
            r.asset, r.model, str(int(r.fold)),
            f"{r.mae:.5f}", f"{r.rmse:.5f}",
            f"{r.directional_accuracy:.2%}",
            r.validation_start, r.validation_end,
        ])

    ft = Table(fold_rows, repeatRows=1,
               colWidths=[90, 105, 35, 60, 60, 65, 100, 100])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26384A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F3F6")]),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ft)

    story += [
        PageBreak(),
        Paragraph("5. Methodology and Limitations", styles["Heading1"]),
    ]
    notes = [
        "All models use the same common forecast-origin dates, targets and outer validation folds.",
        "The target is the compounded return over the next 21 observations.",
        "VAR is fitted on daily returns through the final training date. During validation, its coefficients remain fixed; observed returns available at each forecast origin update the initial conditions.",
        "VAR forecasts 21 daily returns and compounds them into a 21-observation return.",
        "AR(2) uses only the individual asset's own daily return history.",
        "Historical mean and median are calculated from training-fold forward-return labels only.",
        "Ridge scaling is fitted on training data only. HistGradientBoosting uses fixed parameters and no early stopping.",
        "VAR lag configurations are compared on the outer folds for research. Final lag selection should use nested time-series validation, not these same outer results.",
        "The forward targets overlap, so forecast errors can be serially dependent. Statistical comparisons should account for this dependence.",
        "Historical forecasting performance does not guarantee future investment performance.",
    ]
    for note in notes:
        story.append(Paragraph(f"• {note}", styles["BodyText"]))
        story.append(Spacer(1, 5))

    doc.build(story)

    # Remove temporary chart files after ReportLab has embedded them.
    for chart in chart_paths:
        chart.unlink(missing_ok=True)


# ---------------- EXPERIMENT ----------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT)

    market = get_synchronized_market_data().sort_index()
    if market.index.has_duplicates or not market.index.is_monotonic_increasing:
        raise ValueError("Market index must be unique and sorted.")
    if market.isna().any().any():
        raise ValueError("Market data contains missing values.")
    if not np.isfinite(market.to_numpy(dtype=float)).all():
        raise ValueError("Market data contains non-finite values.")

    assets = list(market.columns)
    feature_map, targets, common_dates = prepare_common_data(market)
    splits = list(TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP).split(common_dates))

    print("VAR RESEARCH")
    print("Assets:", assets)
    print("Common observations:", len(common_dates))
    print("VAR lags:", VAR_LAGS)

    fold_rows = []
    prediction_rows = []

    with mlflow.start_run(run_name="var_research_parent") as parent:
        mlflow.log_params({
            "horizon": HORIZON,
            "n_splits": N_SPLITS,
            "gap": GAP,
            "var_lags": ",".join(map(str, VAR_LAGS)),
            "ar_lags": AR_LAGS,
            "assets": ",".join(assets),
            "common_observations": len(common_dates),
        })

        # VAR(p): one joint model per fold and lag order.
        for lag in VAR_LAGS:
            print(f"\nVAR({lag})")
            lag_fold_rows = []
            lag_pred_rows = []

            for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
                train_end = common_dates[train_idx[-1]]
                start = time.perf_counter()
                fitted = VAR(market.loc[:train_end]).fit(
                    maxlags=lag, ic=None, trend="c"
                )
                fit_seconds = time.perf_counter() - start

                for idx in valid_idx:
                    date = common_dates[idx]
                    observed = market.loc[:date].to_numpy(dtype=float)
                    pred_vector = compound_var_forecast(fitted, observed)
                    actual_vector = targets.loc[date].to_numpy(dtype=float)

                    for j, asset in enumerate(assets):
                        lag_pred_rows.append({
                            "model": f"VAR({lag})", "asset": asset,
                            "fold": fold, "date": str(date.date()),
                            "actual": float(actual_vector[j]),
                            "predicted": float(pred_vector[j]),
                        })

                current = pd.DataFrame([
                    x for x in lag_pred_rows if x["fold"] == fold
                ])
                for asset in assets:
                    p = current[current.asset == asset]
                    m = metrics(p.actual, p.predicted)
                    row = {
                        "model": f"VAR({lag})", "asset": asset,
                        "fold": fold, **m,
                        "training_seconds": fit_seconds,
                        "validation_start": str(common_dates[valid_idx[0]].date()),
                        "validation_end": str(common_dates[valid_idx[-1]].date()),
                    }
                    lag_fold_rows.append(row)
                    fold_rows.append(row)

                print(f"  Fold {fold} complete")

            lag_fold_df = pd.DataFrame(lag_fold_rows)
            lag_pred_df = pd.DataFrame(lag_pred_rows)
            with mlflow.start_run(run_name=f"VAR_lag_{lag}", nested=True):
                mlflow.log_param("lag_order", lag)
                for asset in assets:
                    p = lag_pred_df[lag_pred_df.asset == asset]
                    m = metrics(p.actual, p.predicted)
                    mlflow.log_metrics({
                        f"{asset}_pooled_mae": m["mae"],
                        f"{asset}_pooled_rmse": m["rmse"],
                        f"{asset}_directional_accuracy": m["directional_accuracy"],
                    })
                save_csv_artifact(lag_fold_df, "fold_metrics.csv")
                save_csv_artifact(lag_pred_df, "predictions.csv")

            prediction_rows.extend(lag_pred_rows)

        # Individual AR(2), historical baselines, Ridge and HGB.
        for asset in assets:
            print(f"\nBenchmarks: {asset}")
            X = feature_map[asset]
            y = targets[asset]

            for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
                Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
                ytr, yva = y.iloc[train_idx], y.iloc[valid_idx]
                dates = common_dates[valid_idx]

                predictions = {
                    "Historical_Mean": np.full(len(valid_idx), ytr.mean()),
                    "Historical_Median": np.full(len(valid_idx), ytr.median()),
                }
                times = {
                    "Historical_Mean": 0.0,
                    "Historical_Median": 0.0,
                }

                ridge = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ])
                t0 = time.perf_counter()
                ridge.fit(Xtr, ytr)
                times["Ridge"] = time.perf_counter() - t0
                predictions["Ridge"] = ridge.predict(Xva)

                hgb = HistGradientBoostingRegressor(
                    max_iter=150, learning_rate=0.03,
                    max_leaf_nodes=15, min_samples_leaf=25,
                    l2_regularization=2.0, early_stopping=False,
                    random_state=SEED + fold,
                )
                t0 = time.perf_counter()
                hgb.fit(Xtr, ytr)
                times["HistGradientBoosting"] = time.perf_counter() - t0
                predictions["HistGradientBoosting"] = hgb.predict(Xva)

                train_end = common_dates[train_idx[-1]]
                daily_train = market.loc[:train_end, asset].to_numpy(dtype=float)
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar_fit = AutoReg(
                        daily_train, lags=AR_LAGS, trend="c"
                    ).fit()
                times["AR(2)"] = time.perf_counter() - t0

                ar_pred = []
                for date in dates:
                    observed = market.loc[:date, asset].to_numpy(dtype=float)
                    ar_pred.append(compound_ar_forecast(ar_fit.params, observed))
                predictions["AR(2)"] = np.asarray(ar_pred)

                for name, pred in predictions.items():
                    actual = yva.to_numpy(dtype=float)
                    m = metrics(actual, pred)
                    row = {
                        "model": name, "asset": asset, "fold": fold,
                        **m, "training_seconds": times[name],
                        "validation_start": str(dates[0].date()),
                        "validation_end": str(dates[-1].date()),
                    }
                    fold_rows.append(row)

                    for date, a, p in zip(dates, actual, pred):
                        prediction_rows.append({
                            "model": name, "asset": asset, "fold": fold,
                            "date": str(date.date()),
                            "actual": float(a), "predicted": float(p),
                        })

        folds = pd.DataFrame(fold_rows)
        predictions = pd.DataFrame(prediction_rows)

        # Summary metrics calculated from pooled OOF predictions.
        summaries = []
        for (asset, model_name), part in predictions.groupby(["asset", "model"]):
            m = metrics(part.actual, part.predicted)
            fp = folds[(folds.asset == asset) & (folds.model == model_name)]
            summaries.append({
                "asset": asset, "model": model_name,
                "fold_mae_mean": float(fp.mae.mean()),
                "fold_rmse_mean": float(fp.rmse.mean()),
                "pooled_mae": m["mae"],
                "pooled_rmse": m["rmse"],
                "directional_accuracy_mean": float(fp.directional_accuracy.mean()),
                "pooled_directional_accuracy": m["directional_accuracy"],
                "mean_training_seconds": float(fp.training_seconds.mean()),
            })

        summary = pd.DataFrame(summaries).sort_values(["asset", "pooled_mae"])

        metadata = {
            "experiment": EXPERIMENT,
            "parent_run_id": parent.info.run_id,
            "assets": assets,
            "var_lags": VAR_LAGS,
            "ar_lags": AR_LAGS,
            "horizon": HORIZON,
            "n_splits": N_SPLITS,
            "gap": GAP,
            "n_common_dates": len(common_dates),
            "market_start": str(market.index.min().date()),
            "market_end": str(market.index.max().date()),
            "common_start": str(common_dates.min().date()),
            "common_end": str(common_dates.max().date()),
        }

        summary_path = OUT / "var_research_summary.csv"
        folds_path = OUT / "var_research_folds.csv"
        predictions_path = OUT / "var_research_predictions.csv"
        metadata_path = OUT / "var_research_metadata.json"
        pdf_path = OUT / "var_research_report.pdf"

        summary.to_csv(summary_path, index=False)
        folds.to_csv(folds_path, index=False)
        predictions.to_csv(predictions_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        create_pdf(summary, folds, metadata, pdf_path)

        for path in [summary_path, folds_path, predictions_path, metadata_path, pdf_path]:
            mlflow.log_artifact(str(path))

        mlflow.log_metrics({
            "asset_count": len(assets),
            "common_observation_count": len(common_dates),
            "var_lag_configuration_count": len(VAR_LAGS),
        })

    print("\nVAR RESEARCH COMPLETE")
    print(summary.to_string(index=False))
    print("\nPDF:", pdf_path)
    print("Outputs:", OUT)
    print("MLflow:", EXPERIMENT)


if __name__ == "__main__":
    main()
