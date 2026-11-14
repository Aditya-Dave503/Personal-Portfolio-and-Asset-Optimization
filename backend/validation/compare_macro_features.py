"""Compare technical-only vs technical+G-Sec macro features.

Run from backend:
    python validation/compare_macro_features.py

Both configurations use identical observations, folds, model settings, and
21-observation purge gaps. No production model files are modified.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ml.indian_market_data import get_synchronized_market_data, resolve_data_directory
from ml.ml_engine import (
    extract_asset_features,
    extract_forward_targets,
    load_macro_yield_features,
)

BACKEND = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BACKEND / "outputs" / "ml_macro_comparison"
HORIZON = 21
N_SPLITS = 5
RANDOM_STATE = 42


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


def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
    }


def evaluate_configuration(asset, X, y, feature_set_name, feature_columns):
    X = X[feature_columns]
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=HORIZON)
    fold_rows = []
    oof_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        model = build_model(fold)
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        zero_predictions = np.zeros(len(y_valid), dtype=float)
        mean_predictions = np.full(len(y_valid), float(y_train.mean()))

        model_metrics = score(y_valid, predictions)
        zero_metrics = score(y_valid, zero_predictions)
        mean_metrics = score(y_valid, mean_predictions)

        fold_rows.append({
            "asset": asset,
            "feature_set": feature_set_name,
            "fold": fold,
            "feature_count": len(feature_columns),
            "train_observations": len(train_idx),
            "validation_observations": len(valid_idx),
            "train_end": str(X.index[train_idx[-1]].date()),
            "validation_start": str(X.index[valid_idx[0]].date()),
            "row_separation": int(valid_idx[0] - train_idx[-1]),
            "model_mae": model_metrics["mae"],
            "model_rmse": model_metrics["rmse"],
            "model_directional_accuracy": model_metrics["directional_accuracy"],
            "zero_baseline_mae": zero_metrics["mae"],
            "zero_baseline_rmse": zero_metrics["rmse"],
            "historical_mean_mae": mean_metrics["mae"],
            "historical_mean_rmse": mean_metrics["rmse"],
            "historical_mean_directional_accuracy": mean_metrics["directional_accuracy"],
        })

        for date, actual, prediction, mean_prediction in zip(
            X.index[valid_idx], y_valid.to_numpy(), predictions, mean_predictions
        ):
            oof_rows.append({
                "asset": asset,
                "feature_set": feature_set_name,
                "fold": fold,
                "date": str(date.date()),
                "actual_return": float(actual),
                "prediction": float(prediction),
                "residual": float(actual - prediction),
                "historical_mean_prediction": float(mean_prediction),
            })

    folds_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    summary = {
        "asset": asset,
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "labeled_observations_common_sample": len(X),
        "folds": N_SPLITS,
        "gap_observations": HORIZON,
        "model_mae_mean": float(folds_df.model_mae.mean()),
        "model_rmse_mean": float(folds_df.model_rmse.mean()),
        "model_directional_accuracy_mean": float(folds_df.model_directional_accuracy.mean()),
        "zero_baseline_mae_mean": float(folds_df.zero_baseline_mae.mean()),
        "zero_baseline_rmse_mean": float(folds_df.zero_baseline_rmse.mean()),
        "historical_mean_baseline_mae_mean": float(folds_df.historical_mean_mae.mean()),
        "historical_mean_baseline_rmse_mean": float(folds_df.historical_mean_rmse.mean()),
        "historical_mean_baseline_directional_accuracy_mean": float(
            folds_df.historical_mean_directional_accuracy.mean()
        ),
        "oof_residual_variance": float(oof_df.residual.var(ddof=1)),
        "oof_predictions": len(oof_df),
    }
    return folds_df, oof_df, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market = get_synchronized_market_data().sort_index()
    macro = load_macro_yield_features(resolve_data_directory())

    if market.index.has_duplicates or not market.index.is_monotonic_increasing:
        raise ValueError("Market data dates must be unique and sorted.")
    if market.isna().any().any() or not np.isfinite(market.to_numpy(dtype=float)).all():
        raise ValueError("Market returns contain missing or non-finite values.")

    all_fold_results = []
    all_oof_results = []
    all_summaries = []

    print("MACRO FEATURE ABLATION EXPERIMENT")
    print(f"Market observations: {len(market)}")
    print(f"Macro observations: {len(macro)}")
    print(f"Assets: {list(market.columns)}")
    print(f"CV folds: {N_SPLITS} | gap: {HORIZON} observations")
    print("Both feature configurations use the exact same labeled dates.")
    print("early_stopping=False; production models will not be modified.\n")

    for asset in market.columns:
        technical = extract_asset_features(market[asset])
        with_macro = extract_asset_features(market[asset], macro)
        target = extract_forward_targets(market[asset], HORIZON)

        # Build one common sample so the comparison differs only by features,
        # not by dates or number of observations.
        combined = pd.concat(
            [technical, with_macro[["gsec_yield_level", "gsec_yield_change_bps"]], target],
            axis=1
        ).dropna()

        technical_cols = list(technical.columns)
        macro_cols = ["gsec_yield_level", "gsec_yield_change_bps"]
        y = combined[target.name]

        for label, columns in [
            ("technical_only", technical_cols),
            ("technical_plus_gsec_macro", technical_cols + macro_cols),
        ]:
            print(f"Evaluating {asset} | {label} ...")
            folds_df, oof_df, summary = evaluate_configuration(
                asset, combined, y, label, columns
            )
            all_fold_results.append(folds_df)
            all_oof_results.append(oof_df)
            all_summaries.append(summary)
            print(
                f"  MAE: {summary['model_mae_mean']:.6f} | "
                f"RMSE: {summary['model_rmse_mean']:.6f} | "
                f"Directional accuracy: "
                f"{summary['model_directional_accuracy_mean']:.2%}"
            )

    folds_out = pd.concat(all_fold_results, ignore_index=True)
    oof_out = pd.concat(all_oof_results, ignore_index=True)
    summary_out = pd.DataFrame(all_summaries)

    folds_out.to_csv(OUTPUT_DIR / "macro_comparison_fold_metrics.csv", index=False)
    oof_out.to_csv(OUTPUT_DIR / "macro_comparison_oof_predictions.csv", index=False)
    summary_out.to_csv(OUTPUT_DIR / "macro_comparison_summary.csv", index=False)

    metadata = {
        "method": "5-fold expanding TimeSeriesSplit",
        "gap_observations": HORIZON,
        "horizon_observations": HORIZON,
        "feature_configurations": [
            "technical_only",
            "technical_plus_gsec_macro",
        ],
        "same_labeled_dates_for_both_configurations": True,
        "early_stopping": False,
        "production_models_modified": False,
    }
    (OUTPUT_DIR / "macro_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nCOMPARISON COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print("  macro_comparison_summary.csv")
    print("  macro_comparison_fold_metrics.csv")
    print("  macro_comparison_oof_predictions.csv")
    print("  macro_comparison_metadata.json")


if __name__ == "__main__":
    main()
