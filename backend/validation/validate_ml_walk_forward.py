"""Independent ML walk-forward validation. Run from backend:
python validation/validate_ml_walk_forward.py
Does not overwrite production models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from ml.indian_market_data import get_synchronized_market_data
from ml.ml_engine import extract_asset_features, extract_forward_targets

BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND / "outputs" / "ml_validation"
HORIZON = 21
FOLDS = 5

def model_for_fold(fold):
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.03, max_leaf_nodes=15,
        min_samples_leaf=25, l2_regularization=2.0,
        early_stopping=False, random_state=42 + fold
    )

def metrics(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return (float(mean_absolute_error(y, pred)),
            float(np.sqrt(mean_squared_error(y, pred))),
            float(np.mean(np.sign(y) == np.sign(pred))))

def validate_asset(name, returns):
    returns = returns.sort_index().astype(float)
    features = extract_asset_features(returns)
    target = extract_forward_targets(returns, HORIZON)
    labeled = pd.concat([features, target], axis=1).dropna()
    X, y = labeled[list(features.columns)], labeled[target.name]
    splitter = TimeSeriesSplit(n_splits=FOLDS)
    fold_rows, prediction_rows = [], []

    for fold, (tr, va) in enumerate(splitter.split(X), 1):
        Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
        fitted = model_for_fold(fold)
        fitted.fit(Xtr, ytr)
        pred = fitted.predict(Xva)
        zero = np.zeros(len(yva))
        historical_mean = np.full(len(yva), float(ytr.mean()))
        mm, zm, hm = metrics(yva, pred), metrics(yva, zero), metrics(yva, historical_mean)
        fold_rows.append({
            "asset": name, "fold": fold, "train_n": len(tr), "validation_n": len(va),
            "train_end": str(X.index[tr[-1]].date()),
            "validation_start": str(X.index[va[0]].date()),
            "row_separation": int(va[0] - tr[-1]),
            "model_mae": mm[0], "model_rmse": mm[1], "model_directional_accuracy": mm[2],
            "zero_mae": zm[0], "zero_rmse": zm[1], "zero_directional_accuracy": zm[2],
            "historical_mean_mae": hm[0], "historical_mean_rmse": hm[1],
            "historical_mean_directional_accuracy": hm[2],
        })
        for date, actual, p, z, h in zip(X.index[va], yva.to_numpy(), pred, zero, historical_mean):
            prediction_rows.append({
                "asset": name, "date": str(date.date()), "actual_return": float(actual),
                "model_prediction": float(p), "zero_prediction": float(z),
                "historical_mean_prediction": float(h), "model_residual": float(actual-p)
            })

    fdf, pdf = pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)
    summary = {
        "asset": name, "labeled_observations": len(X), "features": X.shape[1],
        "horizon_observations": HORIZON, "gap_observations": HORIZON, "folds": FOLDS,
        "model_mae_mean": fdf.model_mae.mean(), "model_rmse_mean": fdf.model_rmse.mean(),
        "model_directional_accuracy_mean": fdf.model_directional_accuracy.mean(),
        "zero_baseline_mae_mean": fdf.zero_mae.mean(),
        "zero_baseline_rmse_mean": fdf.zero_rmse.mean(),
        "zero_baseline_directional_accuracy_mean": fdf.zero_directional_accuracy.mean(),
        "historical_mean_baseline_mae_mean": fdf.historical_mean_mae.mean(),
        "historical_mean_baseline_rmse_mean": fdf.historical_mean_rmse.mean(),
        "historical_mean_baseline_directional_accuracy_mean": fdf.historical_mean_directional_accuracy.mean(),
        "oof_residual_variance": pdf.model_residual.var(ddof=1),
        "oof_predictions": len(pdf),
    }
    return fdf, pdf, summary

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = get_synchronized_market_data()
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("Input dates must be sorted and unique.")
    if data.isna().any().any() or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Input contains missing or non-finite values.")

    print("INDEPENDENT WALK-FORWARD VALIDATION")
    print(f"Input: {len(data)} observations, {data.index.min().date()} to {data.index.max().date()}")
    print(f"Assets: {list(data.columns)} | horizon={HORIZON} | gap={HORIZON} | folds={FOLDS}")
    print("Early stopping disabled for controlled chronological validation.")
    print("Production model files will not be modified.\n")

    fold_tables, prediction_tables, summaries = [], [], []
    for asset in data.columns:
        print(f"Validating {asset} ...")
        fdf, pdf, summary = validate_asset(asset, data[asset])
        fold_tables.append(fdf)
        prediction_tables.append(pdf)
        summaries.append(summary)
        print(f"  MAE model/zero/mean: {summary['model_mae_mean']:.6f} / "
              f"{summary['zero_baseline_mae_mean']:.6f} / "
              f"{summary['historical_mean_baseline_mae_mean']:.6f}")
        print(f"  RMSE: {summary['model_rmse_mean']:.6f} | "
              f"directional accuracy: {summary['model_directional_accuracy_mean']:.2%}")
        print(f"  OOF residual variance: {summary['oof_residual_variance']:.8f}\n")

    pd.concat(fold_tables, ignore_index=True).to_csv(OUT / "ml_validation_fold_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(OUT / "ml_validation_oof_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "ml_validation_summary.csv", index=False)
    metadata = {
        "input_observations": len(data), "start_date": str(data.index.min().date()),
        "end_date": str(data.index.max().date()), "assets": list(data.columns),
        "horizon_observations": HORIZON, "gap_observations": HORIZON,
        "folds": FOLDS, "early_stopping": False, "production_models_modified": False
    }
    (OUT / "ml_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Results saved in: {OUT}")

if __name__ == "__main__":
    main()
