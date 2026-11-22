"""Evaluate simple freight rate baselines on a chronological split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = ["distance", "sqrt_distance", "weight", "day_sin", "day_cos"]


def features(rows):
    """Use fields available in both the shipment and December files."""
    result = rows[CATEGORICAL + ["distance", "weight"]].copy()
    result["weight"] = result["weight"].where(result["weight"] > 0)
    result["sqrt_distance"] = np.sqrt(result["distance"])
    weekday_angle = pd.to_datetime(rows["date"]).dt.dayofweek * (2 * np.pi / 7)
    result["day_sin"] = np.sin(weekday_angle)
    result["day_cos"] = np.cos(weekday_angle)
    return result


def ridge_baseline():
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(preprocess, Ridge(alpha=20.0, solver="lsqr"))


def metrics(actual, predicted):
    return {
        "mae_usd": round(mean_absolute_error(actual, predicted), 2),
        "rmse_usd": round(mean_squared_error(actual, predicted) ** 0.5, 2),
        "r2": round(r2_score(actual, predicted), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    rows = pd.read_csv(args.data_dir / "train_test.csv")
    dates = pd.to_datetime(rows["date"])
    fit = rows.loc[dates < "2025-09-01"]
    holdout = rows.loc[dates >= "2025-09-01"]
    target = holdout["posted_rate"]

    model = ridge_baseline()
    model.fit(features(fit), fit["posted_rate"])
    prediction = np.maximum(model.predict(features(holdout)), 0.01)
    by_month = holdout["date"].str[:7]
    report = {
        "split": {"holdout_start": "2025-09-01", "fit_rows": len(fit), "holdout_rows": len(holdout)},
        "training_median": metrics(target, np.full(len(target), fit["posted_rate"].median())),
        "ridge": metrics(target, prediction),
        "ridge_by_month": {
            month: metrics(target.loc[by_month == month], prediction[(by_month == month).to_numpy()])
            for month in sorted(by_month.unique())
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
