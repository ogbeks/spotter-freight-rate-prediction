"""Compare freight rate models on rolling date windows and build the submission files."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from baseline import CATEGORICAL, features, metrics, ridge_baseline

# Each window trains on every earlier month and tests on two later months.
WINDOWS = [("2025-05", "2025-06"), ("2025-07", "2025-08"), ("2025-09", "2025-10")]
REFERENCE = "boosted_shared"
BASELINES = ["ridge_baseline", "huber_shared"]
OPTIONS = ["boosted_shared", "boosted_market", "boosted_blend", "boosted_shared_day",
           "boosted_market_day", "boosted_shared_3mo"]
# The December inputs have no market index.
CHART_OPTIONS = ["boosted_shared", "boosted_shared_day", "boosted_shared_3mo"]


def locate_cities(rows):
    """Look up coordinates for December inputs using labeled shipment metadata."""
    endpoints = [
        rows[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for side in ("pickup", "delivery")
    ]
    return pd.concat(endpoints).groupby("city")[["lat", "lon"]].median()


def make_features(rows, city_coords):
    result = features(rows)
    for side in ("pickup", "delivery"):
        for axis in ("lat", "lon"):
            name = f"{side}_{axis}"
            result[name] = rows[name] if name in rows else result[side].map(city_coords[axis])
    for equipment in ("Dry Van", "Reefer", "Flatbed"):
        result[f"distance_{equipment}"] = result["distance"] * result["equipment"].eq(equipment)
    result["market_distance"] = rows["market_index"] * rows["distance"] if "market_index" in rows else np.nan
    result["day"] = pd.to_datetime(rows["date"]).dt.dayofyear
    return result


def robust_linear(columns):
    preprocess = ColumnTransformer([
        ("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), columns),
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(preprocess, HuberRegressor(epsilon=1.35, max_iter=1500))


def boosted_trees(columns):
    preprocess = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
        ("numbers", "passthrough", columns),
    ])
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error", categorical_features=[0, 1, 2], max_iter=350,
        max_leaf_nodes=23, min_samples_leaf=55, l2_regularization=8,
        learning_rate=0.08, early_stopping=False, random_state=11,
    )
    return make_pipeline(preprocess, regressor)


def predict(model, data):
    return np.maximum(model.predict(data), 0.01)


def fit_predict(name, x, y, month, x_new):
    """Train one named option and return its predictions."""
    if name == "boosted_blend":
        parts = [fit_predict(part, x, y, month, x_new) for part in ("boosted_shared", "boosted_market")]
        return sum(parts) / 2
    common = [c for c in x if c not in CATEGORICAL + ["market_distance", "day"]]
    if name == "ridge_baseline":
        model = ridge_baseline()
    elif name == "huber_shared":
        model = robust_linear(common)
    else:
        extra = {
            "boosted_market": ["market_distance"],
            "boosted_shared_day": ["day"],
            "boosted_market_day": ["market_distance", "day"],
        }.get(name, [])
        model = boosted_trees(common + extra)
    if name == "boosted_shared_3mo":
        keep = month.isin(sorted(month.unique())[-3:])
        x, y = x[keep], y[keep]
    return predict(model.fit(x, y), x_new)


def select(results, options):
    """Replace the reference only if an option has a lower average and a lower worst MAE."""
    reference = results[REFERENCE]
    better = [
        name for name in options
        if results[name]["average_mae"] < reference["average_mae"]
        and results[name]["worst_mae"] < reference["worst_mae"]
    ]
    return min(better, key=lambda name: results[name]["average_mae"]) if better else REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    data = args.data_dir
    train = pd.read_csv(data / "train_test.csv")
    validation = pd.read_csv(data / "validation.csv")
    template = pd.read_csv(data / "validation_predictions_template.csv")
    december = pd.read_csv(data / "december_chart_inputs.csv")
    if not validation.load_id.is_unique or not template.load_id.is_unique:
        raise ValueError("Shipment IDs must be unique in the prediction inputs.")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Shipment IDs must match between validation data and template.")

    month = train.date.str[:7]
    results = {name: {"windows": {}} for name in BASELINES + OPTIONS}
    for start, end in WINDOWS:
        fit = train[month < start]
        holdout = train[(month >= start) & (month <= end)]
        coords = locate_cities(fit)
        x_fit, x_holdout = make_features(fit, coords), make_features(holdout, coords)
        for name in results:
            estimated = fit_predict(name, x_fit, fit.posted_rate, month[fit.index], x_holdout)
            results[name]["windows"][f"{start} to {end}"] = metrics(holdout.posted_rate, estimated)
    for result in results.values():
        errors = [window["mae_usd"] for window in result["windows"].values()]
        result["average_mae"] = round(sum(errors) / len(errors), 2)
        result["worst_mae"] = max(errors)

    freight_model = select(results, OPTIONS)
    chart_model = select(results, CHART_OPTIONS)
    coords = locate_cities(train)
    x_train = make_features(train, coords)
    rates = fit_predict(freight_model, x_train, train.posted_rate, month, make_features(validation, coords))
    template["predicted_rate"] = template.load_id.map(pd.Series(rates, index=validation.load_id))
    december["predicted_rate"] = fit_predict(
        chart_model, x_train, train.posted_rate, month, make_features(december, coords)
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    template.to_csv(out / "validation_predictions.csv", index=False, float_format="%.2f")
    december.to_csv(out / "december_predictions.csv", index=False, float_format="%.2f")
    report = {"freight_model": freight_model, "chart_model": chart_model, "models": results}
    (out / "model_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame({
        name: {**{window: scores["mae_usd"] for window, scores in result["windows"].items()},
               "average": result["average_mae"], "worst": result["worst_mae"]}
        for name, result in results.items()
    }).T
    print("MAE in US dollars")
    print(table.to_string())
    print(f"\nValidation model: {freight_model}\nDecember model: {chart_model}")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
