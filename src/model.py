"""Compare freight rate models and build the submission files."""
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

COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CUTOFF = "2025-09-01"


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
    fit = train.loc[train.date < CUTOFF]
    holdout = train.loc[train.date >= CUTOFF]
    holdout_cities = locate_cities(fit)
    x_fit, x_holdout = (make_features(split, holdout_cities) for split in (fit, holdout))
    common = [name for name in x_fit if name not in CATEGORICAL + ["market_distance"]]
    models = {
        "ridge_baseline": ridge_baseline(),
        "huber_shared": robust_linear(common),
        "huber_market": robust_linear(common + ["market_distance"]),
        "boosted_shared": boosted_trees(common),
        "boosted_market": boosted_trees(common + ["market_distance"]),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_fit, fit.posted_rate)
        estimated = predict(model, x_holdout)
        results[name] = metrics(holdout.posted_rate, estimated)
        results[name]["by_month"] = {
            month: metrics(holdout.posted_rate[mask], estimated[mask.to_numpy()])
            for month in ("2025-09", "2025-10")
            for mask in [holdout.date.str.startswith(month)]
        }

    freight_model = min(models, key=lambda name: results[name]["mae_usd"])
    chart_model = min(
        ("ridge_baseline", "huber_shared", "boosted_shared"),
        key=lambda name: results[name]["mae_usd"],
    )
    if not validation.load_id.is_unique or not template.load_id.is_unique:
        raise ValueError("Shipment IDs must be unique in the prediction inputs.")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Shipment IDs must match between validation data and template.")
    coords = locate_cities(train)
    x_train = make_features(train, coords)
    for name in dict.fromkeys((freight_model, chart_model)):
        models[name].fit(x_train, train.posted_rate)

    rates = predict(models[freight_model], make_features(validation, coords))
    template["predicted_rate"] = template.load_id.map(pd.Series(rates, index=validation.load_id))
    december["predicted_rate"] = predict(models[chart_model], make_features(december, coords))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    template.to_csv(out / "validation_predictions.csv", index=False, float_format="%.2f")
    december.to_csv(out / "december_predictions.csv", index=False, float_format="%.2f")
    report = {
        "cutoff": CUTOFF, "fit_rows": len(fit), "holdout_rows": len(holdout),
        "freight_model": freight_model, "chart_model": chart_model, "models": results,
    }
    (out / "model_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
