# Freight Rate Prediction

Predicts `posted_rate` in US dollars for 12,000 loads from November and December 2025, using 48,000 labeled loads from January through October 2025. It also prices one fixed load for each day of December.

## Run

Requires Python 3.11 or 3.12. Tested with Python 3.12.13 on macOS. Create a `data/` folder in the repository root and add the assessment files:

- `train_test.csv`
- `validation.csv`
- `validation_predictions_template.csv`
- `december_chart_inputs.csv`

Use these exact names. The data is not included in this repository.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/model.py
cp outputs/validation_predictions.csv validation_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
```

`model.py` writes the predictions and metrics to `outputs/`. `score.py` is the supplied checker and creates `scorer_results/candidate_december.png`.

Optional: `python src/explore.py` prints the data checks and `python src/baseline.py` prints the baseline results.

## Approach

**Split.** Models train on January through August (38,477 loads) and are compared on September and October (9,523 loads), because the final predictions are for later months. The selected model is refit on all 48,000 loads.

**Data cleaning.** Missing weights (300) and nonpositive weights (292) are treated as missing. Extreme rates are kept.

**Features.** Pickup and delivery city, equipment, distance and its square root, distance by equipment, weight, city coordinates, and weekday. Coordinates help with eight validation cities not seen in training. `quote_signal` is excluded because its meaning is undocumented.

## Holdout results

| Model | MAE | RMSE |
| --- | --- | --- |
| Ridge baseline | $138.25 | $635.88 |
| Huber | $109.69 | $633.50 |
| Huber with market index | $166.75 | $652.19 |
| **Boosted trees (selected)** | **$104.01** | **$631.91** |
| Boosted trees with market index | $168.95 | $652.50 |

The selected model is histogram gradient boosting with absolute-error loss. Its MAE is 24.8% lower than the ridge baseline. Boosted results may vary slightly between platforms.

## Limitations

- Weekday is the only date feature, so the model does not estimate seasonal or holiday effects. The December chart repeats weekly.
- A small number of extreme rates dominate RMSE.
- The holdout was used to select the model, so its MAE may be slightly optimistic.
