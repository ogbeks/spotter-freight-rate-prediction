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

Optional: `python src/explore.py` prints the data checks and `python src/baseline.py` prints the ridge baseline on September and October.

## Approach

**Validation.** Models are tested on three later periods, each time training on all earlier months: May to June, July to August, and September to October. The final predictions are for later months, so a date split matches the task.

**Selection.** The reference model is kept unless another option has both a lower average MAE and a lower worst-period MAE. If several qualify, the lowest average wins. The selected models are refit on the labeled data before predicting.

**Data cleaning.** Missing weights (300) and nonpositive weights (292) are treated as missing. Extreme rates are kept.

**Features.** Pickup and delivery city, city coordinates, equipment, distance and its square root, distance by equipment, weight and weekday. The validation model adds market index times distance and day of year.

**Excluded.** `quote_signal` has a rank correlation of about 0.96 or stronger with rate per mile in nine of ten months, but the sign changes between months and it is near zero in August. Its direction in November and December cannot be known in advance.

## Results

MAE in US dollars:

| Model | May-Jun | Jul-Aug | Sep-Oct | Average | Worst |
| --- | --- | --- | --- | --- | --- |
| Ridge baseline | 188.91 | 139.10 | 138.25 | 155.42 | 188.91 |
| Huber | 199.10 | 112.38 | 109.69 | 140.39 | 199.10 |
| Boosted (reference) | 199.78 | 109.43 | 104.01 | 137.74 | 199.78 |
| Boosted, market index | 123.50 | 107.07 | 168.95 | 133.17 | 168.95 |
| Boosted, blend of the two above | 154.29 | 98.38 | 126.37 | 126.35 | 154.29 |
| Boosted, day of year | 123.52 | 187.79 | 109.36 | 140.22 | 187.79 |
| **Boosted, market index and day of year (validation model)** | **122.11** | **136.31** | **119.07** | **125.83** | **136.31** |
| Boosted, latest three months (December model) | 174.60 | 121.65 | 111.20 | 135.82 | 174.60 |

All boosted models use histogram gradient boosting with absolute-error loss. The December inputs have no market index, so that file uses the best qualifying option without it. Boosted results may vary slightly between platforms.

## Limitations

- The validation model beats the reference on average and in the worst period, but not in July to August or September to October.
- Three test periods is a small sample, and they were used to select the models, so the average may be slightly optimistic.
- Neither model estimates holiday effects. The December chart repeats weekly.
