"""Explore the freight data before modeling."""
import pandas as pd

train = pd.read_csv("data/train_test.csv")
validation = pd.read_csv("data/validation.csv")

print("Rows and dates")
for name, rows in (("train", train), ("validation", validation)):
    print(f"  {name}: {len(rows):,} rows, {rows.date.min()} to {rows.date.max()}")

print("\nMissing values")
missing = pd.DataFrame({"train": train.isna().sum(), "validation": validation.isna().sum()})
print(missing[missing.sum(axis=1) > 0])

print("\nData quality")
print("  Nonpositive weights in train:", (train.weight <= 0).sum())
print("  Duplicate load IDs in train:", train.load_id.duplicated().sum())

known = set(train.pickup) | set(train.delivery)
unseen = sorted((set(validation.pickup) | set(validation.delivery)) - known)
affected = validation.pickup.isin(unseen) | validation.delivery.isin(unseen)
print("  Validation cities not in train:", unseen)
print("  Validation rows affected:", affected.sum())

print("\nPosted rate")
print(train.posted_rate.describe().round(2))

per_mile = train.posted_rate / train.distance
print("\nRates above $10 per mile:", (per_mile > 10).sum())
print("\nMedian rate per mile by month")
print(per_mile.groupby(train.date.str[:7]).median().round(2))
print("\nMedian rate per mile by equipment")
print(per_mile.groupby(train.equipment).median().round(2))

print("\nCorrelation with posted rate")
columns = ["distance", "weight", "market_index", "quote_signal"]
print(train[columns].corrwith(train.posted_rate).round(3))
