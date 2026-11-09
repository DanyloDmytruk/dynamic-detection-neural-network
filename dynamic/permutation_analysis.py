from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "dataset.csv"
OUTPUT_PATH = ROOT / "results" / "permutation_importance.csv"

RANDOM_STATE = 42

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["sample_id", "label"])
y = df["label"].astype(int)

# Reproduce the original split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val, _, y_val, _ = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

# Fit feature selector only on training data
selector = VarianceThreshold(threshold=0.0)

X_train_selected = selector.fit_transform(X_train)
X_val_selected = selector.transform(X_val)

feature_names = X.columns[selector.get_support()]

# Train the same Random Forest configuration
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train_selected, y_train)

# Calculate permutation importance on validation data
result = permutation_importance(
    model,
    X_val_selected,
    y_val,
    scoring="average_precision",
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
})

importance = importance.sort_values(
    "importance_mean",
    ascending=False,
).reset_index(drop=True)

print("=== PERMUTATION IMPORTANCE ===")
print(importance.to_string(index=False))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
importance.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
