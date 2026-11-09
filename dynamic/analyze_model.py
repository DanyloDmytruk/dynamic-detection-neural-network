from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "results" / "best_model.joblib"
OUTPUT_PATH = ROOT / "results" / "feature_importance.csv"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["selected_features"]

if not hasattr(model, "feature_importances_"):
    raise TypeError("The selected model does not support feature_importances_")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

importance["importance_percent"] = (
    importance["importance"] * 100
)

print("=== RANDOM FOREST FEATURE IMPORTANCE ===")
print(importance.to_string(index=False))

importance.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
