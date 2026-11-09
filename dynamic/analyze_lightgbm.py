"""Calculate LightGBM Gain and SHAP importance on the held-out test split."""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/baselines/lightgbm.pkl"
TEST = ROOT / "data/splits/test.csv"
OUT = ROOT / "models/analysis"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL)
    df = pd.read_csv(TEST)
    X = df.drop(columns="label")
    gain = pd.DataFrame({"feature": X.columns, "gain": model.booster_.feature_importance(importance_type="gain")})
    gain["gain_percent"] = gain.gain / gain.gain.sum() * 100 if gain.gain.sum() else 0
    gain.sort_values("gain", ascending=False).to_csv(OUT / "gain_importance.csv", index=False)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    # Binary LightGBM may return either an ndarray or a list by SHAP version.
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    shap_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    shap_df["share_percent"] = shap_df.mean_abs_shap / shap_df.mean_abs_shap.sum() * 100
    shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(OUT / "shap_importance.csv", index=False)
    print("Gain importance:")
    print(gain.sort_values("gain", ascending=False).head(20).to_string(index=False))
    print("\nMean absolute SHAP:")
    print(shap_df.sort_values("mean_abs_shap", ascending=False).head(20).to_string(index=False))
    print(f"Saved analysis to {OUT}")

if __name__ == "__main__":
    main()
