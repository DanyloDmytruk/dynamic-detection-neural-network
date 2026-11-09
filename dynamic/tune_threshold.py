"""Select classification threshold on validation data only."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = joblib.load(ROOT / "models/baselines/lightgbm.pkl")
    val = pd.read_csv(ROOT / "data/splits/validation.csv")
    X = val.drop(columns="label")
    y = val.label.astype(int)
    prob = model.predict_proba(X)[:, 1]
    rows = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        pred = (prob >= threshold).astype(int)
        rows.append({"threshold": round(float(threshold), 2),
                     "accuracy": accuracy_score(y, pred),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0)})
    df = pd.DataFrame(rows)
    best = df.loc[df.f1.idxmax()]
    out = ROOT / "models"
    df.to_csv(out / "validation_threshold_analysis.csv", index=False)
    (out / "threshold.json").write_text(json.dumps({"threshold": float(best.threshold)}, indent=2), encoding="utf-8")
    print(best.to_string())

if __name__ == "__main__":
    main()
