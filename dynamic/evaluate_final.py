"""One final test evaluation using a threshold selected on validation data."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = joblib.load(ROOT / "models/baselines/lightgbm.pkl")
    test = pd.read_csv(ROOT / "data/splits/test.csv")
    threshold_data = json.loads((ROOT / "models/threshold.json").read_text(encoding="utf-8"))
    threshold = float(threshold_data["threshold"])
    X, y = test.drop(columns="label"), test.label.astype(int).to_numpy()
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    result = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }
    print(json.dumps(result, indent=2))
    (ROOT / "models/final_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
