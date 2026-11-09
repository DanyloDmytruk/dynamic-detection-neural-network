"""Train Logistic Regression, Random Forest and LightGBM baselines."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits"
OUT = ROOT / "models/baselines"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    df = pd.read_csv(SPLITS / f"{name}.csv")
    return df.drop(columns="label"), df.label.astype(int).to_numpy()


def pick_threshold(y, p):
    rows = []
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (p >= t).astype(int)
        rows.append((f1_score(y, pred, zero_division=0), float(t)))
    return max(rows)[1]


def score(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def main():
    Xtr, ytr = load("train"); Xv, yv = load("validation"); Xt, yt = load("test")
    if list(Xtr.columns) != list(Xv.columns) or list(Xtr.columns) != list(Xt.columns):
        raise ValueError("Feature columns differ")
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                                   max_depth=-1, min_child_samples=20, subsample=0.9,
                                   colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0,
                                   class_weight="balanced", random_state=42, n_jobs=-1,
                                   verbosity=-1),
    }
    all_metrics = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(Xtr, ytr)
        val_prob = model.predict_proba(Xv)[:, 1]
        threshold = pick_threshold(yv, val_prob)
        test_prob = model.predict_proba(Xt)[:, 1]
        all_metrics[name] = score(yt, test_prob, threshold)
        joblib.dump(model, OUT / f"{name}.pkl")
        print(name, all_metrics[name])
    (OUT / "baseline_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": k, **v} for k, v in all_metrics.items()]).to_csv(OUT / "baseline_comparison.csv", index=False)
    print(f"Saved baseline models and metrics to {OUT}")

if __name__ == "__main__":
    main()
