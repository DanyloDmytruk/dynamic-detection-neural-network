"""Ablation experiment for LightGBM. Threshold is selected on validation only."""
from pathlib import Path
import argparse, json, joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remove", nargs="+", required=True, help="Feature names to exclude")
    p.add_argument("--name", default="ablation_custom")
    args = p.parse_args()
    split_dir = ROOT / "data/splits"
    out = ROOT / "models/baselines" / args.name
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = [pd.read_csv(split_dir / f"{n}.csv") for n in ("train", "validation", "test")]
    feature_names = [c for c in train.columns if c != "label" and c not in args.remove]
    missing = sorted(set(args.remove) - set(train.columns))
    if missing:
        raise SystemExit(f"Requested feature(s) not found: {missing}")
    Xtr, ytr = train[feature_names], train.label.astype(int)
    Xv, yv = val[feature_names], val.label.astype(int)
    Xt, yt = test[feature_names], test.label.astype(int)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                           max_depth=-1, min_child_samples=20, subsample=0.9,
                           colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0,
                           class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(Xtr, ytr)
    joblib.dump(model, out / "lightgbm_ablation.pkl")
    pv = model.predict_proba(Xv)[:, 1]
    thresholds = np.arange(0.05, 0.951, 0.01)
    best_t = max(thresholds, key=lambda t: f1_score(yv, (pv >= t).astype(int), zero_division=0))
    pt = model.predict_proba(Xt)[:, 1]
    pred = (pt >= best_t).astype(int)
    result = {
        "removed_features": args.remove,
        "feature_count": len(feature_names),
        "validation_threshold": float(best_t),
        "validation_f1": float(f1_score(yv, (pv >= best_t).astype(int), zero_division=0)),
        "test": {
            "accuracy": float(accuracy_score(yt, pred)),
            "precision": float(precision_score(yt, pred, zero_division=0)),
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, pt)),
            "pr_auc": float(average_precision_score(yt, pt)),
        },
    }
    importance = pd.DataFrame({"feature": feature_names,
                               "gain": model.booster_.feature_importance(importance_type="gain")})
    importance["gain_percent"] = importance.gain / importance.gain.sum() * 100 if importance.gain.sum() else 0
    importance.sort_values("gain", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    (out / "ablation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(importance.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
