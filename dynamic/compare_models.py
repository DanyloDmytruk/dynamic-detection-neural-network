
from pathlib import Path

import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "dataset.csv"
OUTPUT_PATH = ROOT / "results" / "model_comparison.csv"

RANDOM_STATE = 42
N_SPLITS = 5

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["sample_id", "label"])
y = df["label"].astype(int)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    ),
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

all_results = []

print("=== DATASET ===")
print(f"Samples: {len(df)}")
print(f"Benign: {(y == 0).sum()}")
print(f"Malicious: {(y == 1).sum()}")
print(f"Features before filtering: {X.shape[1]}")
print(f"Cross-validation folds: {N_SPLITS}")

for name, classifier in models.items():

    pipeline = Pipeline([
        ("selector", VarianceThreshold(threshold=0.0)),
        ("classifier", classifier),
    ])

    print(f"\n=== {name} ===")

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    row = {"model": name}

    for metric in scoring:
        scores = results[f"test_{metric}"]

        mean = scores.mean()
        std = scores.std()

        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std

        print(f"{metric:12s}: {mean:.4f} (+/- {std:.4f})")

    all_results.append(row)

comparison = pd.DataFrame(all_results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_PATH, index=False)

print("\n=== FINAL COMPARISON ===")
print(comparison.to_string(index=False))

print(f"\nSaved to: {OUTPUT_PATH}")
