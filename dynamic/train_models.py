import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "dataset.csv"
OUTPUT_DIR = ROOT / "results"

RANDOM_STATE = 42


def evaluate_model(model, X, y, name):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=["sample_id", "label"])
    y = df["label"].astype(int)

    # 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Split temporary data equally: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("=== DATA SPLIT ===")
    print(f"Train:      {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")

    # Remove constant features using training data only
    selector = VarianceThreshold(threshold=0.0)

    X_train_selected = selector.fit_transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    selected_features = X.columns[selector.get_support()].tolist()
    removed_features = X.columns[~selector.get_support()].tolist()

    print(f"\nSelected features: {len(selected_features)}")
    print(f"Removed features: {len(removed_features)}")
    print("Removed:", removed_features)

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
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    validation_results = []

    print("\n=== VALIDATION RESULTS ===")

    for name, model in models.items():
        model.fit(X_train_selected, y_train)

        metrics = evaluate_model(
            model,
            X_val_selected,
            y_val,
            name,
        )

        validation_results.append(metrics)

        print(f"\n{name}")
        for key, value in metrics.items():
            if key not in ("model", "confusion_matrix"):
                print(f"  {key}: {value:.4f}")

        print("  Confusion matrix:")
        print(metrics["confusion_matrix"])

    # Select model using validation F1
    best_result = max(
        validation_results,
        key=lambda result: result["f1"],
    )

    best_name = best_result["model"]
    best_model = models[best_name]

    print(f"\nSelected model: {best_name}")

    # Refit selected model on train + validation
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    final_selector = VarianceThreshold(threshold=0.0)
    X_train_val_selected = final_selector.fit_transform(X_train_val)
    X_test_final = final_selector.transform(X_test)

    best_model.fit(X_train_val_selected, y_train_val)

    # Final evaluation on untouched test set
    test_metrics = evaluate_model(
        best_model,
        X_test_final,
        y_test,
        best_name,
    )

    print("\n=== FINAL TEST RESULTS ===")

    for key, value in test_metrics.items():
        if key not in ("model", "confusion_matrix"):
            print(f"{key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nClassification report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test_final),
        target_names=["Benign", "Malicious"],
        zero_division=0,
    ))

    # Save model and feature information
    joblib.dump(
        {
            "selector": final_selector,
            "model": best_model,
            "features": X.columns.tolist(),
            "selected_features": X_train_val.columns[
                final_selector.get_support()
            ].tolist(),
        },
        OUTPUT_DIR / "best_model.joblib",
    )

    report = {
        "validation_results": validation_results,
        "test_results": test_metrics,
        "selected_model": best_name,
        "selected_features": selected_features,
        "removed_features": removed_features,
    }

    with open(OUTPUT_DIR / "training_report.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\nModel saved:", OUTPUT_DIR / "best_model.joblib")
    print("Report saved:", OUTPUT_DIR / "training_report.json")


if __name__ == "__main__":
    main()
