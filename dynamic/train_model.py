"""Train the reconstructed MLP dynamic classifier."""
from pathlib import Path
import json
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits"
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def load_split(name):
    df = pd.read_csv(SPLITS / f"{name}.csv")
    if "label" not in df:
        raise ValueError(f"{name}.csv has no label column")
    return df.drop(columns=["label"]), df.label.astype(int).to_numpy()


def main():
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")
    if list(X_train.columns) != list(X_val.columns) or list(X_train.columns) != list(X_test.columns):
        raise ValueError("Feature columns differ between splits")
    if X_train.isna().any().any() or X_val.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values detected; clean/impute before training")
    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.astype("float32"))
    X_val_s = scaler.transform(X_val.astype("float32"))
    X_test_s = scaler.transform(X_test.astype("float32"))

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(k): float(v) for k, v in zip(classes, weights)}

    model = Sequential([
        Input(shape=(len(feature_names),)),
        Dense(64, activation="relu"),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="roc_auc"),
                 tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    history = model.fit(
        X_train_s, y_train, validation_data=(X_val_s, y_val),
        epochs=100, batch_size=32, class_weight=class_weight,
        callbacks=callbacks, verbose=1,
    )

    val_prob = model.predict(X_val_s, verbose=0).ravel()
    thresholds = np.arange(0.05, 0.951, 0.01)
    best_threshold = max(
        thresholds,
        key=lambda t: f1_score(y_val, (val_prob >= t).astype(int), zero_division=0),
    )
    test_prob = model.predict(X_test_s, verbose=0).ravel()
    test_pred = (test_prob >= best_threshold).astype(int)
    metrics = {
        "threshold_from_validation": float(best_threshold),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "features": feature_names,
    }
    model.save(OUT / "dynamic_mlp.keras")
    joblib.dump(scaler, OUT / "scaler.pkl")
    (OUT / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (OUT / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "threshold.json").write_text(json.dumps({"threshold": float(best_threshold)}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and metadata to {OUT}")

if __name__ == "__main__":
    main()
