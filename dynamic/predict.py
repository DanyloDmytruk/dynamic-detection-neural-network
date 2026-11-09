
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "results" / "best_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    return (
        bundle["selector"],
        bundle["model"],
        bundle["features"],
        bundle["selected_features"],
    )


def predict(features: dict):
    selector, model, expected_features, selected_features = load_model()

    # Check for missing features
    missing = set(expected_features) - set(features)

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    # Keep only features used during training
    features = {
        name: features[name]
        for name in expected_features
    }

    # Build input in exactly the same order as during training
    X = pd.DataFrame(
        [[features[name] for name in expected_features]],
        columns=expected_features,
    )
    
    # Validate numeric values
    X = X.apply(pd.to_numeric, errors="raise")

    if X.isna().any().any():
        raise ValueError("Features cannot contain NaN values.")

    if not X.map(lambda value: float("-inf") < value < float("inf")).all().all():
        raise ValueError("Features must contain finite values.")

    if (X < 0).any().any():
        raise ValueError("Features cannot contain negative values.")

    # Apply the same feature selection used during training
    X_selected = selector.transform(X)

    # Get probability of malicious class (label = 1)
    probabilities = model.predict_proba(X_selected)[0]

    malicious_index = list(model.classes_).index(1)
    benign_index = list(model.classes_).index(0)

    malicious_probability = float(probabilities[malicious_index])
    benign_probability = float(probabilities[benign_index])

    predicted_label = int(model.predict(X_selected)[0])

    return {
        "prediction": (
            "malicious" if predicted_label == 1 else "benign"
        ),
        "malicious_probability": malicious_probability,
        "benign_probability": benign_probability,
        "malicious_probability_percent": round(
            malicious_probability * 100, 2
        ),
        "benign_probability_percent": round(
            benign_probability * 100, 2
        ),
        "features_used": len(selected_features),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict malware probability using Random Forest."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file containing behavioral features.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain an object.")

    # Support both raw feature dictionaries and collector JSON
    if isinstance(data.get("features"), dict):
        features = data["features"]
    else:
        features = data

    result = predict(features)

    print("\n=== MALWARE PREDICTION ===")
    print(f"Prediction:          {result['prediction']}")
    print(
        f"Malicious probability: "
        f"{result['malicious_probability_percent']}%"
    )
    print(
        f"Benign probability:    "
        f"{result['benign_probability_percent']}%"
    )
    print(f"Features used:       {result['features_used']}")

    print("\nJSON result:")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
