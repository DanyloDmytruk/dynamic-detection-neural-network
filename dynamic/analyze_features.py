
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/dataset.csv"

df = pd.read_csv(DATASET)

features = df.drop(columns=["sample_id", "label"])

print("=== FEATURE ANALYSIS BY CLASS ===")

summary = features.groupby(df["label"]).agg(
    ["mean", "std", "min", "max"]
)

for feature in features.columns:
    benign_mean = features.loc[df["label"] == 0, feature].mean()
    malicious_mean = features.loc[df["label"] == 1, feature].mean()

    benign_nonzero = (
        features.loc[df["label"] == 0, feature].gt(0).sum()
    )
    malicious_nonzero = (
        features.loc[df["label"] == 1, feature].gt(0).sum()
    )

    print(f"\n{feature}")
    print(f"  Benign mean:       {benign_mean:.4f}")
    print(f"  Malicious mean:    {malicious_mean:.4f}")
    print(f"  Benign nonzero:    {benign_nonzero}/480")
    print(f"  Malicious nonzero: {malicious_nonzero}/175")

print("\n=== CLASS-SPECIFIC FEATURE VARIANCE ===")

for label, name in [(0, "BENIGN"), (1, "MALICIOUS")]:
    class_features = features.loc[df["label"] == label]

    constant = class_features.columns[
        class_features.nunique() <= 1
    ].tolist()

    print(f"\n{name}:")
    print("Constant features:", constant)
