"""Group-aware train/validation/test split for the dynamic dataset."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dynamic_dataset.csv"
OUT = ROOT / "data/splits"


def main():
    df = pd.read_csv(DATA)
    if "label" not in df:
        raise SystemExit("Dataset must contain a 'label' column")
    features = [c for c in df.columns if c != "label"]
    # Group equal behavioral vectors so an exact vector cannot leak across splits.
    group_keys = pd.util.hash_pandas_object(df[features].astype(str), index=False).astype(str)
    group_label_counts = pd.DataFrame({"group": group_keys, "label": df.label}).groupby("group").label.nunique()
    if (group_label_counts > 1).any():
        raise SystemExit("Conflicting labels remain for identical feature vectors; rerun dataset_builder.py")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(df[features], df.label, groups=group_keys))
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()
    temp_groups = group_keys.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(gss2.split(temp[features], temp.label, groups=temp_groups))
    validation = temp.iloc[val_rel].copy()
    test = temp.iloc[test_rel].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    validation.to_csv(OUT / "validation.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)

    def summary(part):
        counts = part.label.value_counts().to_dict()
        return {"samples": len(part), "benign": int(counts.get(0, 0)), "malicious": int(counts.get(1, 0))}
    meta = {
        "feature_count": len(features), "features": features,
        "split": {"train": summary(train), "validation": summary(validation), "test": summary(test)},
        "method": "GroupShuffleSplit; exact duplicate feature vectors grouped by hash",
        "random_state": 42,
    }
    (OUT / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def group_set(part):
        return set(pd.util.hash_pandas_object(part[features].astype(str), index=False).astype(str))
    a, b, c = group_set(train), group_set(validation), group_set(test)
    print(f"Input samples: {len(df)}; features: {len(features)}")
    for name, part in (("Train", train), ("Validation", validation), ("Test", test)):
        print(f"{name}: {len(part)} | benign={int((part.label == 0).sum())} | malicious={int((part.label == 1).sum())}")
    print(f"Group intersections: train/val={len(a & b)}, train/test={len(a & c)}, val/test={len(b & c)}")
    print(f"Saved splits to {OUT}")

if __name__ == "__main__":
    main()
