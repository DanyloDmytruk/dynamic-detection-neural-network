"""Build a labeled dynamic-analysis dataset from Sysmon behavior JSON files."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
from feature_extractor import extract_record, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def discover_records(results_dir: Path):
    paths = sorted(results_dir.rglob("*.json"))
    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] {path}: invalid JSON ({exc})")
            continue
        if not isinstance(obj, dict) or not ("events" in obj or "features" in obj):
            continue
        label = None
        parts = {p.lower() for p in path.parts}
        if "benign" in parts:
            label = 0
        elif "malicious" in parts or "malware" in parts:
            label = 1
        if label is None:
            label = obj.get("label")
        try:
            label = int(label)
        except (TypeError, ValueError):
            label = -1
        yield path, obj, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "data/dynamic_dataset.csv")
    parser.add_argument("--conflicts", type=Path, default=ROOT / "data/conflicting_samples.csv")
    args = parser.parse_args()

    rows, skipped_unknown = [], 0
    for path, record, label in discover_records(args.results):
        if label not in (0, 1):
            skipped_unknown += 1
            continue
        row = extract_record(record, label)
        row["source_json"] = str(path)
        rows.append(row)

    if not rows:
        raise SystemExit("No labeled behavior JSON records found. Expected results/benign and results/malicious.")

    df = pd.DataFrame(rows)
    feature_cols = [c for c in FEATURE_NAMES if c != "process_success" and c in df.columns]
    # Drop columns that are constant in this collection, matching the original workflow.
    constants = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    df.drop(columns=constants, inplace=True)
    feature_cols = [c for c in feature_cols if c not in constants]

    # Exact feature vectors with both labels are ambiguous; remove all rows in those groups.
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        key = tuple(row[c] for c in feature_cols)
        groups[key].append(idx)
    conflict_indices = set()
    conflict_rows = []
    for key, indices in groups.items():
        labels = {int(df.at[i, "label"]) for i in indices}
        if len(labels) > 1:
            conflict_indices.update(indices)
            conflict_rows.extend(df.loc[indices].to_dict("records"))

    conflict_df = pd.DataFrame(conflict_rows)
    clean = df.drop(index=list(conflict_indices)).copy()
    clean.drop(columns=["sample_path", "source_json"], errors="ignore", inplace=True)
    clean = clean[feature_cols + ["label", "sample"]]
    clean.drop(columns=["sample"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.conflicts.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    conflict_df.to_csv(args.conflicts, index=False)

    print(f"Initial labeled samples: {len(df)}")
    print(f"Skipped unlabeled samples: {skipped_unknown}")
    print(f"Constant features removed ({len(constants)}): {constants}")
    print(f"Conflicting samples removed: {len(conflict_indices)}")
    print(f"Usable samples: {len(clean)}")
    print("Class distribution:")
    print(clean.label.value_counts().sort_index().rename(index={0: "benign", 1: "malicious"}))
    print(f"Saved dataset: {args.output}")
    print(f"Saved conflicts: {args.conflicts}")

if __name__ == "__main__":
    main()
