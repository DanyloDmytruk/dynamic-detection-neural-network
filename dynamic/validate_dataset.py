
import json
from collections import Counter
from pathlib import Path

# Paths to collected behavior JSON files
DATASETS = {
    "benign": {
        "path": Path("../results/benign/benign"),
        "label": 0,
    },
    "malicious": {
        "path": Path("../results/malicious"),
        "label": 1,
    },
}


def validate_class(class_name, config):
    folder = config["path"]
    expected_label = config["label"]

    files = sorted(folder.rglob("*_behavior.json"))

    valid = 0
    invalid = []
    feature_sets = []
    return_codes = Counter()
    event_counts = []
    feature_types = {}

    sample_ids = set()

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")

            features = data.get("features")

            if not isinstance(features, dict):
                raise ValueError("Missing or invalid features")

            if data.get("label") != expected_label:
                raise ValueError(
                    f"Wrong label: {data.get('label')}, "
                    f"expected: {expected_label}"
                )

            if data.get("dataset_class") != class_name:
                raise ValueError(
                    f"Wrong dataset_class: {data.get('dataset_class')}"
                )

            # Save feature schema
            feature_sets.append(set(features.keys()))

            # Collect feature data types
            for name, value in features.items():
                feature_types.setdefault(name, Counter())[
                    type(value).__name__
                ] += 1

            return_codes[str(data.get("return_code"))] += 1

            event_count = data.get("event_count")

            if isinstance(event_count, (int, float)):
                event_counts.append(event_count)

            # Extract sample ID from filename
            sample_id = path.name.replace("_behavior.json", "")
            sample_ids.add(sample_id)

            valid += 1

        except Exception as exc:
            invalid.append({
                "file": str(path),
                "error": str(exc),
            })

    # Print class report
    print("\n" + "=" * 60)
    print(f"CLASS: {class_name.upper()}")
    print("=" * 60)

    print(f"Directory: {folder}")
    print(f"Total files: {len(files)}")
    print(f"Valid records: {valid}")
    print(f"Invalid records: {len(invalid)}")

    if event_counts:
        print("\nEvent count statistics:")
        print(f"  Min: {min(event_counts)}")
        print(f"  Max: {max(event_counts)}")
        print(f"  Average: {sum(event_counts) / len(event_counts):.2f}")
        print(f"  Zero events: {sum(x == 0 for x in event_counts)}")

    print("\nReturn code distribution:")

    for code, count in return_codes.most_common():
        print(f"  {code}: {count}")

    if feature_types:
        print("\nFeature types:")

        for name in sorted(feature_types):
            types = ", ".join(
                f"{kind}={count}"
                for kind, count in feature_types[name].items()
            )
            print(f"  {name}: {types}")

    if feature_sets:
        reference_schema = feature_sets[0]

        inconsistent = sum(
            current != reference_schema
            for current in feature_sets
        )

        print("\nFeature schema:")
        print(f"  Features: {len(reference_schema)}")
        print(f"  Inconsistent records: {inconsistent}")

        if inconsistent:
            print("  WARNING: Some records have different feature schemas")

    if invalid:
        print("\nFirst invalid records:")

        for item in invalid[:10]:
            print(f"  File: {item['file']}")
            print(f"  Error: {item['error']}")

    return {
        "class_name": class_name,
        "files": len(files),
        "valid": valid,
        "invalid": invalid,
        "feature_sets": feature_sets,
        "sample_ids": sample_ids,
    }


def main():
    results = {}

    for class_name, config in DATASETS.items():
        results[class_name] = validate_class(class_name, config)

    benign = results["benign"]
    malicious = results["malicious"]

    print("\n" + "=" * 60)
    print("CROSS-CLASS VALIDATION")
    print("=" * 60)

    if benign["feature_sets"] and malicious["feature_sets"]:
        benign_schema = benign["feature_sets"][0]
        malicious_schema = malicious["feature_sets"][0]

        print(f"Benign features: {len(benign_schema)}")
        print(f"Malicious features: {len(malicious_schema)}")
        print(f"Common features: {len(benign_schema & malicious_schema)}")

        print("\nOnly in benign:")
        print(sorted(benign_schema - malicious_schema))

        print("\nOnly in malicious:")
        print(sorted(malicious_schema - benign_schema))

        benign_inconsistent = sum(
            schema != benign_schema
            for schema in benign["feature_sets"]
        )

        malicious_inconsistent = sum(
            schema != malicious_schema
            for schema in malicious["feature_sets"]
        )

        print("\nInconsistent schemas:")
        print(f"  Benign: {benign_inconsistent}")
        print(f"  Malicious: {malicious_inconsistent}")

    overlap = benign["sample_ids"] & malicious["sample_ids"]

    print("\nSample ID overlap between classes:", len(overlap))

    if overlap:
        print("Examples:", sorted(overlap)[:10])

    total_files = benign["files"] + malicious["files"]
    total_valid = benign["valid"] + malicious["valid"]
    total_invalid = benign["invalid"] + malicious["invalid"]

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Total files: {total_files}")
    print(f"Valid records: {total_valid}")
    print(f"Invalid records: {len(total_invalid)}")
    print(f"Benign: {benign['valid']}")
    print(f"Malicious: {malicious['valid']}")

    # Save validation errors
    error_report = {
        "benign": benign["invalid"],
        "malicious": malicious["invalid"],
    }

    report_path = Path("validation_errors.json")

    report_path.write_text(
        json.dumps(error_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nValidation errors saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
