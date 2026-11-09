
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

SOURCES = {
    "benign": {
        "path": PROJECT_ROOT / "results/benign/benign",
        "label": 0,
    },
    "malicious": {
        "path": PROJECT_ROOT / "results/malicious",
        "label": 1,
    },
}

OUTPUT_DIR = ROOT / "data"

FEATURES = [
    "create_remote_thread_count",
    "created_dll",
    "created_exe",
    "created_script",
    "deleted_dll",
    "deleted_exe",
    "dns_query_count",
    "driver_load_count",
    "file_create_count",
    "file_delete_count",
    "file_stream_hash_count",
    "image_load_count",
    "network_connection_count",
    "pipe_connect_count",
    "pipe_create_count",
    "process_access_count",
    "process_create_count",
    "process_tampering_count",
    "process_terminate_count",
    "registry_create_delete_count",
    "registry_rename_count",
    "registry_value_set_count",
    "run_key_changes",
    "service_registry_changes",
    "unique_dns_domains",
    "unique_loaded_images",
    "unique_remote_ips",
    "unique_remote_ports",
    "wmi_activity_count",
]


def load_records():
    records = []
    metadata = []
    sample_ids = set()
    class_counts = Counter()

    for class_name, config in SOURCES.items():
        folder = config["path"]
        expected_label = config["label"]

        files = sorted(folder.rglob("*_behavior.json"))

        print(f"\nReading {class_name}: {len(files)} files")

        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSON object: {path}")

            if data.get("label") != expected_label:
                raise ValueError(f"Incorrect label: {path}")

            if data.get("dataset_class") != class_name:
                raise ValueError(f"Incorrect class: {path}")

            features = data.get("features")

            if not isinstance(features, dict):
                raise ValueError(f"Invalid features: {path}")

            missing = set(FEATURES) - set(features)

            if missing:
                raise ValueError(
                    f"Missing features in {path}: {sorted(missing)}"
                )

            sample_id = path.name.replace("_behavior.json", "")

            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")

            sample_ids.add(sample_id)

            row = {
                "sample_id": sample_id,
                "label": expected_label,
            }

            for feature in FEATURES:
                value = features[feature]

                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(
                        f"Non-numeric feature {feature} in {path}: {value}"
                    )

                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite feature {feature} in {path}: {value}"
                    )

                if value < 0:
                    raise ValueError(
                        f"Negative feature {feature} in {path}: {value}"
                    )

                row[feature] = value

            records.append(row)

            metadata.append({
                "sample_id": sample_id,
                "label": expected_label,
                "dataset_class": class_name,
                "return_code": data.get("return_code"),
                "event_count": data.get("event_count"),
                "execution_duration_seconds": features.get(
                    "execution_duration_seconds"
                ),
                "source_file": str(path.relative_to(PROJECT_ROOT)),
            })

            class_counts[class_name] += 1

    return records, metadata, class_counts


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records, metadata, class_counts = load_records()

    # Deterministic ordering by sample ID
    records.sort(key=lambda row: row["sample_id"])
    metadata.sort(key=lambda row: row["sample_id"])

    dataset_fields = ["sample_id", "label"] + FEATURES

    metadata_fields = [
        "sample_id",
        "label",
        "dataset_class",
        "return_code",
        "event_count",
        "execution_duration_seconds",
        "source_file",
    ]

    dataset_path = OUTPUT_DIR / "dataset.csv"
    metadata_path = OUTPUT_DIR / "metadata.csv"

    save_csv(dataset_path, records, dataset_fields)
    save_csv(metadata_path, metadata, metadata_fields)

    label_counts = Counter(row["label"] for row in records)

    report = {
        "total_samples": len(records),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "classes": {
            "benign": {
                "label": 0,
                "samples": class_counts["benign"],
            },
            "malicious": {
                "label": 1,
                "samples": class_counts["malicious"],
            },
        },
        "label_distribution": {
            str(label): count
            for label, count in sorted(label_counts.items())
        },
        "dataset_file": str(dataset_path),
        "metadata_file": str(metadata_path),
    }

    report_path = OUTPUT_DIR / "dataset_report.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("DATASET BUILD SUMMARY")
    print("=" * 55)

    print(f"Total samples: {len(records)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Benign: {class_counts['benign']}")
    print(f"Malicious: {class_counts['malicious']}")

    print("\nGenerated files:")
    print(f"  {dataset_path}")
    print(f"  {metadata_path}")
    print(f"  {report_path}")

    print("\nDataset successfully created.")


if __name__ == "__main__":
    main()
