"""Batch Sysmon collection wrapper for an isolated Windows VM."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTOR = HERE / "collector.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, choices=(0, 1), required=True, help="0=benign, 1=malicious")
    parser.add_argument("--input", type=Path, help="Defaults to benign_samples or malicious_samples")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    source = args.input or Path("benign_samples" if args.label == 0 else "malicious_samples")
    target_dir = args.results / ("benign" if args.label == 0 else "malicious")
    target_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted(source.rglob("*.exe"))
    if args.limit > 0:
        samples = samples[:args.limit]
    failures = []
    success = 0
    print(f"Found {len(samples)} EXE files in {source}")
    for index, sample in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {sample}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(COLLECTOR), str(sample), "--timeout", str(args.timeout), "--output", str(args.results)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                failures.append({"sample": str(sample), "error": result.stderr[-1500:]})
                print("  FAILED:", result.stderr.strip()[-500:])
                continue
            generated = args.results / f"{sample.stem}_behavior.json"
            if not generated.exists():
                failures.append({"sample": str(sample), "error": "collector output not found"})
                continue
            record = json.loads(generated.read_text(encoding="utf-8"))
            record["label"] = args.label
            record["dataset_class"] = "benign" if args.label == 0 else "malicious"
            destination = target_dir / generated.name
            destination.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            success += 1
        except Exception as exc:
            failures.append({"sample": str(sample), "error": str(exc)})
        if args.delay > 0:
            import time
            time.sleep(args.delay)
    failure_path = args.results / ("collection_failures_benign.json" if args.label == 0 else "collection_failures_malicious.json")
    failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Total: {len(samples)} | Successful: {success} | Failed: {len(failures)}")
    print(f"Failure report: {failure_path}")

if __name__ == "__main__":
    main()
