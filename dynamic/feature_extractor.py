"""Convert a Sysmon behavior JSON record into numeric features.

This is a reconstructed implementation based on the feature schema preserved
in the project history. Keep this file in sync with collector.py and the model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVENT_FEATURES = {
    1: "process_create_count", 3: "network_connection_count",
    5: "process_terminate_count", 6: "driver_load_count",
    7: "image_load_count", 8: "create_remote_thread_count",
    10: "process_access_count", 11: "file_create_count",
    12: "registry_create_delete_count", 13: "registry_value_set_count",
    14: "registry_rename_count", 15: "file_stream_hash_count",
    17: "pipe_create_count", 18: "pipe_connect_count",
    19: "wmi_activity_count", 20: "wmi_activity_count",
    21: "wmi_activity_count", 22: "dns_query_count",
    23: "file_delete_count", 25: "process_tampering_count",
}

FEATURE_NAMES = [
    "create_remote_thread_count", "created_dll", "created_exe",
    "created_script", "deleted_dll", "deleted_exe", "dns_query_count",
    "driver_load_count", "event_count", "execution_duration_seconds",
    "file_create_count", "file_delete_count", "file_stream_hash_count",
    "has_dns_activity", "has_file_activity", "has_injection_activity",
    "has_network_activity", "has_registry_activity", "has_wmi_activity",
    "image_load_count", "network_connection_count", "pipe_connect_count",
    "pipe_create_count", "process_access_count", "process_count",
    "process_create_count", "process_success", "process_tampering_count",
    "process_terminate_count", "registry_create_delete_count",
    "registry_rename_count", "registry_value_set_count", "run_key_changes",
    "service_registry_changes", "unique_dns_domains", "unique_loaded_images",
    "unique_parent_processes", "unique_processes", "unique_remote_ips",
    "unique_remote_ports", "wmi_activity_count",
]

CONSTANT_CANDIDATES = {
    "create_remote_thread_count", "created_script", "deleted_dll",
    "deleted_exe", "driver_load_count", "file_stream_hash_count",
    "has_wmi_activity", "pipe_connect_count", "pipe_create_count",
    "process_tampering_count", "process_terminate_count",
    "registry_rename_count", "wmi_activity_count",
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _event_id(event: dict[str, Any]) -> int:
    return _int(event.get("event_id", event.get("EventID", event.get("eventid", 0))))


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data", event.get("EventData", {}))
    return data if isinstance(data, dict) else {}


def _uniq(values: list[Any]) -> set[str]:
    return {str(v).strip().lower() for v in values if v not in (None, "", "-", "unknown")}


def extract_record(record: dict[str, Any], label: int | None = None) -> dict[str, Any]:
    events = record.get("events") or []
    raw = record.get("features") or {}
    if not isinstance(raw, dict):
        raw = {}

    counts = {name: 0 for name in set(EVENT_FEATURES.values())}
    processes, parents, dns, images, ips, ports = set(), set(), set(), set(), set(), set()
    created_dll = created_exe = created_script = deleted_dll = deleted_exe = 0
    run_key_changes = service_registry_changes = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        eid = _event_id(event)
        if eid in EVENT_FEATURES:
            counts[EVENT_FEATURES[eid]] += 1
        data = _event_data(event)
        image = data.get("Image") or data.get("image")
        parent = data.get("ParentImage") or data.get("ParentProcessName")
        if image:
            processes.add(str(image).lower())
        if parent:
            parents.add(str(parent).lower())
        if eid == 7:
            loaded = data.get("ImageLoaded") or data.get("Image")
            if loaded:
                images.add(str(loaded).lower())
        elif eid == 22:
            domain = data.get("QueryName") or data.get("query_name")
            if domain:
                dns.add(str(domain).lower())
        elif eid == 3:
            ip = data.get("DestinationIp") or data.get("DestinationIP")
            port = data.get("DestinationPort")
            if ip:
                ips.add(str(ip))
            if port not in (None, ""):
                ports.add(str(port))
        elif eid in (11, 23):
            path = str(data.get("TargetFilename") or data.get("target_filename") or "").lower()
            ext = Path(path).suffix
            is_create = eid == 11
            if ext == ".dll":
                created_dll += int(is_create)
                deleted_dll += int(not is_create)
            elif ext == ".exe":
                created_exe += int(is_create)
                deleted_exe += int(not is_create)
            elif ext in {".ps1", ".bat", ".cmd", ".vbs", ".js", ".jse", ".wsf", ".hta"} and is_create:
                created_script += 1
        elif eid in (12, 13, 14):
            target = str(data.get("TargetObject") or "").lower()
            if "\\run" in target or "\\runonce" in target:
                run_key_changes += 1
            if "\\services\\" in target and "\\services\\bam\\" not in target:
                service_registry_changes += 1

    def count(name: str) -> int:
        # Collector feature counters take precedence; event-derived values are fallback.
        return _int(raw.get(name), counts.get(name, 0))

    # Support older collector variants that stored unique values as lists.
    dns.update(_uniq(raw.get("dns_domains", []) if isinstance(raw.get("dns_domains", []), list) else []))
    images.update(_uniq(raw.get("loaded_images", []) if isinstance(raw.get("loaded_images", []), list) else []))
    ips.update(_uniq(raw.get("remote_ips", []) if isinstance(raw.get("remote_ips", []), list) else []))
    ports.update(_uniq(raw.get("remote_ports", []) if isinstance(raw.get("remote_ports", []), list) else []))

    event_count = _int(record.get("event_count"), len(events))
    duration = _float(raw.get("execution_duration_seconds", record.get("execution_duration_seconds", 0)))
    if duration <= 0:
        duration = _float(record.get("observation_duration_seconds", 0))

    values = {
        "create_remote_thread_count": count("create_remote_thread_count"),
        "created_dll": _int(raw.get("created_dll"), created_dll),
        "created_exe": _int(raw.get("created_exe"), created_exe),
        "created_script": _int(raw.get("created_script"), created_script),
        "deleted_dll": _int(raw.get("deleted_dll"), deleted_dll),
        "deleted_exe": _int(raw.get("deleted_exe"), deleted_exe),
        "dns_query_count": count("dns_query_count"),
        "driver_load_count": count("driver_load_count"),
        "event_count": event_count,
        "execution_duration_seconds": duration,
        "file_create_count": count("file_create_count"),
        "file_delete_count": count("file_delete_count"),
        "file_stream_hash_count": count("file_stream_hash_count"),
        "has_dns_activity": int(count("dns_query_count") > 0),
        "has_file_activity": int(count("file_create_count") + count("file_delete_count") > 0),
        "has_injection_activity": int(count("create_remote_thread_count") + count("process_access_count") > 0),
        "has_network_activity": int(count("network_connection_count") > 0),
        "has_registry_activity": int(count("registry_create_delete_count") + count("registry_value_set_count") + count("registry_rename_count") > 0),
        "has_wmi_activity": int(count("wmi_activity_count") > 0),
        "image_load_count": count("image_load_count"),
        "network_connection_count": count("network_connection_count"),
        "pipe_connect_count": count("pipe_connect_count"),
        "pipe_create_count": count("pipe_create_count"),
        "process_access_count": count("process_access_count"),
        "process_count": max(_int(raw.get("process_count"), 0), len(processes), 1 if record.get("root_pid") else 0),
        "process_create_count": count("process_create_count"),
        "process_success": int(_int(record.get("return_code"), -1) == 0),
        "process_tampering_count": count("process_tampering_count"),
        "process_terminate_count": count("process_terminate_count"),
        "registry_create_delete_count": count("registry_create_delete_count"),
        "registry_rename_count": count("registry_rename_count"),
        "registry_value_set_count": count("registry_value_set_count"),
        "run_key_changes": _int(raw.get("run_key_changes"), run_key_changes),
        "service_registry_changes": _int(raw.get("service_registry_changes"), service_registry_changes),
        "unique_dns_domains": _int(raw.get("unique_dns_domains"), len(dns)),
        "unique_loaded_images": _int(raw.get("unique_loaded_images"), len(images)),
        "unique_parent_processes": _int(raw.get("unique_parent_processes"), len(parents)),
        "unique_processes": _int(raw.get("unique_processes"), len(processes)),
        "unique_remote_ips": _int(raw.get("unique_remote_ips"), len(ips)),
        "unique_remote_ports": _int(raw.get("unique_remote_ports"), len(ports)),
        "wmi_activity_count": count("wmi_activity_count"),
    }

    assigned_label = label if label in (0, 1) else record.get("label")
    if assigned_label not in (0, 1, "0", "1"):
        assigned_label = -1
    values["label"] = int(assigned_label)
    values["sample"] = str(record.get("sample") or record.get("sample_path") or "unknown")
    values["sample_path"] = str(record.get("sample_path") or "")
    return values


def extract_file(path: str | Path, label: int | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    return extract_record(record, label=label)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description="Extract features from behavior JSON files")
    parser.add_argument("input", type=Path)
    parser.add_argument("--label", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path, default=Path("features.csv"))
    args = parser.parse_args()
    rows = [extract_file(p, args.label) for p in args.input.rglob("*.json")]
    if not rows:
        raise SystemExit(f"No JSON files found in {args.input}")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} records to {args.output}")
