"""Reconstructed Sysmon collector for an isolated Windows analysis VM.

Run ONLY inside a disposable, isolated Windows VM. This script executes the
provided sample. It is intended for defensive sandbox telemetry collection.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_NAME = "Microsoft-Windows-Sysmon/Operational"
EVENT_IDS = [1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26]
COUNTER_BY_ID = {
    1: "process_create_count", 3: "network_connection_count", 5: "process_terminate_count",
    6: "driver_load_count", 7: "image_load_count", 8: "create_remote_thread_count",
    10: "process_access_count", 11: "file_create_count", 12: "registry_create_delete_count",
    13: "registry_value_set_count", 14: "registry_rename_count", 15: "file_stream_hash_count",
    17: "pipe_create_count", 18: "pipe_connect_count", 19: "wmi_activity_count",
    20: "wmi_activity_count", 21: "wmi_activity_count", 22: "dns_query_count",
    25: "process_tampering_count", 26: "file_delete_count",
}


def powershell(script: str, timeout: int = 120) -> str:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"PowerShell failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def latest_record_id() -> int:
    output = powershell(
        f"$e=Get-WinEvent -LogName '{LOG_NAME}' -MaxEvents 1 -ErrorAction SilentlyContinue; "
        "if ($null -eq $e) { '0' } else { [string]$e.RecordId }"
    )
    try:
        return int(output.splitlines()[-1].strip()) if output else 0
    except ValueError:
        return 0


def read_events_after(record_id: int) -> list[dict]:
    event_filter = " or ".join(f"EventID={event_id}" for event_id in EVENT_IDS)

    xpath = (
        f"*[System[(EventRecordID > {record_id}) "
        f"and ({event_filter})]]"
    )

    script = f"""
$log = '{LOG_NAME}'
$xpath = "{xpath}"

$events = @(
    Get-WinEvent -LogName $log -FilterXPath $xpath -ErrorAction SilentlyContinue |
    Sort-Object RecordId |
    ForEach-Object {{
        $xml = [xml]$_.ToXml()
        $data = @{{}}

        foreach ($item in $xml.Event.EventData.Data) {{
            if ($null -ne $item.Name) {{
                $data[[string]$item.Name] = [string]$item.'#text'
            }}
        }}

        [pscustomobject]@{{
            event_id = [int]$_.Id
            record_id = [long]$_.RecordId
            time_created = $_.TimeCreated.ToUniversalTime().ToString('o')
            data = $data
        }}
    }}
)

ConvertTo-Json -InputObject $events -Depth 7 -Compress
"""

    output = powershell(script, timeout=60)

    if not output:
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not decode Sysmon JSON: {exc}; "
            f"output starts: {output[:300]}"
        )

    if isinstance(parsed, dict):
        return [parsed]

    return parsed if isinstance(parsed, list) else []


def _pid(value) -> int | None:
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None


def filter_process_tree(events: list[dict], root_pid: int) -> tuple[list[dict], list[int]]:
    process_ids = {root_pid}
    changed = True
    while changed:
        changed = False
        for event in events:
            if int(event.get("event_id", 0)) != 1:
                continue
            data = event.get("data") or {}
            pid = _pid(data.get("ProcessId"))
            parent = _pid(data.get("ParentProcessId"))
            if pid is not None and parent in process_ids and pid not in process_ids:
                process_ids.add(pid)
                changed = True
    selected = []
    for event in events:
        data = event.get("data") or {}
        eid = int(event.get("event_id", 0))
        actor = _pid(data.get("ProcessId"))
        if eid == 10:
            actor = _pid(data.get("SourceProcessId"))
        if actor in process_ids:
            selected.append(event)
    return selected, sorted(process_ids)


def aggregate(events: list[dict], duration: float) -> dict:
    from collections import Counter
    counts = Counter()
    domains, images, ips, ports, processes, parents = set(), set(), set(), set(), set(), set()
    created_dll = created_exe = created_script = deleted_dll = deleted_exe = 0
    run_changes = service_changes = 0
    for event in events:
        eid = int(event.get("event_id", 0))
        if eid in COUNTER_BY_ID:
            counts[COUNTER_BY_ID[eid]] += 1
        data = event.get("data") or {}
        for key, target in (("Image", processes), ("ParentImage", parents)):
            if data.get(key):
                target.add(str(data[key]).lower())
        if eid == 7:
            value = data.get("ImageLoaded") or data.get("Image")
            if value: images.add(str(value).lower())
        if eid == 22 and data.get("QueryName"):
            domains.add(str(data["QueryName"]).lower())
        if eid == 3:
            if data.get("DestinationIp"): ips.add(str(data["DestinationIp"]))
            if data.get("DestinationPort"): ports.add(str(data["DestinationPort"]))
        if eid in (11, 23):
            path = str(data.get("TargetFilename", "")).lower()
            ext = Path(path).suffix
            if eid == 11 and ext == ".dll": created_dll += 1
            if eid == 11 and ext == ".exe": created_exe += 1
            if eid == 11 and ext in {".ps1", ".bat", ".cmd", ".vbs", ".js", ".jse", ".wsf", ".hta"}: created_script += 1
            if eid == 23 and ext == ".dll": deleted_dll += 1
            if eid == 23 and ext == ".exe": deleted_exe += 1
        if eid in (12, 13, 14):
            target = str(data.get("TargetObject", "")).lower()
            if "\\run" in target or "\\runonce" in target: run_changes += 1
            if "\\services\\" in target and "\\services\\bam\\" not in target: service_changes += 1
    return {
        **{name: int(counts[name]) for name in set(COUNTER_BY_ID.values())},
        "event_count": len(events),
        "execution_duration_seconds": float(duration),
        "process_count": len(processes),
        "created_dll": created_dll, "created_exe": created_exe,
        "created_script": created_script, "deleted_dll": deleted_dll, "deleted_exe": deleted_exe,
        "run_key_changes": run_changes, "service_registry_changes": service_changes,
        "dns_domains": sorted(domains), "loaded_images": sorted(images),
        "remote_ips": sorted(ips), "remote_ports": sorted(ports),
        "unique_dns_domains": len(domains), "unique_loaded_images": len(images),
        "unique_parent_processes": len(parents), "unique_processes": len(processes),
        "unique_remote_ips": len(ips), "unique_remote_ports": len(ports),
    }


def run_sample(sample_path: Path, timeout: int, output_dir: Path) -> Path:
    sample_path = sample_path.resolve()
    if not sample_path.is_file() or sample_path.suffix.lower() != ".exe":
        raise ValueError(f"Expected an existing .exe file: {sample_path}")
    start_id = latest_record_id()
    started = datetime.now(timezone.utc)
    start_clock = time.monotonic()
    proc = subprocess.Popen([str(sample_path)], cwd=str(sample_path.parent))
    try:
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait(timeout=5)
    wall_duration = time.monotonic() - start_clock
    time.sleep(1.5)  # allow Sysmon to flush recent events
    events_all = read_events_after(start_id)
    events, process_tree = filter_process_tree(events_all, proc.pid)
    latest_event = max((e.get("time_created", "") for e in events), default="")
    observation_duration = wall_duration
    features = aggregate(events, observation_duration)
    record = {
        "sample": sample_path.name,
        "sample_path": str(sample_path),
        "start_time": started.isoformat(),
        "root_pid": proc.pid,
        "process_tree": process_tree,
        "return_code": int(return_code),
        "event_count": len(events),
        "observation_duration_seconds": round(observation_duration, 6),
        "latest_event_time": latest_event,
        "features": features,
        "events": events,
        "collector_note": "Reconstructed collector. Validate telemetry and feature parity against the original implementation.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_path.stem}_behavior.json"
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Collect Sysmon behavior for one EXE in an isolated Windows VM")
    parser.add_argument("sample", type=Path)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if os.name != "nt":
        raise SystemExit("This collector must run inside the Windows analysis VM.")
    try:
        result = run_sample(args.sample, args.timeout, args.output)
    except Exception as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Saved: {result}")

if __name__ == "__main__":
    main()
