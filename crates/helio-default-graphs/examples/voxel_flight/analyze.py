"""Summarize flight source frames and asynchronous GPU timings; stdlib only.

python analyze.py RUN_DIRECTORY [--compare-images BASELINE_DIRECTORY]
Quantiles use nearest rank. A warm frame has complete, current residency.
GPU stages are separate scopes: do not add children to their parent graph.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def statistics(values):
    values = sorted(values)
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    return {"n": len(values), "p50": values[math.ceil(len(values) * .5) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1], "max": values[-1]}


def summarize(path):
    source_frames = read_csv(path / "frames.csv")
    frames = {int(row["frame"]): row for row in source_frames}
    assert len(frames) == len(source_frames), "duplicate source frame"
    groups = defaultdict(list)
    for frame in frames.values():
        warm = (frame["ready"] == "true" and frame["refining"] == "false"
                and frame["planning"] == "false" and int(frame["pending"]) == 0)
        for phase in ["all"] + (["warm"] if warm else []):
            for metric in ["sync_frame_ms", "cpu_submit_ms", "gpu_wait_ms"]:
                if metric in frame:
                    groups[frame["stage"], phase, metric].append(float(frame[metric]))
    seen = set()
    coverage = defaultdict(set)
    diagnostics = defaultdict(lambda: {"max_lag_frames": 0, "readback_drops": 0, "query_overflows": 0})
    epoch_counters = defaultdict(lambda: {"readback_drops": 0, "query_overflows": 0})
    for row in read_csv(path / "gpu.csv"):
        key = (row["domain"], int(row["epoch"]), int(row["engine_frame"]), row["pass"])
        assert key not in seen, f"duplicate sample {key}"
        seen.add(key)
        index = int(row["flight_frame"])
        frame = frames[index]
        assert frame["stage"] == row["stage"], f"incorrect source stage: {row}"
        coverage[row["domain"]].add(index)
        domain = diagnostics[row["domain"]]
        counters = epoch_counters[row["domain"], int(row["epoch"])]
        for field in ["readback_drops", "query_overflows"]:
            counters[field] = max(counters[field], int(row[field]))
        domain["max_lag_frames"] = max(domain["max_lag_frames"], int(row["lag_frames"]))
        warm = (frame["ready"] == "true" and frame["refining"] == "false"
                and frame["planning"] == "false" and int(frame["pending"]) == 0)
        for phase in ["all"] + (["warm"] if warm else []):
            groups[row["stage"], phase, row["pass"] + "_gpu_ms"].append(float(row["gpu_ms"]))
    rows = [{"stage": key[0], "phase": key[1], "metric": key[2], **statistics(value)}
            for key, value in sorted(groups.items())]
    with (path / "analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, indices in coverage.items():
        diagnostics[name]["frames_with_samples"] = len(indices)
        diagnostics[name]["frames_without_samples"] = sorted(frames.keys() - indices)
    for (name, _), counters in epoch_counters.items():
        for field, count in counters.items():
            diagnostics[name][field] += count
    report = {"frame_count": len(frames), "gpu_readbacks": dict(diagnostics),
              "memory": read_csv(path / "memory.csv")}
    (path / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare-images", type=Path)
    args = parser.parse_args()
    report = summarize(args.run)
    if args.compare_images:
        # Exact PNG file equality is sufficient; unequal files need pixel or
        # visual inspection before concluding that the image actually changed.
        comparison = {}
        for image in sorted(args.run.glob("*.png")):
            baseline = args.compare_images / image.name
            if baseline.is_file():
                comparison[image.name] = hashlib.sha256(image.read_bytes()).digest() == hashlib.sha256(baseline.read_bytes()).digest()
        report["identical_png_files"] = comparison
        (args.run / "image-comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
