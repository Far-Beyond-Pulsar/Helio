"""Compare seeded HLFS GPU audit exports; run from the repository root.

Usage: python scripts/compare_hlfs_rt_audits.py BASELINE CANDIDATE OUTPUT.json
Grid order is allowed to differ. Overflow cases must be bit-identical.
"""
import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    data = path.read_bytes()
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite or empty audit: {path}")
    return data, values


def compare(baseline, candidate):
    result = {}
    names = ["local-grid", "local-overflow", "mixed-grid", "mixed-overflow"]
    for optional in ("packed-id-overflow", "hdr-material-overflow"):
        if any((p / f"{optional}.f32").exists() for p in (baseline, candidate)):
            names.append(optional)
    for name in names:
        a, av = read(baseline / f"{name}.f32")
        b, bv = read(candidate / f"{name}.f32")
        if len(a) != len(b):
            raise ValueError(f"Audit dimensions differ: {name}")
        count = len(av)
        mean_a, mean_b = sum(av) / count, sum(bv) / count
        result[name] = {
            "float_count": count,
            "baseline_sha256": hashlib.sha256(a).hexdigest(),
            "candidate_sha256": hashlib.sha256(b).hexdigest(),
            "bit_identical": a == b,
            "changed_floats": sum(x != y for x, y in zip(av, bv)),
            "baseline_mean": mean_a,
            "candidate_mean": mean_b,
            "relative_mean_error": abs(mean_b - mean_a) / max(abs(mean_a), 1e-20),
            "nrmse": math.sqrt(sum((x-y)**2 for x,y in zip(av,bv))/count) / max(abs(mean_a),1e-20),
        }
    return result


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    result = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(v["bit_identical"] for k,v in result.items() if k.endswith("overflow")):
        raise SystemExit("Overflow audit differs; inspect before accepting an equivalence claim.")
