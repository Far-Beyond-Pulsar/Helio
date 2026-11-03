#!/usr/bin/env python3
"""Co-change report: how often a change to one unit also changes others.

A unit is a feature group, or the layer of a non-feature crate (core, scene,
spine, renderer, preset), from architecture/layers.toml. Host crates
(examples, demos) are left out unless --include-hosts. Crates that no longer
exist count as their own unit.

The rewrite's phase-4 exit metric was the `voxel` row: commits touching voxel
that also touched another unit < 10% (docs/architecture/layered-rewrite.md).
Voxel's crates left this repository at phase 5 step 6.6 (they are the
pulsar.voxel plugin's, in Pulsar's plugins/voxel/); their history here counts
under their old crate names.

Usage: python scripts/architecture/co_change.py [--since 2026-10-01] [REV_RANGE]
       [--include-hosts] [--unit voxel]
"""

import argparse
import re
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = re.compile(r"^(?:crates/passes/[23]d|crates|crates_other)/([^/]+)/")


def unit_of(path, crates, renamed, include_hosts):
    m = CRATE_DIR.match(path)
    if not m:
        return None
    name = renamed.get(m.group(1), m.group(1))
    entry = crates.get(name)
    if entry is None:
        return name
    if entry["layer"] == "host":
        return name if include_hosts else None
    if entry["layer"] == "feature":
        return entry["group"]
    return entry["layer"]


def commits(args):
    cmd = ["git", "log", "--no-merges", "--format=@@%h", "--name-only"]
    if args.since:
        cmd.append(f"--since={args.since}")
    if args.range:
        cmd.append(args.range)
    out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    current = None
    for line in out.splitlines():
        if line.startswith("@@"):
            if current is not None:
                yield current
            current = set()
        elif line.strip():
            current.add(line.strip())
    if current is not None:
        yield current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("range", nargs="?")
    parser.add_argument("--since")
    parser.add_argument("--include-hosts", action="store_true")
    parser.add_argument("--unit", help="print only this unit's row")
    args = parser.parse_args()

    layers = tomllib.loads((ROOT / "architecture" / "layers.toml").read_text(encoding="utf-8"))
    crates, renamed = layers["crates"], layers.get("renamed", {})
    touched, spread, wide, partners = Counter(), Counter(), Counter(), {}
    for files in commits(args):
        units = {u for f in files if (u := unit_of(f, crates, renamed, args.include_hosts))}
        for u in units:
            touched[u] += 1
            if len(units) > 1:
                spread[u] += 1
                partners.setdefault(u, Counter()).update(units - {u})
            if len(units) >= 4:
                wide[u] += 1

    rows = [args.unit] if args.unit else [u for u, _ in touched.most_common()]
    print(f"{'unit':<16}{'commits':>8}{'+others':>9}{'%':>6}{'4+ units':>10}  top partners")
    for u in rows:
        n = touched[u]
        pct = 100.0 * spread[u] / n if n else 0.0
        top = ", ".join(f"{p} {c}" for p, c in partners.get(u, Counter()).most_common(5))
        print(f"{u:<16.16}{n:>8}{spread[u]:>9}{pct:>6.0f}{wide[u]:>10}  {top}")


if __name__ == "__main__":
    main()
