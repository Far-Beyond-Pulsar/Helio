#!/usr/bin/env python3
"""Layer gate for the Helio workspace (docs/architecture/layered-rewrite.md).

Reads architecture/layers.toml and `cargo metadata`, and fails when:
- a workspace crate has no layer;
- a normal/build dependency breaks the layer rules and is not in
  architecture/layer-violations.txt (new coupling);
- a recorded violation no longer exists (shrink the list with --update);
- a crate outside a sealed group ([sealed_groups]) depends on one of its
  crates (normal/build; never recordable);
- a budget in [budgets] is exceeded, or has slack (lower it).

Usage: python scripts/architecture/check_layers.py [--update]
"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYERS = ROOT / "architecture" / "layers.toml"
VIOLATIONS = ROOT / "architecture" / "layer-violations.txt"
SCENE_STORE = re.compile(r"#\[derive\(([^)]*)\)\]", re.S)


def workspace():
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    meta = json.loads(out)
    names = {p["name"] for p in meta["packages"]}
    deps, dirs = {}, {}
    for p in meta["packages"]:
        deps[p["name"]] = sorted({
            d["name"] for d in p["dependencies"]
            if d["name"] in names and d["kind"] in (None, "build")
        })
        dirs[p["name"]] = Path(p["manifest_path"]).resolve().parent
    return deps, dirs


def violations(config, deps):
    layers, crates = config["layers"], config["crates"]
    sealed = config.get("sealed_groups", {})
    errors, found = [], []
    for name in sorted(deps):
        if name not in crates:
            errors.append(f"{name}: no layer in architecture/layers.toml")
            continue
        a = crates[name]
        for dep in deps[name]:
            b = crates.get(dep)
            if b is None:
                continue  # reported above
            same_group = a.get("group") and a.get("group") == b.get("group")
            if same_group:
                continue
            if b.get("group") in sealed:
                errors.append(f"{name} -> {dep}: {dep} is in the sealed group "
                              f"'{b['group']}' ({sealed[b['group']]}); only its own "
                              f"crates may depend on it")
                continue
            if b["layer"] not in layers[a["layer"]]["may_use"]:
                found.append(f"{name} -> {dep}  ({a['layer']} -> {b['layer']})")
            elif a["layer"] == b["layer"] == "feature":
                found.append(f"{name} -> {dep}  (feature {a['group']} -> feature {b['group']})")
    return errors, found


def scene_store_derives(config, dirs):
    """Counts derives in non-host crates; hosts (examples, demos) may define
    their own components. A file belongs to the package with the deepest
    directory containing it, so files of excluded crates fall to the root
    package (a host) and are not counted."""
    by_depth = sorted(dirs.items(), key=lambda kv: len(kv[1].parts), reverse=True)
    count, sites = 0, []
    for top in ("crates", "crates_other", "voxel", "src"):
        for path in sorted((ROOT / top).resolve().rglob("*.rs")):
            if "target" in path.parts:
                continue
            owner = next((n for n, d in by_depth if path.is_relative_to(d)), None)
            if owner is None or config["crates"].get(owner, {}).get("layer", "host") == "host":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            n = sum("SceneStore" in m.group(1) for m in SCENE_STORE.finditer(text))
            if n:
                count += n
                sites.append(f"  {n:3} {path.relative_to(ROOT.resolve()).as_posix()}")
    return count, sites


def main():
    update = "--update" in sys.argv[1:]
    config = tomllib.loads(LAYERS.read_text(encoding="utf-8"))
    deps, dirs = workspace()
    errors, found = violations(config, deps)
    recorded = set()
    if VIOLATIONS.exists():
        recorded = {
            line.strip() for line in VIOLATIONS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }

    new = sorted(set(found) - recorded)
    gone = sorted(recorded - set(found))
    if update and not errors:
        header = (
            "# Layer violations that exist today (scripts/architecture/check_layers.py).\n"
            "# This list may only shrink. Do not add to it: fix the dependency.\n"
        )
        VIOLATIONS.write_text(header + "".join(f"{v}\n" for v in sorted(found)), encoding="utf-8")
        new, gone = [], []

    budget = config.get("budgets", {}).get("scene_store_derives")
    derives, sites = scene_store_derives(config, dirs)
    budget_errors = []
    if budget is not None and derives > budget:
        budget_errors.append(f"#[derive(SceneStore)] in Helio: {derives} > budget {budget}. "
                             "Components belong to plugins, not Helio:")
        budget_errors.extend(sites)
    elif budget is not None and derives < budget:
        budget_errors.append(f"#[derive(SceneStore)] in Helio: {derives} < budget {budget}. "
                             "Lower scene_store_derives in architecture/layers.toml.")

    print(f"layer violations: {len(found)} recorded-and-present, "
          f"{len(new)} new, {len(gone)} fixed; SceneStore derives: {derives}")
    for e in errors:
        print(f"error: {e}")
    for v in new:
        print(f"new violation: {v}")
    for v in gone:
        print(f"fixed (remove from layer-violations.txt, or run --update): {v}")
    for e in budget_errors:
        print(e)
    sys.exit(1 if errors or new or gone or budget_errors else 0)


if __name__ == "__main__":
    main()
