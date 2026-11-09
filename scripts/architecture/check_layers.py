#!/usr/bin/env python3
"""Layer gate for a Cargo workspace (docs/architecture/layered-rewrite.md).

Helio's gate (`just arch`), and Pulsar-Native's: Pulsar runs this script from
its Helio submodule with `--root .` over its own architecture/layers.toml, so
both repositories hold their layers to one implementation.

Reads <root>/architecture/layers.toml and `cargo metadata`, and fails when:
- a workspace crate has no layer (named in [crates], or by the longest
  matching directory prefix in [paths]);
- a dependency breaks the layer rules and is not in
  architecture/layer-violations.txt (new coupling). Normal and build
  dependencies are checked; dev dependencies too when [check]
  dev_dependencies is true. A path dependency outside the workspace (an
  excluded crate) is checked by its directory;
- a recorded violation no longer exists (shrink the list with --update);
- a budget in [budgets] is exceeded, or has slack (lower it);
- code outside the test host (a crate with `test_host = true`, or in a
  layer with `names_kinds = true`) names an object kind ([kinds]) where
  architecture/kind-names.txt does not record it, or a recorded name is gone
  (docs/architecture/plugin-migration.md, "C. Helio repo is core only").
  [scan] dirs lists the directories read (default: crates, src).

Usage: python scripts/architecture/check_layers.py [--root DIR] [--update]

--update rewrites layer-violations.txt and drops kind-names.txt lines whose
names are gone; it never adds a kind name.
"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

def root_arg():
    args = sys.argv[1:]
    if "--root" in args:
        return Path(args[args.index("--root") + 1]).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = root_arg()
LAYERS = ROOT / "architecture" / "layers.toml"
VIOLATIONS = ROOT / "architecture" / "layer-violations.txt"
KIND_NAMES = ROOT / "architecture" / "kind-names.txt"
SCENE_STORE = re.compile(r"#\[derive\(([^)]*)\)\]", re.S)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STRING = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)
KEYLIKE = re.compile(r"[A-Za-z0-9_./:-]+")
WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
CFG_TEST_MOD = re.compile(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")


def workspace(config):
    """(deps, dirs): each member's checked dependencies as (name, dir, dev),
    and each member's directory. A dependency counts when it is a member or
    a path dependency (an excluded crate, checked by its directory)."""
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    meta = json.loads(out)
    dev = config.get("check", {}).get("dev_dependencies", False)
    dirs = {p["name"]: Path(p["manifest_path"]).resolve().parent for p in meta["packages"]}
    deps = {}
    for p in meta["packages"]:
        found = set()
        for d in p["dependencies"]:
            kind = d["kind"]
            if kind not in (None, "build") and not (dev and kind == "dev"):
                continue
            if d["name"] in dirs:
                found.add((d["name"], dirs[d["name"]], kind == "dev"))
            elif d.get("path"):
                found.add((d["name"], Path(d["path"]).resolve(), kind == "dev"))
        # A crate both used and dev-used is reported once, as used.
        used = {(n, at) for n, at, is_dev in found if not is_dev}
        deps[p["name"]] = sorted(
            (n, at, is_dev) for n, at, is_dev in found if not (is_dev and (n, at) in used)
        )
    return deps, dirs


def crate_info(config, name, path):
    """A crate's entry: [crates] by name, else the longest [paths] prefix
    containing its directory, else None."""
    if name in config.get("crates", {}):
        return config["crates"][name]
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    best = None
    for prefix, entry in config.get("paths", {}).items():
        prefix = prefix.rstrip("/")
        if rel == prefix or rel.startswith(prefix + "/"):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, entry)
    return best[1] if best else None


def exempt_from_kinds(config, info):
    """The test host: a crate marked `test_host`, or one in a layer whose
    crates may name kinds (`names_kinds`)."""
    if info is None:
        return False
    return bool(info.get("test_host") or config["layers"][info["layer"]].get("names_kinds"))


def violations(config, deps, dirs):
    layers = config["layers"]
    errors, found = [], []
    for name in sorted(deps):
        a = crate_info(config, name, dirs[name])
        if a is None:
            errors.append(f"{name}: no layer in architecture/layers.toml")
            continue
        for dep, path, is_dev in deps[name]:
            b = crate_info(config, dep, path)
            if b is None:
                continue  # a member without a layer is reported above
            same_group = a.get("group") and a.get("group") == b.get("group")
            if same_group:
                continue
            dev = " dev" if is_dev else ""
            if b["layer"] not in layers[a["layer"]]["may_use"]:
                found.append(f"{name} -> {dep}  ({a['layer']} ->{dev} {b['layer']})")
            elif a["layer"] == b["layer"] == "feature":
                found.append(f"{name} -> {dep}  (feature {a['group']} ->{dev} feature {b['group']})")
    return errors, found


def scene_store_derives(config, dirs):
    """Counts derives in non-host crates; hosts (examples, demos) may define
    their own components. A file belongs to the package with the deepest
    directory containing it, so files of excluded crates fall to the root
    package (a host) and are not counted."""
    by_depth = sorted(dirs.items(), key=lambda kv: len(kv[1].parts), reverse=True)
    count, sites = 0, []
    for top in ("crates", "crates_other", "src"):
        for path in sorted((ROOT / top).resolve().rglob("*.rs")):
            if "target" in path.parts:
                continue
            owner = next((n for n, d in by_depth if path.is_relative_to(d)), None)
            info = crate_info(config, owner, dirs[owner]) if owner else None
            if info is None or info.get("layer", "host") == "host":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            n = sum("SceneStore" in m.group(1) for m in SCENE_STORE.finditer(text))
            if n:
                count += n
                sites.append(f"  {n:3} {path.relative_to(ROOT.resolve()).as_posix()}")
    return count, sites


def words(ident):
    """`MESH_BOUNDS`, `meshBounds`, `MeshBoundsRow` -> lower-case words."""
    out = []
    for part in ident.split("_"):
        out += [w.lower() for w in WORD.findall(part)]
    return out


def strip_comments(text):
    """The code of a Rust or WGSL file: comments blanked, strings kept (a
    `//` inside a string literal is not a comment)."""
    out, i, n = [], 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\":
                out.append(text[i + 1:i + 2])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def checked_files(config, dirs):
    """The `.rs`/`.wgsl` files of every crate but the test host, without
    tests: `tests/` and `benches/`, modules declared `#[cfg(test)] mod x;`
    and what follows a `#[cfg(test)]` module in a file."""
    by_depth = sorted(dirs.items(), key=lambda kv: len(kv[1].parts), reverse=True)
    files = {}
    for top in config.get("scan", {}).get("dirs", ["crates", "src"]):
        for path in sorted((ROOT / top).resolve().rglob("*")):
            if path.suffix not in (".rs", ".wgsl") or "target" in path.parts:
                continue
            owner = next((n for n, d in by_depth if path.is_relative_to(d)), None)
            if owner is None or exempt_from_kinds(config, crate_info(config, owner, dirs[owner])):
                continue
            rel = path.relative_to(dirs[owner])
            if rel.parts[0] in ("tests", "benches", "examples"):
                continue
            files[path] = path.read_text(encoding="utf-8", errors="replace")
    # A `#[cfg(test)] mod x;` is x.rs, or the directory x/ with every file
    # under it (its own submodules).
    test_files, test_dirs = set(), set()
    for path, text in files.items():
        for name in CFG_TEST_MOD.findall(text):
            base = path.parent if path.name in ("lib.rs", "main.rs", "mod.rs") else path.with_suffix("")
            test_files.add(base / f"{name}.rs")
            test_dirs.add(base / name)
    for path in sorted(files):
        if path in test_files or any(path.is_relative_to(d) for d in test_dirs):
            continue
        text = files[path]
        cut = re.search(r"^\s*#\[cfg\(test\)\]\s*(?:pub\s+)?mod\s+\w+\s*\{", text, re.M)
        yield path, text[:cut.start()] if cut else text


def kind_names(config, dirs):
    """(path, kind) -> the identifiers and key strings naming that kind."""
    terms = [(kind, t.split()) for kind, ts in config.get("kinds", {}).items() for t in ts]
    found = {}
    for path, text in checked_files(config, dirs):
        code = strip_comments(text)
        tokens = IDENT.findall(STRING.sub(" ", code))
        tokens += [s for s in STRING.findall(code) if KEYLIKE.fullmatch(s)]
        rel = path.relative_to(ROOT.resolve()).as_posix()
        for token in tokens:
            ws = words(token)
            for kind, term in terms:
                if any(ws[i:i + len(term)] == term for i in range(len(ws) - len(term) + 1)):
                    found.setdefault((rel, kind), set()).add(token)
    return found


def recorded_kind_names():
    """(path, kind) -> reason, from architecture/kind-names.txt."""
    recorded, errors = {}, []
    if KIND_NAMES.exists():
        for n, line in enumerate(KIND_NAMES.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) < 3 or not parts[2]:
                errors.append(f"architecture/kind-names.txt:{n}: `path | kind | reason` with a reason")
                continue
            recorded[(parts[0], parts[1])] = parts[2]
    return recorded, errors


def main():
    update = "--update" in sys.argv[1:]
    config = tomllib.loads(LAYERS.read_text(encoding="utf-8"))
    deps, dirs = workspace(config)
    errors, found = violations(config, deps, dirs)
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
    derives, sites = scene_store_derives(config, dirs) if budget is not None else (0, [])
    budget_errors = []
    if budget is not None and derives > budget:
        budget_errors.append(f"#[derive(SceneStore)] below the host layer: {derives} > budget {budget}. "
                             "Components belong to plugins, not Helio:")
        budget_errors.extend(sites)
    elif budget is not None and derives < budget:
        budget_errors.append(f"#[derive(SceneStore)] below the host layer: {derives} < budget {budget}. "
                             "Lower scene_store_derives in architecture/layers.toml.")

    names = kind_names(config, dirs)
    recorded_names, name_errors = recorded_kind_names()
    new_names = sorted(set(names) - set(recorded_names))
    gone_names = sorted(set(recorded_names) - set(names))
    if update and gone_names and not name_errors:
        # Shrinking only: drop the lines whose names are gone, keep the rest
        # (and every comment) as written.
        dropped = set(gone_names)
        kept = [
            line for line in KIND_NAMES.read_text(encoding="utf-8").splitlines()
            if line.startswith("#") or not line.strip()
            or tuple(p.strip() for p in line.split("|", 2)[:2]) not in dropped
        ]
        KIND_NAMES.write_text("\n".join(kept) + "\n", encoding="utf-8")
        gone_names = []

    print(f"layer violations: {len(found)} recorded-and-present, "
          f"{len(new)} new, {len(gone)} fixed; SceneStore derives: {derives}; "
          f"kind names outside the test host: {len(names)} recorded, "
          f"{len(new_names)} new, {len(gone_names)} gone")
    for e in name_errors:
        print(f"error: {e}")
    for key in new_names:
        print(f"names a kind outside the test host: {key[0]} | {key[1]} "
              f"({', '.join(sorted(names[key]))}); the kind's contract is its plugin's")
    for key in gone_names:
        print(f"no longer names {key[1]} (remove from architecture/kind-names.txt): {key[0]}")
    for e in errors:
        print(f"error: {e}")
    for v in new:
        print(f"new violation: {v}")
    for v in gone:
        print(f"fixed (remove from layer-violations.txt, or run --update): {v}")
    for e in budget_errors:
        print(e)
    sys.exit(1 if errors or new or gone or budget_errors or name_errors or new_names or gone_names else 0)


if __name__ == "__main__":
    main()
