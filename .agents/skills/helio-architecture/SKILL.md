---
name: helio-architecture
description: Mandatory for any work that touches Helio (the renderer) or Pulsar's rendering side: Helio crates, Pulsar plugins under plugins/vendor/*, SceneDB components that publish GPU rows, render features and passes, WGSL, render graphs and presets, scene derivations, or editor tools for an object kind. Use it before reading, designing, editing, reviewing or briefing a session or subagent about such work, including bug fixes, perf work, tests and small changes. It makes the agent read the code first, keep the layered plugin architecture, avoid duplicating any concept, and hold the design to AAA quality, with no naive solutions or cheap tricks.
---

# Helio architecture discipline

Helio is meant to be an AAA renderer, and Pulsar its engine and editor. Every
change has to leave the architecture cleaner than it found it. A fix that
works but is naive, duplicated, special-cased or put in the wrong layer is a
regression, even when the tests pass.

Follow the steps in order. Don't write code before step 3.

## 1. Read before you design

The architecture is written down. Read it, then read the code it describes.
Never design from memory, from a summary, or from what the code "probably" does.

- **Plan docs (Helio, `docs/architecture/`):**
  - `layered-rewrite.md`: layers, rules, buffer contracts, extension points, why this shape.
  - `plugin-migration.md`: every object kind is a plugin, what Helio is, the plugin table, and the "As built" sections for what actually landed.
- **Layer gate:** `architecture/layers.toml`, plus `architecture/layer-violations.txt` (this list only shrinks).
- **Pulsar agent docs:** `AGENTS.md` and the relevant `.agents/*.md` (SceneDB, ECS, render graph, plugin system, frames, lens, LOD, memory, and so on).
- **The code itself:**
  - The owning plugin's crates.
  - The api crate of each concept you will touch.
  - The frame and spine contracts.
  - Every current caller of what you change.
  - Read whole functions and files, not grep hits.
  - Follow the data: where a value is produced, published (key, row ABI), consumed, and shown.
- **Existing tests for the area:** they tell you the intended behaviour, and what is already broken.

Write down, for yourself, what owns the thing you are changing and how data flows through it. If you can't, you haven't read enough.

## 2. Find the owner, and check nothing already does it

**Every concept has exactly one owner.**

| Concept | Owner |
|---|---|
| Object kinds: mesh, light, material, sky, fog, water, voxel, foliage, VG, decal, portal, lens, post, GI, particles, sprites, splines | Their plugin in Pulsar `plugins/vendor/<kind>/`: `api`, `component`, `feature`, `editor`, `examples` |
| Kind-agnostic pieces: views, cameras, jitter, motion vectors, draw phases, G-buffer layout, scene colour chain, lighting-input contracts, uploads, the graph | The frame and spine in Helio core |

Before adding anything (a struct, WGSL function, constant, key, buffer, pass, setting, registry, trait or helper), search for an existing one:

- `rg` the concept's names and synonyms across Helio and Pulsar, including WGSL snippets, `*-api` crates, `helio-frame-api`, the prelude, and `crates/core`.
- Look for an existing seam before inventing one. Seams that exist include:
  - feature registrar methods;
  - scene derivations;
  - extension points (`GBufferContributor`, `ShadowCaster`, `RayProxySource`, `LightingContributor`, `ParticipatingMedia`, `PostEffect`, `SurfaceQuery`, `HistoryInvalidation`);
  - draw phases;
  - the editor's `ui_level_editor::extend` registrations;
  - the asset loader and reload registrations;
  - by-name component methods.
- If it exists, use it: include the owner's WGSL snippet and depend on the owner's api crate.
- If it exists in the wrong place, move it to its owner in the same change, and delete the old one.

Never:
- copy a struct, constant, WGSL function or row layout;
- keep a private fallback copy for when the owner is absent (absent owner = a stand-in through the contract);
- name another owner's key by string literal (use its `BufferContract`/`PoolContract`/key const);
- add a second path that does what an existing path does.

Pulsar's `crates/core/one_owner` test catches WGSL and key copies. Its `exceptions.txt` only shrinks. Never add a line to get a change through.

## 3. Hold the design to AAA, at scale

Before writing code, write a short design note answering four questions:

1. **How do the best engines solve this?** Name concrete techniques and where they come from (e.g. UE5 Nanite/Lumen/VSM/RDG, Frostbite FrameGraph, Unity SRP/HDRP, Decima, id Tech, Bevy, published GDC/SIGGRAPH work). Use real knowledge, not vague references. Say what each costs and where each breaks.
2. **Why is ours at least as good for our constraints?** The engine is generic: it serves any game, any world type and any mix of plugins, and no single setup is its target. Engine-wide constraints are:
   - tens of thousands of objects, many lights, volumes and views;
   - large worlds (f64 world positions rebased to the frame origin);
   - any plugin may be absent;
   - hardware ranges from modest to high-end. A design must scale up to use more cores, VRAM and GPU throughput, and down through quality settings. It is never capped at, or tuned to, the machine it was developed on.

   Each kind adds its own constraints. For voxels, that means Earth- and Moon-scale planets (6371 km), 10 cm voxels and full destruction. Judge a design against the engine-wide constraints plus those of the kinds it touches, and never shape engine code around one kind's or one example's needs.

   If a better known design exists, use it instead of the weaker one, even when it is more work.
3. **What scales with what?** CPU, GPU and memory cost versus objects, lights, views, screen size, world size and edits. Per-frame work must scale with what changed or what is visible, never with everything that exists.
4. **Where does it live?** The layer and crate (step 2), and why it can't live lower or more generically.

### Red flags

Each of these means the design is wrong; redesign instead of shipping it:

**Special-casing**
- Magic constants or thresholds tuned until one scene or one test looks right.
- A fix that only works in one example (voxel_craft, one level) instead of the engine path all setups share.
- Kind names, `if kind == ...`, or downcasts to a concrete pass in core, the spine, presets or a host.
- Renderer setters or mailboxes for one kind. Component methods are the only API, and the editor and scripts call the same ones.

**Per-frame cost and stalls**
- Per-frame O(all objects) CPU scans, per-frame GPU buffer or bind-group re-creation, or per-frame allocation in hot paths.
- Blocking readbacks, `device.poll(Wait)` in the frame, sleeps, or "wait N frames" instead of a real readiness signal.
- Picking "the first directional light" or "the first anything" instead of an explicit contract.

**Hiding problems**
- Hiding an artefact (bias, blur, clamp, fade, extra samples) without fixing its cause.
- Loosening a test threshold, disabling or ignoring a test, or catching and dropping an error so a change passes.
- Env-var knobs or feature flags as a substitute for explicit config.

**Leftovers and private copies**
- `v2`/compat paths or old code kept "just in case".
- Copying code "for now" with a TODO to dedupe.
- A private graph, private G-buffer drawer or private resource that duplicates what the frame provides.

## 4. Architecture invariants

The full rules are in the plan docs; a change must keep all of these:

1. **Dependencies point down only:** gpu → graph → scene-input → feature-api → spine → features → presets → facade. A feature impl crate depends only on core and other kinds' `*-api` crates, never on another feature's impl crate.
2. **Helio knows no object kind and defines no SceneDB components.** Helio core declares contracts (key, row ABI, version), and features consume what components publish under them.
3. **A plugin owns everything about its kind:** its component (data plus scriptable methods), its feature (passes, nodes, settings, debug views, scene derivation), its contracts (api crate: keys, row ABIs, WGSL snippets), its file formats, importers and migrations, and its editor UI through extension registrations.
4. **The consumer's api crate owns a row ABI.** The component asserts that its layout matches, and Rust/WGSL layout tests guard it.
5. **`*-api` and `feature-api` crates have no `static`, `thread_local!` or `inventory::collect!`.**
6. **Graphs are data.** Code registers nodes; presets name no kind.
7. **A disabled plugin is absent.** Every registration carries `origin: pulsar_plugins::origin!()` and is iterated through `pulsar_plugins::registered::<T>()`. No host links a plugin crate by name (`plugins/installed` is generated).
8. **Kinds meet only at frame contracts, extension points, draw phases, and each other's api crates.**

## 5. Implement

- Make the smallest change that puts the logic in the right place with the right design. A big change is fine when the right place needs it.
- **Delete what you replace in the same change:** old paths, dead code, stale tests, env switches, and docs that describe the old way. Keep a copied baseline exe for A/B instead of old code.
- **Fix what you find in your scope** (stale tests you touch, copies, wrong owners). Don't leave "follow-up" lists for things you could fix now. Hand off only another phase's design decision, with the reason.
- **Update the docs that describe what you changed:** the plan doc's "As built" sections and the relevant `.agents/*.md`. Do it in the same change.

## 6. Verify before you claim it works

- **Gates:**
  - `just arch` (Helio layer gate);
  - `cargo test -p one_owner` (Pulsar);
  - the contract layout tests;
  - clippy on the crates you touched;
  - the crates' own tests.
- **Rendering changes:**
  - Compare the preset pass lists before and after.
  - Check real frames, not only synthetic tests. On voxel_craft, A/B against a baseline exe, plus the flicker burst for anything temporal.
  - Check every setup the change affects, not just one example. Include a large-world setup when the code depends on scale. For voxel work, that means an Earth- or Moon-scale planet (voxel_flight presets, the voxel_planet level).
  - Check that an example actually wires the system you changed before trusting its result.
- **Perf changes:** measure GPU timings with pipelined frames and an interleaved A/B, not one synced run.
- **Your machine's rules:** follow your environment's build and GPU rules (shared locks, disk limits, target dirs). They live in your agent's user-level instructions, not in this skill.
- Never chain commit/push after a filtered test pipeline. Read the results, then commit.

## 7. Report

End every piece of work, and every session or subagent brief, with:

- **Design:** what you built, where it lives, and why there (owner, layer).
- **AAA comparison:** how others solve it, why ours is at least as good, and what it scales with.
- **Deleted:** the old paths, copies and exceptions this change removed.
- **Verified:** which gates, tests, frames and A/Bs ran, with results. Name what was not verified.

When briefing another session, subagent or agent on Helio work, tell it to load this skill first, and include the report section above in what you ask it to return.

This skill follows the open Agent Skills format, so any agent can use it: Claude Code, Codex/GPT, or others. It must stay agent-neutral, with no tool names or machine paths. When the architecture changes, update this file. It is the single source; agent entry files (`AGENTS.md`, `CLAUDE.md`) only point to it.
