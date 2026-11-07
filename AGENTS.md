# Helio

Helio is the renderer: the frame (views, draw phases, the G-buffer, the scene
colour chain, lighting-input contracts), the render graph, GPU uploads, shaders
and profiling. It knows no object kind. Meshes, lights, materials, sky, fog,
water, voxels, foliage and every other kind are Pulsar plugins
(`plugins/vendor/<kind>/` in Pulsar-Native) that bring their own components,
features, passes and contracts.

## Required for every agent

Before reading, designing, editing, reviewing or briefing any change here,
load the **`helio-architecture`** skill:
[`.agents/skills/helio-architecture/SKILL.md`](.agents/skills/helio-architecture/SKILL.md).
That applies to bug fixes, perf work, tests and small changes too.

Follow the skill's steps in order: read the code and the plan docs, find the
concept's one owner, write the AAA design note, keep the architecture
invariants, delete what you replace, verify, and end with its report section
(Design, AAA comparison, Deleted, Verified). Anything you brief (a subagent,
another session or another agent) must load the skill too.

The skill is the single source of these rules. This file and `CLAUDE.md`
only point to it, so change the skill rather than copying rules here.

## Architecture docs

| File | What |
|---|---|
| `docs/architecture/layered-rewrite.md` | Layers, rules, buffer contracts, extension points, and why this shape |
| `docs/architecture/plugin-migration.md` | Every object kind is a plugin; what Helio is; the plugin table; "As built" notes per phase |
| `architecture/layers.toml` | The layer gate (`just arch`); `architecture/layer-violations.txt` only shrinks |

## Key commands

```
just ci          fmt-check, layer gate, check, clippy, nextest
just arch        layer gate
just co-change   how often a change to one unit also changed others
```
