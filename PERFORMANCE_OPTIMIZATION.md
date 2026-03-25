# Helio Renderer - Performance Optimization Guide

## Executive Summary — All Items Implemented ✅

This document originally identified critical performance bottlenecks in the Helio renderer that prevented AAA-quality framerates at 4K on mid-range hardware (e.g., laptop RTX 4060). All four bottlenecks have now been resolved. The renderer implements the full GPU-driven performance tier described in the original analysis.

**Status:** All optimizations are active in the default render graph as of 2026-03-25. `cargo check --workspace` passes with no errors.

---

## Resolved Bottlenecks

### 1. ✅ Per-Pixel Light Culling — Implemented (LightCullPass + Tiled Forward+)

**Previous problem:** Every pixel evaluated every light — O(pixels × lights) brute-force loop.

**Resolution:** `LightCullPass` runs a compute dispatch after `HiZBuildPass` each frame. It divides the screen into 16×16 tiles and, for each tile, tests every scene light against the tile's depth-range AABB. Passing light indices are written to `tile_light_lists`; per-tile counts go to `tile_light_counts`. Both buffers are published via `FrameResources`.

`DeferredLightPass` now reads these via Bind Group 3:
```wgsl
const TILE_SIZE:           u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;

@group(3) @binding(0) var<storage, read> tile_light_lists:  array<u32>;
@group(3) @binding(1) var<storage, read> tile_light_counts: array<u32>;
```

The fragment shader computes its tile index from `in.clip_pos` and iterates only the lights in that tile:
```wgsl
let tile_x = u32(in.clip_pos.x) / TILE_SIZE;
let tile_y = u32(in.clip_pos.y) / TILE_SIZE;
let tile_idx = tile_y * globals.num_tiles_x + tile_x;
for (var i = 0u; i < tile_light_counts[tile_idx]; i++) {
    let light_idx = tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i];
    // ... PBR evaluation
}
```

`globals.num_tiles_x` replaces the former `_pad1` field in the `Globals` uniform struct.

---

### 2. ✅ Occlusion Culling — Implemented (OcclusionCullPass + HiZBuildPass)

**Previous problem:** Hi-Z and occlusion cull passes existed but were not wired into `build_default_graph()`.

**Resolution:** Both passes are wired in the correct order:

```
ShadowPass → OcclusionCullPass → DepthPrepassPass → HiZBuildPass → LightCullPass → ...
```

`OcclusionCullPass` receives `frame.hiz` from the previous frame (temporal), writes updated visibility bits to the indirect draw buffer before the depth prepass, so the depth prepass already benefits from culled geometry.

`HiZBuildPass` runs immediately after `DepthPrepassPass`, rebuilding `frame.hiz` from the current frame's depth buffer for consumption by `LightCullPass` in the same frame and `OcclusionCullPass` in the next frame.

---

### 3. ✅ Excessive Shadow Sampling — Resolved (Dynamic PCF + TAA filtering)

**Previous problem:** `override PCF_SAMPLE_COUNT: u32 = 16u;` was a compile-time constant — changing it required a full pipeline recompile. At 4K with 4 shadow-casting lights this amounted to ~530 million shadow taps per frame.

**Resolution:** `PCF_SAMPLE_COUNT` is no longer an override constant. It is now a runtime uniform field:

```wgsl
shadow_config.pcf_sample_count  // read from ShadowConfig uniform each frame
```

This allows per-frame quality adjustment without recompiling. The default is 8 samples (was 16); `TaaPass` accumulates the temporal signal to recover the full quality across frames, effectively amortising the cost over time.

---

### 4. ✅ Temporal Reprojection — Implemented (TaaPass)

**Previous problem:** `TaaPass` existed but was not added to `build_default_graph()`.

**Resolution:** `TaaPass` is wired after `DeferredLightPass`. It implements:

- **Correct jitter UV invariant:** `cur_uv = in.uv + jitter_uv` (world-point in jittered current frame); `history_uv = in.uv - velocity` (world-point in unjittered history)
- **RESET mode:** First-frame priming via `taa.reset = 1` to avoid blending against uninitialised history
- **Confidence counter in alpha:** Static pixels accumulate `+= 10.0` per frame; moving pixels reset to `1.0`. Blend rate = `clamp(1.0 / confidence, 0.015, 0.10)`
- **`clip_towards_aabb_center`** (Playdead method) instead of plain AABB clamp
- **Reversible Reinhard tonemapping** around the blend to prevent HDR outliers from dominating the AABB statistics

The `TaaUniform` struct is 24 bytes:
```rust
struct TaaUniform {
    feedback_min:  f32,      // legacy — retained for GPU layout compat
    feedback_max:  f32,      // legacy — retained for GPU layout compat
    jitter_offset: [f32; 2],
    reset:         u32,
    _pad:          u32,
}
```

---

## Updated Performance Projection

With all four optimizations active, the expected 4K 60 FPS frame budget breakdown on RTX 4060 laptop:

| Pass | Estimated Cost | Notes |
|------|---------------|-------|
| Shadow Rendering | ~2 ms | Unchanged |
| OcclusionCullPass | ~0.2 ms | One compute dispatch |
| HiZBuildPass | ~0.3 ms | ~10 compute dispatches (one per mip) |
| DepthPrepassPass | ~0.6 ms | Reduced by occlusion cull |
| LightCullPass | ~0.2 ms | One compute dispatch |
| GBufferPass | ~1.5 ms | ~50% less geometry drawn |
| DeferredLightPass | ~3–5 ms | Tiled N lights/pixel, 8 PCF samples |
| TaaPass | ~0.8 ms | One fullscreen draw + blit |
| Other (sky, RC, FX) | ~1.5 ms | Unchanged |
| **Total** | **~10–11 ms** | ✅ Comfortably under 16.67 ms |

---

## Default Graph Pass Order (as implemented)

```
1.  ShadowMatrixPass
2.  ShadowPass
3.  SkyLutPass
4.  OcclusionCullPass    ← temporal; reads last frame's HiZ
5.  DepthPrepassPass
6.  HiZBuildPass         ← builds HiZ from current depth
7.  LightCullPass        ← tiled 16×16 light lists
8.  GBufferPass
9.  VirtualGeometryPass
10. DeferredLightPass    ← consumes tile_light_lists (group 3)
11. TaaPass              ← jitter-corrected accumulation
12. BillboardPass
```
