# VR / XR Forward Rendering Support

**Status: In Progress** — OpenXR integration (`helio-xr`) and multi-eye rendering infrastructure landed; render-loop integration and demo pending.

## Problem

Helio needs a first-class VR path. A headset requires per-eye stereo rendering at
HMD refresh rates (90/120 Hz) with minimal added latency, which the current
single-camera deferred pipeline is ill-suited for. Building on the existing
[forward rendering work](#82), we add:

1. **Multi-eye rendering** — both eyes rendered in a single geometry pass using
   Vulkan multiview (2-layer array textures + `multiview_mask`), avoiding the
   ~2× vertex cost of naive double-pass stereo.
2. **OpenXR integration** — headset tracking, pose, per-eye asymmetric FOV, and
   swapchain compositing via the `openxr` crate (Vulkan backend).
3. **Transparent activation** — a single init-time builder
   (`RendererConfig::enable_xr` / `RenderGraph::with_xr_mode(true)`) flips the
   entire renderer into stereo mode. No per-pass changes; the same pipelines,
   shaders, and graph builders work unchanged.

## Design principle

The renderer is already forward-capable (`ForwardOpaque` / `ForwardOnly` render
modes). VR should be *an activation of that same pipeline*, not a parallel one:

| Layer | Desktop (mono) | XR (stereo) |
|---|---|---|
| Render targets | 2D textures | 2D-array textures, 2 layers |
| `multiview_mask` | `None` (per pass) | `Some(0b11)`, injected by graph executor |
| Camera data | single view/proj | `array<Camera, 2>` storage buffer indexed by `view_index` |
| Swapchain | wgpu surface | OpenXR swapchain (VkImage wrapped as wgpu textures) |

## Current state

### Landed

- **`crates/helio-core`** — `RenderGraph::with_xr_mode(true)`:
  - `xr_active` flag patches `multiview_mask → 0b11` at both render-pass
    creation sites in the graph executor.
  - `GraphTexturePool::set_xr_mode(true)` allocates internal colour/depth
    targets as 2-layer `D2Array` textures.
- **`crates/libhelio`** — camera converted from a single `var<uniform> Camera`
  to `var<storage, read> cameras: array<Camera, 2>`:
  - 49 WGSL shaders updated (`cameras[0].field` — mono-safe, `view_index` ready).
  - All pass bind group layouts updated `Uniform → Storage { read_only }`.
  - Camera storage buffer doubled (2 × `GpuCameraUniforms`), `upload_stereo()`
    helper added.
- **`crates/helio-xr`** (new) — OpenXR lifecycle crate, compiles standalone:
  - `openxr::Graphics` impl for `WgpuGraphics` (Vulkan: `Format = u32`,
    `SwapchainImage = u64`).
  - `XrInstance` — `Entry`, instance + VULKAN extension, debug utils, system.
  - `XrSession` — session create from wgpu hal handles
    (`device.as_hal::<wgpu::hal::vulkan::Api>()`), event polling, frame lifecycle.
  - `XrSwapchain` — 2-layer array swapchain (multiview) with 1-layer fallback;
    wraps every `VkImage` via `vulkan::Device::texture_from_raw` with
    `TextureMemory::External` (wgpu never frees OpenXR-owned images). Image pool
    is created once; `acquire_image` returns a pre-built index (no frame leak).
  - `camera.rs` — `ViewPose`, `xr_view_to_camera` → `[GpuCameraUniforms; 2]`
    using the OpenXR asymmetric-frustum convention (matches glam
    `perspective_rh`).

### Pending

- **Render loop integration** (`crates/helio/src/renderer/render.rs`): when
  `enable_xr`, drive the frame from OpenXR — `wait_frame`/`begin_frame`,
  `locate_views` → per-eye cameras, upload stereo camera, execute the graph
  against the XR swapchain array view, `end_frame`.
- **`RendererConfig::enable_xr`** plumbing from config → graph builder →
  `graph.with_xr_mode(true)`.
- **VR demo** (`crates/examples/vr/` folder) — headset rendering, HMD tracking,
  minimal scene, controller input.

## Required work

### 1. Render-loop XR path

In `Renderer::render`, branch when XR is active:

1. `xr.wait_frame()` → `begin_frame()`; handle `SessionStateChanged` events
   (Ready/Focused/Visible — pause rendering when not visible).
2. `locate_views(stage_space, display_time)` → per-eye `ViewPose` + FOV.
3. `xr_view_to_camera(left, right, near, far)` → `[GpuCameraUniforms; 2]`;
   `upload_stereo()` to the scene camera buffer.
4. `swapchain.acquire_image()` → index into the pre-built array views.
5. Execute the graph against `swapchain.views[index]` (multiview target) +
   depth array.
6. `xr.end_frame(swapchain, views)` with the projection layer anchored to stage.

No per-pass changes: the graph's `multiview_mask` injection + 2-layer targets
handle the rest.

### 2. Config plumbing

```rust
// RendererConfig
pub enable_xr: bool,   // default false

pub fn with_xr_mode(mut self, active: bool) -> Self {
    self.enable_xr = active;
    self
}
```

Graph builders forward it to `RenderGraph::with_xr_mode(true)` and the renderer
keeps an `enable_xr` flag. The demo still renders to a normal wgpu surface in
desktop mode (window shows one eye) and to the headset in XR mode.

### 3. VR demo — `crates/examples/vr/`

Folder-based demo (larger than single-file examples; same pattern as
`crates/examples/voxel/`):

- `vr/main.rs` — winit window + OpenXR bootstrap; desktop fallback renders one
  eye to the window so the demo runs without a headset.
- Scene: forward-lit cubes/ground + lights + sky (reuse `v3_demo_common`).
- Input: head tracking always; optional thumbstick/teleport locomotion.
- Toggle: `F1` switches the mirror window between left/right/centre eye.

## Quality gates

- Both eyes render correct perspective (no duplicated same-eye image).
- Forward-lit VR matches the desktop forward renderer within 0.5% RMSE for the
  same scene/camera (multiview must not change shading).
- Single geometry pass per eye-pair (one draw per object, `view_index` selects
  camera) — no 2× vertex cost.
- No regression in desktop mode: mono renders identical to pre-XR branch.
- 90/120 Hz frame pacing: `wait_frame` back-pressure respected; no CPU work
  between `begin_frame` and `end_frame` beyond command recording.
- Transparent objects render correctly in VR (alpha composited per-eye).
- `TextureMemory::External` swapchain pool: no per-frame texture wrapping
  (bounded allocation).

## References

- Forward rendering path: #82
- Reference integration: philpax/wgpu-openxr-example (archived, wgpu 0.13-era;
  pattern only — modern path uses `as_hal` + `texture_from_raw`)
- OpenXR crate: Ralith/openxrs (0.21.x)
