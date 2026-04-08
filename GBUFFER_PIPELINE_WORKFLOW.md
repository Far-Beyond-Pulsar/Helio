# GBuffer Pipeline Workflow - Complete Documentation

## Overview
This document details the **complete workflow** for how GBuffer textures flow through the Helio rendering engine, from creation to consumption. Use this as a reference for adding the 5th GBuffer target (lightmap UVs).

---

## 1. GBuffer Texture Creation & Storage

### File: `crates/helio-pass-gbuffer/src/lib.rs`

#### Lines 79-90: GBufferPass Structure  
The 4 GBuffer textures are stored as owned fields in the `GBufferPass` struct:

```rust
pub struct GBufferPass {
    pipeline: wgpu::RenderPipeline,
    // ... other fields ...
    
    // ── GBuffer textures (owned; exposed for downstream passes) ───────────────
    pub albedo_tex: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
    pub normal_tex: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub orm_tex: wgpu::Texture,
    pub orm_view: wgpu::TextureView,
    pub emissive_tex: wgpu::Texture,
    pub emissive_view: wgpu::TextureView,
}
```

**PATTERN FOR 5TH TARGET:**  
Add these fields after `emissive_view`:
```rust
pub lightmap_uv_tex: wgpu::Texture,
pub lightmap_uv_view: wgpu::TextureView,
```

---

#### Lines 279-311: Texture Creation in `new()`

Textures are created using the `gbuffer_texture()` helper:

```rust
// ── GBuffer textures ──────────────────────────────────────────────────
let (albedo_tex, albedo_view) = gbuffer_texture(
    device,
    width,
    height,
    wgpu::TextureFormat::Rgba8Unorm,
    "GBuffer/Albedo",
);
let (normal_tex, normal_view) = gbuffer_texture(
    device,
    width,
    height,
    wgpu::TextureFormat::Rgba16Float,
    "GBuffer/Normal",
);
let (orm_tex, orm_view) = gbuffer_texture(
    device,
    width,
    height,
    wgpu::TextureFormat::Rgba8Unorm,
    "GBuffer/ORM",
);
let (emissive_tex, emissive_view) = gbuffer_texture(
    device,
    width,
    height,
    wgpu::TextureFormat::Rgba16Float,
    "GBuffer/Emissive",
);
```

**PATTERN FOR 5TH TARGET:**  
Add after emissive texture creation:
```rust
let (lightmap_uv_tex, lightmap_uv_view) = gbuffer_texture(
    device,
    width,
    height,
    wgpu::TextureFormat::Rg16Float,  // 2-channel UV coordinates
    "GBuffer/LightmapUV",
);
```

---

#### Lines 602-622: Helper Function `gbuffer_texture()`

Creates textures with proper usage flags:

```rust
fn gbuffer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
```

**NO CHANGES NEEDED** - This helper is generic and will work for the 5th target.

---

## 2. Pipeline Configuration (Render Targets)

### File: `crates/helio-pass-gbuffer/src/lib.rs`

#### Lines 234-259: Fragment Pipeline Target Configuration

The 4 render targets are declared in the pipeline's `FragmentState`:

```rust
fragment: Some(wgpu::FragmentState {
    module: &shader,
    entry_point: Some("fs_main"),
    compilation_options: Default::default(),
    targets: &[
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,      // Target 0: Albedo
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,    // Target 1: Normal
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,     // Target 2: ORM
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,    // Target 3: Emissive
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ],
}),
```

**PATTERN FOR 5TH TARGET:**  
Add after the 4th target:
```rust
Some(wgpu::ColorTargetState {
    format: wgpu::TextureFormat::Rg16Float,          // Target 4: Lightmap UV
    blend: None,
    write_mask: wgpu::ColorWrites::ALL,
}),
```

---

## 3. Render Pass Execution (Attachment Binding)

### File: `crates/helio-pass-gbuffer/src/lib.rs`

#### Lines 489-528: `execute()` - Render Pass Creation

The 4 textures are bound as color attachments in `begin_render_pass()`:

```rust
let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    label: Some("GBuffer"),
    color_attachments: &[
        Some(wgpu::RenderPassColorAttachment {
            view: &self.albedo_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        }),
        Some(wgpu::RenderPassColorAttachment {
            view: &self.normal_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        }),
        Some(wgpu::RenderPassColorAttachment {
            view: &self.orm_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        }),
        Some(wgpu::RenderPassColorAttachment {
            view: &self.emissive_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        }),
    ],
    depth_stencil_attachment: Some(/* ... */),
    // ...
});
```

**PATTERN FOR 5TH TARGET:**  
Add after the 4th attachment:
```rust
Some(wgpu::RenderPassColorAttachment {
    view: &self.lightmap_uv_view,
    resolve_target: None,
    depth_slice: None,
    ops: wgpu::Operations {
        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
        store: wgpu::StoreOp::Store,
    },
}),
```

---

## 4. Publishing to FrameResources

### File: `crates/helio-pass-gbuffer/src/lib.rs`

#### Lines 343-350: `publish()` Method

The `publish()` method exposes texture views to downstream passes via `FrameResources`:

```rust
fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
    frame.gbuffer = Some(libhelio::GBufferViews {
        albedo: &self.albedo_view,
        normal: &self.normal_view,
        orm: &self.orm_view,
        emissive: &self.emissive_view,
    });
}
```

**PATTERN FOR 5TH TARGET:**  
Add field to the struct initialization:
```rust
frame.gbuffer = Some(libhelio::GBufferViews {
    albedo: &self.albedo_view,
    normal: &self.normal_view,
    orm: &self.orm_view,
    emissive: &self.emissive_view,
    lightmap_uv: &self.lightmap_uv_view,  // ADD THIS
});
```

---

### File: `crates/libhelio/src/frame.rs`

#### Lines 24-32: `GBufferViews` Struct Definition

The struct holding borrowed texture view references:

```rust
#[derive(Clone, Copy)]
pub struct GBufferViews<'a> {
    /// Albedo (RGB) + alpha (A) — `Rgba8Unorm`
    pub albedo: &'a wgpu::TextureView,
    /// World normal (RGB) + F0.r (A) — `Rgba16Float`
    pub normal: &'a wgpu::TextureView,
    /// AO, roughness, metallic, F0.g — `Rgba8Unorm`
    pub orm: &'a wgpu::TextureView,
    /// Emissive (RGB) + F0.b (A) — `Rgba16Float`
    pub emissive: &'a wgpu::TextureView,
}
```

**PATTERN FOR 5TH TARGET:**  
Add field after `emissive`:
```rust
/// Lightmap UV coordinates (RG) — `Rg16Float`
pub lightmap_uv: &'a wgpu::TextureView,
```

---

#### Line 72: `FrameResources` Field

The `FrameResources` struct stores the optional `GBufferViews`:

```rust
pub struct FrameResources<'a> {
    /// GBuffer textures (populated after GBufferPass)
    pub gbuffer: Option<GBufferViews<'a>>,
    // ... other fields ...
}
```

**NO CHANGES NEEDED** - The `gbuffer` field already holds `GBufferViews`, which you'll update.

---

## 5. Consuming GBuffer Textures (Deferred Lighting)

### File: `crates/helio-pass-deferred-light/src/lib.rs`

#### Lines 135-171: Bind Group Layout 1 (GBuffer Inputs)

The deferred lighting pass declares texture bindings for all 4 GBuffer targets + depth + AO:

```rust
let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("DeferredLight BGL1"),
    entries: &[
        texture_entry(0, wgpu::TextureSampleType::Float { filterable: false }), // Albedo
        texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }), // Normal
        texture_entry(2, wgpu::TextureSampleType::Float { filterable: false }), // ORM
        texture_entry(3, wgpu::TextureSampleType::Float { filterable: false }), // Emissive
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        texture_entry(5, wgpu::TextureSampleType::Float { filterable: true }), // SSAO
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ],
});
```

**PATTERN FOR 5TH TARGET:**  
Add after binding 5 (before sampler):
```rust
texture_entry(7, wgpu::TextureSampleType::Float { filterable: false }), // Lightmap UV
```
*(Note: Adjust sampler binding from 6 to 8)*

---

#### Lines 529-570: Execute - Bind Group Creation

The bind group is created with texture views from `FrameResources`:

```rust
fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
    let gbuffer = ctx.resources.gbuffer.as_ref().ok_or_else(|| {
        helio_v3::Error::InvalidPassConfig(
            "DeferredLight requires published gbuffer resources".to_string(),
        )
    })?;

    let ao_view = ctx.resources.ssao.unwrap_or(&self.fallback_ao_view);

    let gbuffer_key = (
        gbuffer.albedo as *const _ as usize,
        gbuffer.normal as *const _ as usize,
        gbuffer.orm as *const _ as usize,
        gbuffer.emissive as *const _ as usize,
        ctx.depth as *const _ as usize,
        ao_view as *const _ as usize,
    );
    
    if self.bind_group_1_key != Some(gbuffer_key) {
        self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DeferredLight BG1"),
            layout: &self.bgl_1,
            entries: &[
                texture_view_entry(0, gbuffer.albedo),
                texture_view_entry(1, gbuffer.normal),
                texture_view_entry(2, gbuffer.orm),
                texture_view_entry(3, gbuffer.emissive),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ctx.depth),
                },
                texture_view_entry(5, ao_view),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.fallback_ao_sampler),
                },
            ],
        }));
        self.bind_group_1_key = Some(gbuffer_key);
    }
    // ...
}
```

**PATTERN FOR 5TH TARGET:**  
1. Add `lightmap_uv` to the cache key tuple `gbuffer_key`
2. Add bind group entry:
```rust
texture_view_entry(7, gbuffer.lightmap_uv),
```

---

## 6. Shader Output (WGSL Fragment Shader)

### File: `crates/helio-pass-gbuffer/shaders/gbuffer.wgsl`

#### Lines 165-170: Fragment Output Structure (CURRENT - INCORRECT)

**CRITICAL BUG FOUND:** The current shader has a syntax error with duplicate `@location(3)`:

```wgsl
struct GBufferOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,  // ← BUG: This should write to target 4!
    @location(3) emissive:    vec4<f32>,  // ← Duplicate location!
}
```

**CORRECTED PATTERN:**
```wgsl
struct GBufferOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(3) emissive:    vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,  // 5th target (Rg16Float)
}
```

#### Lines 323-328: Fragment Main Return (CURRENT)

```wgsl
var out: GBufferOutput;
out.albedo = vec4<f32>(albedo.rgb, alpha);
out.normal = vec4<f32>(N, specular_f0.r);
out.orm = vec4<f32>(ao, roughness, metallic, specular_f0.g);
out.emissive = vec4<f32>(emissive, specular_f0.b);
out.lightmap_uv = input.lightmap_uv;
return out;
```

**NO CHANGES NEEDED** - Already writes `lightmap_uv` (from vertex shader).

---

## 7. Resize Support

### File: `crates/helio-pass-gbuffer/src/lib.rs`

#### Lines 561-590: `resize()` Method

When the window resizes, all GBuffer textures must be recreated:

```rust
pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
    let (albedo_tex, albedo_view) = gbuffer_texture(
        device, width, height,
        wgpu::TextureFormat::Rgba8Unorm,
        "GBuffer/Albedo",
    );
    let (normal_tex, normal_view) = gbuffer_texture(
        device, width, height,
        wgpu::TextureFormat::Rgba16Float,
        "GBuffer/Normal",
    );
    let (orm_tex, orm_view) = gbuffer_texture(
        device, width, height,
        wgpu::TextureFormat::Rgba8Unorm,
        "GBuffer/ORM",
    );
    let (emissive_tex, emissive_view) = gbuffer_texture(
        device, width, height,
        wgpu::TextureFormat::Rgba16Float,
        "GBuffer/Emissive",
    );
    
    self.albedo_tex = albedo_tex;
    self.albedo_view = albedo_view;
    self.normal_tex = normal_tex;
    self.normal_view = normal_view;
    self.orm_tex = orm_tex;
    self.orm_view = orm_view;
    self.emissive_tex = emissive_tex;
    self.emissive_view = emissive_view;
}
```

**PATTERN FOR 5TH TARGET:**  
Add texture recreation and assignment:
```rust
let (lightmap_uv_tex, lightmap_uv_view) = gbuffer_texture(
    device, width, height,
    wgpu::TextureFormat::Rg16Float,
    "GBuffer/LightmapUV",
);
self.lightmap_uv_tex = lightmap_uv_tex;
self.lightmap_uv_view = lightmap_uv_view;
```

---

## Summary of Required Changes

### Files to Modify (in order):

1. **`crates/libhelio/src/frame.rs`**  
   - Add `lightmap_uv: &'a wgpu::TextureView` to `GBufferViews` struct

2. **`crates/helio-pass-gbuffer/src/lib.rs`**  
   - Add `lightmap_uv_tex` and `lightmap_uv_view` fields to `GBufferPass` struct
   - Create lightmap UV texture in `new()` (after emissive)
   - Add 5th target to pipeline `FragmentState`
   - Add 5th attachment in `execute()` render pass descriptor
   - Update `publish()` to include `lightmap_uv` field
   - Add texture recreation in `resize()` method

3. **`crates/helio-pass-gbuffer/shaders/gbuffer.wgsl`**  
   - **FIX BUG:** Change `@location(4) lightmap_uv` to `@location(3)` and fix emissive duplicate
   - Or reorder: Keep `@location(3) emissive`, set `@location(4) lightmap_uv` (if pipeline supports it)

4. **`crates/helio-pass-deferred-light/src/lib.rs`**  
   - Add binding 7 to `bgl_1` for lightmap UV texture
   - Update `gbuffer_key` tuple to include `lightmap_uv` pointer
   - Add bind group entry for lightmap UV texture
   - Update deferred lighting shader to consume lightmap UVs

---

## Key Architecture Patterns

### Texture Creation Pattern
```rust
let (tex, view) = gbuffer_texture(device, width, height, FORMAT, "Label");
```

### FrameResources Publishing Pattern
```rust
fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
    frame.gbuffer = Some(libhelio::GBufferViews {
        field1: &self.field1_view,
        field2: &self.field2_view,
        // ...
    });
}
```

### Downstream Consumption Pattern
```rust
let gbuffer = ctx.resources.gbuffer.as_ref().ok_or_else(|| {
    helio_v3::Error::InvalidPassConfig("Pass requires gbuffer".to_string())
})?;

// Use: gbuffer.albedo, gbuffer.normal, etc.
```

### Cache Invalidation Pattern
```rust
let key = (
    gbuffer.albedo as *const _ as usize,
    gbuffer.normal as *const _ as usize,
    // ... all dependent pointers
);

if self.bind_group_key != Some(key) {
    // Rebuild bind group
    self.bind_group_key = Some(key);
}
```

---

## Testing Checklist

After implementing the 5th target:

- [ ] GBuffer pass creates lightmap UV texture
- [ ] GBuffer pass publishes lightmap UV view
- [ ] Deferred lighting pass receives lightmap UV
- [ ] Shader compiles with 5 `@location` outputs
- [ ] Resize updates all 5 textures
- [ ] Cache invalidation works correctly
- [ ] No crashes on window resize
- [ ] Lightmap UVs sample correctly in lighting shader

---

**Document Created:** 2026-04-08  
**Engine Version:** Helio (Rust + wgpu)  
**Purpose:** Add 5th GBuffer target (lightmap UVs, Rg16Float)
