# Billboard Rendering System

## Overview

The billboard rendering system adds PNG texture support and camera-facing billboards to the Helio engine. The system is designed for two primary use cases:

1. **Standalone scene actors** - Billboards as first-class scene objects (particles, sprites, etc.)
2. **Editor gizmos** - Visual representation for lights and other no-body scene actors

## Architecture

The implementation follows Helio's modular feature architecture:

```
helio-core
├── TextureManager - General-purpose PNG loading & GPU texture management
├── TextureId - Opaque texture handle
├── BillboardVertex - Simple vertex format (position + UV)
└── create_billboard_quad() - Quad mesh generator

helio-feature-billboards
├── BillboardFeature - Implements Feature trait
├── BillboardData - Per-billboard instance data
└── BlendMode - Opaque or transparent rendering

Shaders
└── billboard.wgsl - Spherical billboard shader (always faces camera)
```

## Key Components

### TextureManager (helio-core)

General-purpose texture loading and management, designed for use by both billboards and meshes:

```rust
use helio_core::TextureManager;
use std::sync::Arc;

// Create manager with GPU context
let mut texture_manager = TextureManager::new(gpu_context.clone());

// Load PNG file
let texture_id = texture_manager.load_png("assets/icon.png")?;

// Or upload raw RGBA data
let texture_data = TextureData::from_rgba_bytes(256, 256, rgba_bytes)?;
let texture_id = texture_manager.upload_texture(texture_data)?;

// Get GPU texture
if let Some(gpu_texture) = texture_manager.get(texture_id) {
    // Access: gpu_texture.texture, gpu_texture.view, gpu_texture.sampler
}

// Remove texture when done
texture_manager.remove(texture_id);
```

**Note**: Texture data upload to GPU is not yet implemented. Textures are created but will be uninitialized. This needs to be completed by finding the correct blade-graphics API for texture data upload.

### BillboardData

Represents a single billboard instance:

```rust
use helio_feature_billboards::{BillboardData, BlendMode};

let billboard = BillboardData::new(
    [0.0, 2.0, 0.0],  // world position
    [1.0, 1.0],        // scale (width, height)
    texture_id,        // texture to display
).with_blend_mode(BlendMode::Transparent);
```

### BillboardFeature

The feature that manages billboard rendering:

```rust
use helio_feature_billboards::BillboardFeature;

// Create and add to registry
let mut billboard_feature = BillboardFeature::new();
billboard_feature.set_texture_manager(Arc::new(texture_manager));

// Add to feature registry (during renderer setup)
let registry = FeatureRegistry::builder()
    .with_feature(BaseGeometry::new())
    .with_feature(billboard_feature)
    .build();
```

## Current Status

### ✅ Implemented

1. **TextureManager** - PNG loading, GPU texture creation
2. **Billboard vertex format** - Simple position + UV
3. **Billboard quad geometry** - Reusable quad mesh
4. **BillboardFeature** - Feature trait skeleton
5. **Shader code** - Camera-facing vertex transformation
6. **Data structures** - BillboardData with blend modes

### ⚠️ Incomplete

1. **Texture data upload** - Textures created but data not uploaded (needs correct blade-graphics API)
2. **Rendering pipeline** - Billboard feature needs full pipeline creation (deferred for API compatibility)
3. **Renderer integration** - FeatureRenderer needs billboard draw call support
4. **Example code** - No working example yet

## Next Steps

To complete the implementation:

1. **Fix texture upload** - Research blade-graphics texture data upload API
2. **Complete rendering pipeline** - Follow blade-graphics patterns from other features
3. **Integrate with FeatureRenderer** - Add billboard rendering pass
4. **Create example** - Standalone billboards + light gizmos demo
5. **Performance optimization** - Instancing, depth sorting for transparency

## Usage Example (Future)

```rust
// Setup (once)
let mut texture_manager = TextureManager::new(context.clone());
let light_icon = texture_manager.load_png("assets/light_icon.png")?;

let mut billboard_feature = BillboardFeature::new();
billboard_feature.set_texture_manager(Arc::new(texture_manager));

// Runtime (each frame)
let billboards = vec![
    BillboardData::new([0.0, 2.0, 0.0], [0.5, 0.5], light_icon)
        .with_blend_mode(BlendMode::Transparent),
];

// Render (in main render loop)
renderer.render_with_billboards(&mut encoder, target, camera, &meshes, &billboards, delta_time);
```

## Design Decisions

### Why Separate from Main Pipeline?

Billboards use a fundamentally different rendering approach:
- Custom vertex shader for camera-facing transformation
- Different vertex format (no normals/tangents)
- Separate blend modes (transparent pass)
- Texture sampling (not yet in material system)

### Why TextureManager in helio-core?

Texture loading will be needed by multiple systems:
- Billboards (icons, sprites)
- Materials (PBR textures)
- UI (future)
- Post-processing (LUTs, noise)

Placing it in core makes it reusable across all features.

### Future-Proofing for Mesh Textures

The TextureManager API is designed to support PBR material textures:
```rust
// Future material system:
let material = Material {
    base_color_texture: texture_manager.load_png("diffuse.png")?,
    normal_texture: texture_manager.load_png("normal.png")?,
    metallic_roughness_texture: texture_manager.load_png("metallic_roughness.png")?,
};
```

## Shader Details

The billboard shader (`billboard.wgsl`) implements spherical billboards:

1. Extract camera right/up vectors from view matrix
2. Transform quad vertices to face camera
3. Scale by billboard size
4. Sample texture with UV coordinates
5. Discard fully transparent pixels

This ensures billboards always face the camera regardless of orientation.

## Known Limitations

1. **No texture data upload** - Critical missing feature
2. **No depth sorting** - Transparent billboards may z-fight
3. **No instancing** - Each billboard is a separate draw call
4. **No LOD system** - All billboards rendered at full resolution
5. **Fixed spherical mode** - No cylindrical or axial billboards

## Contributing

To complete this feature:
- See `D:\Documents\GitHub\Helio\crates\helio-core\src\texture.rs` for texture upload TODO
- See `D:\Documents\GitHub\Helio\crates\helio-feature-billboards\src\lib.rs` for rendering pipeline
- Check blade-graphics examples for correct API usage
