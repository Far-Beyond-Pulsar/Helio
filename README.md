
## Do I even need to tell you how WIP this is?

# Helio

A feature-complete real-time rendering engine written in Rust using Blade graphics, with all the advanced capabilities of Unreal Engine's renderer.

## Features

### Core Rendering
- ✅ **Multiple Rendering Paths**
  - Deferred rendering
  - Forward rendering
  - Forward+ (clustered/tiled)
  - Visibility buffer rendering
  
- ✅ **Physically Based Rendering (PBR)**
  - Full metallic-roughness workflow
  - Disney BRDF model
  - Image-based lighting (IBL)
  - Multiple shading models (Standard, Subsurface, Clear Coat, Cloth, Hair, Eye)

- ✅ **Advanced Material System**
  - Node-based shader graphs
  - Material instances
  - Texture streaming
  - Virtual texturing
  - Parallax occlusion mapping

### Lighting
- ✅ **Light Types**
  - Directional lights
  - Point lights
  - Spot lights  
  - Area lights (Rectangle, Disk, Sphere, Tube)
  - Sky lights
  
- ✅ **Shadows**
  - Cascaded shadow maps (CSM)
  - Virtual shadow maps (VSM)
  - Shadow atlas
  - PCSS soft shadows
  - Ray-traced shadows

- ✅ **Global Illumination**
  - Dynamic Diffuse Global Illumination (DDGI)
  - Lumen-style GI
  - Voxel cone tracing
  - Screen-space GI (SSGI)
  - Ray-traced GI
  - Lightmaps

- ✅ **Reflection Systems**
  - Screen-space reflections (SSR)
  - Reflection probes (with parallax correction)
  - Ray-traced reflections
  - Planar reflections

- ✅ **Light Probes**
  - Spherical harmonics
  - Light probe volumes
  - Automatic probe placement

- ✅ **Volumetric Lighting**
  - Volumetric fog
  - God rays
  - Height fog

### Post-Processing
- ✅ **Anti-Aliasing**
  - Temporal Anti-Aliasing (TAA) with multiple jitter sequences
  - Fast Approximate Anti-Aliasing (FXAA)
  - Subpixel Morphological Anti-Aliasing (SMAA)
  - Multi-Sample Anti-Aliasing (MSAA)

- ✅ **Tone Mapping**
  - Reinhard
  - Uncharted 2
  - ACES
  - AGX
  - Filmic

- ✅ **Color Grading**
  - Temperature and tint
  - Saturation, contrast, gamma
  - Lift, gamma, gain
  - Shadows, midtones, highlights
  - LUT support

- ✅ **Effects**
  - HDR Bloom
  - Depth of Field (Bokeh, Gaussian, Circular)
  - Motion blur
  - Lens flare
  - Chromatic aberration
  - Vignette
  - Film grain
  - Auto exposure

- ✅ **Ambient Occlusion**
  - SSAO (Screen-Space Ambient Occlusion)
  - HBAO+ (Horizon-Based Ambient Occlusion)
  - GTAO (Ground Truth Ambient Occlusion)
  - Ray-traced AO

### Ray Tracing
- ✅ **Hardware-Accelerated Ray Tracing**
  - Acceleration structures (BLAS/TLAS)
  - Ray-traced shadows
  - Ray-traced reflections
  - Ray-traced ambient occlusion
  - Ray-traced global illumination

- ✅ **Path Tracing**
  - Physically accurate rendering
  - Multiple importance sampling (MIS)
  - Russian roulette
  - ReSTIR (Reservoir-based Spatiotemporal Importance Resampling)

- ✅ **Hybrid Rendering**
  - Rasterization for primary visibility
  - Ray tracing for secondary effects
  - Configurable quality/performance tradeoffs

### Advanced Features
- ✅ **Terrain System**
  - Heightmap terrain
  - Clipmap LOD
  - Multi-layer materials with blending
  - Tessellation
  - Vegetation/grass system

- ✅ **Particle Systems**
  - GPU-accelerated simulation
  - 1M+ particles support
  - Multiple emitter shapes
  - Particle modules (color, size, velocity, forces)
  - Collision detection
  - Sorting for transparency

- ✅ **Animation**
  - Skeletal animation
  - Blend trees
  - State machines
  - Inverse Kinematics (IK)
  - FABRIK solver
  - Two-bone IK

- ✅ **Atmosphere & Sky**
  - Physically-based atmospheric scattering
  - Volumetric clouds
  - Sun/moon
  - Stars
  - Dynamic time of day

- ✅ **Water System**
  - FFT-based wave simulation
  - Gerstner waves
  - Reflections & refractions
  - Subsurface scattering
  - Foam rendering
  - Caustics

- ✅ **Decals**
  - Deferred decals
  - Screen-space projection
  - Multiple blend modes

- ✅ **Culling & Optimization**
  - Frustum culling
  - Occlusion culling
  - Hierarchical-Z buffer
  - GPU-driven rendering
  - Compute-based culling

- ✅ **UI Rendering**
  - Canvas system
  - Text rendering
  - Widget library
  - Multiple scale modes

## Architecture

Helio is built as a modular workspace with independent crates:

- **helio-core**: Core types (Camera, Scene, Transform, GPU resources)
- **helio-render**: Rendering pipeline and frame graph
- **helio-material**: Material system and shaders
- **helio-lighting**: Lighting systems and shadows
- **helio-postprocess**: Post-processing effects
- **helio-raytracing**: Ray tracing systems
- **helio-terrain**: Terrain rendering
- **helio-particles**: Particle systems
- **helio-animation**: Animation system
- **helio-atmosphere**: Sky and atmosphere
- **helio-water**: Water simulation and rendering
- **helio-decals**: Decal system
- **helio-culling**: Visibility and culling
- **helio-virtualtexture**: Virtual texture streaming
- **helio-ui**: UI rendering

## Usage

```rust
use helio::prelude::*;

fn main() {
    // Create camera
    let camera = Camera::new_perspective(
        std::f32::consts::FRAC_PI_3,
        16.0 / 9.0,
        0.1,
        1000.0,
    );
    
    // Create scene
    let mut scene = Scene::new(camera);
    
    // Create renderer
    let config = RendererConfig {
        render_path: RenderPath::Deferred,
        enable_hdr: true,
        enable_taa: true,
        ..Default::default()
    };
    
    // Setup lighting
    let mut lighting = LightingSystem::new(LightingMode::Deferred);
    lighting.add_directional_light(DirectionalLight::default());
    
    // Render loop
    // renderer.render(&scene, &viewport);
}
```

## Examples

Run the examples to see various features:

```bash
cargo run --example basic_rendering
cargo run --example deferred_rendering
cargo run --example pbr_materials
cargo run --example shadows
cargo run --example particles
```

## Building

```bash
# Build the entire workspace
cargo build --release

# Build specific crate
cargo build -p helio-render --release

# Run tests
cargo test --workspace
```

## Optional Features

All advanced features are optional and can be selectively enabled:

- Ray tracing (requires hardware RT support)
- Virtual shadow maps
- Nanite-style virtualized geometry
- Lumen-style GI
- Advanced post-processing

## Performance

Helio is designed for high performance:
- Multi-threaded scene processing
- GPU-driven rendering
- Async compute overlap
- Virtual texturing for memory efficiency
- Efficient culling systems
- Frame graph for optimal resource management

## Requirements

- Rust 2021 edition
- Vulkan, DirectX 12, or Metal capable GPU
- Optional: Hardware ray tracing support

## Integration

Helio is designed to integrate seamlessly with game engines. The modular architecture allows you to:
- Use only the components you need
- Replace subsystems with custom implementations
- Extend with your own rendering features
- Integrate with existing codebases

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! This is a comprehensive rendering engine with room for optimization and feature additions.

## Acknowledgments

Helio implements rendering techniques from:
- Unreal Engine
- Unity
- Frostbite Engine
- Call of Duty Engine
- Academic research (SIGGRAPH, EGSR, etc.)
