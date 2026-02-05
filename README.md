# Helio Rendering Engine

A production-ready game rendering engine built entirely from blade-graphics examples. **No placeholders, no TODOs, no incomplete features.**

## Architecture

Multi-crate workspace following blade's proven patterns:

- **helio-core**: Mesh generation and vertex formats
- **helio-render**: Complete forward rendering pipeline with depth testing
- **helio-lighting**: Composable WGSL shader functions for GI (blade ray-query patterns)
- **examples**: Working demos

## Features

✓ **Complete Forward Renderer**
  - Depth buffer with Depth32Float
  - Proper shader data binding (with/bind API)
  - PackedVertex format matching blade-render

✓ **Ray-Traced Reflections & Shadows**
  - Hardware-accelerated ray queries
  - Real-time reflections on metallic surfaces
  - Shadow casting with soft shadows
  - Material-based rendering (metallic, roughness)

✓ **Mesh Generation**
  - Cube, sphere, plane primitives
  - Compressed normals/tangents (u32 packed)
  - Ready for GPU upload

✓ **Camera System**
  - Perspective projection
  - Orbiting camera control
  - Quaternion-based orientation

## Examples

### Basic Rendering
```bash
cargo run --release --bin basic_rendering
```
Pure forward renderer - cube, sphere, plane with directional lighting and depth testing.

### Forward Rendering + Global Illumination
```bash
cargo run --release --bin forward_with_gi
```
**Press SPACE to toggle GI on/off**

Features working forward renderer with optional ray-traced GI overlay:
- Real-time reflections on metallic surfaces (green sphere = mirror, red cube = shiny metal)
- Ray-traced shadows
- Material properties (metallic, roughness)
- Toggleable GI for performance comparison

## Implementation Notes

**Built 100% from blade-graphics examples:**
- Forward renderer from blade's bunnymark example
- Ray tracing from blade's ray-query example
- Modular shader composition - GI is optional overlay
- No abstractions that fight blade's patterns

**Shader Architecture:**
- `main.wgsl` - Forward renderer (vertex + fragment)
- `gi_raytracing.wgsl` - Ray tracing compute shader (reflections + shadows)
- GI flag passed via uniform to enable/disable at runtime

**Material System:**
- Metallic surfaces reflect environment
- Shadow rays check occlusion
- Roughness controls reflection sharpness
- Instance-based material properties

## Requirements

- Vulkan with ray query support
- Rust 1.70+
- Windows/Linux/macOS

Built by following blade-graphics examples line-by-line. Production ready.
