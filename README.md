# Helio - Modular GPU Rendering Engine

Helio is a modular, feature-based rendering engine built on top of blade-graphics. It allows you to compose rendering pipelines from individual feature crates, each providing their own shaders and GPU functionality.

# ⭐ Support the Project

If Pulsar Engine aligns with how you think game engines should be built, consider supporting the project on GitHub:
- ⭐ Star the repo to help others discover Pulsar
- 👀 Watch for updates to follow major architectural changes and milestones
- 🍴 Fork if you want to experiment or contribute
Stars and watches directly influence visibility and help justify continued deep-systems work on the engine.

<img width="1922" height="1109" alt="image" src="https://github.com/user-attachments/assets/032640eb-0c1a-4ed2-b00c-2e204c4c614c" />

## Key Features

✨ **Modular Architecture**: Each rendering feature is a separate crate  
🔧 **Composable Shaders**: Features combine through shader injection, not inheritance  
📝 **Well-Documented**: Comprehensive API docs and examples  
⚡ **High Performance**: Zero-cost abstractions with compile-time optimization  
🎯 **Type-Safe**: Strong typing with Result-based error handling  
🔍 **Developer-Friendly**: Builder patterns, debug output, clear error messages  

## Architecture

### Core Crates

- **helio-core**: Fundamental data structures (vertices, meshes, primitives)
- **helio-render**: Core rendering infrastructure with `FeatureRenderer`
- **helio-features**: Feature system trait definitions and registry
- **helio-lighting**: Ray-traced global illumination (standalone)

### Feature Crates

Each feature is a separate crate that implements the `Feature` trait:

- **helio-feature-base-geometry**: Base geometry rendering with UV coordinate pass-through
- **helio-feature-lighting**: Basic diffuse + ambient lighting
- **helio-feature-materials**: Procedural texture materials with checkerboard pattern
- **helio-feature-procedural-shadows**: Shadow mapping with configurable resolution

## Quick Start

### Using the Feature System

```rust
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_lighting::BasicLighting;
use helio_features::FeatureRegistry;
use helio_render::FeatureRenderer;

// Create base geometry feature and get its shader template
let base_geometry = BaseGeometry::new();
let base_shader = base_geometry.shader_template();

// Register features using builder pattern
let registry = FeatureRegistry::builder()
    .with_feature(base_geometry)
    .with_feature(BasicLighting::new())
    .with_feature(BasicMaterials::new())
    .debug_output(true) // Optional: write composed shader to file
    .build();

// Create renderer with composed pipeline
let renderer = FeatureRenderer::new(
    gpu_context,
    surface_format,
    width,
    height,
    registry,
    base_shader,
);

// Render each frame
renderer.render(&mut encoder, target_view, camera, &meshes, delta_time);

// Toggle features at runtime (requires pipeline rebuild)
renderer.registry_mut().toggle_feature("basic_lighting")?;
renderer.rebuild_pipeline();
```

## Feature System

### How It Works

1. **Features provide shader code**: Each feature returns `ShaderInjection` objects specifying where their WGSL code should be inserted
2. **Shader composition**: The `FeatureRegistry` composes features into a final shader by replacing injection point markers
3. **Modular pipelines**: Features can add render passes before/after the main pass
4. **Runtime toggling**: Features can be enabled/disabled with pipeline rebuild
5. **Proper cleanup**: Resources are automatically cleaned up via Drop trait

### Injection Points

Features inject code at these points in the rendering pipeline:

| Injection Point | Usage | Example |
|----------------|-------|---------|
| `VertexPreamble` | Global declarations, helper functions | Struct definitions, vertex utilities |
| `VertexMain` | Start of vertex main | Early vertex processing |
| `VertexPostProcess` | After position calculation | Passing data to fragment shader |
| `FragmentPreamble` | Global declarations, bindings | Texture samplers, utility functions |
| `FragmentMain` | Start of fragment main | Material setup, early discard |
| `FragmentColorCalculation` | Color computation | Lighting, materials, textures |
| `FragmentPostProcess` | Final post-processing | Tone mapping, color grading |

### Creating a Custom Feature

```rust
use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

pub struct MyFeature {
    enabled: bool,
    my_texture: Option<gpu::Texture>,
}

impl Feature for MyFeature {
    fn name(&self) -> &str {
        "my_feature"
    }

    fn init(&mut self, context: &FeatureContext) {
        // Create GPU resources once during initialization
        self.my_texture = Some(context.gpu.create_texture(...));
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            // Use static strings for zero-copy
            ShaderInjection::new(
                ShaderInjectionPoint::FragmentPreamble,
                include_str!("../shaders/my_functions.wgsl"),
            ),
            // Or use String for dynamic code with custom priority
            ShaderInjection::with_priority(
                ShaderInjectionPoint::FragmentColorCalculation,
                "    final_color = my_function(final_color);",
                10,  // Higher priority = runs later
            ),
        ]
    }
    
    fn prepare_frame(&mut self, context: &FeatureContext) {
        // Update per-frame data (uniforms, animation state, etc.)
    }
    
    fn cleanup(&mut self, context: &FeatureContext) {
        // Release GPU resources
        if let Some(texture) = self.my_texture.take() {
            context.gpu.destroy_texture(texture);
        }
    }
}
```

## Examples

The `crates/examples` directory contains demonstrations of the feature system:

### 1. Base Geometry (`feature_geometry`)
```bash
cargo run --bin feature_geometry --release
```
Demonstrates the base geometry feature without any lighting. Objects appear flat with a uniform gray color.

**Controls:**
- `1` - Toggle base geometry
- `ESC` - Exit

### 2. Geometry + Lighting (`feature_with_lighting`)
```bash
cargo run --bin feature_with_lighting --release
```
Adds basic diffuse + ambient lighting to the geometry. Objects now have shading that responds to a directional light.

**Controls:**
- `1` - Toggle base geometry
- `2` - Toggle lighting
- `ESC` - Exit

### 3. Complete Pipeline (`feature_complete`)
```bash
cargo run --bin feature_complete --release
```
Combines all features: geometry, lighting, materials, and shadows. Objects display a checkerboard pattern with proper lighting and shadow mapping. This demonstrates how features compose to create a complete rendering pipeline.

**Controls:**
- `1` - Toggle base geometry
- `2` - Toggle lighting
- `3` - Toggle materials
- `4` - Toggle shadows
- `ESC` - Exit

### 4. Basic Rendering (`basic_rendering`)
```bash
cargo run --bin basic_rendering --release
```
The original non-feature-based example using the legacy `Renderer` class. Useful for comparison with the feature-based approach.

## Building Custom Pipelines

Mix and match any combination of features:

```rust
// Minimal: Just geometry
let registry = FeatureRegistry::builder()
    .with_feature(BaseGeometry::new())
    .build();

// Lighting pipeline
let registry = FeatureRegistry::builder()
    .with_feature(BaseGeometry::new())
    .with_feature(BasicLighting::new())
    .build();

// Full featured with custom shadow resolution
let registry = FeatureRegistry::builder()
    .with_feature(BaseGeometry::new())
    .with_feature(BasicLighting::new())
    .with_feature(BasicMaterials::new())
    .with_feature(ProceduralShadows::new().with_size(1024))
    .build();
```

## Shader Composition Example

Base geometry shader template (`base_geometry.wgsl`):
```wgsl
// INJECT_FRAGMENTPREAMBLE

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // INJECT_FRAGMENTMAIN

    var final_color = in.color;

    // INJECT_FRAGMENTCOLORCALCULATION

    return vec4<f32>(final_color, 1.0);
}
```

Lighting feature injects:
```wgsl
// At INJECT_FRAGMENTPREAMBLE:
fn apply_basic_lighting(normal: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ndotl = max(dot(normal, light_dir), 0.0);
    return color * (ndotl + 0.2); // diffuse + ambient
}

// At INJECT_FRAGMENTCOLORCALCULATION:
final_color = apply_basic_lighting(normalize(in.world_normal), final_color);
```

## Feature Priority

When multiple features inject at the same point, `priority` controls the order:
- **Lower priority** (e.g., -10) = injected first → for base setup like material bindings
- **Higher priority** (e.g., 10) = injected later → for post-processing like shadows
- **Default priority** (0) = for most features

## API Design Philosophy

Helio is designed to be:

1. **Modular**: Each rendering feature is a separate crate with clear boundaries
2. **Composable**: Features combine through shader injection, not inheritance  
3. **Explicit**: No hidden magic - you see exactly what features are active
4. **Extensible**: Create your own features without modifying the engine
5. **Lightweight**: Examples are minimal - Helio and features do all heavy lifting
6. **Well-Documented**: Every public API has comprehensive documentation
7. **Error-Aware**: Result types for operations that can fail
8. **Resource-Safe**: Automatic cleanup via Drop trait

## Advanced Features

### Debug Output

Enable debug shader output to inspect composed shaders:

```rust
let mut registry = FeatureRegistry::new();
registry.set_debug_output(true);
// Composed shader will be written to composed_shader_debug.wgsl
```

### Error Handling

The feature system uses Result types for better error handling:

```rust
match renderer.registry_mut().toggle_feature("my_feature") {
    Ok(new_state) => {
        println!("Feature is now: {}", if new_state { "ON" } else { "OFF" });
        renderer.rebuild_pipeline();
    }
    Err(e) => eprintln!("Failed to toggle feature: {}", e),
}
```

### Feature Lifecycle

Features follow a clear lifecycle:

1. **Creation**: Feature created with `new()` or builder
2. **Registration**: Added to registry via `register()` or builder
3. **Initialization**: `init()` called once - create GPU resources here
4. **Per-Frame**:
   - `prepare_frame()` - Update uniforms, animation
   - `pre_render_pass()` - Shadow maps, etc.
   - Main render pass (shader injections active)
   - `post_render_pass()` - Post-processing
5. **Cleanup**: `cleanup()` called when renderer drops

## Performance Tips

- Use static strings (`include_str!`) for shader code - zero runtime cost
- Shadow map size significantly impacts performance (1024-2048 recommended)
- Disable unused features to reduce shader complexity
- Features are checked for enabled state each frame - keep the check lightweight
- Use `debug_output` only during development

## Roadmap

Potential future features:
- ✅ Shadow mapping (DONE!)
- Cascaded shadow maps for better quality
- PBR materials with texture support
- Screen-space ambient occlusion (SSAO)
- Bloom and HDR tonemapping
- Deferred rendering pipeline
- Compute-based particle systems
- Volumetric lighting/fog

## Contributing

## Android Build (Experimental)

Helio can be built for Android using [cargo-apk](https://github.com/rust-windowing/cargo-apk) and the `helio-android`/`helio-android-apk` crates:

### Prerequisites
- Install [Android NDK](https://developer.android.com/ndk/downloads) and set `ANDROID_NDK_HOME` environment variable
- Install Rust Android targets:
    ```sh
    rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
    ```
- Install cargo-apk:
    ```sh
    cargo install cargo-apk
    ```

### Build and Run
From the workspace root:
```sh
cargo apk run -p helio-android-apk --target aarch64-linux-android
```
This will build and deploy the APK to a connected Android device or emulator.

**Note:**
- The Android integration is a stub. See `crates/helio-android/src/lib.rs` and winit's android example for a full event loop and renderer integration.
- You may need to adjust the Android manifest and icons for your app.


To add a new feature:

1. Create `crates/helio-feature-yourfeature/`
2. Implement the `Feature` trait with proper docs
3. Add shaders to `shaders/` subdirectory
4. Implement `cleanup()` for GPU resource management
5. Add the crate to workspace `Cargo.toml`
6. Create an example demonstrating your feature
7. Update this README

See existing features for reference implementations.

## License

See LICENSE file for details.
