# Helio - Modular GPU Rendering Engine

Helio is a modular, feature-based rendering engine built on top of blade-graphics. It allows you to compose rendering pipelines from individual feature crates, each providing their own shaders and GPU functionality.

## Architecture

### Core Crates

- **helio-core**: Fundamental data structures (vertices, meshes, primitives)
- **helio-render**: Core rendering infrastructure with `FeatureRenderer`
- **helio-features**: Feature system trait definitions and registry
- **helio-lighting**: Ray-traced global illumination (standalone)

### Feature Crates

Each feature is a separate crate that implements the `Feature` trait:

- **helio-feature-base-geometry**: Base geometry rendering (no lighting)
- **helio-feature-lighting**: Basic diffuse + ambient lighting
- **helio-feature-materials**: Material system with color/metallic/roughness properties

## Feature System

### How It Works

1. **Features provide shader code**: Each feature returns `ShaderInjection` objects that specify where their WGSL code should be inserted
2. **Shader composition**: The `FeatureRegistry` composes features into a final shader by replacing injection point markers
3. **Modular pipelines**: Features can add render passes before/after the main pass
4. **Runtime toggling**: Features can be enabled/disabled (requires pipeline rebuild)

### Injection Points

Features can inject code at these points in the rendering pipeline:

- `VertexPreamble`: Before vertex shader main function
- `VertexMain`: Start of vertex main
- `VertexPostProcess`: After position calculation
- `FragmentPreamble`: Before fragment shader main (for functions/bindings)
- `FragmentMain`: Start of fragment main
- `FragmentColorCalculation`: For color computation
- `FragmentPostProcess`: Final post-processing

### Creating a Custom Feature

```rust
use helio_features::{Feature, FeatureContext, ShaderInjection, ShaderInjectionPoint};

pub struct MyFeature {
    enabled: bool,
}

impl Feature for MyFeature {
    fn name(&self) -> &str {
        "my_feature"
    }

    fn init(&mut self, _context: &FeatureContext) {
        // Initialize GPU resources here
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn shader_injections(&self) -> Vec<ShaderInjection> {
        vec![
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentPreamble,
                code: include_str!("../shaders/my_functions.wgsl").to_string(),
                priority: 0,
            },
            ShaderInjection {
                point: ShaderInjectionPoint::FragmentColorCalculation,
                code: "    final_color = my_function(final_color);".to_string(),
                priority: 0,
            },
        ]
    }
}
```

### Using the Feature System

```rust
use helio_feature_base_geometry::BaseGeometry;
use helio_feature_lighting::BasicLighting;
use helio_features::FeatureRegistry;
use helio_render::FeatureRenderer;

// Create features
let base_geometry = BaseGeometry::new();
let base_shader = base_geometry.shader_template().to_string();

// Register features
let mut registry = FeatureRegistry::new();
registry.register(base_geometry);
registry.register(BasicLighting::new());

// Create renderer with composed pipeline
let renderer = FeatureRenderer::new(
    context.clone(),
    surface_format,
    width,
    height,
    registry,
    &base_shader,
);

// Render
renderer.render(&mut encoder, target_view, camera, &meshes, delta_time);
```

## Examples

The `crates/examples` directory contains demonstrations of the feature system:

### 1. Base Geometry (`feature_geometry`)
```bash
cargo run --bin feature_geometry --release
```
Demonstrates the base geometry feature without any lighting. Objects appear flat with a uniform gray color.

### 2. Geometry + Lighting (`feature_with_lighting`)
```bash
cargo run --bin feature_with_lighting --release
```
Adds basic diffuse + ambient lighting to the geometry. Objects now have shading that responds to a directional light.

### 3. Complete Pipeline (`feature_complete`)
```bash
cargo run --bin feature_complete --release
```
Combines all three features: geometry, lighting, and materials. This demonstrates how features compose to create a complete rendering pipeline.

### 4. Basic Rendering (`basic_rendering`)
```bash
cargo run --bin basic_rendering --release
```
The original non-feature-based example using the legacy `Renderer` class.

## Building Custom Pipelines

You can mix and match any combination of features:

```rust
// Minimal: Just geometry
registry.register(BaseGeometry::new());

// Lighting only
registry.register(BaseGeometry::new());
registry.register(BasicLighting::new());

// Full featured
registry.register(BaseGeometry::new());
registry.register(BasicLighting::new());
registry.register(BasicMaterials::new());
registry.register(MyCustomFeature::new());
```

## Shader Injection Example

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
- **Lower priority** = injected first (e.g., -10 for materials)
- **Higher priority** = injected later (e.g., 10 for post-processing)

## Design Philosophy

Helio is designed to be:

1. **Modular**: Each rendering feature is a separate crate
2. **Composable**: Features combine through shader injection, not inheritance
3. **Explicit**: No hidden magic - you see exactly what features are active
4. **Extensible**: Create your own features without modifying the engine
5. **Lightweight**: Examples are minimal - Helio and features do all heavy lifting

## Roadmap

Potential future features:
- Shadow mapping
- PBR materials with texture support
- Screen-space ambient occlusion (SSAO)
- Bloom and HDR tonemapping
- Deferred rendering pipeline
- Compute-based particle systems

## Contributing

To add a new feature:

1. Create `crates/helio-feature-yourfeature/`
2. Implement the `Feature` trait
3. Add shaders to `shaders/` subdirectory
4. Add the crate to workspace `Cargo.toml`
5. Create an example demonstrating your feature

## License

See LICENSE file for details.
