# Renderer API

## Overview

TheRenderer is the central orchestrator of Helio's rendering system. It manages the GPU device and queue, owns all rendering resources (textures, buffers, bind groups), hosts the render graph that determines pass execution order, and provides the public API for submitting meshes, lights, and debug geometry. Every frame, you submitcontent to the renderer through its draw methods, then call `render()` or `render_scene()` to execute the frame.

Helio's architecture separates scene description from rendering execution. The Renderer doesn't know or care about your game loop, windowing system, or asset loading—it simply exposes a clean interface for uploading meshes to GPUs, configuring materials, and executing render passes. This makes it trivial to integrate into any application, from real-time games to offline rendering tools or automated test harnesses.

## Creating a Renderer

The renderer is constructed with a `RendererConfig`, which specifies the output resolution, surface format, and any optional features you want to enable:

```rust
use helio_render_v2::{Renderer, RendererConfig, FeatureRegistry};

let config = RendererConfig {
    width: 1920,
    height: 1080,
    surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,  // match your surface
    features: FeatureRegistry::new(),
};

let renderer = Renderer::new(config)?;
```

The construction process is quite involved under the hood. The renderer creates the GPU device and queue, builds bind group layouts for global uniforms, materials, lighting, and G-buffer reads, initializes the pipeline cache, creates depth and G-buffer textures, registers all built-in render passes (sky LUT, sky, G-buffer write, deferred lighting, debug draw), and builds the render graph dependency tree. All of this happens automatically—by the time `new()` returns, the renderer is fully ready to accept content and render frames.

### Surface Format

The `surface_format` should match the texture format of your swapchain or window surface. Common values are `Bgra8UnormSrgb` (Windows, most desktop platforms) or `Rgba8UnormSrgb` (some Linux configurations). If you're rendering offscreen for screenshots or video encoding, you might use `Rgba16Float` for HDR output. The renderer doesn't create the surface itself—it just needs to know what format to target so it can configure the final render pass correctly.

### Feature Configuration

The `features` field is a `FeatureRegistry` that holds optional rendering features like lighting, billboards, radiance cascades, or post-processing effects. You can register features before passing the config to the renderer:

```rust
use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BillboardsFeature,
};

let mut features = FeatureRegistry::new();
features.register("lighting", Box::new(LightingFeature::new()))?;
features.register("billboards", Box::new(BillboardsFeature::new()))?;

let config = RendererConfig {
    width: 1920,
    height: 1080,
    surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
    features,
};
```

Each feature implements the `Feature` trait, which provides hooks for initialization, per-frame preparation, and registration of its own render passes. For example, `LightingFeature` creates the shadow atlas, uploads light data to a storage buffer, and registers the `ShadowPass`. Features are modular—if you don't register lighting, the renderer still works but won't cast shadows or apply PBR shading (it'll use unlit or ambient-only rendering).

## Render Modes

Helio supports two complementary workflows: scene-based rendering and immediate-mode submission. You can use either one exclusively or mix them in a single frame.

### Scene-Based Rendering

Scene-based rendering is the high-level workflow where you build a `Scene` struct containing all objects, lights, and atmosphere configuration, then pass it to `render_scene()`:

```rust
let scene = Scene::new()
    .add_object(floor_mesh)
    .add_object_with_material(character_mesh, character_material)
    .add_light(SceneLight::directional([0.3, -1.0, 0.5], [1.0, 0.95, 0.85], 30.0))
    .with_sky_atmosphere(SkyAtmosphere::new());

renderer.render_scene(&scene, &camera, &surface_view, delta_time)?;
```

This method is declarative and composable. The renderer iterates over the scene's objects, extracts lights, and configures sky parameters automatically. It's perfect for games with structured levels or applications where you want a clear separation between world definition and rendering.

### Immediate-Mode Rendering

Immediate mode lets you submit geometry directly without constructing a scene. This is useful for tools, debug overlays, or procedurally generated content that changes every frame:

```rust
renderer.draw_mesh(&mesh_a);
renderer.draw_mesh_with_material(&mesh_b, custom_material);
renderer.render(&camera, &surface_view, delta_time)?;
```

Each `draw_mesh()` call queues a draw command in an internal list. When you call `render()`, the renderer executes the G-buffer pass with those commands, then clears the list for the next frame. This is lower-level than scene-based rendering but more flexible—you can conditionally add meshes based on runtime logic, sort them manually, or interleave drawing with other operations.

### Hybrid Workflow

You can combine both modes in a single frame. For example, render your main scene with `render_scene()`, then use immediate-mode debug draw to overlay wireframes or gizmos:

```rust
// Render main scene
renderer.render_scene(&level_scene, &camera, &surface_view, delta_time)?;

// Add debug overlay
renderer.debug_line(start, end, [1.0, 0.0, 0.0, 1.0], 2.0);
renderer.debug_sphere(light_pos, light_range, [1.0, 1.0, 0.0, 0.5], 1.0);
```

The debug geometry is rendered in a separate pass after the deferred lighting, so it overlays on top of the scene with depth testing. See the Debug Drawing section of the documentation for details.

## Drawing Meshes

Meshes are uploaded to the GPU with the `GpuMesh::new()` constructor, which takes a wgpu device, vertex slice, and index slice:

```rust
use helio_render_v2::mesh::{PackedVertex, GpuMesh};

let vertices = vec![
    PackedVertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0]),
    PackedVertex::new([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
    PackedVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
];
let indices = vec![0, 1, 2];

let gpu_mesh = GpuMesh::new(&renderer.device, &vertices, &indices);
```

The `PackedVertex` format matches the G-buffer shader's vertex input layout exactly: position (3 floats), bitangent sign (1 float), texture coordinates (2 floats), and packed normal/tangent vectors (2 u32 values encoding 8-bit signed normalized components). Normals and tangents are compressed to save bandwidth—each occupies 4 bytes instead of 12. The `new()` constructor automatically computes a default tangent perpendicular to the normal if you don't provide one explicitly.

Once you have a `GpuMesh`, you submit it to the renderer with `draw_mesh()` or `draw_mesh_with_material()`:

```rust
// Use default white material
renderer.draw_mesh(&gpu_mesh);

// Use custom PBR material
let material = renderer.create_material(&Material::new()
    .with_base_color([0.8, 0.2, 0.1, 1.0])
    .with_metallic(1.0)
    .with_roughness(0.3));

renderer.draw_mesh_with_material(&gpu_mesh, material.bind_group);
```

The draw call is queued in an internal vector and executed during the G-buffer pass. The renderer binds the provided material bind group before issuing the GPU draw command. If you don't provide a material, it uses a built-in default white material with metallic=0, roughness=0.5, and no texture maps.

### Primitive Meshes

Helio provides factory methods for common procedural shapes. These are useful for rapid prototyping, debug visualization, or placeholder geometry:

```rust
// Unit cube centered at (0, 0, 0) with half-extent 0.5
let cube = GpuMesh::cube(&renderer.device, [0.0, 0.0, 0.0], 0.5);

// Rectangular box with independent half-extents per axis
let box_mesh = GpuMesh::rect3d(&renderer.device, [0.0, 0.5, 0.0], [2.0, 0.5, 1.0]);

// Ground plane (XZ quad centered at origin)
let plane = GpuMesh::plane(&renderer.device, [0.0, 0.0, 0.0], 10.0);
```

All primitive meshes have correct normals, tangents, and UV coordinates. The cube uses a standard box unwrap (each face is a square in UV space), and the plane uses a simple 0-1 mapping. These are great for quickly building test scenes or visualizing abstract data.

### Mesh Bounds

Every `GpuMesh` stores a bounding sphere defined by a center point and radius. This is computed automatically during construction by finding the centroid of all vertices and the maximum distance from that point. The bounding sphere is used for shadow map culling—if a mesh is entirely outside a light's frustum, it's skipped during the shadow pass. This is an important optimization for scenes with thousands of objects and multiple shadow-casting lights.

```rust
println!("Mesh bounds: center={:?}, radius={}", 
    gpu_mesh.bounds_center, 
    gpu_mesh.bounds_radius);
```

The bounding sphere is slightly conservative (it uses the centroid rather than a true minimal bounding sphere), but this keeps the computation fast and the extra padding rarely matters in practice.

## Materials

Materials define the physical appearance of surfaces via PBR (Physically Based Rendering) parameters. Helio's material system supports base color (albedo), metallic/roughness/occlusion maps, normal maps, emissive color, and opacity. Materials are uploaded to the GPU as bind groups that the G-buffer pass binds before drawing each object.

### Creating Materials

The `Material` struct is a builder-pattern description of PBR properties. You configure it, then call `renderer.create_material()` to upload it to the GPU:

```rust
use helio_render_v2::Material;

let gold_material = Material::new()
    .with_base_color([1.0, 0.85, 0.4, 1.0])  // golden yellow
    .with_metallic(1.0)                       // fully metallic
    .with_roughness(0.1)                      // very smooth (polished)
    .with_emissive_factor(0.0);               // no glow

let gpu_material = renderer.create_material(&gold_material);
```

The resulting `GpuMaterial` contains an `Arc<wgpu::BindGroup>` that you pass to `draw_mesh_with_material()` or `Scene::add_object_with_material()`. Materials are cheap to clone (Arc-counted) and can be shared across multiple objects—if you have twenty gold coins, create one gold material and reuse it for all of them.

### Material Parameters

- **base_color**: RGBA color multiplier for albedo. The alpha channel controls transparency (1.0 = opaque, 0.0 = fully transparent). If you provide an albedo texture, this color is multiplied with the texture sample.
- **metallic**: 0.0 for dielectrics (plastic, wood, stone), 1.0 for metals (iron, gold, aluminum). Intermediate values are physically dubious but sometimes useful for artistic effects (rusty metal = 0.7, oxidized copper = 0.6).
- **roughness**: 0.0 for mirror-like surfaces (polished chrome, water), 1.0 for perfectly diffuse (chalk, clay). This controls the width of specular highlights.
- **emissive_factor**: Multiplier for emissive color. Set to 0.0 for non-glowing surfaces, or >0.0 for light-emitting materials (neon signs, lava, screens). Emissive objects don't cast light on other surfaces—they just add brightness to themselves.
- **emissive_color**: RGB color of the glow. Multiplied by `emissive_factor` and added to the final pixel color after PBR shading.
- **ao** (ambient occlusion): A baked shadowing term in [0, 1]. Values <1.0 darken the surface in crevices or cavities. Typically comes from AO maps; the default is 1.0 (no AO).

### Texture Maps

Materials can include up to five texture maps: albedo, normal, ORM (occlusion-roughness-metallic packed), emissive, and opacity. Textures are loaded separately and referenced by handle or path:

```rust
let brick_material = Material::new()
    .with_albedo_texture(albedo_handle)       // diffuse color RGB
    .with_normal_texture(normal_handle)       // tangent-space normals (DirectX format: +Y up)
    .with_orm_texture(orm_handle)             // R=AO, G=roughness, B=metallic
    .with_emissive_texture(emissive_handle)   // emissive RGB glow map
    .with_base_color([1.0, 1.0, 1.0, 1.0]);   // tint multiplier

let gpu_material = renderer.create_material(&brick_material);
```

The texture handle types depend on your asset system—Helio doesn't provide asset loading out of the box. You typically integrate with an external crate like `image` or `wgpu-utils` to load PNG/JPEG files, then pass `wgpu::TextureView` references to the material builder.

If you don't provide a texture, Helio uses sensible defaults: albedo defaults to white (1,1,1), normal defaults to flat up (0,0,1 in tangent space), ORM defaults to full AO, medium roughness, zero metallic, and emissive defaults to black (no glow). This means you can mix textureless and textured materials freely without worrying about missing bindings.

## The Render Loop

A typical frame looks like this:

```rust
loop {
    // 1. Handle input, update camera, update game state
    let delta_time = calculate_delta_time();
    camera.position += movement_delta;

    // 2. Submit geometry via immediate mode or scene
    renderer.draw_mesh(&floor);
    renderer.draw_mesh_with_material(&character, character_material.bind_group);

    // 3. (Optional) Submit debug geometry
    renderer.debug_line(start, end, [1.0, 0.0, 0.0, 1.0], 2.0);

    // 4. Render the frame
    let surface_texture = surface.get_current_texture()?;
    let view = surface_texture.texture.create_view(&Default::default());
    renderer.render(&camera, &view, delta_time)?;

    // 5. Present
    surface_texture.present();
}
```

The renderer doesn't own the surface or manage vsync—that's your responsibility. Helio is designed to integrate with any windowing library (winit, SDL, glutin) or headless environment. The `render()` method writes to whatever texture view you provide, whether it's a swapchain image, an offscreen framebuffer, or a compute shader output.

### Delta Time

The `delta_time` parameter is the elapsed time since the last frame in seconds. It's uploaded to the GPU as part of the `Globals` uniform buffer and made available to shaders for animations, time-based effects, or motion blur. The camera also has a `time` field that accumulates across frames—you can increment it by delta_time each frame or set it to an absolute clock value.

## Debug Drawing

Debug geometry is submitted through dedicated methods that tessellate shapes on the CPU and render them as a translucent overlay after the deferred lighting pass. This system is perfect for visualizing physics bounding volumes, AI navigation paths, light frustums, or any runtime diagnostic data:

```rust
// Line segment
renderer.debug_line(
    Vec3::new(0.0, 0.0, 0.0),  // start
    Vec3::new(5.0, 2.0, 3.0),  // end
    [1.0, 0.0, 0.0, 1.0],      // red color (RGBA)
    2.0                         // thickness in pixels
);

// Wireframe sphere
renderer.debug_sphere(
    Vec3::new(10.0, 3.0, 0.0),  // center
    2.5,                         // radius
    [0.0, 1.0, 1.0, 0.5],       // cyan with 50% opacity
    1.5                          // line thickness
);

// Wireframe box (with rotation)
renderer.debug_box(
    Vec3::new(-5.0, 1.0, 0.0),  // center
    Vec3::new(1.0, 0.5, 2.0),   // half-extents (x, y, z)
    Quat::from_rotation_y(0.5), // rotation quaternion
    [1.0, 1.0, 0.0, 1.0],       // yellow
    2.0                          // thickness
);

// Cone (for spotlights or visual cues)
renderer.debug_cone(
    Vec3::new(3.0, 5.0, 0.0),    // apex
    Vec3::new(0.0, -1.0, 0.0),   // direction (should be normalized)
    3.0,                          // height
    1.0,                          // base radius
    [1.0, 0.5, 0.0, 0.8],        // orange
    1.5                           // thickness
);

// Capsule (for character controllers)
renderer.debug_capsule(
    Vec3::new(0.0, 0.5, 0.0),    // start (bottom center)
    Vec3::new(0.0, 1.8, 0.0),    // end (top center)
    0.3,                          // radius
    [0.0, 1.0, 0.0, 0.7],        // green
    1.0                           // thickness
);
```

All shapes are rendered with depth testing, so they occlude correctly behind geometry. The thickness parameter controls the width of the wireframe lines—values around 1.0-2.0 are typical for readability. Colors use RGBA format where the alpha channel controls transparency (1.0 = fully opaque, 0.0 = invisible).

Debug shapes are transient—they're cleared every frame after rendering. If you want persistent visualization, you need to re-submit them each frame. This keeps the API simple and prevents accidental accumulation of thousands of debug primitives over time. If you need to clear debug shapes manually mid-frame (for example, to visualize different stages of an algorithm), call `renderer.clear_debug_shapes()`.

### Debug Geometry Implementation

Internally, debug shapes are tessellated into triangle meshes on the CPU and uploaded to a transient GPU buffer each frame. Lines become rectangular prisms (tubes) aligned along the line direction with a radius proportional to thickness. Spheres are subdivided icosahedrons. Boxes are 12 edge segments. Cones and capsules are built from cylinder segments and spherical caps. The tessellation is low-poly (spheres are ~100 triangles) to keep the CPU cost minimal—this system is designed for hundreds of debug shapes, not millions.

The debug draw pass runs after the deferred lighting pass in the render graph, ensuring it appears on top of all scene geometry. It uses the global bind group to access camera transforms, so debug shapes automatically follow the camera viewpoint without additional configuration.

## Feature Management

Features are optional rendering subsystems that extend the renderer's capabilities. Examples include lighting (shadows, PBR shading), billboards (camera-facing quads), radiance cascades (global illumination), and post-processing (bloom, FXAA). Features register render passes, allocate GPU resources, and hook into the per-frame update loop.

### Enabling and Disabling Features at Runtime

Features can be toggled dynamically without recreating the renderer:

```rust
// Enable a feature
renderer.enable_feature("lighting")?;

// Disable a feature
renderer.disable_feature("billboard")?;
```

When you enable a feature, its shader defines are added to the pipeline cache's active flags, and any pipelines requesting those defines are recompiled lazily on next use. When you disable a feature, the defines are cleared and pipelines fall back to simpler codepaths. This is useful for performance profiling (disable features to measure their cost) or adapting rendering quality to hardware capability (disable shadows on low-end GPUs).

### Accessing Features

If you need to mutate feature state (for example, updating shadow map resolution or changing billboard texture atlases), use `get_feature_mut()`:

```rust
if let Some(lighting) = renderer.get_feature_mut::<LightingFeature>("lighting") {
    lighting.set_shadow_resolution(2048);
    lighting.enable_cascaded_shadows(true);
}
```

The type parameter is the concrete feature struct (must implement the `Feature` trait). The method returns `Option<&mut T>`, so you can safely query features that might not be registered without panicking.

## Resizing

When your window or rendering target changes size, call `resize()` to reallocate internal framebuffers:

```rust
renderer.resize(new_width, new_height);
```

This recreates the depth texture and G-buffer render targets at the new resolution. Existing meshes and materials are unaffected—only GPU textures owned by the renderer are resized. If you have a swapchain, you'll also need to reconfigure it separately (Helio doesn't manage the swapchain).

After resizing, the camera's aspect ratio is typically stale, so update it too:

```rust
renderer.resize(1600, 900);

camera = Camera::perspective(
    camera.position,
    camera_target,
    Vec3::Y,
    60.0_f32.to_radians(),
    1600.0 / 900.0,  // new aspect ratio
    0.1,
    1000.0,
    camera.time,
);
```

Forgetting to update the camera causes stretching—objects appear squished or elongated because the projection matrix doesn't match the render target's dimensions.

## Performance Profiling

The renderer includes a GPU profiler that measures the execution time of each render pass using timestamp queries. This is invaluable for identifying bottlenecks and verifying optimizations.

### Enabling Debug Timing

Call `debug_key_pressed()` to toggle timing printouts:

```rust
// In your keyboard input handler:
if key_code == KeyCode::Key4 {
    renderer.debug_key_pressed();
}
```

When enabled, GPU pass timings are printed to stderr approximately once per second (every 60 frames). The output shows the duration of each pass in milliseconds, plus total GPU time, total CPU time (command encoding), and frame-to-frame latency:

```
━━━━━━━━━━━━━━━━━━━━ FRAME TIMINGS ━━━━━━━━━━━━━━━━━━━━
 SkyLutPass         0.08 ms
 SkyPass            0.12 ms
 ShadowPass         2.34 ms
 GBufferPass        3.56 ms
 DeferredLighting   1.89 ms
 DebugDrawPass      0.03 ms
 
 TOTAL GPU:         8.02 ms
 TOTAL CPU:         0.45 ms
 Frame time:       16.67 ms (CPU render() call duration)
 Frame-to-frame:   16.72 ms (time since last frame start)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The profiler uses wgpu's `TIMESTAMP_QUERY` feature, which is only available on devices that support it (most desktop GPUs). If timestamp queries aren't supported, the profiler is disabled and `debug_key_pressed()` does nothing.

### Programmatic Timing Access

You can also query timing data directly from your application:

```rust
let timings = renderer.last_frame_timings();
for pass in timings {
    println!("{}: {:.2}ms", pass.name, pass.duration_ms);
}

println!("Frame {}: total GPU {:.2}ms", 
    renderer.frame_count(),
    timings.iter().map(|p| p.duration_ms).sum::<f32>());
```

This is useful for building custom profiling UIs, exporting performance logs, or implementing automatic quality scaling based on framerate.

## Renderer Architecture

Understanding the internal structure helps when debugging issues or extending Helio with custom features.

### Render Graph

The renderer uses a dependency-based render graph to automatically sequence passes. Each pass declares which resources it reads and writes (for example, the G-buffer pass writes to the albedo/normal/ORM/emissive textures, and the deferred lighting pass reads them). The graph sorts passes topologically so dependencies are satisfied—if pass B reads a texture that pass A writes, pass A executes first.

This design eliminates manual ordering boilerplate. When you register a custom feature that adds a new pass, you just declare its dependencies and the graph handles scheduling. The graph also validates that there are no cycles (A depends on B, B depends on A) and reports errors if resources are missing.

### Bind Groups and Layouts

Helio organizes GPU resources into four bind groups:

1. **Global (group 0)**: Camera uniform, globals uniform (frame count, delta time, light count, ambient color). Bound once per frame.
2. **G-buffer Read (group 1)**: Albedo, normal, ORM, emissive texture views plus depth. Used by the deferred lighting pass to reconstruct surface properties.
3. **Lighting (group 2)**: Light storage buffer, shadow atlas, shadow sampler, environment cube map, shadow matrices, radiance cascade texture. Used by shading passes that need light data.
4. **Material (group 3)**: Per-material textures (albedo, normal, ORM, emissive, opacity) plus material uniform buffer. Bound per draw call in the G-buffer pass.

This layout minimizes bind group switches. Most passes only bind their specific group(s), and common resources (camera, lights) are in group 0 so they're available everywhere without rebinding.

### Deferred Rendering Pipeline

Helio uses a deferred shading architecture with the following passes:

1. **Sky LUT Pass**: Renders a small panoramic lookup texture (200×100 pixels) containing pre-integrated atmospheric scattering. This is sampled by the sky pass to avoid per-pixel raymarching.
2. **Sky Pass**: Fills the background with the physical atmosphere by sampling the LUT. Runs before geometry so occluded pixels can be depth-culled later.
3. **Shadow Pass**: Renders depth maps from light viewpoints into a shadow atlas. Supports directional (cascaded), point (cubemap), and spot (single face) lights.
4. **G-Buffer Pass**: Writes albedo, view-space normals, occlusion/roughness/metallic, and emissive into four render targets. Also writes depth. This is the geometry pass where all opaque meshes are drawn.
5. **Features (e.g., Billboard Pass, Radiance Cascades)**: Custom feature passes run here, after G-buffer but before lighting. Billboards render camera-facing quads on top of geometry; radiance cascades build a light probe grid.
6. **Deferred Lighting Pass**: Fullscreen quad that reads the G-buffer, applies PBR shading (Cook-Torrance BRDF) with all lights, and writes the final HDR color to a render target.
7. **Debug Draw Pass**: Renders wireframe overlay shapes on top of the lit scene.

The final color target from the deferred lighting pass is what you see. It's already tone-mapped and ready to present, though you could add post-processing (bloom, FXAA) in additional passes if desired.

### Pipeline Cache

The `PipelineCache` lazily compiles render pipelines on-demand. When a shader is first used, it's compiled with the active feature defines (e.g., `USE_SHADOWS`, `USE_BILLBOARDS`). The compiled pipeline is cached so subsequent frames reuse it without recompilation. If you enable/disable a feature, pipelines are recompiled with updated defines the next time they're requested.

Shader source is embedded via `include_str!()` macros, so there's no runtime file I/O. This keeps the binary self-contained and eliminates shader path issues, though it does mean you need to rebuild to change shaders during development.

## Advanced Usage

### Custom Render Passes

If you need rendering behavior that isn't covered by built-in features, you can implement custom render passes by defining a struct that implements the `RenderPass` trait and registering it in the graph:

```rust
use helio_render_v2::graph::RenderPass;

struct MyCustomPass {
    pipeline: Arc<wgpu::RenderPipeline>,
}

impl RenderPass for MyCustomPass {
    fn name(&self) -> &str { "MyCustomPass" }
    
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &mut GraphContext,
    ) -> Result<()> {
        // Create render pass, bind pipeline, issue draws
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("My Custom Pass"),
            color_attachments: &[/* ... */],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&self.pipeline);
        // ... draw logic
        Ok(())
    }
    
    fn resources(&self) -> Vec<ResourceDeclaration> {
        vec![
            ResourceDeclaration::Read("color_target"),
            ResourceDeclaration::Write("custom_output"),
        ]
    }
}
```

Then register it in the renderer's graph:

```rust
renderer.graph.add_pass(MyCustomPass::new(device, layout));
renderer.graph.build()?;
```

The graph will automatically schedule your pass according to its resource dependencies.

### Render-to-Texture

If you need to render to an offscreen texture (for reflections, portals, or picture-in-picture effects), create a separate texture view and pass it to `render()`:

```rust
let offscreen_texture = device.create_texture(&wgpu::TextureDescriptor {
    label: Some("Offscreen RT"),
    size: wgpu::Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: wgpu::TextureFormat::Rgba16Float,
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    view_formats: &[],
});

let offscreen_view = offscreen_texture.create_view(&Default::default());

renderer.render(&reflection_camera, &offscreen_view, delta_time)?;
```

The renderer doesn't care whether the view comes from a swapchain or an offscreen texture—it just writes pixels to whatever you provide.

## Common Pitfalls

### Forgetting to Call render()

The draw methods (`draw_mesh()`, `debug_line()`, etc.) only queue commands—they don't actually execute anything. You must call `render()` or `render_scene()` each frame to submit the GPU work:

```rust
// ❌ WRONG: nothing renders
renderer.draw_mesh(&mesh);

// ✅ CORRECT: queue + execute
renderer.draw_mesh(&mesh);
renderer.render(&camera, &view, delta_time)?;
```

### Surface Format Mismatch

If the renderer's `surface_format` doesn't match your window surface, you'll get validation errors or incorrect colors. Always query your surface's preferred format and pass it to `RendererConfig`:

```rust
let surface_format = surface.get_capabilities(&adapter).formats[0];
let config = RendererConfig {
    width: 1920,
    height: 1080,
    surface_format,  // use the surface's native format
    features: FeatureRegistry::new(),
};
```

### Resize Without Camera Update

Resizing the renderer doesn't update the camera's aspect ratio. If you resize the window but keep the old camera, objects will appear stretched:

```rust
// ❌ WRONG: aspect ratio is stale
renderer.resize(1600, 900);
// camera still uses old aspect 16:9

// ✅ CORRECT: update both
renderer.resize(1600, 900);
camera = Camera::perspective(
    camera.position, target, Vec3::Y,
    60.0_f32.to_radians(),
    1600.0 / 900.0,  // new aspect
    0.1, 1000.0, camera.time,
);
```

### Arc<BindGroup> Lifetime Confusion

Materials return `Arc<wgpu::BindGroup>`, which is reference-counted. If you pass it to `draw_mesh_with_material()`, the draw call stores a clone of the Arc, so the bind group stays alive even if you drop the `GpuMaterial`. This is intentional—draw calls are queued and executed later, so they need to own their dependencies. Just be aware that cloning a material is cheap (incrementing a refcount), not duplicating GPU resources.

## Summary

The Renderer is the heart of Helio's rendering system. It manages the GPU, orchestrates render passes, and provides a clean API for submitting geometry, materials, and debug shapes. Key points:

- **Construction**: Create with `Renderer::new(config)` specifying resolution, surface format, and features.
- **Rendering Modes**: Use `render_scene()` for declarative scene graphs or `render()` for immediate-mode submission.
- **Materials**: Build with `Material`, upload with `renderer.create_material()`, bind with `draw_mesh_with_material()`.
- **Debug Drawing**: Submit transient wireframe shapes with `debug_line()`, `debug_sphere()`, `debug_box()`, etc.
- **Profiling**: Toggle GPU timing with `debug_key_pressed()` or query via `last_frame_timings()`.
- **Resizing**: Call `resize()` when the window dimensions change, and update the camera aspect ratio too.

This architecture keeps your application code simple and focused on content creation rather than low-level GPU plumbing. Build your scenes, configure your materials, and let Helio handle the rest.
