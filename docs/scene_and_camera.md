# Scene and Camera Systems

## Overview

The scene and camera systems form the foundation of Helio's content pipeline. The Scene acts as the authoritative database for everything the renderer draws—objects, lights, atmospheric effects, and billboards—while the Camera defines the viewpoint and projection transform used to render that content. Together, these systems provide a clean separation between what you want to draw and how you want to view it.

Unlike many engines where renderer state is scattered across multiple subsystems, Helio centralizes all renderable content in a single `Scene` struct. This design makes it trivial to swap scenes, clone them for multi-viewport rendering, or serialize them for level streaming. The camera is similarly self-contained: it holds the view-projection matrix, world position, and inverse transforms needed by shaders, all packaged in a GPU-ready format that can be uploaded directly to a uniform buffer.

## Camera System

### The Camera Structure

The `Camera` struct represents a complete camera configuration ready for GPU consumption. It's marked with `#[repr(C)]` and derives `Pod` and `Zeroable` from bytemuck, which means you can transmute it directly into bytes and upload it to a uniform buffer without manual serialization. Every frame, the renderer writes this structure to the GPU so shaders can access the view-projection matrix and camera position.

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    /// Combined view-projection matrix
    pub view_proj: Mat4,
    /// Camera position in world space
    pub position: Vec3,
    /// Elapsed time in seconds
    pub time: f32,
    /// Inverse of view_proj (needed by sky shader to reconstruct world ray dirs)
    pub view_proj_inv: Mat4,
}
```

The `view_proj` matrix transforms world-space positions into clip space. Shaders multiply vertex positions by this matrix to project them onto the screen. The `position` field stores the camera's location in world coordinates, which is essential for lighting calculations—specular highlights, for example, depend on the view direction from each surface point to the camera. The `time` field provides a global elapsed-time value that shaders can use for animations, procedural effects, or temporal anti-aliasing jitter. Finally, `view_proj_inv` is the inverse of the view-projection matrix, used primarily by the sky shader to reconstruct world-space ray directions from screen-space pixels.

### Creating Cameras

Helio provides two primary factory methods for creating cameras: perspective and orthographic projections. The perspective camera mimics how the human eye or a real-world camera works, with foreshortening and a vanishing point. The orthographic camera produces parallel projection lines, which is useful for UI overlays, shadow maps, or isometric views.

#### Perspective Camera

The perspective camera constructor takes a position, target point, up vector, field of view, aspect ratio, near plane, far plane, and time value. It internally constructs a right-handed view matrix using `look_at_rh` and a perspective projection matrix, then combines them into a single `view_proj` transform:

```rust
let camera = Camera::perspective(
    Vec3::new(0.0, 2.0, 5.0),  // position: stand 2 units up, 5 units back
    Vec3::new(0.0, 0.0, 0.0),  // target: look at the origin
    Vec3::Y,                    // up: standard Y-up orientation
    60.0_f32.to_radians(),      // vertical field of view (60 degrees)
    1920.0 / 1080.0,            // aspect ratio (16:9)
    0.1,                        // near plane (clip anything closer than 0.1)
    1000.0,                     // far plane (clip anything beyond 1000.0)
    0.0,                        // time: start at zero
);
```

The field of view controls how wide the camera's cone of vision is. Smaller values create a telephoto effect (zoomed in), while larger values produce a wide-angle or fisheye look. The aspect ratio should match your render target's width/height to avoid distortion. The near and far planes define the depth range that gets mapped to the depth buffer—choosing these carefully is important for precision, especially in large scenes where z-fighting can occur if the range is too wide.

#### Orthographic Camera

The orthographic constructor is similar but takes explicit left, right, bottom, and top bounds instead of a field of view and aspect ratio. This defines a rectangular frustum in world space that maps directly to screen space without perspective distortion:

```rust
let ortho_camera = Camera::orthographic(
    Vec3::new(0.0, 10.0, 0.0),  // position: top-down view
    Vec3::ZERO,                 // target: look at origin
    Vec3::Z,                    // up: screen-up is world +Z
    -10.0, 10.0,                // left, right: 20-unit horizontal span
    -10.0, 10.0,                // bottom, top: 20-unit vertical span
    0.1, 100.0,                 // near, far: depth range
    0.0,                        // time
);
```

Orthographic cameras are commonly used for shadow cascades, debugging overlays, or 2D rendering where perspective would be inappropriate. Objects don't get smaller with distance—a cube ten units away looks the same size as one right in front of the camera.

#### Manual Construction

If you already have a pre-computed view-projection matrix (for example, from a third-party camera controller or VR headset), you can construct a Camera directly with the low-level `new` method:

```rust
let custom_vp = compute_custom_view_proj();
let camera = Camera::new(custom_vp, Vec3::new(1.0, 2.0, 3.0), 42.0);
```

This computes the inverse matrix automatically and packages everything into the GPU-ready format.

### Camera Methods

The Camera provides a `forward()` method that extracts the normalized forward direction vector from the inverse view-projection matrix. This is useful when you need to cast rays, implement camera-relative movement, or align objects to face the camera:

```rust
let fwd = camera.forward();
println!("Camera is looking in direction: {:?}", fwd);
```

The forward vector points in the direction the camera is facing, in world space. It's computed by extracting the negative Z-axis from the view matrix and normalizing it. This follows the right-handed coordinate convention where -Z is forward.

## Scene System

### The Scene Database

The `Scene` struct is the single source of truth for everything the renderer draws. It contains vectors of objects, lights, and billboards, plus global state like ambient lighting and sky configuration. When you call `renderer.render_scene()`, the renderer iterates over these collections, uploads them to the GPU, and executes the appropriate render passes:

```rust
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub lights: Vec<SceneLight>,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub billboards: Vec<BillboardInstance>,
    pub sky_color: [f32; 3],
    pub sky_atmosphere: Option<SkyAtmosphere>,
    pub skylight: Option<Skylight>,
}
```

The design philosophy here is builder-pattern construction: you create an empty scene with `Scene::new()`, then chain method calls to add content. This produces readable code and makes it easy to conditionally add elements based on runtime logic.

### Building a Scene

Here's a typical scene setup showing how the builder pattern flows naturally:

```rust
use helio_render_v2::{Scene, SceneLight, Material};

let scene = Scene::new()
    .with_sky([0.6, 0.8, 1.0])  // light blue background
    .with_ambient([0.1, 0.1, 0.15], 0.3)  // dim blue ambient
    .add_light(SceneLight::directional(
        [0.3, -1.0, 0.2],        // direction: from upper-right
        [1.0, 0.95, 0.9],        // warm white sunlight
        25.0                     // intensity
    ))
    .add_object(floor_mesh)
    .add_object_with_material(
        character_mesh,
        renderer.create_material(&character_material)
    );
```

Each method consumes `self` and returns a modified scene, allowing you to chain calls indefinitely. The scene doesn't actually render anything until you pass it to the renderer—it's just a data container.

### Scene Objects

A `SceneObject` wraps a `GpuMesh` and an optional material bind group. If you don't provide a material, the renderer uses its built-in default white material:

```rust
pub struct SceneObject {
    pub mesh: GpuMesh,
    pub material: Option<Arc<wgpu::BindGroup>>,
}
```

You add objects to the scene with `add_object(mesh)` for default material or `add_object_with_material(mesh, material)` for custom PBR properties. The `GpuMaterial` returned by `renderer.create_material()` contains a bind group that you pass in here:

```rust
let material = Material::new()
    .with_base_color([0.8, 0.2, 0.1, 1.0])  // reddish metal
    .with_metallic(1.0)
    .with_roughness(0.3);

let gpu_material = renderer.create_material(&material);

let scene = Scene::new()
    .add_object_with_material(mesh, gpu_material);
```

Internally, the renderer iterates the `objects` vector during the G-buffer pass and issues a draw call for each one, binding the per-object material before drawing.

### Lights

Helio supports three light types: directional, point, and spot. Each is constructed with a dedicated factory method on `SceneLight`:

#### Directional Light

Directional lights represent infinitely distant sources like the sun. They have a direction but no position (position is ignored). The light rays are parallel across the entire scene:

```rust
let sun = SceneLight::directional(
    [0.3, -1.0, 0.2],   // direction: normalized vector the light travels along
    [1.0, 0.95, 0.85],  // color: warm daylight
    30.0                // intensity: brightness multiplier
);
```

The direction vector should be normalized for correct shadowing, though the shader will typically normalize it again. Directional lights are cheap to shadow because they use cascaded shadow maps that cover the camera frustum directly, without needing cubemap faces.

#### Point Light

Point lights emit in all directions from a position. They're perfect for light bulbs, torches, explosions, or any omnidirectional source:

```rust
let torch = SceneLight::point(
    [5.0, 2.0, 3.0],    // position: world-space location
    [1.0, 0.7, 0.4],    // color: orange flame
    15.0,               // intensity
    10.0                // range: falloff distance
);
```

The range parameter controls both the attenuation distance and the shadow map bounding volume. Beyond this distance, the light contributes zero illumination. Point lights require six shadow map faces (a cubemap) if shadowing is enabled, which makes them significantly more expensive than directional or spot lights.

#### Spot Light

Spot lights emit a cone of light from a position along a direction. They're used for flashlights, car headlights, stage spotlights, or any focused beam:

```rust
let flashlight = SceneLight::spot(
    [0.0, 1.5, 0.0],     // position: camera or hand position
    [0.0, 0.0, -1.0],    // direction: forward vector
    [1.0, 1.0, 1.0],     // color: white light
    20.0,                // intensity
    15.0,                // range
    0.7,                 // inner_angle: sharp cone (radians)
    1.0                  // outer_angle: soft falloff edge (radians)
);
```

The `inner_angle` and `outer_angle` define the cone shape. Inside the inner angle, the light is full strength. Between inner and outer, it smoothly fades to zero. Spot lights use a single shadow map (no cubemap), making them cheaper than point lights but slightly more expensive than directional for shadow rendering.

### Ambient Lighting

Ambient light provides a baseline illumination that affects all surfaces uniformly, preventing areas from going completely black. It's a hacky approximation of global illumination—real-world scenes are rarely pitch dark because light bounces off surfaces:

```rust
let scene = Scene::new()
    .with_ambient(
        [0.05, 0.05, 0.08],  // color: very dark blue tint
        0.5                   // intensity: dim baseline brightness
    );
```

Ambient is added to every pixel after PBR shading. It's not physically based, but it's cheap and effective for stylized looks or ensuring readability in dark areas. The color lets you tint the ambient to match your scene's mood—cool blue for nighttime, warm orange for sunset.

### Sky Configuration

Helio supports two sky modes: a simple solid color background or a physically-based atmospheric scattering simulation.

#### Solid Color Sky

The simplest option is a uniform background color that's rendered wherever no geometry exists:

```rust
let scene = Scene::new()
    .with_sky([0.53, 0.81, 0.92]);  // light blue sky color
```

This is extremely cheap (no shader cost) and sufficient for many stylized games or when the skybox is textured separately.

#### Atmospheric Scattering Sky

For outdoor scenes that need realistic sunsets, day/night cycles, or aerial perspective, Helio provides a full physically-based atmosphere model. The `SkyAtmosphere` struct simulates Rayleigh scattering (blue sky), Mie scattering (haze and sun glow), and an accurate sun disc:

```rust
let atmosphere = SkyAtmosphere::new()
    .with_sun_intensity(22.0)   // brightness of the sun disc
    .with_exposure(4.0)         // tone mapping exposure
    .with_mie_g(0.76);          // forward scattering (sun glow)

let scene = Scene::new()
    .with_sky_atmosphere(atmosphere);
```

The atmosphere integrates with the first directional light in the scene to determine the sun's position and color. As you rotate the light direction, the sky automatically adjusts—sunrise produces orange/red hues near the horizon, midday gives deep blue overhead, and sunset mirrors the sunrise tones. This is implemented through a two-pass rendering system: the `SkyLutPass` precomputes a panoramic lookup texture each frame, then the `SkyPass` samples that texture to fill the background.

##### Atmospheric Parameters

The `SkyAtmosphere` struct exposes physical parameters that control the appearance:

- **rayleigh_scatter**: Per-wavelength scattering coefficients for air molecules. Defaults to Earth atmosphere values `[5.8e-3, 13.5e-3, 33.1e-3]` in km⁻¹ for R/G/B. Higher values make the sky more vivid; lower values desaturate it.
- **rayleigh_h_scale**: Scale height as a fraction of atmosphere thickness (default 0.08). Controls how quickly air density falls off with altitude. Lower = sharper horizon gradient.
- **mie_scatter**: Aerosol scattering strength (default 2.1e-3). Governs the brightness of haze and the glow around the sun.
- **mie_h_scale**: Aerosol scale height (default 0.012). Haze layer thickness.
- **mie_g**: Henyey-Greenstein phase function asymmetry (-1 to 1, default 0.76). Positive values create forward scattering (bright halo around sun); negative values produce backscattering.
- **sun_intensity**: Brightness multiplier for the sun disc (default 22.0).
- **sun_disk_angle**: Angular radius of the sun in radians (default 0.0045, matching the real sun). Larger values make the sun appear bigger in the sky.
- **earth_radius** and **atm_radius**: Planetary geometry in kilometers (defaults 6360 and 6420). Affect horizon curvature—larger radii flatten the horizon.
- **exposure**: Tone mapping exposure for the sky (default 4.0). Increase to brighten the sky, decrease for darker twilight or space-like scenes.

##### Volumetric Clouds

The atmosphere can optionally include a volumetric cloud layer. Clouds are rendered as a 3D density field between `base_height` and `top_height`, animated by wind:

```rust
let clouds = VolumetricClouds::new()
    .with_coverage(0.6)          // 60% cloud coverage
    .with_density(0.8)           // opaque clouds
    .with_layer(800.0, 1800.0)   // altitude range in world units
    .with_wind([1.0, 0.5], 0.1); // wind direction and speed

let atmosphere = SkyAtmosphere::new()
    .with_clouds(clouds);
```

The cloud system uses Perlin-style noise to generate procedural cloud shapes. Higher coverage means more sky is filled with clouds; higher density makes them more opaque. Wind direction is a 2D XZ vector (not normalized), and the magnitude of the wind speed controls animation rate. This creates convincing cloud motion without requiring texture atlases or precomputed data.

### Skylight

When you have a physical atmosphere enabled, you can add a `Skylight` to derive ambient lighting from the sky color. The skylight samples the atmospheric scattering model and uses the result as a dynamic ambient term:

```rust
let skylight = Skylight::new()
    .with_intensity(1.2)              // boost ambient contribution
    .with_tint([1.0, 0.95, 0.9]);    // warm tint

let scene = Scene::new()
    .with_sky_atmosphere(SkyAtmosphere::new())
    .with_skylight(skylight);
```

This replaces the static `with_ambient()` configuration with a dynamic one that changes as the sun moves. At noon, the skylight will be bright blue; at sunset, it shifts to orange/red. The tint is multiplied on top of the computed color, allowing you to artistically adjust the mood without breaking physical correctness.

Note that skylight requires `with_sky_atmosphere()`—it does nothing if you only have a solid color sky.

### Billboards

Billboards are special objects that always face the camera. They're commonly used for particles, foliage imposters, or UI elements in world space. The `BillboardInstance` struct (defined in the billboard feature module) contains position, size, color, and texture coordinates:

```rust
let particle = BillboardInstance {
    position: [0.0, 1.0, 0.0],
    size: [0.5, 0.5],
    color: [1.0, 1.0, 1.0, 0.8],  // white with 80% opacity
    uv_rect: [0.0, 0.0, 1.0, 1.0], // full texture
    rotation: 0.0,                 // no in-plane rotation
};

let scene = Scene::new()
    .add_billboard(particle);
```

Billboards are rendered by the `BillboardPass`, which is registered as a feature. The pass generates two triangles per billboard on the GPU, transforming them to always face the camera while preserving their world-space position.

## Rendering the Scene

Once you've constructed a scene and camera, rendering is a single method call:

```rust
renderer.render_scene(&scene, &camera, target_view, delta_time)?;
```

Internally, the renderer extracts lights from the scene, uploads them to a storage buffer, updates the camera uniform, iterates over scene objects to populate the draw list, sends billboards to the billboard feature, configures the sky pass, and executes the render graph. From the user's perspective, it's beautifully simple: build a scene, point a camera at it, render.

## Example: Complete Scene Setup

Here's a full example demonstrating how to build a rich scene with multiple lights, a physical atmosphere, and objects with custom materials:

```rust
use helio_render_v2::{
    Renderer, RendererConfig, Camera, Scene, SceneLight,
    SkyAtmosphere, Skylight, Material, Mesh, GpuMesh,
};
use glam::{Vec3, Quat};

fn create_scene(renderer: &Renderer) -> Scene {
    // Create meshes
    let ground_mesh = renderer.upload_mesh(
        &Mesh::plane(50.0, 50.0, 10, 10)
    );
    
    let sphere_mesh = renderer.upload_mesh(
        &Mesh::sphere(16, 16)
    );

    // Create materials
    let ground_material = renderer.create_material(
        &Material::new()
            .with_base_color([0.3, 0.3, 0.3, 1.0])
            .with_roughness(0.9)
            .with_metallic(0.0)
    );

    let metal_sphere = renderer.create_material(
        &Material::new()
            .with_base_color([0.9, 0.7, 0.5, 1.0])  // bronze
            .with_metallic(1.0)
            .with_roughness(0.2)
    );

    // Build the scene
    Scene::new()
        // Atmosphere
        .with_sky_atmosphere(
            SkyAtmosphere::new()
                .with_sun_intensity(25.0)
                .with_exposure(4.5)
        )
        .with_skylight(
            Skylight::new()
                .with_intensity(1.0)
        )
        
        // Sun
        .add_light(SceneLight::directional(
            [0.3, -1.0, 0.5],
            [1.0, 0.95, 0.85],
            30.0
        ))
        
        // Fill light
        .add_light(SceneLight::point(
            [-5.0, 3.0, 5.0],
            [0.5, 0.6, 0.8],
            10.0,
            15.0
        ))
        
        // Objects
        .add_object_with_material(ground_mesh, ground_material)
        .add_object_with_material(sphere_mesh, metal_sphere)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let renderer = Renderer::new(RendererConfig::default())?;
    let scene = create_scene(&renderer);
    
    let camera = Camera::perspective(
        Vec3::new(5.0, 3.0, 8.0),
        Vec3::ZERO,
        Vec3::Y,
        60.0_f32.to_radians(),
        16.0 / 9.0,
        0.1,
        1000.0,
        0.0
    );

    // In a real application, this would be in your event loop
    renderer.render_scene(&scene, &camera, &surface_view, 0.016)?;
    
    Ok(())
}
```

This creates a complete outdoor scene with physically-based sky, two lights (sun and fill), a ground plane, and a metallic sphere. The scene is fully data-driven—you could serialize it to JSON, load it from a level editor, or procedurally generate it at runtime.

## Scene vs Immediate Mode

It's worth noting that Helio supports both scene-based and immediate-mode workflows. The scene system is perfect for structured content where you define everything upfront and render it repeatedly (game levels, architectural visualization). But you can also call `renderer.draw_mesh()` directly without a scene:

```rust
renderer.draw_mesh(&mesh);
renderer.draw_mesh_with_material(&other_mesh, custom_material);
renderer.render(&camera, target_view, delta_time)?;
```

This immediate mode is useful for tools, debug visualization, or procedurally generated content where maintaining a scene graph would be overkill. The debug draw system (`renderer.debug_line()`, `renderer.debug_sphere()`, etc.) also operates in immediate mode—you submit shapes each frame and they're rendered as a transient overlay.

Both systems can coexist: use `render_scene()` for your main content and `draw_mesh()` / `debug_*()` for overlays or dynamic annotations.

## Performance Considerations

The scene's vectors are heap-allocated and can grow dynamically, but they're iterated every frame. If you have thousands of lights or objects, consider culling objects outside the view frustum before adding them to the scene, or use an immediate-mode approach where you only submit visible content. The renderer itself doesn't perform culling—it draws everything in the scene.

Lights are uploaded to a GPU storage buffer each frame, so adding hundreds of lights has a memory and upload bandwidth cost. The current architecture is optimized for dozens of lights rather than thousands. For large numbers of small lights (like fireflies or candles), consider using billboards with emissive materials instead of actual light sources.

The atmospheric sky rendering is relatively expensive (two fullscreen passes), so if you're targeting low-end hardware or VR framerates, consider baking a cubemap instead or using the solid color sky mode.

## Summary

- **Camera**: Encapsulates view-projection matrix, world position, and time in a GPU-ready format. Use `Camera::perspective()` for 3D games, `Camera::orthographic()` for tools or 2D.
- **Scene**: The single source of truth for all rendered content. Build it with the fluent builder pattern, populating objects, lights, and atmosphere.
- **Lights**: Three types (directional, point, spot) with factory methods. Directional is cheapest for shadows; point requires cubemaps.
- **Atmosphere**: Physically-based sky with Rayleigh/Mie scattering, optional volumetric clouds, and dynamic ambient via `Skylight`.
- **Rendering**: Call `renderer.render_scene()` to draw the entire scene, or use immediate-mode `draw_mesh()` for dynamic content.

This architecture keeps your rendering code simple and declarative. You describe what exists in the world, where the camera is, and Helio handles the rest.
