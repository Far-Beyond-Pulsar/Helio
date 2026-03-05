# Debug Drawing System

## Overview

The debug drawing system provides runtime visualization of geometric primitives as wireframe overlays. It's designed for diagnosing physics simulations, visualizing AI logic, displaying bounding volumes, annotating world-space data, and any other scenario where you need to draw simple shapes without creating full mesh assets. The system tessellates primitives on the CPU each frame, uploads them to a transient GPU buffer, and renders them with depth testing after the main lighting pass.

Unlike production geometry which uses static GPU meshes and PBR materials, debug shapes are intentionally ephemeral. You submit them each frame with a single function call, they render as colored wireframes with configurable thickness and transparency, and they're automatically cleared before the next frame. This immediate-mode design keeps the API simple and prevents accidental accumulation of thousands of debug objects over time—if you forget to clear debug geometry in other engines, you often end up with a visual mess. Helio avoids this by making debug shapes transient by default.

## Supported Shapes

The system supports five primitive types, each with dedicated submission methods:

### Lines

Lines are the simplest debug primitive: a straight segment from point A to point B. They're perfect for raycasts, graph connections, path visualizations, or any linear relationship:

```rust
renderer.debug_line(
    Vec3::new(0.0, 0.0, 0.0),     // start point
    Vec3::new(5.0, 2.0, -3.0),    // end point
    [1.0, 0.0, 0.0, 1.0],         // red color (RGBA)
    2.0                            // thickness in pixels
);
```

The thickness parameter controls the visual width of the line on screen. Unlike mathematical lines which have zero width, rendered lines are drawn as rectangular tubes aligned along the line direction with a radius proportional to thickness. Values around 1.0 to 3.0 pixels provide good visibility without being overly chunky. Thickness is specified in pixels regardless of distance from the camera—a line 100 units away looks the same width as one right in front of you. This differs from world-space line width, which would shrink with distance and become invisible at far ranges.

The color uses RGBA format where the alpha channel controls transparency. An alpha of 1.0 produces a fully opaque line; 0.5 gives translucent rendering useful for de-emphasizing less important data; 0.0 makes the line invisible (though there's no reason to submit invisible geometry). Typical usage is opaque primary colors (red, green, blue, yellow) for maximum contrast against scene geometry.

### Spheres

Spheres are rendered as three perpendicular wireframe circles forming a cage around the center point. This provides good visual coverage from any viewing angle without filling the entire volume:

```rust
renderer.debug_sphere(
    Vec3::new(10.0, 3.0, 0.0),   // center position
    2.5,                          // radius
    [0.0, 1.0, 1.0, 0.5],        // cyan with 50% opacity
    1.5                           // line thickness
);
```

Spheres are commonly used for light range visualization, trigger volumes, explosion radii, or any spherical bounding shape. The wireframe representation makes them distinguishable from actual sphere meshes in the scene—debug spheres are hollow cages, not solid geometry. The three circles are aligned with the world-space XY, XZ, and YZ planes, giving you orthogonal slices through the volume.

The radius can be any positive value. Very small spheres (radius <0.001) are clamped to avoid degenerate geometry. Very large spheres (hundreds of units) work fine but may have jagged circles if you're close to them—the tessellation uses a fixed 32-segment resolution which is a compromise between smoothness and performance.

### Boxes

Boxes are axis-aligned or rotated rectangular volumes, rendered as twelve wireframe edges connecting eight corners:

```rust
renderer.debug_box(
    Vec3::new(-5.0, 1.0, 0.0),     // center position
    Vec3::new(1.0, 0.5, 2.0),      // half-extents (width, height, depth)
    Quat::from_rotation_y(0.5),    // rotation quaternion
    [1.0, 1.0, 0.0, 1.0],          // yellow color
    2.0                             // line thickness
);
```

The half-extents define the box dimensions before rotation. A half-extent of (1.0, 0.5, 2.0) means the box is 2 units wide, 1 unit tall, and 4 units deep. This matches the convention used by physics engines and bounding box representations—half-extents are easier to work with mathematically than full dimensions.

The rotation quaternion is applied around the center point. For axis-aligned boxes, use `Quat::IDENTITY`. For oriented boxes, construct quaternions with `from_rotation_x()`, `from_rotation_y()`, `from_rotation_z()`, or `from_axis_angle()` from the glam crate. This flexibility makes boxes suitable for visualizing oriented bounding boxes (OBBs) from physics simulations or character controller capsules that have been transformed.

Boxes are the workhorse of debug visualization. Use them for trigger volumes, room boundaries, AI navigation grid cells, physics collision shapes, or any rectangular region. The wireframe edges make overlapping boxes readable even when they intersect.

### Cones

Cones are directional shapes with an apex point, a base circle, and straight edges connecting them. They're perfect for visualizing spotlights, projectile launch trajectories, vision cones, or any tapered volume:

```rust
renderer.debug_cone(
    Vec3::new(3.0, 5.0, 0.0),      // apex (tip of the cone)
    Vec3::new(0.0, -1.0, 0.0),     // direction (should be normalized)
    3.0,                            // height (distance from apex to base)
    1.0,                            // base radius
    [1.0, 0.5, 0.0, 0.8],          // orange with slight transparency
    1.5                             // line thickness
);
```

The direction vector points from the apex toward the base and should be normalized for correct results (the implementation normalizes it internally if you forget, but it's good practice to pass unit vectors). The height is the distance along the direction axis from apex to base center. The base radius controls the cone's spread—a larger radius produces a wider cone, while a smaller radius creates a narrow, needle-like shape.

Internally, the cone is tessellated as a wireframe base circle plus eighteen radial edges from the apex to points on the base. This gives reasonable coverage without excessive geometry. If you need a smooth cone appearance, you'll need to implement a filled cone with proper shading—debug cones are intentionally simple wireframes.

A common use case is visualizing spotlight bounds. Match the cone's apex to the light position, direction to the light direction, height to the light range, and radius derived from the outer cone angle. This lets you see exactly what volume the light affects, which is invaluable when debugging shadow issues or positioning lights.

### Capsules

Capsules are cylinder-with-hemispheres shapes defined by two end caps (start and end points) and a radius. They're the go-to shape for character collision volumes because they handle slopes and stairs smoothly:

```rust
renderer.debug_capsule(
    Vec3::new(0.0, 0.5, 0.0),     // start (bottom center, e.g., feet position)
    Vec3::new(0.0, 1.8, 0.0),     // end (top center, e.g., head position)
    0.3,                           // radius (body thickness)
    [0.0, 1.0, 0.0, 0.7],         // green with transparency
    1.0                            // line thickness
);
```

The start and end points define the central axis of the capsule. The shape is a cylinder of that height with hemispherical caps on each end. The radius applies to both the cylinder and the caps, so the total height of the capsule is `||end - start|| + 2 * radius` (the axis length plus two cap radii).

Capsules are rendered as two end circles (one at each cap) plus four longitudinal lines connecting them. This sparse wireframe keeps the visual noise low while clearly indicating the volume bounds. Unlike spheres which are fully symmetric, capsules have a clear orientation defined by the start-to-end axis, making them easier to interpret when visualizing upright characters or moving objects.

A typical character controller uses a capsule with the start point at ground level, the end point at shoulder or head height, and a radius equal to the character's body thickness. Visualizing this lets you immediately see if the collision shape matches the visual mesh and whether the character can fit through doorways or under obstacles.

## Rendering Behavior

Debug shapes are rendered after the deferred lighting pass in a dedicated `DebugDrawPass`. This means they overlay on top of all scene geometry, materials, and lighting effects. The pass uses depth testing, so shapes are correctly occluded by closer geometry—if a debug sphere is behind a wall, you'll only see the portion that's visible through any gaps or openings.

### Transparency and Blending

Colors with alpha <1.0 are rendered with alpha blending enabled. The blend mode is standard transparency (source alpha over destination), so overlapping transparent shapes composite correctly. If you draw multiple debug spheres in the same location with 50% opacity, each successive sphere darkens the result appropriately rather than overwriting previous shapes.

Fully opaque shapes (alpha = 1.0) still write to the depth buffer, so they can occlude other debug geometry. If you want debug shapes to always render on top regardless of depth, you'd need to modify the debug pass to disable depth testing, but the default behavior (depth-tested overlay) is usually what you want.

### Thickness Implementation

The thickness parameter doesn't just scale a 2D screen-space line width (which would break at grazing angles). Instead, debug lines are tessellated as 3D cylindrical tubes with actual geometry. A thickness of 2.0 creates a tube with a radius of 1.0 pixel (thickness is diameter, not radius). These tubes are built from multiple quads arranged radially around the line axis.

This implementation has trade-offs. The advantage is correct perspective rendering—lines maintain consistent visual thickness regardless of viewing angle or distance. The disadvantage is higher geometry cost compared to line primitives or screen-space lines. For a few hundred debug shapes, this is negligible. For thousands of shapes, the CPU tessellation and GPU vertex processing can become a bottleneck.

If you're hitting performance issues with debug drawing, reduce the number of shapes or simplify complex shapes (use boxes instead of spheres, which have fewer edges). The system isn't designed for rendering entire scenes as debug geometry—it's meant for dozens to hundreds of diagnostic primitives, not architectural wireframes.

## Frame Lifecycle

Debug geometry is transient and cleared automatically every frame after rendering. The workflow is:

1. **Before frame**: Debug shape list is empty.
2. **Submission**: Your application calls `debug_line()`, `debug_sphere()`, etc. to queue shapes.
3. **Batch build**: At the start of `render()` or `render_scene()`, the renderer calls `build_batch()` to tessellate all queued shapes into a single vertex/index buffer pair.
4. **Rendering**: The debug draw pass binds the batch and renders it as a single draw call.
5. **Clear**: After rendering, the shape list is automatically cleared.
6. **Next frame**: Back to step 1 with an empty list.

This means you need to re-submit debug shapes every frame if you want them to persist. For persistent visualization, call the debug methods inside your main loop. For one-shot annotations (e.g., "show this raycast hit point"), submit them only when the event occurs and they'll disappear on the next frame.

### Manual Clearing

If you need to clear debug geometry mid-frame (for example, to visualize different algorithm stages without accumulation), call `clear_debug_shapes()`:

```rust
// Visualize stage 1
renderer.debug_sphere(pos_a, 1.0, [1.0, 0.0, 0.0, 1.0], 1.0);
// Submit intermediate result for debugging, then clear
renderer.render(&camera, &view, delta_time)?;

renderer.clear_debug_shapes();

// Visualize stage 2
renderer.debug_sphere(pos_b, 1.0, [0.0, 1.0, 0.0, 1.0], 1.0);
renderer.render(&camera, &view, delta_time)?;
```

This is uncommon—most applications just submit shapes before each render call and rely on automatic clearing.

## Tessellation Details

Understanding how shapes are converted to triangles helps optimize debug geometry or diagnose unexpected visual artifacts.

### Line Tubes

Lines are rendered as cylindrical tubes with eight radial sides. Each tube is a series of rectangular quads wrapped around the line axis. The local coordinate frame is computed using `basis_from_dir()`, which finds two perpendicular vectors to the line direction via cross products. These form the U and V axes for circle generation.

Eight sides is a compromise—fewer sides (e.g., four) create obviously square tubes, while more sides (e.g., sixteen) are smoother but double the triangle count. For thickness values of 1-3 pixels, eight sides are sufficient for a round appearance.

### Circles

Circles (used for sphere wireframes, cone bases, and capsule caps) are approximated as 32-segment polygons. Each segment is connected with a tube, forming a closed ring. The segment count is hardcoded to balance smoothness versus geometry complexity. At typical viewing distances and radii, 32 segments produce smooth circles without visible faceting.

If you zoom extremely close to a large debug sphere, you may notice the individual segments. This is acceptable for debug geometry—we prioritize performance and simplicity over perfect visual quality.

### Sphere Wireframes

Spheres are three orthogonal circles aligned with the XY, XZ, and YZ planes. Each circle is 32 segments connected by tubes, so a sphere is roughly 3 × 32 × 8 × 2 = 1,536 triangles (three circles, thirty-two segments each, eight sides per tube, two triangles per quad). This is significantly more geometry than a line, which is why drawing hundreds of spheres can impact performance.

If you need cheaper sphere visualization, consider using a single circle (essentially a flat disc from one angle) or a bounding box approximation.

### Box Edges

Boxes have twelve edges (four bottom, four top, four vertical). Each edge is a tube, so the triangle count is 12 × 8 × 2 = 192 triangles per box. This is much cheaper than spheres, making boxes a good choice when performance matters.

### Cone Wireframes

Cones consist of one base circle (32 segments) plus eighteen radial spokes from the apex to the base perimeter. The base circle is 32 × 8 × 2 = 512 triangles. Each spoke is another eight-sided tube, adding 18 × 8 × 2 = 288 triangles. Total: approximately 800 triangles per cone.

### Capsule Wireframes

Capsules are two end circles plus four longitudinal edges. Each circle is 24 segments (slightly fewer than spheres for performance), and each edge is an eight-sided tube. Triangle count: 2 × 24 × 8 × 2 + 4 × 8 × 2 = 832 triangles. Capsules are comparable in cost to cones.

## Usage Patterns

### Visualizing Physics

Debug drawing is invaluable for physics debugging. Render collision shapes, raycasts, and forces to understand simulation behavior:

```rust
// Character collision capsule
renderer.debug_capsule(
    character_pos,
    character_pos + Vec3::new(0.0, 1.8, 0.0),
    0.3,
    [0.0, 1.0, 0.0, 0.5],
    1.0
);

// Ground raycast
if let Some(hit) = physics.raycast(character_pos, Vec3::NEG_Y, 10.0) {
    renderer.debug_line(
        character_pos,
        hit.point,
        [1.0, 1.0, 0.0, 1.0],
        2.0
    );
    renderer.debug_sphere(hit.point, 0.1, [1.0, 0.0, 0.0, 1.0], 1.5);
}
```

This pattern is typical: use capsules for character shapes, lines for raycasts, spheres for hit points, and boxes for static collision geometry.

### Visualizing AI

AI systems often work with abstract spatial relationships (navigation graphs, threat zones, patrol paths). Debug drawing makes the invisible visible:

```rust
// Patrol path
for waypoint in patrol_path.windows(2) {
    renderer.debug_line(waypoint[0], waypoint[1], [0.0, 0.5, 1.0, 0.7], 1.5);
}

// Threat cone (enemy vision)
renderer.debug_cone(
    enemy_pos + Vec3::new(0.0, 1.5, 0.0),  // eye height
    enemy_forward,
    10.0,                                   // vision range
    5.0,                                    // cone spread
    [1.0, 0.0, 0.0, 0.3],                  // red, very transparent
    1.0
);
```

Transparent shapes are perfect for "soft" information like vision cones or probability fields where you want to indicate presence without completely obscuring the scene.

### Visualizing Lighting

Debugging light placement and falloff is much easier when you can see the bounding volumes:

```rust
for light in &scene.lights {
    match light.light_type {
        LightType::Point => {
            renderer.debug_sphere(
                Vec3::from(light.position),
                light.range,
                [1.0, 1.0, 0.0, 0.2],  // yellow, subtle
                1.0
            );
        }
        LightType::Spot { inner_angle, outer_angle } => {
            let height = light.range;
            let radius = height * outer_angle.tan();
            renderer.debug_cone(
                Vec3::from(light.position),
                Vec3::from(light.direction),
                height,
                radius,
                [1.0, 0.8, 0.4, 0.2],
                1.0
            );
        }
        _ => {}
    }
}
```

This visualizes point light spheres and spotlight cones directly overlaid on the scene. The subtle transparency lets you see both the debug shapes and the actual lighting effect simultaneously.

### Annotating World Data

For level editors, profiling tools, or data visualization, debug drawing can annotate arbitrary world-space data:

```rust
// Show spawn points
for (i, spawn) in spawn_points.iter().enumerate() {
    renderer.debug_sphere(spawn.position, 0.5, [0.3, 0.3, 1.0, 1.0], 1.5);
    
    // Forward indicator
    let fwd_end = spawn.position + spawn.forward * 2.0;
    renderer.debug_line(spawn.position, fwd_end, [0.5, 0.5, 1.0, 1.0], 2.5);
}
```

This creates a visual representation of editor data that would otherwise be invisible at runtime.

## Performance Considerations

Debug drawing is intentionally CPU-driven because it's transient and the geometry is simple. The tessellation happens once per frame, and the resulting vertex/index buffers are uploaded to a staging buffer that's recycled on the next frame. This avoids dynamic GPU allocation overhead while keeping the implementation straightforward.

The performance cost breaks down as:

1. **CPU Tessellation**: Dominant cost for complex shapes like spheres. Each sphere is ~1,500 triangles of computation.
2. **GPU Upload**: Relatively cheap—uploading a few thousand vertices is negligible compared to texture streaming.
3. **GPU Rendering**: Single draw call with depth testing and alpha blending. Negligible unless you have tens of thousands of triangles.

Practical guidance:

- **Dozens of shapes**: Zero performance impact.
- **Hundreds of shapes**: Measurable but usually <0.1ms per frame.
- **Thousands of shapes**: Potentially visible cost (1-2ms CPU, 0.5ms GPU). Consider simplifying or culling.

If you're hitting limits, use simpler shapes (boxes instead of spheres), reduce thickness (fewer tube sides), or implement frustum culling to skip off-screen geometry.

## Shader Implementation

The debug draw shader is minimal. The vertex shader transforms positions by the camera view-projection matrix and passes through vertex colors:

```wgsl
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}
```

The fragment shader outputs the interpolated color directly—no lighting, no textures:

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
```

This simplicity is intentional. Debug geometry doesn't need PBR shading or complex materials—it just needs to be visible and correctly positioned.

## Comparison to Immediate-Mode GUIs

Debug drawing is similar to immediate-mode GUI libraries like Dear ImGui in philosophy: you describe what you want each frame without managing persistent state. The differences:

- **3D vs 2D**: Debug shapes exist in world space with depth testing; ImGui draws in screen space.
- **Geometry vs Widgets**: Debug drawing renders geometric primitives; ImGui renders UI panels and controls.
- **Rendering Integration**: Debug drawing plugs into the main render graph; ImGui typically runs in a separate pass.

Both systems excel at rapid iteration and simple APIs. Use debug drawing for spatial visualization and ImGui for numerical readouts, settings panels, and user interaction.

## Advanced Techniques

### Multi-Color Shapes

Each shape uses a single color, but you can simulate gradients or multi-color objects by submitting multiple shapes:

```rust
// Traffic light: three stacked spheres
renderer.debug_sphere(pos + Vec3::new(0.0, 2.0, 0.0), 0.3, [1.0, 0.0, 0.0, 1.0], 1.0);  // red
renderer.debug_sphere(pos + Vec3::new(0.0, 1.0, 0.0), 0.3, [1.0, 1.0, 0.0, 1.0], 1.0);  // yellow
renderer.debug_sphere(pos, 0.3, [0.0, 1.0, 0.0, 1.0], 1.0);                              // green
```

### Animated Shapes

Since shapes are re-submitted each frame, animating them is trivial:

```rust
let t = (time * 2.0).sin() * 0.5 + 0.5;  // oscillate 0..1
let radius = 1.0 + t * 2.0;               // pulse radius
renderer.debug_sphere(center, radius, [1.0, 0.0, 0.0, 1.0], 1.5);
```

This creates a pulsing sphere. Use time-based math to create blinking colors, moving indicators, or orbiting markers.

### Conditional Visibility

Toggle debug geometry based on application state:

```rust
if show_physics_debug {
    for shape in &physics_shapes {
        renderer.debug_box(shape.center, shape.extents, shape.rotation, [0.0, 1.0, 0.0, 0.5], 1.0);
    }
}

if show_ai_debug {
    for agent in &ai_agents {
        renderer.debug_cone(agent.pos, agent.facing, 5.0, 2.0, [1.0, 0.5, 0.0, 0.3], 1.0);
    }
}
```

This lets you enable/disable debug layers independently, which is essential for complex projects where multiple systems produce debug visualization.

## Summary

Helio's debug drawing system provides transient wireframe visualization of lines, spheres, boxes, cones, and capsules. Key points:

- **Immediate Mode**: Submit shapes each frame; they're automatically cleared after rendering.
- **Shapes**: Five primitives with dedicated methods (`debug_line()`, `debug_sphere()`, etc.).
- **Rendering**: Depth-tested overlay after lighting, supporting transparency via RGBA colors.
- **Thickness**: Pixel-based line width converted to 3D tubes for correct perspective.
- **Performance**: CPU tessellation followed by single-draw-call GPU rendering; suitable for hundreds of shapes.
- **Use Cases**: Physics debugging, AI visualization, lighting bounds, editor annotations.

The system prioritizes simplicity and ergonomics over raw performance or visual quality. It's a diagnostic tool, not a production rendering system. Use it liberally during development, then remove debug calls or gate them behind compile-time flags for release builds.
