# Materials and Meshes

## Overview

Materials and meshes are the fundamental building blocks of rendered geometry in Helio. A mesh defines the shape—vertex positions, normals, texture coordinates, and connectivity—while a material defines the appearance—colors, reflectivity, roughness, and emissive glow. Together, they form the complete description of a drawn object. The renderer combines them during the G-buffer pass, writing surface properties into textures that the deferred lighting pass later shades with physically-based rendering.

Helio uses a vertex format optimized for modern GPUs, packing normals and tangents into compressed representations to save memory bandwidth. Materials follow the metallic-roughness PBR workflow popularized by tools like Substance and glTF, making them compatible with industry-standard asset pipelines. The system separates CPU-side descriptions (Material, vertex arrays) from GPU-resident resources (GpuMaterial, GpuMesh), giving you control over when and how data is uploaded.

## Mesh System

### PackedVertex Format

Every vertex in Helio uses the `PackedVertex` struct, which packs all necessary attributes into exactly 32 bytes. This matches the G-buffer shader's input layout and provides cache-friendly memory access patterns:

```rust
#[repr(C)]
pub struct PackedVertex {
    pub position: [f32; 3],      // 12 bytes
    pub bitangent_sign: f32,     //  4 bytes (handedness for tangent-space)
    pub tex_coords: [f32; 2],    //  8 bytes
    pub normal: u32,             //  4 bytes (packed SNORM8x4)
    pub tangent: u32,            //  4 bytes (packed SNORM8x4)
}                                // Total: 32 bytes
```

The position and texture coordinates are stored as full floats for maximum precision. The normal and tangent use compressed 8-bit signed normalized encoding (SNORM8x4), which packs four float values into a single u32. This reduces bandwidth by 75% compared to uncompressed vectors with minimal quality loss—normals are unit vectors, so the quantization error is barely perceptible.

The `bitangent_sign` field stores the handedness of the tangent-space basis (+1 or -1). Instead of storing the full bitangent vector, we reconstruct it in the shader via cross product: `bitangent = cross(normal, tangent) * bitangent_sign`. This saves 12 bytes per vertex without losing information.

#### Creating Vertices

The simplest constructor auto-computes a tangent perpendicular to the normal:

```rust
let v = PackedVertex::new(
    [0.0, 1.0, 0.0],      // position
    [0.0, 1.0, 0.0],      // normal (straight up)
    [0.5, 0.5]            // UV at center
);
```

For correct normal mapping, you should provide explicit tangents aligned with your UV layout:

```rust
let v = PackedVertex::new_with_tangent(
    [0.0, 1.0, 0.0],      // position
    [0.0, 1.0, 0.0],      // normal
    [0.5, 0.5],           // UV
    [1.0, 0.0, 0.0]       // tangent (UV u-direction)
);
```

The tangent should point in the direction of increasing U (horizontal texture axis). For a quad with UVs at (0,0), (1,0), (1,1), (0,1), the tangent is the direction from the first corner to the second, which is the +X or +U axis. Getting tangents wrong causes normal maps to look flipped or rotated—you'll see bumps where there should be dents.

The `bitangent_sign` is automatically set to `1.0` in both constructors. If you're importing meshes from DCC tools like Blender or Maya, check whether they use left-handed or right-handed tangent spaces. Helio expects right-handed (OpenGL/glTF convention).

### GpuMesh

The `GpuMesh` struct represents geometry that's been uploaded to video memory. It owns Arc-counted vertex and index buffers, making it cheap to clone and share across multiple draw calls:

```rust
pub struct GpuMesh {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
    pub vertex_count: u32,
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}
```

The bounding sphere (center and radius) is computed automaticallyduring construction and used for shadow map culling. When the renderer builds shadow maps for a light, it skips meshes whose bounds are entirely outside the light's frustum. This is an essential optimization for scenes with thousands of objects.

#### Uploading Meshes

Create a `GpuMesh` by providing a wgpu device, vertex slice, and index slice:

```rust
use helio_render_v2::mesh::{PackedVertex, GpuMesh};

let vertices = vec![
    PackedVertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0]),
    PackedVertex::new([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
    PackedVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
];

let indices = vec![0u32, 1, 2];

let mesh = GpuMesh::new(&renderer.device, &vertices, &indices);
```

The constructor creates wgpu buffers with `create_buffer_init()`, which performs a blocking upload. For large meshes (hundreds of thousands of vertices), this can stall if done mid-frame. Best practice is to upload all assets during a loading screen or in a background thread, then keep `Gpu Mesh` handles alive for the duration of the level.

The vertex buffer usage is `VERTEX` plus optionally `BLAS_INPUT` if hardware ray tracing is enabled on the device. The latter allows meshes to participate in ray-traced acceleration structures, though Helio's current renderer doesn't expose ray tracing APIs publicly.

Indices must be u32 (not u16). This supports meshes with more than 65,535 vertices without index overflow. For small meshes, this wastes bandwidth, but it simplifies the pipeline and avoids edge cases where large models silently corrupt due to index wrapping.

#### Bounding Sphere Computation

The bounding sphere is computed as the centroid of all vertex positions plus the maximum distance from that centroid:

```rust
let centroid = vertices.iter().map(|v| v.position).sum() / vertex_count;
let radius = vertices.iter()
    .map(|v| distance(v.position, centroid))
    .max();
```

This is a conservative approximation—the true minimal bounding sphere might be smaller and offset from the centroid. However, computing the exact minimal sphere is expensive (requires iterative optimization), and the extra padding from using the centroid is negligible in practice. Shadow culling doesn't need pixel-perfect bounds; it just needs to avoid grossly incorrect rejections.

### Procedural Primitives

Helio provides factory methods for common geometric shapes. These are useful for rapid prototyping, placeholder geometry, or procedurally generated content.

#### Cube

A cube is six square faces arranged into a closed box. The `cube()` method takes a center position and half-extent (cube extends ±half_size on each axis):

```rust
let cube = GpuMesh::cube(
    &renderer.device,
    [0.0, 0.0, 0.0],  // center at origin
    0.5                // half-extent (full width = 1.0)
);
```

Each face has correct  normals pointing outward, UVs mapped to (0,0)-(1,1) per face, and tangents aligned with the UV u-axis. The cube uses 24 vertices (four per face, duplicated at shared corners to allow distinct normals per face) and 36 indices (three per triangle, two triangles per face, six faces).

The winding order is counter-clockwise from the outside, which is the standard convention for right-handed coordinate systems. This ensures correct backface culling—faces pointing away from the camera are skipped, improving performance.

#### Rectangular Box

The `rect3d()` method creates a box with independent half-extents per axis, allowing thin slabs, beams, or any non-uniform rectangular volume:

```rust
let beam = GpuMesh::rect3d(
    &renderer.device,
    [0.0, 0.5, 0.0],        // center
    [2.0, 0.1, 0.5]         // half-extents (wide thin beam)
);
```

This is functionally identical to `cube()` but with axis-aligned scaling. It's commonly used for floors (large XZ, small Y), walls (large XY, small Z), or architectural elements.

#### Plane

A plane is a single quad aligned with the XZ axes (horizontal ground plane). It's centered at the given position and extends ±half_extent on both X and Z:

```rust
let ground = GpuMesh::plane(
    &renderer.device,
    [0.0, 0.0, 0.0],  // center at origin
    10.0               // half-extent (20×20 total)
);
```

The normal points up (+Y), the tangent points right (+X), and UVs map (0,0) to (-X,+Z) corner and (1,1) to (+X,-Z) corner. This matches the typical ground texture orientation where U increases rightward and V increases forward.

Planes are useful for floors, water surfaces, or any flat horizontal geometry. If you need a vertical wall or ceiling, create a plane and manually transform it in your scene (rotate via quaternion or apply a model matrix).

### Custom Mesh Loading

For complex meshes (characters, props, terrain), you'll typically load them from asset files (glTF, OBJ, FBX). Helio doesn't include asset loading utilities—that's the responsibility of your asset pipeline. The integration pattern is:

1. Load mesh data from file using a crate like `gltf`, `obj-rs`, or `russimp`.
2. Convert to `PackedVertex` format (extract positions, UVs, normals, and compute or load tangents).
3. Upload to GPU with `GpuMesh::new()`.

Here's a sketch for glTF loading:

```rust
let (gltf, buffers, _) = gltf::import("model.glb")?;

for mesh in gltf.meshes() {
    for primitive in mesh.primitives() {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        
        let positions = reader.read_positions().unwrap().collect::<Vec<_>>();
        let normals = reader.read_normals().unwrap().collect::<Vec<_>>();
        let uvs = reader.read_tex_coords(0).unwrap().into_f32().collect::<Vec<_>>();
        let tangents = reader.read_tangents().unwrap().collect::<Vec<_>>();
        let indices = reader.read_indices().unwrap().into_u32().collect::<Vec<_>>();

        let vertices = positions.iter().enumerate().map(|(i, &pos)| {
            PackedVertex::new_with_tangent(
                pos,
                normals[i],
                uvs[i],
                [tangents[i][0], tangents[i][1], tangents[i][2]]
            )
        }).collect::<Vec<_>>();

        let gpu_mesh = GpuMesh::new(&renderer.device, &vertices, &indices);
    }
}
```

This is simplified—real asset loaders handle skinning, morph targets, multi-material meshes, and other complexities. Consider using an existing integration like `bevy_gltf` or writing your own parser.

### Tangent Computation

If your source data lacks tangents, you can compute them via the MikkTSpace algorithm (the industry standard) or use a simpler per-triangle approach:

```rust
fn compute_tangent(p0: [f32; 3], p1: [f32; 3], uv0: [f32; 2], uv1: [f32; 2]) -> [f32; 3] {
    let edge = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let duv = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
    
    let tangent = if duv[0].abs() > 1e-6 {
        let scale = 1.0 / duv[0];
        [edge[0] * scale, edge[1] * scale, edge[2] * scale]
    } else {
        [1.0, 0.0, 0.0]  // fallback
    };
    
    normalize(tangent)
}
```

This computes the tangent as the direction of increasing U across a triangle edge. For smooth shading across multiple triangles, you'd average tangents at shared vertices and re-normalize. For proper normal mapping quality, use a library like `mikktspace` which handles edge cases correctly.

## Material System

### Material Structure

The `Material` struct is a CPU-side description of PBR properties. It follows the metallic-roughness workflow where metals have high metallic values (1.0) and zero roughness means mirror-like, while dielectrics have low metallic (0.0) and roughness controls matte versus glossy appearance:

```rust
pub struct Material {
    pub base_color: [f32; 4],          // Albedo tint (RGBA, linear)
    pub metallic: f32,                  // 0.0 = dielectric, 1.0 = metal
    pub roughness: f32,                 // 0.0 = smooth, 1.0 = rough
    pub ao: f32,                        // Ambient occlusion factor
    pub emissive_color: [f32; 3],      // Glow color (RGB, linear)
    pub emissive_factor: f32,           // Glow strength
    
    pub base_color_texture: Option<TextureData>,  // Albedo map (sRGB)
    pub normal_map: Option<TextureData>,           // Tangent-space normals
    pub orm_texture: Option<TextureData>,          // Packed ORM (linear)
    pub emissive_texture: Option<TextureData>,     // Emissive map (sRGB)
}
```

All texture fields are optional. If you omit a texture, the renderer uses a 1×1 fallback (white for albedo, flat up-normal for normal maps, default ORM values, black for emissive). This means textured and texture-less materials use the same shader and bind group layout, simplifying the pipeline.

### Creating Materials

Use the builder pattern to construct materials:

```rust
use helio_render_v2::Material;

let gold = Material::new()
    .with_base_color([1.0, 0.85, 0.4, 1.0])  // golden yellow
    .with_metallic(1.0)                       // fully metallic
    .with_roughness(0.1);                     // polished (almost mirror)

let rough_stone = Material::new()
    .with_base_color([0.6, 0.6, 0.6, 1.0])  // gray
    .with_metallic(0.0)                      // non-metal
    .with_roughness(0.9);                    // very rough (matte)

let glowing_ember = Material::new()
    .with_base_color([0.3, 0.1, 0.0, 1.0])  // dark red
    .with_emissive([1.0, 0.3, 0.0], 5.0);   // bright orange glow
```

The base color alpha channel controls transparency, though Helio's current deferred renderer doesn't support order-independent transparency—alpha is written to the G-buffer but not blended. For proper transparency, you'd need a forward rendering pass or OIT techniques.

### PBR Parameters Explained

#### Base Color

The base color (albedo) is the intrinsic color of the material before lighting. For metals, this represents the specular reflection tint (gold is yellowish, copper is orangeish, iron is grayish). For dielectrics, it's the diffuse color you see under white light. Base colors should be in linear color space, not gamma-corrected—the renderer applies gamma correction during tone mapping.

Physically correct albedo values for dielectrics are typically in the 0.2-0.8 range. Values below 0.05 produce unrealistically dark materials (charcoal is ~0.04). Values above 0.9 are rare in nature (fresh snow is ~0.9, most materials are darker).

#### Metallic

Metallic encodes whether a surface is a conductor (1.0) or insulator (0.0). At metallic=1.0, the base color defines the specular reflection tint and diffuse reflection is zero. At metallic=0.0, specular is colorless (Fresnel F0 ~0.04 for most dielectrics) and diffuse uses the base color. Intermediate values blend between the two, which isn't physically meaningful but can be useful for artistic effects like tarnished metal (metallic=0.7).

Real-world materials are almost never exactly 0.0 or 1.0—oxidized metals, painted surfaces, or contaminated dielectrics have intermediate values. But for most content, stick to the extremes: 0.0 for wood/plastic/stone, 1.0 for iron/gold/chrome.

#### Roughness

Roughness controls microfacet surface detail that scatters reflected light. At roughness=0.0, the surface is a perfect mirror with sharp specular highlights. At roughness=1.0, the surface is completely diffuse with no visible highlights. Intermediate values produce glossy to matte appearances.

Roughness is perceptually linear when you square it in the shader (which Helio does via the GGX BRDF). This is why "roughness" is preferred over "glossiness" or "smoothness"—artists can intuitively adjust it without worrying about non-linear falloffs.

Few real materials have roughness below 0.05 (polished chrome, wet surfaces) or above 0.95 (chalk, heavily weathered stone). Most everyday materials sit in the 0.3-0.7 range.

#### Ambient Occlusion

AO is a baked shadowing term that darkens crevices, corners, and cavities. It's typically generated by baking ambient light accessibility from a 3D model in a DCC tool. An AO value of 1.0 means fully lit (no occlusion), while 0.0 means fully shadowed (blocked from all directions).

AO is a hack—it's not physically based. Real ambient occlusion would depend on the lighting environment, but baking it into a texture is fast and looks good for static geometry. Dynamic objects shouldn't use baked AO unless you implement SSAO (screen-space ambient occlusion) or a voxel-based GI technique.

#### Emissive

Emissive materials glow with self-illumination. The emissive color is the RGB tint of the glow, and the emissive factor is a brightness multiplier. The final emissive contribution is:

```
emissive_output = emissive_color * emissive_factor * emissive_texture_sample
```

Emissive objects shine in the dark without needing lights. They don't cast light on other objects in the deferred pipeline (they're not actual light sources), but they can be bright enough to bloom if you add a post-processing pass. Typical use cases: neon signs, lava, computer screens, fireflies, tron-style glowing edges.

High emissive factors (>10) can blow out to pure white after tone mapping, creating an HDR glow effect. Low factors (0.1-1.0) produce subtle illumination.

### Texture Maps

Helio supports four texture types that augment the scalar material parameters.

#### Base Color Texture

An sRGB RGBA texture that multiplies the base_color tint. This is your standard diffuse or albedo map. The RGB channels define color variation across the surface; the alpha channel controls transparency (though deferred rendering doesn't handle alpha blending correctly—use it for alpha-test cutouts like foliage).

Loading a base color texture:

```rust
use image;

let img = image::open("albedo.png")?.to_rgba8();
let width = img.width();
let height = img.height();
let data = img.into_raw();

let material = Material::new()
    .with_base_color_texture(TextureData::new(data, width, height));
```

The texture is automatically uploaded to the GPU when you call `renderer.create_material()`. It uses `Rgba8UnormSrgb` format, which performs sRGB-to-linear conversion on read. This is correct for albedo maps, which are typically authored in sRGB color space.

#### Normal Map

A tangent-space normal map stored as an RGBA texture in linear color space. The RGB channels encode the normal direction (mapped from -1..1 to 0..1 via `n * 0.5 + 0.5`). The alpha channel is unused. Helio expects DirectX/OpenGL-style normal maps where Y-up is positive (green channel = surface up). If your normal maps use Y-down (some tools produce this), you'll need to flip the green channel.

```rust
let material = Material::new()
    .with_base_color_texture(albedo_data)
    .with_normal_map(normal_data);
```

Normal maps add high-frequency surface detail without additional geometry. They perturb the shading normal, creating the illusion of bumps, dents, or grooves. Without a normal map, surfaces look flat even if they have color variation.

Technical note: normal maps are in tangent space, meaning the normal vectors are relative to the surface's UV frame (tangent, bitangent, normal). The G-buffer shader transforms them to view space using the TBN matrix built from vertex normal and tangent. This is why correct tangents are critical—wrong tangents cause normal maps to point the wrong direction.

#### ORM Texture

An "ORM" texture packs three grayscale maps into RGB channels: occlusion (R), roughness (G), metallic (B). This reduces texture count and memory usage compared to storing each map separately:

```rust
let material = Material::new()
    .with_orm_texture(orm_data)
    .with_metallic(1.0)      // ORM metallic channel is multiplied by this
    .with_roughness(1.0)     // ORM roughness channel is multiplied by this
    .with_ao(1.0);           // ORM AO channel is multiplied by this
```

The scalar material parameters act as multipliers for the texture samples:

```
final_ao = ao_factor * orm_texture.r
final_roughness = roughness_factor * orm_texture.g
final_metallic = metallic_factor * orm_texture.b
```

If you don't provide an ORM texture, the renderer uses a 1×1 white texture, so the scalar factors are used directly. This workflow is glTF-compatible—glTF 2.0 specifies the same ORM channel packing.

The ORM texture uses `Rgba8Unorm` format (linear, not sRGB). This is correct for physically-measured quantities like roughness and metallic.

#### Emissive Texture

An sRGB RGBA texture that defines spatially-varying emission. The RGB channels are multiplied by `emissive_color * emissive_factor`:

```
final_emissive = emissive_color * emissive_factor * emissive_texture.rgb
```

Use emissive textures for complex glowing patterns—circuit boards, animated UI elements, or magic runes. Pair them with low base color and high emissive factor to create materials that glow brightly without being reflective (like a computer monitor).

### Uploading Materials to the GPU

Once you've configured a `Material`, upload it with `renderer.create_material()`:

```rust
let material = Material::new()
    .with_base_color([0.8, 0.2, 0.1, 1.0]);

let gpu_material = renderer.create_material(&material);
```

This creates a `GpuMaterial`, which wraps an `Arc<wgpu::BindGroup>`. The bind group contains:

- A uniform buffer with scalar parameters (base color, metallic, roughness, AO, emissive).
- Five texture bindings (albedo, normal, ORM, emissive, plus a sampler).

Textures are uploaded synchronously via `create_texture_with_data()`, which can block if the textures are large. For production, consider uploading textures asynchronously or batching uploads during load screens.

The `GpuMaterial` is cheap to clone (Arc-counted) and can be shared across multiple draw calls. If you have a hundred identical objects, create the material once and reuse it.

### Default Materials

The renderer provides a built-in default white material with no textures. This is used when you call `renderer.draw_mesh()` without providing a material:

```rust
renderer.draw_mesh(&mesh);  // Uses default white material
```

The default has `base_color=[1,1,1,1]`, `metallic=0`, `roughness=0.5`, and no emissive. It's useful for placeholder geometry or testing lighting without worrying about material setup.

### Material Sharing and Lifetime

Materials are reference-counted via Arc, so cloning is cheap and safe. The bind group (and its textures/buffers) stays alive as long as any `GpuMaterial` or `DrawCall` references it. When you drop the last reference, the GPU resources are freed automatically.

This design simplifies memory management—you don't need to manually track which materials are in use or defer deletion until the GPU is idle. Just clone materials freely and let Rust's ownership handle cleanup.

## Draw Calls

A `DrawCall` bundles a mesh with a material for rendering:

```rust
pub struct DrawCall {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
    pub material_bind_group: Arc<wgpu::BindGroup>,
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}
```

Draw calls are created internally when you call `renderer.draw_mesh()` or `renderer.render_scene()`. You rarely construct them manually—the renderer handles batching and submission.

Each draw call binds the material bind group (group 3), then issues an indexed draw command. The G-buffer shader reads the material uniform and samples textures to compute albedo, normal, roughness, metallic, and emissive. These are written to the G-buffer textures and later consumed by the deferred lighting pass.

## Texture Conventions

Helio follows standard PBR texture conventions:

- **Albedo/Base Color**: sRGB RGBA. RGB = color, A = opacity.
- **Normal Map**: Linear RGBA. RGB = tangent-space normal (0..1 mapped from -1..1), A = unused.
- **ORM**: Linear RGBA. R = AO, G = roughness, B = metallic, A = unused.
- **Emissive**: sRGB RGBA. RGB = emissive color, A = unused.

If your assets use different conventions (e.g., separate roughness/metallic textures, or different channel packing), you'll need to convert them during loading. Many texture tools like Substance Painter can export in glTF-compatible ORM format directly.

## Performance Tips

### Vertex Budget

Modern GPUs can handle millions of vertices per frame, but vertex shading cost scales with vertex count and attribute complexity. The PackedVertex format (32 bytes) is already optimized—larger formats would waste bandwidth. If you're hitting vertex throughput limits (unlikely unless rendering 10+ million triangles), consider:

- Level-of-detail (LOD): Swap high-poly meshes for lower-poly variants at distance.
- Frustum culling: Don't submit off-screen objects.
- Occlusion culling: Skip objects hidden behind others.

### Index Buffer Compression

Helio uses u32 indices, which waste space for small meshes. Consider using u16 indices for meshes with <65k vertices and create separate pipelines for each index format. This halves index buffer size and bandwidth.

### Material Batching

Changing material bind groups incurs a small GPU cost (a few nanoseconds per switch). To minimize this, sort draw calls by material before submission. Helio doesn't do this automatically—if you care about optimal batching, maintain your draw list sorted by material ID.

### Texture Atlasing

If you have many small textures (e.g., hundreds of unique 256×256 albedo maps), consider packing them into a single larger atlas. This reduces bind group diversity and descriptor set pressure, improving batch efficiency. However, atlasing complicates UV mapping and filtering at edges—it's a trade-off.

## Example: Textured PBR Material

Here's a complete example loading textures from PNG files and creating a full PBR material:

```rust
use image;
use helio_render_v2::{Material, TextureData};

fn load_texture(path: &str) -> TextureData {
    let img = image::open(path).unwrap().to_rgba8();
    let width = img.width();
    let height = img.height();
    let data = img.into_raw();
    TextureData::new(data, width, height)
}

let material = Material::new()
    .with_base_color([1.0, 1.0, 1.0, 1.0])  // neutral tint
    .with_metallic(0.0)                      // dielectric
    .with_roughness(1.0)                     // roughness multiplier
    .with_base_color_texture(load_texture("brick_albedo.png"))
    .with_normal_map(load_texture("brick_normal.png"))
    .with_orm_texture(load_texture("brick_orm.png"));

let gpu_material = renderer.create_material(&material);
```

This creates a brick material with color, normal, and ORM maps. The scalar parameters act as multipliers, so setting `roughness=1.0` means the ORM roughness channel is used as-is. If you set `roughness=0.5`, the final roughness would be half the ORM value, making the surface glossier.

## Summary

Materials and meshes define what to draw and how it looks. Key points:

- **PackedVertex**: 32-byte vertex format with compressed normals/tangents for bandwidth efficiency.
- **GpuMesh**: GPU-resident mesh with Arc-counted buffers, bounding sphere for culling, and procedural primitives (cube, plane, box).
- **Material**: CPU-side PBR description with base color, metallic, roughness, AO, emissive, and optional texture maps.
- **GpuMaterial**: GPU bind group containing material uniforms and five texture bindings (albedo, normal, ORM, emissive, sampler).
- **Texture Maps**: Follow glTF conventions—sRGB for albedo/emissive, linear for normal/ORM, standard packing.
- **Upload**: Use `GpuMesh::new()` and `renderer.create_material()` to transfer CPU data to GPU.

This system balances simplicity, performance, and compatibility with industry-standard workflows. It's easy to integrate with off-the-shelf assets while maintaining full control over low-level GPU resources.
