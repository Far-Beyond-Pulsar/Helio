//! Tiled light culling (Forward+ style).
//!
//! Divides the screen into TILE_SIZE×TILE_SIZE tiles.  Each workgroup thread
//! is responsible for one tile: it iterates all lights, tests whether each
//! light's sphere of influence intersects the tile frustum (constructed from
//! the tile's four corner NDC coordinates), and writes matching light indices
//! into a flat storage array.
//!
//! CPU cost: O(1) — one dispatch with ceil(num_tiles / 256) workgroups.
//! GPU cost: O(num_tiles × num_lights) in the worst case; in practice much
//!           less because most lights are spatially sparse.
//!
//! Tile data layout (per tile slot of MAX_LIGHTS_PER_TILE entries):
//!   tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i]  = light_index_i
//!   tile_light_counts[tile_idx]                           = number of lights

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;

// ─────────────────────────────────────────────────────────────────────────────
// Bind group 0 — uniforms & scene data
// ─────────────────────────────────────────────────────────────────────────────

struct Camera {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    direction_far: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct LightCullParams {
    num_tiles_x:   u32,
    num_tiles_y:   u32,
    num_lights:    u32,
    screen_width:  u32,
    screen_height: u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
}
@group(0) @binding(1) var<uniform> params: LightCullParams;

struct GpuLight {
    position_range:  vec4<f32>,  // xyz = world pos, w = range
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,        // 0 = directional, 1 = point, 2 = spot
    inner_angle:     f32,
    _pad:            u32,
}
@group(0) @binding(2) var<storage, read> lights: array<GpuLight>;

// Output: flat arrays, one slot per tile
@group(0) @binding(3) var<storage, read_write> tile_light_lists:  array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_light_counts: array<u32>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a view-space frustum plane from an NDC edge.
/// Each plane is stored as (nx, ny, nz, d) where Ax+By+Cz+D >= 0 is inside.
fn make_plane_from_ndc_edge(p0: vec2<f32>, p1: vec2<f32>) -> vec4<f32> {
    // Unproject the two edge points to view-space rays (z = -1 in view space).
    let r0 = normalize(vec3<f32>(
        p0.x / camera.proj[0][0],
        p0.y / camera.proj[1][1],
        -1.0,
    ));
    let r1 = normalize(vec3<f32>(
        p1.x / camera.proj[0][0],
        p1.y / camera.proj[1][1],
        -1.0,
    ));
    // Plane normal = cross of the two rays (points toward inside of frustum).
    let n = normalize(cross(r0, r1));
    return vec4<f32>(n, 0.0); // d = 0 (plane passes through camera origin)
}

/// Test whether a sphere (view-space center + radius) intersects the four
/// lateral frustum planes of a tile.
/// Returns true if the sphere is NOT culled (i.e. it may be visible).
fn sphere_inside_tile_frustum(
    sphere_center_vs: vec3<f32>,
    sphere_radius: f32,
    planes: array<vec4<f32>, 4>,
) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let dist = dot(planes[i].xyz, sphere_center_vs) + planes[i].w;
        if dist < -sphere_radius {
            return false; // Entirely outside this plane
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main kernel — one thread per tile
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_idx = gid.x;
    let total_tiles = params.num_tiles_x * params.num_tiles_y;
    if tile_idx >= total_tiles {
        return;
    }

    let tile_x = tile_idx % params.num_tiles_x;
    let tile_y = tile_idx / params.num_tiles_x;

    // Compute the tile's NDC bounding rectangle.
    // NDC x ∈ [-1, +1] left→right, y ∈ [-1, +1] bottom→top.
    let fw = f32(params.screen_width);
    let fh = f32(params.screen_height);
    let ts = f32(TILE_SIZE);
    let ndc_left  = (f32(tile_x) * ts / fw) * 2.0 - 1.0;
    let ndc_right = min((f32(tile_x + 1u) * ts / fw) * 2.0 - 1.0, 1.0);
    // Note: screen y goes top-down, NDC y goes bottom-up.
    let ndc_top    =  1.0 - (f32(tile_y) * ts / fh) * 2.0;
    let ndc_bottom =  max(1.0 - (f32(tile_y + 1u) * ts / fh) * 2.0, -1.0);

    // Build 4 lateral frustum planes from the tile corners.
    // Plane order: left, right, top, bottom.
    var planes: array<vec4<f32>, 4>;
    planes[0] = make_plane_from_ndc_edge(vec2<f32>(ndc_left, ndc_bottom),  vec2<f32>(ndc_left,  ndc_top));    // left
    planes[1] = make_plane_from_ndc_edge(vec2<f32>(ndc_right, ndc_top),    vec2<f32>(ndc_right, ndc_bottom)); // right
    planes[2] = make_plane_from_ndc_edge(vec2<f32>(ndc_left,  ndc_top),    vec2<f32>(ndc_right, ndc_top));    // top
    planes[3] = make_plane_from_ndc_edge(vec2<f32>(ndc_right, ndc_bottom), vec2<f32>(ndc_left,  ndc_bottom)); // bottom

    // Iterate all lights and test each against this tile's frustum.
    var count = 0u;
    for (var i = 0u; i < params.num_lights; i++) {
        let light = lights[i];

        // Directional lights have infinite range — always visible.
        if light.light_type == 0u {
            if count < MAX_LIGHTS_PER_TILE {
                tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + count] = i;
                count += 1u;
            }
            continue;
        }

        // Transform light position to view space.
        let pos_vs = (camera.view * vec4<f32>(light.position_range.xyz, 1.0)).xyz;
        let range  = light.position_range.w;

        if sphere_inside_tile_frustum(pos_vs, range, planes) {
            if count < MAX_LIGHTS_PER_TILE {
                tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + count] = i;
                count += 1u;
            }
        }
    }

    tile_light_counts[tile_idx] = count;
}
