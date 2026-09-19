struct GridUniform {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> grid: GridUniform;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[index];
    var out: VertexOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

fn grid_line(distance: f32, scale: f32, pixel_width: f32) -> f32 {
    let coordinate = distance / scale;
    // Perspective derivatives become arbitrarily large at the horizon. A
    // raw fwidth here turns every line into a solid band, so cap the
    // anti-alias footprint to a fraction of one cell.
    // Keep the footprint deliberately small. LOD selection controls which
    // scale is visible; a derivative-based visibility factor here causes
    // otherwise continuous major lines to break into horizon-facing dashes.
    let derivative = clamp(fwidth(coordinate), 0.00001, 0.01);
    let line_distance = abs(fract(coordinate - 0.5) - 0.5);
    return 1.0 - smoothstep(pixel_width, pixel_width + derivative, line_distance);
}

fn grid_level(position: vec3<f32>, scale: f32, pixel_width: f32) -> f32 {
    return max(
        grid_line(position.x, scale, pixel_width),
        grid_line(position.z, scale, pixel_width),
    );
}

@fragment
fn fs_main(input: VertexOut) -> FragmentOut {
    let near_clip = grid.inv_view_proj * vec4<f32>(input.ndc, 0.0, 1.0);
    let far_clip = grid.inv_view_proj * vec4<f32>(input.ndc, 1.0, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let far_world = far_clip.xyz / far_clip.w;
    let ray = far_world - near_world;

    // Ground plane is y=0. Discard rays that miss it or point away from it.
    if abs(ray.y) < 0.00001 {
        discard;
    }
    let hit_distance = -near_world.y / ray.y;
    if hit_distance < 0.0 {
        discard;
    }
    let hit = near_world + ray * hit_distance;

    // Logarithmic LOD keeps the line frequency stable as the camera moves.
    // Do not floor this value. A hard floor makes a circular ring where the
    // entire grid changes scale in one frame. Instead, cross-fade the current
    // decade into the next decade over the full interval between them. The
    // coarser line is also present as every tenth line of the finer grid, so
    // this is a continuous change in emphasis rather than a new grid popping
    // into existence.
    let distance_to_camera = max(length(hit - grid.camera_position.xyz), 0.001);
    let lod = max(0.0, log(distance_to_camera / 8.0) / log(10.0));
    let lod_base = floor(lod);
    let lod_blend = smoothstep(0.0, 1.0, fract(lod));
    let minor_scale = pow(10.0, lod_base);
    let next_minor_scale = minor_scale * 10.0;
    let minor_now = grid_level(hit, minor_scale, 0.002);
    let minor_next = grid_level(hit, next_minor_scale, 0.0035);
    let minor = minor_now * (1.0 - lod_blend) + minor_next * lod_blend;

    // Major lines are one decade above the active minor level. Cross-fading
    // them with the next level keeps the visual hierarchy while preventing a
    // band of major lines at each LOD boundary.
    let major_scale = minor_scale * 10.0;
    let next_major_scale = major_scale * 10.0;
    let major_now = grid_level(hit, major_scale, 0.0035);
    let major_next = grid_level(hit, next_major_scale, 0.005);
    let major = major_now * (1.0 - lod_blend) + major_next * lod_blend;

    // At a grazing angle a tiny world-space line covers a large screen area.
    // Fade detail there instead of allowing the perspective derivative to
    // turn the horizon into a visible ring/band.
    let ray_length = max(length(ray), 0.0001);
    let view_plane_alignment = abs(ray.y) / ray_length;
    let horizon_fade = smoothstep(0.015, 0.12, view_plane_alignment);

    // World axes remain stable and readable at every camera position.
    // The axes are world-space lines too, but their derivative must not grow
    // without bound at the horizon or they become giant colored wedges.
    let axis_width = clamp(max(fwidth(hit.x), fwidth(hit.z)) * 2.5, 0.035, 0.18);
    let x_axis = 1.0 - smoothstep(axis_width, axis_width * 2.0, abs(hit.x));
    let z_axis = 1.0 - smoothstep(axis_width, axis_width * 2.0, abs(hit.z));
    let alpha = max(max(minor * 0.22, major * 0.38) * horizon_fade, max(x_axis, z_axis) * 0.8);
    if alpha <= 0.001 {
        discard;
    }

    var color = vec3<f32>(0.42, 0.45, 0.50);
    if x_axis > 0.001 {
        color = vec3<f32>(0.95, 0.25, 0.22);
    } else if z_axis > 0.001 {
        color = vec3<f32>(0.25, 0.85, 0.35);
    }
    let projected = grid.view_proj * vec4<f32>(hit, 1.0);
    let depth = projected.z / projected.w;
    var out: FragmentOut;
    out.color = vec4<f32>(color, alpha);
    out.depth = depth;
    return out;
}
