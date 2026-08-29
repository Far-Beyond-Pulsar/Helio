use wgpu::util::DeviceExt;

/// Resolution of each bokeh shape texture slice.
const BOKEH_TEX_SIZE: u32 = 64;

/// Number of slices in the bokeh shape texture array (blade counts 3..=11).
const BOKEH_SLICES: u32 = 9;

/// Generate a CPU-side bokeh shape texture array and upload it to the GPU.
///
/// Each slice is a 64×64 R8Unorm texture encoding the aperture's light
/// transmission mask for a given blade count (slice 0 = 3 blades, slice 8 = 11
/// blades). White = fully open aperture, black = occluded by blades.
///
/// The texture is a 2D array usable as a sampled texture in WGSL compute shaders.
pub fn create_bokeh_shape_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let tex_data = generate_bokeh_slices();

    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("Bokeh Shape Texture Array"),
            size: wgpu::Extent3d {
                width: BOKEH_TEX_SIZE,
                height: BOKEH_TEX_SIZE,
                depth_or_array_layers: BOKEH_SLICES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &tex_data,
    );

    texture
}

/// Generate all bokeh shape slices as raw R8Unorm texel data.
///
/// Returns a flat Vec<u8> with stride = BOKEH_TEX_SIZE * BOKEH_TEX_SIZE per slice,
/// concatenated for all 9 slices.
fn generate_bokeh_slices() -> Vec<u8> {
    let slice_bytes = (BOKEH_TEX_SIZE * BOKEH_TEX_SIZE) as usize;
    let mut data = Vec::with_capacity(slice_bytes * BOKEH_SLICES as usize);

    for blades in 3..=11 {
        generate_bokeh_slice(blades, &mut data);
    }

    data
}

/// Generate one bokeh shape slice for the given blade count.
fn generate_bokeh_slice(blades: u32, out: &mut Vec<u8>) {
    let half = BOKEH_TEX_SIZE as f32 * 0.5;
    let inv_half = 1.0 / half;
    let angle_step = std::f32::consts::TAU / blades as f32;
    // Rotate so one vertex is at the top (subtract half step so the flat side
    // of an odd-blade polygon faces up, which looks more natural).
    let rot_offset = -std::f32::consts::FRAC_PI_2;

    // Pre-compute polygon vertices (inscribed in a circle of radius = half - 1).
    let radius = half - 1.5;
    let verts: Vec<[f32; 2]> = (0..blades)
        .map(|i| {
            let angle = rot_offset + i as f32 * angle_step;
            [half + radius * angle.cos(), half + radius * angle.sin()]
        })
        .collect();

    for y in 0..BOKEH_TEX_SIZE {
        for x in 0..BOKEH_TEX_SIZE {
            let px = x as f32;
            let py = y as f32;

            // Distance from center
            let dx = (px - half) * inv_half;
            let dy = (py - half) * inv_half;
            let dist = (dx * dx + dy * dy).sqrt();

            // Soft circle falloff: 1.0 at center → 0.0 at radius
            let circle = 1.0 - (dist * (half / (half - 1.5))).clamp(0.0, 1.0);
            let circle = (circle * 2.0 - 1.0).clamp(0.0, 1.0);

            // Point-in-polygon test (ray casting)
            let inside = point_in_convex_polygon(px, py, &verts);

            // Soft edge: smoothstep at the polygon boundary using distance
            // to the nearest edge.
            let min_edge_dist = min_distance_to_polygon_edge(px, py, &verts);
            // 0.5-pixel Gaussian transition at the aperture edge
            let edge_transition = 1.0 - (-min_edge_dist * min_edge_dist * 8.0).exp();

            let value = if inside {
                // Inside aperture: full transmission
                (255.0 * edge_transition) as u8
            } else {
                // Outside: partial transmission near edge for soft falloff
                let bleed = (1.0 - (-min_edge_dist * min_edge_dist * 2.0).exp()) * circle;
                (255.0 * bleed.clamp(0.0, 0.5)) as u8
            };

            out.push(value);
        }
    }
}

/// Ray-cast point-in-convex-polygon test.
fn point_in_convex_polygon(px: f32, py: f32, verts: &[[f32; 2]]) -> bool {
    let n = verts.len();
    let mut sign = 0i8;
    for i in 0..n {
        let j = (i + 1) % n;
        let ax = verts[j][0] - verts[i][0];
        let ay = verts[j][1] - verts[i][1];
        let bx = px - verts[i][0];
        let by = py - verts[i][1];
        let cross = ax * by - ay * bx;
        let s = if cross > 0.0 {
            1
        } else if cross < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 {
            if sign == 0 {
                sign = s;
            } else if sign != s {
                return false;
            }
        }
    }
    true
}

/// Minimum Euclidean distance from point (px, py) to any edge of the polygon.
fn min_distance_to_polygon_edge(px: f32, py: f32, verts: &[[f32; 2]]) -> f32 {
    let n = verts.len();
    let mut min_dist = f32::MAX;
    for i in 0..n {
        let j = (i + 1) % n;
        let ax = verts[i][0];
        let ay = verts[i][1];
        let bx = verts[j][0];
        let by = verts[j][1];

        // Vector along edge
        let ex = bx - ax;
        let ey = by - ay;
        let len2 = ex * ex + ey * ey;
        if len2 < 1e-10 {
            continue;
        }

        // Project point onto edge line, clamped to segment
        let t = ((px - ax) * ex + (py - ay) * ey) / len2;
        let t = t.clamp(0.0, 1.0);
        let cx = ax + t * ex;
        let cy = ay + t * ey;
        let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
        min_dist = min_dist.min(dist);
    }
    min_dist
}
