// Texel-streaming debug viewmodes (Helio#238 §5): density heatmap +
// page-miss flash over the finished quarter-res feedback target.
//
// A fullscreen triangle samples the compaction-INPUT cells (`vt_density`,
// still intact after compaction's read-only pass over it) and colors each
// pixel by its cell's packed slot/mip:
//
//   - HEATMAP mode: green→yellow ramp by demanded mip (hotter = finer detail
//     wanted). Zero cells stay untouched (transparent black over the frame).
//   - MISS-FLASH mode: cells whose slot's demanded mip exceeds the meta row's
//     published floor render RED — visible demand the current residency
//     cannot serve, i.e. every place a fetch fell back this frame. Satisfied
//     cells dim green. (Approximate by construction: one cell aggregates 16
//     pixels' maxima.)

struct HeatmapUniforms {
    /// Quarter-res cell grid [width, height, mode(0=heat,1=flash), pad]
    dims_mode: vec4<u32>,
}

@group(0) @binding(0) var<uniform> u: HeatmapUniforms;
@group(0) @binding(1) var vt_cells_tex: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> vt_meta: array<VtMetaRow>;

// Mirrors helio_core::shader::vt_sample::VtMetaRow (kept textually identical).
struct VtMetaRow {
    dims_xy_mips: vec4<u32>,
    floor_flags: vec4<u32>,
    mip_first_rank: array<u32, 8>,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Classic fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    var out: VsOut;
    let x = f32(i32((vi << 1u) & 2u));
    let y = f32(i32(vi & 2u));
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = u.dims_mode.xy;
    let mode = u.dims_mode.z;
    let fb = vec2<u32>(in.pos.xy);
    // Framebuffer pixel → quarter-res cell (target is internal-res; cells are
    // internal/4). Scale by 4 relative to THIS pass's own viewport size via
    // the uniform rather than textureDimensions so XR/upscale paths agree.
    let scale = max(vec2<u32>(1u), vec2<u32>(4u));
    let cell_px = fb / scale;
    if cell_px.x >= dims.x || cell_px.y >= dims.y {
        return vec4<f32>(0.0);
    }
    let cell = textureLoad(vt_cells_tex, vec2<i32>(cell_px), 0).r;
    if cell == 0u {
        return vec4<f32>(0.0);
    }
    let slot = min(cell >> 8u, 255u);
    let mip = (cell & 0xFFu) - 1u;
    let row = vt_meta[slot];
    let floor_mip = row.floor_flags.y;

    if mode == 1u {
        // Miss flash: demand finer than the published floor ⇒ unmet.
        if mip < floor_mip {
            return vec4<f32>(1.0, 0.15, 0.1, 0.55); // hot red: would miss
        }
        return vec4<f32>(0.1, 0.5, 0.15, 0.25); // dim green: served by floor+
    }

    // Heatmap: mip 0 (finest demand) = yellow-hot, coarsest = deep green.
    let t = clamp(f32(mip) / 8.0, 0.0, 1.0);
    let heat = vec3<f32>(1.0 - t * 0.85, 0.25 + t * 0.65, 0.05);
    return vec4<f32>(heat, 0.45);
}
