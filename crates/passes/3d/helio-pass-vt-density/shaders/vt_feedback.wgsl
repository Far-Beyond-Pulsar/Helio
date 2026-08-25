// Texel-streaming demand feedback kernels (Helio#238 §1).
//
// Two entry points over ONE quarter-res R32Uint target ("vt_density" in the
// frame graph), run at opposite ends of the frame:
//
//   cs_clear   — start of frame, BEFORE any geometry: zero every cell so the
//                five raster passes' textureStore feedback writes land on a
//                known-blank slate. A compute store, not a load-op clear:
//                the target lives its whole life as STORAGE_BINDING (it is
//                written from five different fragment stages and read back
//                here), never as an attachment.
//
//   cs_compact — after the LAST texel-producing pass of the frame (the
//                deferred graph's ordering note lives next to the add_pass
//                call in helio-default-graphs): reduce all cells to a
//                per-slot max wanted mip in a 264-word output buffer —
//                words [0..256) hold `pack_feedback(slot, max_wanted)` per
//                slot (libhelio's packing; 0 = untouched sentinel), word
//                [256] counts touched slots via atomicAdd.
//
// Cell addressing must match vt_sample.wgsl exactly: cell = pixel >> 2, i.e.
// this kernel's quarter-res texel (x, y) aggregates internal-res pixels
// [4x..4x+3] × [4y..4y+3].

/// [quarter_width, quarter_height]
struct ClearDims {
    dims: vec2<u32>,
    _pad: vec2<u32>,
}

// Clear entry point's own bindings (group 0). Compaction re-declares its
// identical dims uniform at binding 2 because both entry points share one
// module but must not share a uniform binding with different access patterns.
@group(0) @binding(0) var<uniform> clear_dims: ClearDims;
@group(0) @binding(1) var vt_density_write: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= clear_dims.dims.x || gid.y >= clear_dims.dims.y {
        return;
    }
    textureStore(vt_density_write, vec2<u32>(gid.xy), vec4<u32>(0u, 0u, 0u, 0u));
}

// ── Compaction ───────────────────────────────────────────────────────────────

/// Output words [0..256]: per-slot `pack_feedback` maxima (0 = untouched);
/// word [256]: feedback-store counter. One flat atomic array, one buffer,
/// one binding — two vars could not alias one buffer's ranges in WGSL.
@group(1) @binding(0) var<storage, read_write> out_words: array<atomic<u32>>;

@group(0) @binding(2) var<uniform> clear_dims_c: ClearDims;
@group(0) @binding(3) var vt_cells: texture_storage_2d<r32uint, read>;

@compute @workgroup_size(8, 8, 1)
fn cs_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= clear_dims_c.dims.x || gid.y >= clear_dims_c.dims.y {
        return;
    }
    let cell = textureLoad(vt_cells, vec2<u32>(gid.xy)).r;
    // 0 is the untouched sentinel (pack_feedback biases mips +1), so blank
    // cells contribute nothing and never fabricate slot-0/mip-0 demand.
    if cell == 0u {
        return;
    }
    let slot = cell >> 8u;
    if slot >= 256u {
        return;
    }
    atomicMax(&out_words[slot], cell);
    _ = atomicAdd(&out_words[256u], 1u);
}

