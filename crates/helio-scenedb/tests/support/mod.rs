//! Shared test-only support for the `helio-scenedb` cull-pass suite (M3-b
//! T5/T6): the headless device bootstrap, the byte readback helper, the
//! scene config, the `MeshMetadata` builder, and — new in T6 — small
//! column-major matrix helpers plus an INDEPENDENT CPU reference
//! implementation of the cull term list (design §4/§11/§12), used by both
//! `tests/cull_equality.rs` (M3-b T6 deliverable 3) and `tests/cull_pass.rs`
//! (M3-b T6 deliverable 4, the rotation regression pin).
//!
//! Lives at `tests/support/mod.rs` (a subdirectory, not a top-level
//! `tests/*.rs` file) so cargo does NOT compile it as its own standalone
//! integration-test binary — every consumer pulls it in via
//! `#[path = "support/mod.rs"] mod support;`, the standard Rust idiom for
//! sharing code between integration tests without a separate crate.
//!
//! ## The CPU reference's independence (M3-b T6 deliverable 3's explicit
//! requirement: "written INDEPENDENTLY from the WGSL — do not transliterate
//! the shader")
//!
//! [`cpu_cull_token`] below is derived directly from the design spec's own
//! prose (§3.1 generation validation, the M3-α T4 mesh-bounds note, §11's
//! `|M_3x3|` world-AABB formula, §12's W<=0 near-plane bypass, and the
//! `n.p+d>=0` frustum-plane convention `spatial::Frustum` already
//! documents) — not copied or mechanically adapted from `CULL_WGSL`'s WGSL
//! text in `wgsl.rs`. It necessarily performs the SAME algorithm (that is
//! the point of an equality oracle), but every line here was written by
//! reading the spec sections, not the shader source.
#![allow(dead_code)]

use pulsar_scenedb::gpu::{EngineGpuContext, MeshMetadata};
use pulsar_scenedb::gpu::{RegionClassConfig, SceneGpuConfig};
use pulsar_scenedb::InstanceInfo;

// ---------------------------------------------------------------------
// Device bootstrap + readback (kept verbatim from `cull_pass.rs`'s M3-b T5
// helpers so every T6 test file gets the identical headless harness).
// ---------------------------------------------------------------------

pub fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("helio-scenedb-cull-support-test"),
        // M3-b T4's point: this seam fits under WebGPU-portable default
        // limits — no `adapter.limits()` workaround needed anywhere in T6.
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

pub fn readback(ctx: &EngineGpuContext, buf: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

pub fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 256, max_resident_cells: 4 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

pub fn mesh(center: [f32; 3], extents: [f32; 3], index_count: u32, index_offset: u32, base_vertex: i32) -> MeshMetadata {
    MeshMetadata {
        vertex_offset: 0,
        index_offset,
        index_count,
        base_vertex,
        material_index: 0,
        lod_count: 1, // C5 XOR rule: traditional mesh, no cluster table
        lod_distances: [0.0; 4],
        local_aabb_center: center,
        cluster_table_offset: 0,
        local_aabb_extents: extents,
        meshlet_count: 0,
    }
}

// ---------------------------------------------------------------------
// M3-b T6: column-major matrix helpers (the pinned convention — see
// `pulsar_scenedb::page`'s `Pod for [f32; 16]` doc comment: `array[4*col +
// row] = M[row][col]`, i.e. `to_cols_array()`, no transpose).
// ---------------------------------------------------------------------

pub fn mat3_rot_x(deg: f32) -> [[f32; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

pub fn mat3_rot_z(deg: f32) -> [[f32; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

pub fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

pub fn mat3_transpose(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[j][i];
        }
    }
    r
}

/// Design §11's `|M_3x3|` operator applied directly to a 3x3 (elementwise
/// abs, standard matrix-vector product) — NOT restricted to matrices that
/// came from a 4x4; used both by [`cpu_cull_token`] (on the 4x4's
/// upper-left block) and directly by the rotation-regression fixture in
/// `cull_pass.rs` (on the bare rotation matrix, to compute BOTH the correct
/// and the deliberately-wrong-transposed extents for the self-verifying
/// guard).
pub fn mat3_abs_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let mut r = [0.0f32; 3];
    for i in 0..3 {
        r[i] = m[i][0].abs() * v[0] + m[i][1].abs() * v[1] + m[i][2].abs() * v[2];
    }
    r
}

/// Column-major-flat `mat4x4<f32>` (16 `f32`s) built from a 3x3 rotation
/// `r` (standard `r[row][col]` indexing) and a translation `t` — the
/// pinned M3-β T5-review convention (`array[4*col+row] = M[row][col]`, no
/// transpose). This is the "correct" writer; the equivalent WRONG
/// (transposed) writer used only by the regression fixture's guard swaps
/// `row`/`col` on the rotation block.
pub fn flatten_column_major(r: [[f32; 3]; 3], t: [f32; 3]) -> [f32; 16] {
    let mut flat = [0.0f32; 16];
    for col in 0..3 {
        for row in 0..3 {
            flat[4 * col + row] = r[row][col];
        }
    }
    flat[12] = t[0];
    flat[13] = t[1];
    flat[14] = t[2];
    flat[15] = 1.0;
    flat
}

/// Column-major-flat pure translation (no rotation) — kept here so every
/// T6 test file that wants a plain translate can share one implementation.
pub fn translation(t: [f32; 3]) -> [f32; 16] {
    flatten_column_major([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t)
}

/// Symmetric perspective projection, fovy=90deg, aspect=1 (`view_proj_
/// 90deg`'s exact geometry from `cull_pass.rs` M3-b T5, reproduced here so
/// T6's new files share ONE definition rather than three hand-copies).
/// Column-major-flat, matching WGSL's `mat4x4<f32>` load convention.
pub fn view_proj_90deg() -> [f32; 16] {
    let near = 1.0_f32;
    let far = 100.0_f32;
    let a = far / (near - far);
    let b = (near * far) / (near - far);
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, a, -1.0,
        0.0, 0.0, b, 0.0,
    ];
    m
}

/// The 6 inward-normal (`n.p+d>=0` == inside) world-space planes of
/// [`view_proj_90deg`]'s frustum, independently derived from the fovy=90/
/// aspect=1/near=1/far=100 geometry (NOT extracted from the matrix) —
/// identical to `cull_pass.rs`'s own `frustum_planes_90deg`.
pub fn frustum_planes_90deg() -> [[f32; 4]; 6] {
    [
        [1.0, 0.0, -1.0, 0.0],  // left:   x - z >= 0
        [-1.0, 0.0, -1.0, 0.0], // right: -x - z >= 0
        [0.0, 1.0, -1.0, 0.0],  // bottom: y - z >= 0
        [0.0, -1.0, -1.0, 0.0], // top:   -y - z >= 0
        [0.0, 0.0, -1.0, -1.0], // near:  -z - 1 >= 0  (z <= -1)
        [0.0, 0.0, 1.0, 100.0], // far:    z + 100 >= 0 (z >= -100)
    ]
}

// ---------------------------------------------------------------------
// M3-b T6: a tiny seedable PRNG (splitmix64) — no `rand` dependency exists
// in this crate's `Cargo.toml`, and the equality test's ONLY requirement
// is a deterministic, printable seed, not cryptographic quality.
// ---------------------------------------------------------------------

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Uniform `u32` in `[0, bound)`. `bound` must be nonzero.
    pub fn next_u32(&mut self, bound: u32) -> u32 {
        (self.next_u64() % bound as u64) as u32
    }

    pub fn chance(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }
}

// ---------------------------------------------------------------------
// M3-b T6 deliverable 3: the independent CPU reference cull implementation.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CpuDecision {
    Stale,
    Oob,
    FrustumCulled,
    Visible { near_clip: bool },
}

/// Row-indexed / mesh-indexed input snapshot the CPU reference reads —
/// plain host-side `Vec`s / slices, deliberately NOT `wgpu::Buffer`s (this
/// oracle never touches the device; the equality test constructs BOTH this
/// struct and the real GPU upload from the SAME host-side values, then
/// compares the two independently-computed outputs).
pub struct CpuCullInputs<'a> {
    pub tokens: &'a [u32],
    pub expected_gens: &'a [u32],
    pub slot_mirror: &'a [u32],
    pub generations: &'a [u32],
    pub instance_info: &'a [InstanceInfo],
    pub instances: &'a [[f32; 16]],
    pub mesh_table: &'a [MeshMetadata],
    pub mesh_count: u32,
    pub view_proj: [f32; 16],
    pub planes: [[f32; 4]; 6],
}

/// `M * (p, 1)`, taking only the xyz result — column-major-flat `m`, the
/// pinned convention (`mat4_upper3x3`/this function agree on the same flat
/// layout `flatten_column_major` writes).
pub fn mat4_mul_point(m: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for row in 0..3 {
        let mut s = m[4 * 3 + row]; // translation column (col 3) times w=1
        for col in 0..3 {
            s += m[4 * col + row] * p[col];
        }
        out[row] = s;
    }
    out
}

/// Full `M * v` for a homogeneous `v` (used for the §12 corner-projection
/// through the view-proj matrix — needs the resulting `w`, so `mat4_mul_
/// point`'s xyz-only shortcut doesn't apply here).
pub fn mat4_mul_vec4(m: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        let mut s = 0.0;
        for col in 0..4 {
            s += m[4 * col + row] * v[col];
        }
        out[row] = s;
    }
    out
}

/// The upper-left 3x3 rotation/scale block of a column-major-flat 4x4,
/// as a `[[f32;3];3]` in standard `[row][col]` indexing.
pub fn mat4_upper3x3(m: &[f32; 16]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for col in 0..3 {
        for row in 0..3 {
            r[row][col] = m[4 * col + row];
        }
    }
    r
}

/// The `n.p+d>=0` == inside convention (`spatial::Frustum`'s own
/// convention, per `wgsl.rs`'s module doc): true unless the box is
/// ENTIRELY on the outside of `plane`.
pub fn plane_test(center: [f32; 3], extents: [f32; 3], plane: [f32; 4]) -> bool {
    let r = extents[0] * plane[0].abs() + extents[1] * plane[1].abs() + extents[2] * plane[2].abs();
    let d = plane[0] * center[0] + plane[1] * center[1] + plane[2] * center[2] + plane[3];
    d >= -r
}

/// The design §4 β term list, for ONE token index `i` into `inputs.tokens`:
/// count-guard is the caller's loop bound (not modeled here — every `i` in
/// `0..inputs.tokens.len()` is presumed already within the dispatch count);
/// §3.1 generation validation -> mesh_index bounds check (M3-α T4) -> §11
/// `|M_3x3|` world AABB -> §12 W<=0 near-clip bypass -> frustum planes.
/// Returns `(row, decision)`.
pub fn cpu_cull_token(inputs: &CpuCullInputs, i: usize) -> (u32, CpuDecision) {
    let row = inputs.tokens[i];
    let slot = inputs.slot_mirror[row as usize];
    let live_gen = inputs.generations[slot as usize];
    if live_gen != inputs.expected_gens[i] {
        return (row, CpuDecision::Stale);
    }

    let mesh_index = inputs.instance_info[row as usize].mesh_index;
    if mesh_index >= inputs.mesh_count {
        return (row, CpuDecision::Oob);
    }

    let mesh = &inputs.mesh_table[mesh_index as usize];
    let m = &inputs.instances[row as usize];
    // GROUND TRUTH, not the shader's shortcut (M3-β T6 review, defect 1):
    // transform all eight local-AABB corners through the full instance
    // matrix and take their bounds. The shader uses design §11's `|M₃ₓ₃|`
    // abs-matrix identity instead — mathematically equivalent for affine
    // transforms but a SHORTCUT, and a reference that reimplements the same
    // shortcut can only prove the code agrees with itself. Deriving the
    // world AABB the slow, obviously-correct way makes this an oracle: it
    // would catch an error in the abs-matrix formula itself, which is
    // exactly the class of bug the T5 review's transform-convention probe
    // showed this codebase is capable of harboring undetected.
    let (world_center, world_extents) = {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for c in 0u32..8 {
            let sx = if c & 1 != 0 { 1.0 } else { -1.0 };
            let sy = if c & 2 != 0 { 1.0 } else { -1.0 };
            let sz = if c & 4 != 0 { 1.0 } else { -1.0 };
            let local = [
                mesh.local_aabb_center[0] + sx * mesh.local_aabb_extents[0],
                mesh.local_aabb_center[1] + sy * mesh.local_aabb_extents[1],
                mesh.local_aabb_center[2] + sz * mesh.local_aabb_extents[2],
            ];
            let w = mat4_mul_point(m, local);
            for k in 0..3 {
                lo[k] = lo[k].min(w[k]);
                hi[k] = hi[k].max(w[k]);
            }
        }
        (
            [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5],
            [(hi[0] - lo[0]) * 0.5, (hi[1] - lo[1]) * 0.5, (hi[2] - lo[2]) * 0.5],
        )
    };
    // Cross-check the shader's §11 shortcut against the ground truth above.
    // If these ever diverge, the abs-matrix identity (or the pinned
    // column-major flattening it depends on) is wrong — fail loudly here
    // rather than let the equality test silently compare two copies of the
    // same mistake.
    {
        let shortcut = mat3_abs_mul_vec3(mat4_upper3x3(m), mesh.local_aabb_extents);
        for k in 0..3 {
            let tol = 1e-4 * world_extents[k].abs().max(1.0);
            assert!(
                (shortcut[k] - world_extents[k]).abs() <= tol,
                "design §11 |M₃ₓ₃| shortcut disagrees with 8-corner ground truth on axis {k}: \
                 shortcut {} vs truth {} (row {row}) — the abs-matrix identity or the \
                 column-major flattening convention is wrong",
                shortcut[k],
                world_extents[k]
            );
        }
    }

    let mut near_clip = false;
    for c in 0u32..8 {
        let sx = if c & 1 != 0 { 1.0 } else { -1.0 };
        let sy = if c & 2 != 0 { 1.0 } else { -1.0 };
        let sz = if c & 4 != 0 { 1.0 } else { -1.0 };
        let corner = [
            world_center[0] + sx * world_extents[0],
            world_center[1] + sy * world_extents[1],
            world_center[2] + sz * world_extents[2],
        ];
        let clip = mat4_mul_vec4(&inputs.view_proj, [corner[0], corner[1], corner[2], 1.0]);
        if clip[3] <= 0.0 {
            near_clip = true;
        }
    }

    if near_clip {
        return (row, CpuDecision::Visible { near_clip: true });
    }

    for plane in inputs.planes.iter() {
        if !plane_test(world_center, world_extents, *plane) {
            return (row, CpuDecision::FrustumCulled);
        }
    }
    (row, CpuDecision::Visible { near_clip: false })
}

/// Runs [`cpu_cull_token`] over every token, returning `(row, decision)`
/// for all of them in token order (callers reduce to whatever comparison
/// shape they need — full decision list, visible-only row set, per-category
/// counts for house-law guards, ...).
pub fn cpu_cull_all(inputs: &CpuCullInputs) -> Vec<(u32, CpuDecision)> {
    (0..inputs.tokens.len()).map(|i| cpu_cull_token(inputs, i)).collect()
}
