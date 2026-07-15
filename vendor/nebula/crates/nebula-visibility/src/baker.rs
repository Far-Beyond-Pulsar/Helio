use async_trait::async_trait;
use nebula_core::{
    context::BakeContext, error::NebulaError, progress::ProgressReporter,
    scene::SceneGeometry, traits::BakePass,
};
use crate::{config::PvsConfig, output::PvsOutput};

/// GPU occlusion-query–based Potentially Visible Set baker.
///
/// ## Algorithm
///
/// 1. **Grid construction** — the scene AABB is voxelised into cells of side
///    `config.cell_size`.  Each cell is represented by its centre point.
///
/// 2. **GPU ray casting** — a compute shader fires `config.ray_budget`
///    randomly distributed rays from each cell centre.  For each ray, the
///    scene triangles are tested via Möller–Trumbore intersection (the same
///    kernel used by the AO baker).  When a ray reaches a target cell without
///    hitting any geometry, a visibility bit is written atomically.
///
/// 3. **Conservative dilation** (optional) — a second CPU pass expands the
///    visible set by one cell in each grid direction to ensure zero false
///    negatives at the cost of a slightly larger PVS.
///
/// 4. **Bit-packing** — the binary visibility matrix is stored as a flat
///    packed `Vec<u64>` bitfield.
pub struct PvsBaker;

#[async_trait(?Send)]
impl BakePass for PvsBaker {
    type Input  = PvsConfig;
    type Output = PvsOutput;

    fn name(&self) -> &'static str { "pvs" }

    async fn execute(
        &self,
        scene:    &SceneGeometry,
        config:   &PvsConfig,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<PvsOutput, NebulaError> {
        reporter.begin("pvs", 4);

        reporter.step("pvs", 0, "computing scene AABB and grid");
        let (world_min, world_max, grid_dims) = compute_grid(scene, config);
        let cell_count = grid_dims[0] * grid_dims[1] * grid_dims[2];
        let words_per_cell = cell_count.div_ceil(64);

        reporter.step("pvs", 1, "uploading geometry");
        let (vbuf, ibuf, mbuf, n_meshes, n_indices) = upload_geometry(scene, ctx)?;

        reporter.step("pvs", 2, &format!("running PVS on {} cells", cell_count));
        let bits_raw = run_pvs_gpu(config, ctx, world_min, grid_dims, cell_count, words_per_cell, &vbuf, &ibuf, &mbuf, n_meshes, n_indices)?;

        reporter.step("pvs", 3, "post-processing");
        let bits = if config.conservative {
            apply_conservative_dilation(&bits_raw, grid_dims, words_per_cell)
        } else {
            bits_raw
        };

        reporter.finish("pvs", true, "done");
        let config_json = serde_json::to_string(config).unwrap_or_default();
        Ok(PvsOutput { world_min, world_max, grid_dims, cell_size: config.cell_size, cell_count, words_per_cell, bits, config_json })
    }
}

// ── Grid helper ───────────────────────────────────────────────────────────────

fn compute_grid(scene: &SceneGeometry, config: &PvsConfig) -> ([f32;3], [f32;3], [u32;3]) {
    let (mut mn, mut mx) = ([f32::MAX;3], [f32::MIN;3]);
    for mesh in &scene.meshes {
        for &p in &mesh.positions {
            let wp = mesh.world_transform.0.transform_point3(glam::Vec3::from(p));
            let wp = wp.to_array();
            for i in 0..3 { mn[i]=mn[i].min(wp[i]); mx[i]=mx[i].max(wp[i]); }
        }
    }
    // Expand by one cell in each direction
    let s = config.cell_size;
    let mn = mn.map(|v| v - s);
    let mx = mx.map(|v| v + s);
    let dims: [u32;3] = core::array::from_fn(|i| ((mx[i] - mn[i]) / s).ceil().max(1.0) as u32);
    (mn, mx, dims)
}

// ── Geometry upload ───────────────────────────────────────────────────────────

fn upload_geometry(scene: &SceneGeometry, ctx: &BakeContext) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, u32, u32), NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuVert { pos: [f32;3], _p: f32 }
    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuMesh { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: [[f32;4];4] }

    let mut verts  = Vec::<GpuVert>::new();
    let mut idxs   = Vec::<u32>::new();
    let mut meshes = Vec::<GpuMesh>::new();

    for mesh in &scene.meshes {
        let vb = verts.len() as u32;
        let ib = idxs.len()  as u32;
        for &p in &mesh.positions { verts.push(GpuVert { pos: p, _p: 0.0 }); }
        idxs.extend(mesh.indices.iter().map(|&i| i + vb));
        meshes.push(GpuMesh { idx_off: ib, idx_cnt: mesh.indices.len() as u32, vert_off: vb, _p: 0, xform: mesh.world_transform.0.to_cols_array_2d() });
    }

    let n_indices = idxs.len() as u32;
    let n_meshes  = meshes.len() as u32;
    let mk = |label, data: &[u8]| ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
    Ok((mk("nebula_pvs_vbuf",bytemuck::cast_slice(&verts)), mk("nebula_pvs_ibuf",bytemuck::cast_slice(&idxs)), mk("nebula_pvs_mbuf",bytemuck::cast_slice(&meshes)), n_meshes, n_indices))
}

// ── GPU PVS kernel ────────────────────────────────────────────────────────────

fn run_pvs_gpu(
    config:         &PvsConfig,
    ctx:            &BakeContext,
    world_min:      [f32;3],
    grid_dims:      [u32;3],
    cell_count:     u32,
    words_per_cell: u32,
    vbuf:           &wgpu::Buffer,
    ibuf:           &wgpu::Buffer,
    mbuf:           &wgpu::Buffer,
    n_meshes:       u32,
    n_indices:      u32,
) -> Result<Vec<u64>, NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct PvsParams {
        world_min: [f32;3], cell_size: f32,
        grid_dims: [u32;3], cell_count: u32,
        words_per_cell: u32, ray_budget: u32,
        max_dist: f32, vis_threshold: u32,
        num_meshes: u32, num_indices: u32,
        seed: u32, _p: u32,
    }

    let p = PvsParams { world_min, cell_size: config.cell_size, grid_dims, cell_count, words_per_cell, ray_budget: config.ray_budget, max_dist: config.max_ray_distance, vis_threshold: config.visibility_threshold, num_meshes: n_meshes, num_indices: n_indices, seed: 0xFEEDFACE, _p: 0 };
    let pbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("nebula_pvs_params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });

    let out_words = cell_count as u64 * words_per_cell as u64;
    let out_bytes = out_words * 8; // u64 = 8 bytes
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("nebula_pvs_bits"), size: out_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nebula_pvs_bgl"),
        entries: &[
            bgl_uniform(0), bgl_storage_ro(1), bgl_storage_ro(2), bgl_storage_ro(3),
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nebula_pvs_bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
        ],
    });

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nebula_pvs_cs"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pvs.wgsl").into()) });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("nebula_pvs_pipeline"), layout: Some(&pl), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    // One thread per source cell; each thread fires ray_budget rays
    let wg = 64u32;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nebula_pvs_enc") });
    { let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
      pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(cell_count.div_ceil(wg), 1, 1); }
    ctx.queue.submit(std::iter::once(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    // Readback
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("nebula_pvs_staging"), size: out_bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc2 = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc2.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    ctx.queue.submit(std::iter::once(enc2.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok().and_then(|r| r.ok()).ok_or_else(|| NebulaError::ReadbackTimeout { ms: 30000 })?;
    let raw = staging
        .slice(..)
        .get_mapped_range()
        .expect("Nebula visibility readback buffer should be mapped");
    let bits = bytemuck::cast_slice::<u8, u64>(&*raw).to_vec();
    drop(raw);
    staging.unmap();
    Ok(bits)
}

// ── Conservative dilation ─────────────────────────────────────────────────────

fn apply_conservative_dilation(bits: &[u64], dims: [u32;3], wpc: u32) -> Vec<u64> {
    let [gx, gy, gz] = dims.map(|d| d as usize);
    let mut out = bits.to_vec();
    let cell = |x: usize, y: usize, z: usize| z * gy * gx + y * gx + x;
    let wpc = wpc as usize;
    for z in 0..gz { for y in 0..gy { for x in 0..gx {
        let src = cell(x,y,z);
        // Six-connected neighbours
        for (nx,ny,nz) in [
            (x.wrapping_sub(1),y,z),(x+1,y,z),
            (x,y.wrapping_sub(1),z),(x,y+1,z),
            (x,y,z.wrapping_sub(1)),(x,y,z+1),
        ] {
            if nx < gx && ny < gy && nz < gz {
                let nbr = cell(nx,ny,nz);
                // OR the source cell's bitset into the neighbour's bitset
                for w in 0..wpc {
                    out[nbr * wpc + w] |= bits[src * wpc + w];
                }
            }
        }
    }}}
    out
}

fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bgl_storage_ro(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
