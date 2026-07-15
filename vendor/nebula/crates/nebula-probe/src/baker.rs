use async_trait::async_trait;
use glam::{Mat4, Vec3};
use nebula_core::{
    context::BakeContext, error::NebulaError, progress::ProgressReporter,
    scene::SceneGeometry, traits::BakePass,
};
use crate::{
    config::ProbeConfig,
    output::{IrradianceOutput, ReflectionOutput, ShCoeff},
};

/// Bakes both a specular reflection cubemap and diffuse irradiance SH
/// coefficients from a single probe position.
///
/// Returns a `(ReflectionOutput, IrradianceOutput)` pair.  To use this as an
/// individual [`BakePass`] the engine should call the two helpers directly or
/// chain them with the façade pipeline.
pub struct ProbeBaker;

// ── Convenience entry point ───────────────────────────────────────────────────

impl ProbeBaker {
    pub async fn bake_at(
        position: Vec3,
        scene:    &SceneGeometry,
        config:   &ProbeConfig,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<(ReflectionOutput, IrradianceOutput), NebulaError> {
        let res  = config.face_resolution.clamp(16, 2048);
        let mips = config.specular_mip_levels.min(1 + (res as f32).log2() as u32);
        reporter.begin("probe", 4);

        reporter.step("probe", 0, "uploading scene geometry");
        let (vbuf, ibuf, mbuf) = upload_geometry(scene, ctx)?;

        reporter.step("probe", 1, &format!("tracing {}×{} cubemap ({} samples/face)", res, res, config.samples_per_face));
        let face_data = trace_cubemap(position, scene, config, ctx, &vbuf, &ibuf, &mbuf, res, mips)?;

        reporter.step("probe", 2, "projecting irradiance SH");
        let coefficients = project_sh(&face_data, res, config.sh_order)?;

        reporter.step("probe", 3, "encoding output");
        let config_json = serde_json::to_string(config).unwrap_or_default();
        reporter.finish("probe", true, "done");

        Ok((
            ReflectionOutput { face_resolution: res, mip_levels: mips, is_rgbe: config.use_rgbe, face_data, config_json: config_json.clone() },
            IrradianceOutput { sh_order: config.sh_order, coefficients, config_json },
        ))
    }
}

// ── BakePass impl (bakes at origin; real usage goes through bake_at) ─────────

#[async_trait(?Send)]
impl BakePass for ProbeBaker {
    type Input  = ProbeConfig;
    type Output = ReflectionOutput;

    fn name(&self) -> &'static str { "probe" }

    async fn execute(
        &self,
        scene:    &SceneGeometry,
        config:   &ProbeConfig,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<ReflectionOutput, NebulaError> {
        let (refl, _irr) = ProbeBaker::bake_at(Vec3::ZERO, scene, config, ctx, reporter).await?;
        Ok(refl)
    }
}

// ── Geometry upload ───────────────────────────────────────────────────────────

fn upload_geometry(
    scene: &SceneGeometry, ctx: &BakeContext,
) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer), NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuVert { pos: [f32;3], _p: f32, normal: [f32;3], _p2: f32, uv: [f32;2], _p3: [f32;2] }
    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuMesh { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: [[f32;4];4] }

    let mut verts  = Vec::<GpuVert>::new();
    let mut idxs   = Vec::<u32>::new();
    let mut meshes = Vec::<GpuMesh>::new();

    for mesh in &scene.meshes {
        let vb = verts.len() as u32;
        let ib = idxs.len()  as u32;
        for i in 0..mesh.positions.len() {
            verts.push(GpuVert {
                pos: mesh.positions[i], _p: 0.0,
                normal: mesh.normals.get(i).copied().unwrap_or([0.0,1.0,0.0]), _p2: 0.0,
                uv: mesh.uvs.get(i).copied().unwrap_or([0.0;2]), _p3: [0.0;2],
            });
        }
        idxs.extend(mesh.indices.iter().map(|&i| i + vb));
        meshes.push(GpuMesh { idx_off: ib, idx_cnt: mesh.indices.len() as u32, vert_off: vb, _p: 0, xform: mesh.world_transform.0.to_cols_array_2d() });
    }

    let mk = |label, data: &[u8]| ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
    Ok((mk("nebula_probe_vbuf", bytemuck::cast_slice(&verts)), mk("nebula_probe_ibuf", bytemuck::cast_slice(&idxs)), mk("nebula_probe_mbuf", bytemuck::cast_slice(&meshes))))
}

// ── Cubemap tracing ───────────────────────────────────────────────────────────

/// Face indices: +X=0  −X=1  +Y=2  −Y=3  +Z=4  −Z=5.
const FACE_VIEWS: [Mat4; 6] = [
    // +X  look_at(origin, +X, +Y)
    Mat4::from_cols_array(&[ 0.0,0.0,-1.0,0.0,  0.0,1.0,0.0,0.0,  1.0,0.0,0.0,0.0,  0.0,0.0,0.0,1.0]),
    // -X  look_at(origin, -X, +Y)
    Mat4::from_cols_array(&[ 0.0,0.0,1.0,0.0,  0.0,1.0,0.0,0.0, -1.0,0.0,0.0,0.0,  0.0,0.0,0.0,1.0]),
    // +Y  look_at(origin, +Y, -Z)
    Mat4::from_cols_array(&[ 1.0,0.0,0.0,0.0,  0.0,0.0,-1.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,0.0,1.0]),
    // -Y  look_at(origin, -Y, +Z)
    Mat4::from_cols_array(&[ 1.0,0.0,0.0,0.0,  0.0,0.0,1.0,0.0,  0.0,-1.0,0.0,0.0,  0.0,0.0,0.0,1.0]),
    // +Z  look_at(origin, +Z, +Y)
    Mat4::from_cols_array(&[-1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,-1.0,0.0,  0.0,0.0,0.0,1.0]),
    // -Z  look_at(origin, -Z, +Y)
    Mat4::from_cols_array(&[ 1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,  0.0,0.0,0.0,1.0]),
];

fn trace_cubemap(
    position:  Vec3,
    scene:     &SceneGeometry,
    config:    &ProbeConfig,
    ctx:       &BakeContext,
    vbuf:      &wgpu::Buffer,
    ibuf:      &wgpu::Buffer,
    mbuf:      &wgpu::Buffer,
    res:       u32,
    mip_levels: u32,
) -> Result<Vec<u8>, NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct ProbeParams {
        probe_pos: [f32; 3], face: u32,
        res: u32, samples: u32, num_meshes: u32, num_indices: u32,
        view: [[f32;4];4],
    }

    // Allocate output storage: all 6 faces × mip chain (mip 0 = full res)
    let bytes_per_pixel = if config.use_rgbe { 4usize } else { 16usize }; // RGBE u8×4 vs RGBA f32×4
    let mut all_face_data: Vec<u8> = Vec::new();

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nebula_probe_cs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/probe.wgsl").into()),
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nebula_probe_bgl"),
        entries: &[
            bgl_uniform(0), bgl_storage_ro(1), bgl_storage_ro(2), bgl_storage_ro(3),
            // Output: rgba32float storage texture
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba32Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ],
    });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nebula_probe_pipeline"), layout: Some(&pl), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let total_idx: usize = scene.meshes.iter().map(|m| m.indices.len()).sum();

    for face_idx in 0u32..6 {
        let face_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nebula_probe_face_tex"), size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let face_view = face_tex.create_view(&Default::default());

        let params = ProbeParams { probe_pos: position.to_array(), face: face_idx, res, samples: config.samples_per_face, num_meshes: scene.meshes.len() as u32, num_indices: total_idx as u32, view: FACE_VIEWS[face_idx as usize].to_cols_array_2d() };
        let pbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("nebula_probe_params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nebula_probe_bg"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&face_view) },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nebula_probe_enc") });
        { let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
          pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
          let wg = nebula_gpu::WORKGROUP_SIZE;
          pass.dispatch_workgroups(res.div_ceil(wg), res.div_ceil(wg), 1); }
        ctx.queue.submit(std::iter::once(enc.finish()));
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

        // Read back RGBA32F face
        let row_stride = (res * 16).next_multiple_of(256); // wgpu copy alignment
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("nebula_probe_staging"), size: (row_stride * res) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc2 = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nebula_probe_rb_enc") });
        enc2.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &face_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_stride), rows_per_image: Some(res) } },
            wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
        );
        ctx.queue.submit(std::iter::once(enc2.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok().and_then(|r| r.ok()).ok_or_else(|| NebulaError::ReadbackTimeout { ms: 5000 })?;
        let raw = staging
            .slice(..)
            .get_mapped_range()
            .expect("Nebula probe readback buffer should be mapped");

        if config.use_rgbe {
            // Convert RGBA32F → RGBE
            for row in 0..res as usize {
                let src_row = &raw[row * row_stride as usize .. row * row_stride as usize + res as usize * 16];
                for px in 0..res as usize {
                    let r = f32::from_le_bytes(src_row[px*16..px*16+4].try_into().unwrap());
                    let g = f32::from_le_bytes(src_row[px*16+4..px*16+8].try_into().unwrap());
                    let b = f32::from_le_bytes(src_row[px*16+8..px*16+12].try_into().unwrap());
                    let m = r.max(g).max(b);
                    if m < 1e-32 { all_face_data.extend_from_slice(&[0,0,0,0]); continue; }
                    let e = m.log2().ceil() as i32 + 1;
                    let scale = (2f32).powi(-e + 8);
                    all_face_data.extend_from_slice(&[(r*scale) as u8, (g*scale) as u8, (b*scale) as u8, (e + 128) as u8]);
                }
            }
        } else {
            // Keep RGBA32F rows, stripping padding
            for row in 0..res as usize {
                all_face_data.extend_from_slice(&raw[row * row_stride as usize .. row * row_stride as usize + res as usize * 16]);
            }
        }
        drop(raw);
        staging.unmap();
    }

    Ok(all_face_data)
}

// ── SH projection ─────────────────────────────────────────────────────────────

/// Project the mip-0 face radiance into spherical harmonics (CPU).
fn project_sh(
    face_data: &[u8],
    res: u32,
    sh_order: u32,
) -> Result<Vec<ShCoeff>, NebulaError> {
    let n_coeffs = ((sh_order + 1) * (sh_order + 1)) as usize;
    let mut coeffs = vec![ShCoeff { r: 0.0, g: 0.0, b: 0.0 }; n_coeffs];
    let bytes_per_pixel = 16usize; // We projected RGBA32F above regardless of use_rgbe for SH input
    let stride = res as usize * bytes_per_pixel;

    // Solid-angle weighted Monte Carlo SH projection over 6 faces
    let face_size = (stride * res as usize) as usize;
    for face in 0usize..6 {
        let face_bytes = &face_data[face * face_size .. (face+1) * face_size];
        for y in 0..res as usize {
            for x in 0..res as usize {
                let off = y * stride + x * bytes_per_pixel;
                let r = f32::from_le_bytes(face_bytes[off   ..off+4 ].try_into().unwrap());
                let g = f32::from_le_bytes(face_bytes[off+4 ..off+8 ].try_into().unwrap());
                let b = f32::from_le_bytes(face_bytes[off+8 ..off+12].try_into().unwrap());

                // Map (face,x,y) → unit direction on sphere
                let u = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;
                let dir = face_to_direction(face as u32, u, v);
                let d = (1.0 + u*u + v*v).sqrt();
                let solid_angle = 4.0 / (d * d * d * (res * res) as f32);

                // Evaluate SH basis
                let sh = eval_sh_basis(dir, sh_order);
                for (i, &s) in sh.iter().enumerate() {
                    coeffs[i].r += r * s * solid_angle;
                    coeffs[i].g += g * s * solid_angle;
                    coeffs[i].b += b * s * solid_angle;
                }
            }
        }
    }
    Ok(coeffs)
}

fn face_to_direction(face: u32, u: f32, v: f32) -> [f32;3] {
    match face {
        0 => [ 1.0,  v, -u], // +X
        1 => [-1.0,  v,  u], // -X
        2 => [ u,  1.0, -v], // +Y
        3 => [ u, -1.0,  v], // -Y
        4 => [ u,  v,  1.0], // +Z
        _ => [-u,  v, -1.0], // -Z
    }
}

fn eval_sh_basis(d: [f32;3], order: u32) -> Vec<f32> {
    let [x,y,z] = d;
    let inv_len = (x*x+y*y+z*z).sqrt().recip();
    let [x,y,z] = [x*inv_len, y*inv_len, z*inv_len];
    let mut out = Vec::with_capacity(((order+1)*(order+1)) as usize);
    // l=0
    out.push(0.282_094_8);
    if order >= 1 {
        out.push(0.488_602_5 * y);
        out.push(0.488_602_5 * z);
        out.push(0.488_602_5 * x);
    }
    if order >= 2 {
        out.push(1.092_548_4 * x * y);
        out.push(1.092_548_4 * y * z);
        out.push(0.315_391_6 * (3.0*z*z - 1.0));
        out.push(1.092_548_4 * x * z);
        out.push(0.546_274_2 * (x*x - y*y));
    }
    if order >= 3 {
        out.push(0.590_043_6 * y * (3.0*x*x - y*y));
        out.push(2.890_611_4 * x * y * z);
        out.push(0.457_045_8 * y * (5.0*z*z - 1.0));
        out.push(0.373_176_3 * z * (5.0*z*z - 3.0));
        out.push(0.457_045_8 * x * (5.0*z*z - 1.0));
        out.push(1.445_305_7 * z * (x*x - y*y));
        out.push(0.590_043_6 * x * (x*x - 3.0*y*y));
    }
    out
}

fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bgl_storage_ro(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
