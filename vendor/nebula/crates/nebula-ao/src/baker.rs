use async_trait::async_trait;
use nebula_core::{
    context::BakeContext, error::NebulaError, progress::ProgressReporter,
    scene::SceneGeometry, traits::BakePass,
};
use nebula_gpu::texture::{BakeTexture, TextureFormat2D};
use crate::{config::AoConfig, output::AoOutput};

#[derive(Default)]
pub struct AoBaker;

#[async_trait(?Send)]
impl BakePass for AoBaker {
    type Input  = AoConfig;
    type Output = AoOutput;

    fn name(&self) -> &'static str { "ao" }

    async fn execute(
        &self,
        scene:    &SceneGeometry,
        config:   &AoConfig,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<AoOutput, NebulaError> {
        let res = config.resolution.clamp(64, nebula_gpu::MAX_TEXTURE_DIM);
        reporter.begin("ao", 3);

        reporter.step("ao", 0, "uploading scene geometry");
        let (vbuf, ibuf, mbuf) = upload_geometry(scene, ctx)?;

        reporter.step("ao", 1, &format!("baking {}×{} AO ({} rays)", res, res, config.ray_count));
        let ao_tex = BakeTexture::new(
            &ctx.device, "nebula_ao", res, res, TextureFormat2D::R32F, 1,
            wgpu::TextureUsages::empty(),
        );
        dispatch_ao(scene, config, ctx, &ao_tex, &vbuf, &ibuf, &mbuf)?;

        reporter.step("ao", 2, "reading back");
        let texels = ao_tex.read_back(&ctx.device, &ctx.queue);
        let config_json = serde_json::to_string(config).unwrap_or_default();

        reporter.finish("ao", true, "done");
        Ok(AoOutput { width: res, height: res, texels, config_json })
    }
}

fn upload_geometry(
    scene: &SceneGeometry,
    ctx:   &BakeContext,
) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer), NebulaError> {
    use wgpu::util::DeviceExt;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuVert { pos: [f32;3], _p: f32, normal: [f32;3], _p2: f32, lm_uv: [f32;2], _p3: [f32;2] }

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct GpuMesh { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: [[f32;4];4] }

    let mut verts: Vec<GpuVert> = Vec::new();
    let mut idxs:  Vec<u32>     = Vec::new();
    let mut meshes: Vec<GpuMesh> = Vec::new();

    for mesh in &scene.meshes {
        let vb = verts.len() as u32;
        let ib = idxs.len()  as u32;
        let fallback: Vec<[f32;2]> = mesh.uvs.clone();
        let lm = mesh.lightmap_uvs.as_ref().unwrap_or(&fallback);
        for i in 0..mesh.positions.len() {
            verts.push(GpuVert {
                pos:    mesh.positions[i], _p: 0.0,
                normal: mesh.normals.get(i).copied().unwrap_or([0.0,1.0,0.0]), _p2: 0.0,
                lm_uv:  lm.get(i).copied().unwrap_or([0.0;2]), _p3: [0.0;2],
            });
        }
        idxs.extend(mesh.indices.iter().map(|&i| i + vb));
        meshes.push(GpuMesh { idx_off: ib, idx_cnt: mesh.indices.len() as u32, vert_off: vb, _p: 0, xform: mesh.world_transform.0.to_cols_array_2d() });
    }

    let mk = |label, data: &[u8]| ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE });
    Ok((
        mk("nebula_ao_vbuf",  bytemuck::cast_slice(&verts)),
        mk("nebula_ao_ibuf",  bytemuck::cast_slice(&idxs)),
        mk("nebula_ao_mbuf",  bytemuck::cast_slice(&meshes)),
    ))
}

fn dispatch_ao(
    scene:   &SceneGeometry,
    config:  &AoConfig,
    ctx:     &BakeContext,
    ao_tex:  &BakeTexture,
    vbuf:    &wgpu::Buffer,
    ibuf:    &wgpu::Buffer,
    mbuf:    &wgpu::Buffer,
) -> Result<(), NebulaError> {
    use bytemuck::{Pod, Zeroable};
    use wgpu::util::DeviceExt;

    #[repr(C)] #[derive(Copy,Clone,Pod,Zeroable)]
    struct AoParams { res: u32, ray_count: u32, max_dist: f32, bias: f32, num_meshes: u32, num_indices: u32, seed: u32, _p: u32 }

    let total_idx: usize = scene.meshes.iter().map(|m| m.indices.len()).sum();
    let p = AoParams { res: config.resolution, ray_count: config.ray_count, max_dist: config.max_distance, bias: config.bias, num_meshes: scene.meshes.len() as u32, num_indices: total_idx as u32, seed: 0xDEADBEEF, _p: 0 };

    let pbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("nebula_ao_params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nebula_ao_bgl"),
        entries: &[
            bgl_uniform(0), bgl_storage_ro(1), bgl_storage_ro(2), bgl_storage_ro(3),
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: ao_tex.format, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nebula_ao_bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&ao_tex.view) },
        ],
    });

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nebula_ao_cs"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ao.wgsl").into()) });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("nebula_ao_pipeline"), layout: Some(&pl), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nebula_ao_enc") });
    { let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nebula_ao_pass"), timestamp_writes: None });
      pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
      let wg = nebula_gpu::WORKGROUP_SIZE; let r = config.resolution;
      pass.dispatch_workgroups(r.div_ceil(wg), r.div_ceil(wg), 1); }
    ctx.queue.submit(std::iter::once(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    Ok(())
}

fn bgl_uniform(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bgl_storage_ro(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
