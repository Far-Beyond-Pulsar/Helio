use super::*;

fn validate(label: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
}

#[test]
fn production_shadow_shaders_validate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    for path in [
        "helio-pass-shadow-matrix/shaders/shadow_casters.wgsl",
        "helio-pass-shadow-matrix/shaders/shadow_matrices.wgsl",
        "helio-pass-shadow/shaders/shadow.wgsl",
        "helio-pass-shadow/shaders/depth_clear.wgsl",
        "helio-pass-shadow/shaders/shadow_transmittance.wgsl",
        "helio-pass-shadow-dirty/shaders/shadow_dirty.wgsl",
        "helio-pass-deferred-light/shaders/deferred_lighting.wgsl",
        "helio-pass-volumetric-fog/shaders/volumetric_fog.wgsl",
        "helio-pass-flare/shaders/lens_response.wgsl",
    ] {
        let mut source = std::fs::read_to_string(root.join(path))
            .unwrap()
            .replace("__PP_TAIL_VEC4__", "64");
        if source.contains("//!use pbr_eval") {
            source = format!(
                "{}\n{}",
                std::fs::read_to_string(root.join("../../helio-mats/shaders/pbr_eval.wgsl"))
                    .unwrap(),
                source
            );
        }
        validate(path, &helio_core::shader::resolve(&source));
    }
    let prefix = "const USE_RAY_TRANSMISSION:bool=false; const USE_TILE_PRESAMPLING:bool=false; alias Visibility=f32; alias VisibilityCache=vec4f; fn visibility_nonzero(v:f32)->bool{return v>0.0;} fn visibility_missing(v:f32)->bool{return v<0.0;} fn visibility_from_rgb(v:vec3f)->f32{return v.x;}";
    let mut source = prefix.to_string();
    for name in ["common", "lighting", "shadows", "sample"] {
        source.push_str(
            &std::fs::read_to_string(root.join(format!("helio-pass-hlfs/shaders/{name}.wgsl")))
                .unwrap(),
        );
    }
    validate("HLFS raster fallback", &source);
}

#[test]
fn configuration_obeys_memory_and_update_budgets() {
    for mb in [1, 4, 16, 40, 160, 640] {
        let b = ShadowBudget {
            memory_bytes: mb * 1024 * 1024,
            ..Default::default()
        }
        .validate()
        .unwrap();
        let size = b.atlas_size(16384);
        assert!(u64::from(size).pow(2) * 10 <= b.memory_bytes);
        assert!(size.is_power_of_two());
    }
    assert!(ShadowBudget {
        updates_per_frame: 1,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(ShadowBudget {
        max_resolution: 300,
        ..Default::default()
    }
    .validate()
    .is_err());
    assert_eq!(std::mem::size_of::<GpuShadowMatrix>(), 96);
    assert_eq!(std::mem::size_of::<ShadowResident>(), 128);
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass: ShadowMatrixPass,
    lights: wgpu::Buffer,
    params: CasterParams,
}
impl Gpu {
    fn new(rows: u32, capacity: u32) -> Self {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("GPU required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("{e:?}")));
        let buf = |size| {
            buffer(
                &device,
                "test",
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )
        };
        let lights = buf(rows as u64 * 128);
        let matrices = buf(MAX_SHADOW_FACES as u64 * 96);
        let camera = buf(736);
        let pass = ShadowMatrixPass::new(
            &device,
            &lights,
            &matrices,
            &camera,
            &buf(1024),
            &buf(1024),
            2048,
        );
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        let params = CasterParams {
            row_count: rows,
            caster_capacity: capacity,
            nonce: 1,
            atlas_size: 2048,
            view_proj: identity,
            inv_view_proj: identity,
            camera: [0., 0., 0., 100.],
            tuning: [1024., 2048., 0.2, 0.],
        };
        Self {
            device,
            queue,
            pass,
            lights,
            params,
        }
    }
    fn read(&self, src: &wgpu::Buffer) -> Vec<u8> {
        let staging = buffer(
            &self.device,
            "read",
            src.size(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut e = self.device.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(src, 0, &staging, 0, src.size());
        self.queue.submit([e.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(r) = rx.try_recv() {
                r.unwrap();
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "GPU readback timed out"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
        staging.unmap();
        bytes
    }
    fn run(&self, rows: &[[u32; 32]], active: &ResidencyTable) -> ResidencyTable {
        self.queue
            .write_buffer(&self.lights, 0, bytemuck::cast_slice(rows));
        self.queue
            .write_buffer(&self.pass.committed, 0, bytemuck::bytes_of(active));
        self.queue.write_buffer(
            &self.pass.caster_params_buf,
            0,
            bytemuck::bytes_of(&self.params),
        );
        let mut e = self.device.create_command_encoder(&Default::default());
        self.pass
            .dispatch(&mut e, 0, self.params.row_count.div_ceil(64));
        self.pass.dispatch(&mut e, 1, 1);
        self.pass.dispatch(&mut e, 2, 1);
        self.queue.submit([e.finish()]);
        bytemuck::pod_read_unaligned(&self.read(&self.pass.proposed))
    }
    fn commit(&self, table: &ResidencyTable) -> Vec<u32> {
        self.queue
            .write_buffer(&self.pass.committed, 0, bytemuck::bytes_of(table));
        let mut e = self.device.create_command_encoder(&Default::default());
        self.pass
            .dispatch(&mut e, 3, self.params.row_count.div_ceil(64));
        self.queue.submit([e.finish()]);
        bytemuck::cast_slice::<u8, u32>(&self.read(&self.lights)).to_vec()
    }
}
fn light(x: f32, radius: f32, kind: u32) -> [u32; 32] {
    let mut l = [0u32; 32];
    l[0] = x.to_bits();
    l[2] = 0.5f32.to_bits();
    l[3] = radius.to_bits();
    l[5] = (-1f32).to_bits();
    l[11] = 1f32.to_bits();
    l[13] = kind;
    l
}
fn no_overlap(t: &ResidencyTable, extent: u32) {
    let tiles: Vec<_> = t
        .residents
        .iter()
        .filter(|r| r.owner != 0)
        .flat_map(|r| r.tiles)
        .filter(|t| t.size > 0)
        .collect();
    for (i, a) in tiles.iter().enumerate() {
        assert!(a.x + a.size <= extent && a.y + a.size <= extent);
        for b in &tiles[i + 1..] {
            assert!(
                a.x + a.size <= b.x
                    || b.x + b.size <= a.x
                    || a.y + a.size <= b.y
                    || b.y + b.size <= a.y,
                "overlapping tiles: {a:?} {b:?}"
            );
        }
    }
}
#[test]
fn gpu_ranking_tiers_hysteresis_fades_faces_and_fallback() {
    let mut gpu = Gpu::new(600, 64);
    let mut rows = vec![[0u32; 32]; 600];
    // Hundreds of requests, including sparse rows and offscreen high intensity lights.
    for i in 0..500 {
        rows[i] = light(0., 0.01 + (i as f32) * 0.0001, 2);
    }
    rows[599] = light(100., 1., 2);
    rows[599][11] = 1e9f32.to_bits();
    let table = gpu.run(&rows, &ResidencyTable::default());
    assert_eq!(table.header[0], 64);
    assert_eq!(table.residents.iter().filter(|r| r.owner > 0).count(), 64);
    assert!(table
        .residents
        .iter()
        .filter(|r| r.owner > 0)
        .all(|r| r.owner >= 437 && r.owner <= 500));
    no_overlap(&table, 2048);
    let committed = gpu.commit(&table);
    assert_eq!(committed[12], u32::MAX);
    assert_eq!(
        committed[15] & 15,
        15,
        "nonresident preserves shadow request and RT intent"
    );
    assert_eq!(committed[599 * 32 + 12], u32::MAX);
    // Small perturbation cannot evict an incumbent; substantial change fades it first.
    gpu.params.caster_capacity = 1;
    rows.fill([0; 32]);
    rows[0] = light(0., 0.1, 2);
    rows[1] = light(0., 0.105, 2);
    let mut active = ResidencyTable::default();
    active.residents[0] = ShadowResident {
        owner: 1,
        resolution: 128,
        strength: 1.,
        target: 65535,
        tiles: [ShadowTile {
            x: 0,
            y: 0,
            size: 128,
            valid: 0,
        }; 6],
        ..Default::default()
    };
    for f in 1..6 {
        active.residents[0].tiles[f] = Default::default();
    }
    rows[0][15] = 12;
    rows[0][12] = 0;
    let stable = gpu.run(&rows, &active);
    assert_eq!(stable.residents[0].owner, 1);
    assert!(stable.residents[0].target > 0);
    rows[1][3] = 0.3f32.to_bits();
    let fading = gpu.run(&rows, &active);
    assert_eq!(fading.residents[0].owner, 1);
    assert_eq!(fading.residents[0].target, 0);
    assert_eq!(fading.residents[0].strength, 1.);
    active = fading;
    active.residents[0].strength = 0.;
    let replaced = gpu.run(&rows, &active);
    assert_eq!(replaced.residents[0].owner, 2);
    // Quantized coverage tiers and an author cap.
    gpu.params.caster_capacity = 4;
    rows.fill([0; 32]);
    for (i, r) in [0.08, 0.2, 0.4, 0.8].iter().enumerate() {
        rows[i] = light(0., *r, 2);
    }
    rows[3][15] = 8 << 16;
    let tiers = gpu.run(&rows, &ResidencyTable::default());
    for (owner, res) in [(1, 128), (2, 256), (3, 512), (4, 256)] {
        assert_eq!(
            tiers
                .residents
                .iter()
                .find(|r| r.owner == owner)
                .unwrap()
                .resolution,
            res
        );
    }
    no_overlap(&tiers, 2048);
    // Tier hysteresis retains 256 just above its ordinary 256-pixel cutoff.
    let mut active = ResidencyTable::default();
    active.residents[0] = ShadowResident {
        owner: 1,
        resolution: 256,
        strength: 1.,
        ..Default::default()
    };
    rows.fill([0; 32]);
    rows[0] = light(0., 0.26, 2);
    rows[0][15] = 12;
    let tier = gpu.run(&rows, &active);
    assert_eq!(tier.residents[0].resolution, 256);
    assert!(tier.residents[0].target > 0);
    // Outside the receiver frustum, only the cube face pointing back toward it is resident.
    rows.fill([0; 32]);
    rows[0] = light(3., 2.1, 1);
    let faces = gpu.run(&rows, &ResidencyTable::default());
    let r = faces.residents.iter().find(|r| r.owner == 1).unwrap();
    assert!(r.tiles[1].size > 0);
    assert_eq!(r.tiles[0].size, 0);
    assert!(r.tiles.iter().filter(|t| t.size > 0).count() < 6);
    // Explicitly disabling both caster sets yields no map; RT explicit-off stays off.
    rows.fill([0; 32]);
    rows[0] = light(0., 0.5, 2);
    rows[0][15] = 48 | 1;
    let disabled = gpu.run(&rows, &ResidencyTable::default());
    assert_eq!(disabled.header[0], 0);
    let flags = gpu.commit(&disabled);
    assert_eq!(flags[15] & 3, 1);
}

#[test]
fn gpu_tile_sampling_fades_and_honors_author_policy() {
    let gpu = Gpu::new(1, 1);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow test depth"),
        size: wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let attachment = texture.create_view(&Default::default());
    let sampled = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let mut e = gpu.device.create_command_encoder(&Default::default());
    {
        let _pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &attachment,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.25),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    gpu.queue.submit([e.finish()]);
    let mut faces = [GpuShadowMatrix {
        light_view_proj: [0.; 16],
        atlas: [0., 0., 1., 1.],
        policy: [0, 0, 128, 2],
    }; 6];
    faces[1].atlas[3] = 0.5;
    faces[2].policy[3] = 1;
    faces[3].policy[1] = 16;
    faces[4].policy[1] = 32;
    faces[5].atlas[2] = 0.;
    let matrices = buffer(
        &gpu.device,
        "sampling matrices",
        6 * 96,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    gpu.queue
        .write_buffer(&matrices, 0, bytemuck::cast_slice(&faces));
    let output = buffer(
        &gpu.device,
        "sampling result",
        6 * 8,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let production = include_str!("../../helio-pass-deferred-light/shaders/deferred_lighting.wgsl");
    let helpers = production
        .split("// Logical face metadata")
        .nth(1)
        .unwrap()
        .split("// Bounded screen-space fallback")
        .next()
        .unwrap();
    let helpers = &helpers[helpers.find("fn budget_resolution").unwrap()..];
    let source = format!(
        "{}\n{}\n{}",
        r#"
        struct LightMatrix { mat:mat4x4f,atlas:vec4f,policy:vec4u }
        @group(0) @binding(0) var<storage,read> shadow_matrices:array<LightMatrix>;
        @group(0) @binding(1) var shadow_atlas:texture_depth_2d_array;
        @group(0) @binding(2) var static_shadow_atlas:texture_depth_2d_array;
        @group(0) @binding(3) var shadow_sampler:sampler_comparison;
        @group(0) @binding(4) var<storage,read_write> result:array<vec2f>;
    "#,
        helpers,
        r#"
        @compute @workgroup_size(1) fn main() {
            for(var i=0u;i<6u;i++) {result[i]=vec2f(budget_compare_dynamic(vec2f(0.5),i,0.5),budget_compare_static(vec2f(0.5),i,0.5));}
        }
    "#
    );
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&sampled),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&sampled),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut e = gpu.device.create_command_encoder(&Default::default());
    {
        let mut p = e.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipeline);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit([e.finish()]);
    let values = gpu.read(&output);
    let values: &[f32] = bytemuck::cast_slice(&values);
    assert_eq!(values, &[0., 0., 0.5, 0.5, 1., 1., 0., 1., 1., 0., 1., 1.]);
}
