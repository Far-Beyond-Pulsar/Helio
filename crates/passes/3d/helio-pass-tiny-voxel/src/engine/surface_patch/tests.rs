use super::*;
use crate::world::Edit;

mod primary;

#[test]
fn queued_revisions_teleports_undo_and_grid_changes_never_publish_foreign_materials() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU required");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: adapter.features() & wgpu::Features::CONSERVATIVE_RASTERIZATION,
                ..Default::default()
            })
            .await
            .unwrap();
        let mut patch = Patch::with_mesh(&device, true);
        patch.stats.enabled = true;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("actual surface patch lookup publication audit"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    crate::SHADER,
                    crate::surface_cache::GPU_SHADER,
                    include_str!("../surface_patch.wgsl").to_owned()
                        + r#"
@group(0) @binding(34) var<storage,read> query_cells:array<vec4<i32>>;
@compute @workgroup_size(64)
fn audit(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&query_cells) {return;}
    let cell=query_cells[id.x].xyz;
    let material=stored_cached_material(stored_cached_page(cell),cell);
    primary_hits[id.x]=Hit(cell,material,vec3<f32>(0.0),0.0);
}"#
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("audit"),
            compilation_options: Default::default(),
            cache: None,
        });
        let empty = World::default();
        let mut solid = empty.clone();
        solid
            .apply_edit(Edit {
                cell: [0, 70_000_000, -80],
                radius: 500.0,
                material: 2,
            })
            .unwrap();
        let mut marked = solid.clone();
        marked
            .apply_edit(Edit {
                cell: [1000, 70_000_000, -80],
                radius: 5.0,
                material: 3,
            })
            .unwrap();
        marked
            .apply_edit(Edit {
                cell: [1000, 70_000_000, -80],
                radius: 1.2,
                material: 0,
            })
            .unwrap();
        let mut coarse = marked.clone();
        coarse.set_voxel_size(0.3).unwrap();
        let mut metre = marked.clone();
        metre.set_voxel_size(1.0).unwrap();
        let mut checked = 0usize;
        for (stage, (world, x)) in [
            (empty, 0),
            (solid.clone(), 0),
            (marked, 1000),
            (coarse, 1000),
            (metre, 1000),
            (solid, 0),
        ]
        .into_iter()
        .enumerate()
        {
            let world = Arc::new(world);
            let mut params: Params = bytemuck::Zeroable::zeroed();
            params.origin = [x, 70_000_000, 0, 0];
            params.fraction = [0.25, 0.33, 0.5, 0.0];
            params.forward = [0.0, 0.0, -1.0, 0.0];
            let start = Instant::now();
            loop {
                let before = patch.stats();
                patch.update(&queue, &world, &params);
                let stats = patch.stats();
                assert_eq!(
                    stats.mesh_enabled,
                    device
                        .features()
                        .contains(wgpu::Features::CONSERVATIVE_RASTERIZATION)
                );
                if stats.mesh_enabled {
                    assert_eq!(stats.mesh_ready + stats.mesh_rejected, stats.ready);
                }
                if before.revision == stats.revision {
                    assert!(stats.ready - before.ready <= UPLOADS);
                    assert!(
                        stats.uploaded_bytes - before.uploaded_bytes
                            <= (UPLOADS * (WORDS * 4 + 4)) as u64
                    );
                }
                let mut cells = Vec::<[i32; 4]>::new();
                for tile in 0..TILES {
                    let key = Key([
                        stats.low[0] + (tile % 8) as i32,
                        stats.low[1] + (tile / 8 % 8) as i32,
                        stats.low[2] + (tile / 64) as i32,
                    ]);
                    let low = key.low(world.voxel_step());
                    for corner in 0..8 {
                        let c = std::array::from_fn::<_, 3, _>(|a| {
                            low[a] + ((corner >> a) & 1) * 31 * world.voxel_step() as i32
                        });
                        cells.push([c[0], c[1], c[2], 0]);
                    }
                }
                let queries = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&cells),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let size = cells.len() as u64 * 32;
                let output =
                    super::super::terrain::buffer(&device, "patch publication results", size);
                let read = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let buffers = [
                    (9, &output),
                    (31, &patch.settings),
                    (32, &patch.directory),
                    (33, &patch.words),
                    (34, &queries),
                ];
                let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &buffers.map(|(binding, b)| wgpu::BindGroupEntry {
                        binding,
                        resource: b.as_entire_binding(),
                    }),
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(cells.len() as u32 / 64, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &read, 0, size);
                queue.submit([encoder.finish()]);
                let (tx, rx) = mpsc::channel();
                read.slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = read.slice(..).get_mapped_range().unwrap();
                let hits = bytemuck::cast_slice::<u8, u32>(&bytes);
                let mut resident = 0;
                for (i, c) in cells.iter().enumerate() {
                    let material = hits[i * 8 + 3];
                    if material == u32::MAX {
                        continue;
                    }
                    assert_eq!(
                        material,
                        world.material([c[0], c[1], c[2]]),
                        "stage {stage} revision {} cell {c:?}",
                        stats.revision
                    );
                    resident += 1;
                    checked += 1;
                }
                assert_eq!(resident, stats.ready * 8);
                drop(bytes);
                read.unmap();
                // Supersede early stages while more work/results remain queued.
                // The last source must eventually fill the complete directory.
                if stats.ready >= if stage == 5 { TILES } else { 16 } {
                    break;
                }
                assert!(
                    start.elapsed().as_secs() < 20,
                    "patch publication stalled at {stage}: {stats:?}"
                );
                std::thread::yield_now();
            }
        }
        eprintln!(
            "SURFACE_PATCH_PUBLICATION checked_materials={checked} final_revision={}",
            patch.stats().revision
        );
    });
}
