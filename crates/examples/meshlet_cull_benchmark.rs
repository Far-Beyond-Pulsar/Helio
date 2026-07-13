//! Headless virtual-geometry culling benchmark.
//!
//! Holds total visible meshlet work constant while changing how that work is
//! distributed across objects. This exposes occupancy problems in the current
//! one-workgroup-per-object publication path.
//!
//! Run with:
//! `cargo run --release -p examples --bin meshlet_cull_benchmark`

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use libhelio::{
    GpuCameraUniforms, GpuInstanceData, GpuMeshletEntry, GpuVgDraw, GpuVgObject,
    GpuVgWorkItem, VG_CULL_MESHLETS_PER_WORK_ITEM,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const CULL_SHADER: &str = include_str!("../helio-pass-virtual-geometry/shaders/vg_cull.wgsl");
const TOTAL_MESHLETS: u32 = 1_048_576;
const DEFAULT_WARMUP: u32 = 5;
const DEFAULT_SAMPLES: u32 = 20;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    object_count: u32,
    screen_width: u32,
    screen_height: u32,
    hiz_mip_count: u32,
    draw_capacity: u32,
    lod_error_threshold_px: f32,
    object_dispatch_width: u32,
    work_item_count: u32,
    work_dispatch_width: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceCullData {
    max_scale: f32,
    min_scale: f32,
    cone_cull_enabled: u32,
    valid_transform: u32,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    object_count: u32,
    meshlets_per_object: u32,
}

struct CaseBuffers {
    select_bind_group: wgpu::BindGroup,
    cull_bind_group: wgpu::BindGroup,
    draw_count: wgpu::Buffer,
    draw_count_readback: wgpu::Buffer,
    object_dispatch_width: u32,
    object_dispatch_height: u32,
    work_dispatch_width: u32,
    work_dispatch_height: u32,
    expected_attempts: u32,
    expected_overflow: u32,
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no GPU adapter available");

    let timestamp_features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    assert!(
        adapter.features().contains(timestamp_features),
        "meshlet benchmark requires encoder timestamp queries"
    );
    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Meshlet Cull Benchmark Device"),
            required_features: timestamp_features,
            required_limits: limits.clone(),
            ..Default::default()
        })
        .await
        .expect("failed to create benchmark device");
    device.on_uncaptured_error(std::sync::Arc::new(|error| {
        panic!("[GPU] {error:?}");
    }));

    let info = adapter.get_info();
    let warmup = env_u32("HELIO_BENCH_WARMUP", DEFAULT_WARMUP);
    let samples = env_u32("HELIO_BENCH_SAMPLES", DEFAULT_SAMPLES).max(1);
    println!("adapter,{},backend,{:?}", info.name, info.backend);
    println!("warmup,{warmup},samples,{samples},total_meshlets,{TOTAL_MESHLETS}");
    println!("case,objects,meshlets_per_object,median_ms,p95_ms,meshlets_per_second");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("VG Cull Benchmark Shader"),
        source: wgpu::ShaderSource::Wgsl(CULL_SHADER.into()),
    });
    let select_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("VG Object Select Benchmark Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("cs_select_objects"),
        compilation_options: Default::default(),
        cache: None,
    });
    let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("VG Meshlet Cull Benchmark Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("cs_cull_meshlets"),
        compilation_options: Default::default(),
        cache: None,
    });
    let select_bind_group_layout = select_pipeline.get_bind_group_layout(0);
    let cull_bind_group_layout = cull_pipeline.get_bind_group_layout(0);
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("VG Cull Benchmark Timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VG Cull Benchmark Query Resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VG Cull Benchmark Query Readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let cases = [
        Case { name: "single_huge", object_count: 1, meshlets_per_object: TOTAL_MESHLETS },
        Case { name: "few_huge", object_count: 16, meshlets_per_object: TOTAL_MESHLETS / 16 },
        Case { name: "balanced", object_count: 4096, meshlets_per_object: TOTAL_MESHLETS / 4096 },
        Case { name: "many_small", object_count: 65_536, meshlets_per_object: TOTAL_MESHLETS / 65_536 },
    ];

    for case in cases {
        if !case_fits_limits(case, &limits) {
            println!("{},SKIPPED,storage_binding_limit", case.name);
            continue;
        }
        let buffers = create_case_buffers(
            &device,
            &queue,
            &select_bind_group_layout,
            &cull_bind_group_layout,
            case,
            &limits,
            TOTAL_MESHLETS,
        );

        for _ in 0..warmup {
            let _ = dispatch_and_time(
                &device,
                &queue,
                &select_pipeline,
                &cull_pipeline,
                &buffers,
                &query_set,
                &query_resolve,
                &query_readback,
            );
        }

        let mut timings_ms = Vec::with_capacity(samples as usize);
        for _ in 0..samples {
            timings_ms.push(dispatch_and_time(
                &device,
                &queue,
                &select_pipeline,
                &cull_pipeline,
                &buffers,
                &query_set,
                &query_resolve,
                &query_readback,
            ));
        }

        let counters = read_draw_counters(&device, &queue, &buffers);
        assert_eq!(
            counters[0], buffers.expected_attempts,
            "{} did not attempt every visible meshlet",
            case.name
        );
        assert_eq!(
            counters[1], buffers.expected_overflow,
            "{} reported an unexpected capacity overflow",
            case.name
        );
        timings_ms.sort_by(f64::total_cmp);
        let median_ms = percentile(&timings_ms, 0.50);
        let p95_ms = percentile(&timings_ms, 0.95);
        let meshlets_per_second = f64::from(TOTAL_MESHLETS) / (median_ms / 1000.0);
        println!(
            "{},{},{},{:.4},{:.4},{:.0}",
            case.name,
            case.object_count,
            case.meshlets_per_object,
            median_ms,
            p95_ms,
            meshlets_per_second,
        );
    }

    let overflow_probe = create_case_buffers(
        &device,
        &queue,
        &select_bind_group_layout,
        &cull_bind_group_layout,
        Case {
            name: "overflow_probe",
            object_count: 4096,
            meshlets_per_object: TOTAL_MESHLETS / 4096,
        },
        &limits,
        TOTAL_MESHLETS - 17,
    );
    let _ = dispatch_and_time(
        &device,
        &queue,
        &select_pipeline,
        &cull_pipeline,
        &overflow_probe,
        &query_set,
        &query_resolve,
        &query_readback,
    );
    let counters = read_draw_counters(&device, &queue, &overflow_probe);
    assert_eq!(counters, [TOTAL_MESHLETS, 17]);
    eprintln!("overflow_probe,attempted={},rejected={}", counters[0], counters[1]);
}

fn case_fits_limits(case: Case, limits: &wgpu::Limits) -> bool {
    let meshlet_bytes = u64::from(case.meshlets_per_object)
        * std::mem::size_of::<GpuMeshletEntry>() as u64;
    let object_bytes = u64::from(case.object_count) * std::mem::size_of::<GpuVgObject>() as u64;
    let instance_bytes =
        u64::from(case.object_count) * std::mem::size_of::<GpuInstanceData>() as u64;
    let indirect_bytes = u64::from(TOTAL_MESHLETS) * 20;
    let metadata_bytes =
        u64::from(TOTAL_MESHLETS) * std::mem::size_of::<GpuVgDraw>() as u64;
    let work_item_count = u64::from(case.object_count)
        * u64::from(case.meshlets_per_object.div_ceil(VG_CULL_MESHLETS_PER_WORK_ITEM));
    let work_item_bytes = work_item_count * std::mem::size_of::<GpuVgWorkItem>() as u64;
    let largest = meshlet_bytes
        .max(object_bytes)
        .max(instance_bytes)
        .max(indirect_bytes)
        .max(metadata_bytes)
        .max(work_item_bytes);
    largest <= u64::from(limits.max_storage_buffer_binding_size)
}

fn create_case_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    select_layout: &wgpu::BindGroupLayout,
    cull_layout: &wgpu::BindGroupLayout,
    case: Case,
    limits: &wgpu::Limits,
    draw_capacity: u32,
) -> CaseBuffers {
    let identity = Mat4::IDENTITY.to_cols_array();
    let camera = GpuCameraUniforms::new(
        Mat4::IDENTITY,
        Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0),
        Vec3::ZERO,
        0.1,
        1000.0,
        0,
        [0.0; 2],
        Mat4::IDENTITY,
    );
    let instances = vec![GpuInstanceData {
        model: identity,
        normal_mat: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        bounds: [0.0, 0.0, -10.0, 1.0],
        mesh_id: 0,
        material_id: 0,
        flags: 0,
        lightmap_index: u32::MAX,
    }; case.object_count as usize];
    let instance_cull = vec![InstanceCullData {
        max_scale: 1.0,
        min_scale: 1.0,
        cone_cull_enabled: 0,
        valid_transform: 1,
    }; case.object_count as usize];
    let meshlets = vec![GpuMeshletEntry {
        center: [0.0, 0.0, -10.0],
        radius: 0.25,
        cone_apex: [0.0, 0.0, -10.0],
        cone_cutoff: 2.0,
        cone_axis: [0.0, 0.0, 1.0],
        lod_error: 0.0,
        first_index: 0,
        index_count: 3,
        vertex_offset: 0,
        instance_index: 0,
    }; case.meshlets_per_object as usize];
    let objects: Vec<_> = (0..case.object_count)
        .map(|instance_index| {
            let mut lod_meshlet_counts = [0; 8];
            lod_meshlet_counts[0] = case.meshlets_per_object;
            GpuVgObject {
                instance_index,
                lod_count: 1,
                max_meshlet_count: case.meshlets_per_object,
                reserved: 0,
                local_bounds: [0.0, 0.0, -10.0, 1.0],
                lod_errors: [0.0; 8],
                lod_first_meshlets: [0; 8],
                lod_meshlet_counts,
            }
        })
        .collect();

    let object_workgroups = case.object_count.div_ceil(VG_CULL_MESHLETS_PER_WORK_ITEM);
    let object_dispatch_width = object_workgroups
        .min(limits.max_compute_workgroups_per_dimension)
        .max(1);
    let object_dispatch_height = object_workgroups.div_ceil(object_dispatch_width);
    let work_items: Vec<_> = (0..case.object_count)
        .flat_map(|object_index| {
            (0..case.meshlets_per_object)
                .step_by(VG_CULL_MESHLETS_PER_WORK_ITEM as usize)
                .map(move |local_meshlet_base| GpuVgWorkItem {
                    object_index,
                    local_meshlet_base,
                })
        })
        .collect();
    let work_item_count = u32::try_from(work_items.len()).expect("benchmark work items exceed u32");
    let work_dispatch_width = work_item_count
        .min(limits.max_compute_workgroups_per_dimension)
        .max(1);
    let work_dispatch_height = work_item_count.div_ceil(work_dispatch_width);
    let cull_uniforms = CullUniforms {
        object_count: case.object_count,
        screen_width: 1920,
        screen_height: 1080,
        hiz_mip_count: 1,
        draw_capacity,
        lod_error_threshold_px: 2.0,
        object_dispatch_width,
        work_item_count,
        work_dispatch_width,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let camera_buffer = init_buffer(device, "Benchmark Camera", bytemuck::bytes_of(&camera), wgpu::BufferUsages::UNIFORM);
    let cull_buffer = init_buffer(device, "Benchmark Cull Uniforms", bytemuck::bytes_of(&cull_uniforms), wgpu::BufferUsages::UNIFORM);
    let meshlet_buffer = init_buffer(device, "Benchmark Meshlets", bytemuck::cast_slice(&meshlets), wgpu::BufferUsages::STORAGE);
    let object_buffer = init_buffer(device, "Benchmark Objects", bytemuck::cast_slice(&objects), wgpu::BufferUsages::STORAGE);
    let instance_buffer = init_buffer(device, "Benchmark Instances", bytemuck::cast_slice(&instances), wgpu::BufferUsages::STORAGE);
    let instance_cull_buffer = init_buffer(device, "Benchmark Instance Cull", bytemuck::cast_slice(&instance_cull), wgpu::BufferUsages::STORAGE);
    let work_item_buffer = init_buffer(device, "Benchmark Work Items", bytemuck::cast_slice(&work_items), wgpu::BufferUsages::STORAGE);
    let indirect_buffer = output_buffer(device, "Benchmark Indirect", u64::from(TOTAL_MESHLETS) * 20);
    let metadata_buffer = output_buffer(device, "Benchmark Draw Metadata", u64::from(TOTAL_MESHLETS) * std::mem::size_of::<GpuVgDraw>() as u64);
    let draw_count = output_buffer(device, "Benchmark Draw And Overflow Counters", 8);
    let draw_count_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Benchmark Draw Count Readback"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let hiz = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Benchmark Far HiZ"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &hiz,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255, 255, 255, 255],
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    let hiz_view = hiz.create_view(&wgpu::TextureViewDescriptor::default());
    let hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Benchmark HiZ Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let select_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("VG Object Select Benchmark Bind Group"),
        layout: select_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cull_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: object_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: instance_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: instance_cull_buffer.as_entire_binding() },
        ],
    });
    let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("VG Cull Benchmark Bind Group"),
        layout: cull_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cull_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: meshlet_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: object_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: instance_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: indirect_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: metadata_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: draw_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&hiz_view) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&hiz_sampler) },
            wgpu::BindGroupEntry { binding: 10, resource: instance_cull_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: work_item_buffer.as_entire_binding() },
        ],
    });

    CaseBuffers {
        select_bind_group,
        cull_bind_group,
        draw_count,
        draw_count_readback,
        object_dispatch_width,
        object_dispatch_height,
        work_dispatch_width,
        work_dispatch_height,
        expected_attempts: TOTAL_MESHLETS,
        expected_overflow: TOTAL_MESHLETS.saturating_sub(draw_capacity),
    }
}

fn dispatch_and_time(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    select_pipeline: &wgpu::ComputePipeline,
    cull_pipeline: &wgpu::ComputePipeline,
    buffers: &CaseBuffers,
    query_set: &wgpu::QuerySet,
    query_resolve: &wgpu::Buffer,
    query_readback: &wgpu::Buffer,
) -> f64 {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("VG Cull Benchmark Encoder"),
    });
    encoder.clear_buffer(&buffers.draw_count, 0, None);
    encoder.write_timestamp(query_set, 0);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VG Cull Benchmark Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(select_pipeline);
        pass.set_bind_group(0, &buffers.select_bind_group, &[]);
        pass.dispatch_workgroups(
            buffers.object_dispatch_width,
            buffers.object_dispatch_height,
            1,
        );
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VG Meshlet Cull Benchmark Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(cull_pipeline);
        pass.set_bind_group(0, &buffers.cull_bind_group, &[]);
        pass.dispatch_workgroups(
            buffers.work_dispatch_width,
            buffers.work_dispatch_height,
            1,
        );
    }
    encoder.write_timestamp(query_set, 1);
    encoder.resolve_query_set(query_set, 0..2, query_resolve, 0);
    encoder.copy_buffer_to_buffer(query_resolve, 0, query_readback, 0, 16);
    queue.submit([encoder.finish()]);

    let ticks = read_mapped::<u64>(device, query_readback, 2);
    ticks[1].saturating_sub(ticks[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1_000_000.0
}

fn read_draw_counters(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &CaseBuffers,
) -> [u32; 2] {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("VG Cull Count Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(&buffers.draw_count, 0, &buffers.draw_count_readback, 0, 8);
    queue.submit([encoder.finish()]);
    let values = read_mapped::<u32>(device, &buffers.draw_count_readback, 2);
    [values[0], values[1]]
}

fn read_mapped<T: Pod + Copy>(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map callback dropped").expect("readback mapping failed");
    let data = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, T>(&data)[..count].to_vec();
    drop(data);
    buffer.unmap();
    values
}

fn init_buffer(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}
