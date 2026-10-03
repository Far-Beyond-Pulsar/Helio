//! Production allocator lifecycle under stranded size-class capacity.
mod common;
use common::*;
use std::collections::HashSet;
use wgpu::util::DeviceExt;

fn segment<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    text.split_once(start).unwrap().1.split_once(end).unwrap().0
}

#[test]
fn pressure_recycles_empty_pages_without_reusing_live_allocations() {
    let Some(gpu) = gpu() else { return };
    const UNITS: u32 = 2048;
    const JOBS: u32 = 1536;
    const LIVE: u32 = 1024;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let buffer = |words: &[u32], usage| {
        gpu.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(words),
                usage,
            })
    };
    let words = |buffer: &wgpu::Buffer| -> Vec<u32> {
        read_buffer(&gpu, buffer, buffer.size())
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    };
    let mut allocator = [0u32; 32];
    allocator[0] = 512;
    allocator[1] = 256;
    allocator[2] = 127;
    allocator[26] = 2; // Two completely free assigned pages.
    allocator[30] = 1; // One still-unassigned page.
    let alloc = buffer(&allocator, storage);
    let pages = buffer(&[512, 0, 256, 1, 127, 2, 0, u32::MAX], storage);
    let mut runs = vec![0; UNITS as usize * 2];
    for r in 0..512 {
        runs[r] = r as u32;
    }
    for r in 0..256 {
        runs[UNITS as usize + r] = 512 + r as u32 * 2;
    }
    for r in 0..127 {
        runs[UNITS as usize * 3 / 2 + r] = LIVE + 4 + r as u32 * 4;
    }
    let mut old = buffer(&runs, storage);
    let mut spare = buffer(&vec![0; runs.len()], storage);
    let free_pages = buffer(&[3, 0, 0, 0], storage);
    let counts = buffer(&[0; 16], storage);
    let frame = buffer(
        &[96, 0, 0, UNITS],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let job_out = buffer(&vec![0; JOBS as usize * 24], storage);
    let mut payload = vec![0; UNITS as usize * 16];
    payload[LIVE as usize * 16] = 0x1234abcd;
    let pool = buffer(&payload, storage);

    let recycle = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/allocator_recycle.wgsl").into(),
            ),
        });
    let production = include_str!("../shaders/generate.wgsl");
    let declarations = production.split_once("var<workgroup> g_band").unwrap().0;
    let allocator_functions = format!(
        "fn run_units{}",
        segment(production, "fn run_units", "fn evict")
            .rsplit_once("@compute")
            .unwrap()
            .0
    );
    let column = format!(
        "struct Column{}",
        segment(
            include_str!("../shaders/common.wgsl"),
            "struct Column",
            "struct FaceBrush"
        )
    );
    // Only the frame counts are needed by these unmodified production entry
    // points. Test entries stamp allocations and retire them through free_run.
    let source = format!(
        r#"
        struct Frame {{ counts: vec4<u32> }}
        @group(0) @binding(0) var<uniform> frame: Frame;
        const INFO_RELIEF: u32 = 0x10000000u;
        const INFO_RELIEF_INLINE: u32 = 0x04000000u;
        {column}
        {declarations}
        {allocator_functions}
        @group(0) @binding(18) var<storage, read_write> pool: array<u32>;
        @compute @workgroup_size(64)
        fn stamp(@builtin(global_invocation_id) id: vec3<u32>) {{
            if id.x < frame.counts.x && job_out[id.x].status == 0u {{
                pool[job_out[id.x].run * 16u] = id.x + 1u;
            }}
        }}
        @compute @workgroup_size(64)
        fn retire(@builtin(global_invocation_id) id: vec3<u32>) {{
            if id.x < frame.counts.x && job_out[id.x].status == 0u {{
                var c: Column;
                c.info = job_out[id.x].size_class << 18u;
                c.run = job_out[id.x].run;
                free_run(c);
            }}
        }}
    "#
    );
    let generation = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = |module: &wgpu::ShaderModule, entry| {
        gpu.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    let reclaim = pipeline(&recycle, "reclaim_pages");
    let compact = pipeline(&recycle, "compact_runs");
    let finish = pipeline(&recycle, "finish_recycle");
    let count = pipeline(&generation, "count");
    let refill = pipeline(&generation, "refill");
    let allocate = pipeline(&generation, "allocate");
    let fixup = pipeline(&generation, "fixup");
    let stamp = pipeline(&generation, "stamp");
    let retire = pipeline(&generation, "retire");
    let dispatch = |encoder: &mut wgpu::CommandEncoder,
                    pipeline: &wgpu::ComputePipeline,
                    resources: &[(u32, &wgpu::Buffer)],
                    groups| {
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &resources
                .iter()
                .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    };
    for (cycle, n) in [96, JOBS].into_iter().enumerate() {
        gpu.queue
            .write_buffer(&frame, 0, bytemuck::cast_slice(&[n, 0, 0, UNITS]));
        let mut results = vec![0u32; JOBS as usize * 24];
        for job in results.chunks_exact_mut(24).take(n as usize) {
            job[2] = 1;
            job[3] = if cycle == 0 { 15 } else { 0 }; // 16-unit then 1-unit runs.
        }
        gpu.queue
            .write_buffer(&job_out, 0, bytemuck::cast_slice(&results));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&counts, 0, None);
        dispatch(
            &mut encoder,
            &reclaim,
            &[(0, &alloc), (1, &pages), (4, &free_pages)],
            1,
        );
        dispatch(
            &mut encoder,
            &compact,
            &[
                (0, &alloc),
                (1, &pages),
                (2, &old),
                (3, &spare),
                (5, &counts),
            ],
            (UNITS * 2).div_ceil(256),
        );
        dispatch(&mut encoder, &finish, &[(0, &alloc), (5, &counts)], 1);
        std::mem::swap(&mut old, &mut spare);
        dispatch(
            &mut encoder,
            &count,
            &[(0, &frame), (8, &job_out), (10, &alloc)],
            n.div_ceil(64),
        );
        dispatch(
            &mut encoder,
            &refill,
            &[
                (0, &frame),
                (10, &alloc),
                (11, &old),
                (12, &free_pages),
                (17, &pages),
            ],
            1,
        );
        dispatch(
            &mut encoder,
            &allocate,
            &[
                (0, &frame),
                (8, &job_out),
                (10, &alloc),
                (11, &old),
                (17, &pages),
            ],
            n.div_ceil(64),
        );
        dispatch(&mut encoder, &fixup, &[(10, &alloc)], 1);
        dispatch(
            &mut encoder,
            &stamp,
            &[(0, &frame), (8, &job_out), (18, &pool)],
            n.div_ceil(64),
        );
        gpu.queue.submit([encoder.finish()]);
        let jobs = words(&job_out);
        let mut occupied = HashSet::new();
        for job in jobs.chunks_exact(24).take(n as usize) {
            assert_eq!(job[0], 0, "reallocated job failed in cycle {cycle}");
            assert_eq!(job[5], if cycle == 0 { 4 } else { 0 });
            for unit in job[6]..job[6] + (1 << job[5]) {
                assert!(
                    unit < UNITS && !(1024..1536).contains(&unit),
                    "partially live page was recycled"
                );
                assert!(occupied.insert(unit), "duplicate allocation of unit {unit}");
            }
        }
        let state = words(&alloc);
        assert_eq!(
            state[2], 127,
            "partial page's free runs must survive compaction"
        );
        assert_eq!(state[27], if cycle == 0 { 2 } else { 5 });
        assert_eq!(state[30], 0);
        assert_eq!(
            words(&pool)[LIVE as usize * 16],
            0x1234abcd,
            "live payload overwritten"
        );
        assert_eq!(&words(&pages)[4..6], &[127, 2]);
        if cycle == 0 {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            dispatch(
                &mut encoder,
                &retire,
                &[
                    (0, &frame),
                    (8, &job_out),
                    (10, &alloc),
                    (11, &old),
                    (17, &pages),
                ],
                n.div_ceil(64),
            );
            gpu.queue.submit([encoder.finish()]);
            assert_eq!(
                words(&alloc)[26],
                3,
                "last retire must make each page reclaimable once"
            );
        }
    }
}
