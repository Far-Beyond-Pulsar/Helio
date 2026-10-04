//! Summary handoff uses retained canonical records without regeneration.
mod common;
use common::*;
use helio_pass_voxel_planet::noise::hash3;
use wgpu::util::DeviceExt;

fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(" )).unwrap();
    &source[start..start + source[start..].find("\n}").unwrap() + 2]
}

#[test]
fn production_summary_handoff_rebuilds_retained_edited_records_and_complete_parents() {
    let gpu = gpu().expect("summary handoff regression needs an adapter");
    const REGION: u32 = 16384 + 1024 + 64;
    const MASK: u32 = 16383;
    const VALID: u32 = 0x80000000;
    const OVERFLOW: u32 = 0x40000000;
    const NONE: u32 = u32::MAX;
    let slot = |tier: u32, bi: i32, bj: i32| -> usize {
        let bits = 9 - tier * 2;
        let mask = (1 << bits) - 1;
        let offset = if tier == 1 { 0 } else if tier == 2 { 16384 } else { 17408 };
        (offset + ((bj & mask) << bits) + (bi & mask)) as usize
    };
    let mut records: Vec<[u32; 8]> = Vec::new();
    // Outgoing owner remains resident in the same summary slots.
    for j in 0..4 {
        for i in 0..4 {
            records.push([i, j, 0, VALID | 1, 77, 0, 1, 0]);
        }
    }
    for j in 0..64 {
        for i in 512..576 {
            records.push([i, j, 0, VALID | 1, 99, 0, 1, 0]);
        }
    }
    // Topology-edited record retains its packed exact top and edit reference.
    records[16] = [512, 0, 0, VALID | 0x08000000 | 16 | (5 << 22), 123, 7, 0, 321];
    let canonical = records.clone();
    let mut table = vec![NONE; (MASK + 1) as usize];
    for (record, column) in records.iter().enumerate() {
        let mut at = hash3(column[0] as i32, column[1] as i32, 0x2f6b1d3a, 0x9e3779b9) & MASK;
        while table[at as usize] != NONE { at = (at + 1) & MASK; }
        table[at as usize] = record as u32;
    }
    let mut summaries = vec![-1i32; REGION as usize * 4];
    summaries[slot(1, 0, 0) * 4..slot(1, 0, 0) * 4 + 4].copy_from_slice(&[0, 0, 8, 16]);
    let mut descriptors = Vec::new();
    for bj in 0..16 {
        for bi in 128..144 { descriptors.extend([slot(1, bi, bj) as u32, bi as u32, bj as u32]); }
    }
    for bj in 0..4 {
        for bi in 32..36 { descriptors.extend([slot(2, bi, bj) as u32, bi as u32, bj as u32]); }
    }
    descriptors.extend([slot(3, 8, 0) as u32, 8, 0]);
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let buffer = |bytes: &[u8], usage| gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytes, usage,
    });
    let frame_words = |patches: usize| [0u32, 0, MASK, 0, 0, REGION, patches as u32, 0, 0, 16384, 2, 0];
    let frame = buffer(bytemuck::cast_slice(&frame_words(descriptors.len() / 3)), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    let table_buffer = buffer(bytemuck::cast_slice(&table), storage);
    let record_buffer = buffer(bytemuck::cast_slice(&records), storage);
    let blocks = buffer(bytemuck::cast_slice(&summaries), storage);
    let patches = buffer(bytemuck::cast_slice(&descriptors), storage);
    let common = include_str!("../shaders/common.wgsl");
    let column_start = common.find("struct Column {").unwrap();
    let column = &common[column_start..column_start + common[column_start..].find("\n}").unwrap() + 2];
    let helpers = ["column_slot", "block_slot", "find_column", "column_valid", "band_count", "column_top_cell"]
        .map(|name| function(common, name)).join("\n");
    let noise = function(include_str!("../shaders/noise.wgsl"), "hash3");
    let generation = include_str!("../shaders/generate.wgsl").replace("\r\n", "\n");
    let entries = &generation[generation.find("@compute @workgroup_size(64)\nfn patch_blocks").unwrap()
        ..generation.find("@compute @workgroup_size(64)\nfn publish").unwrap()];
    let shader = format!(r#"
        struct Frame {{ counts:vec4<u32>, extra:vec4<u32>, layer_i:vec4<i32> }}
        {column}
        const NONE:u32=0xffffffffu;
        const TOMBSTONE:u32=0xfffffffeu;
        const INFO_VALID:u32=0x80000000u;
        const INFO_OVERFLOW:u32=0x40000000u;
        const MAX_PROBES:u32=64u;
        @group(0) @binding(0) var<uniform> frame:Frame;
        @group(0) @binding(2) var<storage,read_write> table:array<u32>;
        @group(0) @binding(3) var<storage,read_write> records:array<Column>;
        @group(0) @binding(13) var<storage,read> evictions:array<u32>;
        @group(0) @binding(15) var<storage,read_write> block_state:array<atomic<i32>>;
        {noise}
        {helpers}
        {entries}
    "#);
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production summary handoff"), source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let resources = [(0, &frame), (2, &table_buffer), (3, &record_buffer), (13, &patches), (15, &blocks)];
    let entries = ["patch_blocks", "rebuild_tier1", "rebuild_tier2", "rebuild_tier3"];
    let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &resources.iter().map(|&(binding, _)| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 0 { wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None } }
                else { wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: binding == 13 }, has_dynamic_offset: false, min_binding_size: None } },
            count: None,
        }).collect::<Vec<_>>(),
    });
    let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0,
    });
    let pipelines = entries.map(|entry| gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pipeline_layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None,
    }));
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &layout,
        entries: &resources.iter().map(|&(binding, buffer)| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }).collect::<Vec<_>>(),
    });
    let run = |count: usize| {
        gpu.queue.write_buffer(&frame, 0, bytemuck::cast_slice(&frame_words(count)));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &group, &[]);
            for pipeline in &pipelines {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        read_buffer(&gpu, &blocks, blocks.size()).chunks_exact(4)
            .map(|w| i32::from_le_bytes(w.try_into().unwrap())).collect::<Vec<_>>()
    };
    let assert_entry = |words: &[i32], tier, bi, bj, top, count| {
        let at = slot(tier, bi, bj) * 4;
        assert_eq!(&words[at..at + 4], &[bi, bj, top, count]);
    };
    let full = run(descriptors.len() / 3);
    assert_entry(&full, 1, 128, 0, 123, 16);
    assert_entry(&full, 2, 32, 0, 123, 256);
    assert_entry(&full, 3, 8, 0, 123, 4096);
    assert_ne!(&full[slot(1, 0, 0) * 4..slot(1, 0, 0) * 4 + 2], &[0, 0]);
    assert_eq!(read_buffer(&gpu, &record_buffer, record_buffer.size()), bytemuck::cast_slice::<_, u8>(&canonical));
    // No changed attachments means no resets, preserving every full count.
    let unchanged = run(0);
    assert_eq!(unchanged, full);
    // A detached child leaves its exact records resident. Dirty ancestors are
    // rebuilt even when not selected, retaining all untouched sibling counts.
    let detached = [slot(1, 128, 0) as u32, NONE, NONE,
        slot(2, 32, 0) as u32, 32, 0, slot(3, 8, 0) as u32, 8, 0];
    gpu.queue.write_buffer(&patches, 0, bytemuck::cast_slice(&detached));
    let detached = run(3);
    assert_entry(&detached, 1, 129, 0, 8, 16);
    assert_entry(&detached, 2, 32, 0, 8, 240);
    assert_entry(&detached, 3, 8, 0, 8, 4080);
    assert_eq!(read_buffer(&gpu, &record_buffer, record_buffer.size()), bytemuck::cast_slice::<_, u8>(&canonical));
    // A retry can publish a retained detached record and increment matching
    // parents. Its dirty closure removes that count without touching siblings.
    gpu.queue.write_buffer(&blocks, slot(2, 32, 0) as u64 * 16,
        bytemuck::cast_slice(&[32i32, 0, 123, 241]));
    gpu.queue.write_buffer(&blocks, slot(3, 8, 0) as u64 * 16,
        bytemuck::cast_slice(&[8i32, 0, 123, 4081]));
    let ancestors = [slot(2, 32, 0) as u32, 32, 0, slot(3, 8, 0) as u32, 8, 0];
    gpu.queue.write_buffer(&patches, 0, bytemuck::cast_slice(&ancestors));
    let retried_detached = run(2);
    assert_entry(&retried_detached, 2, 32, 0, 8, 240);
    assert_entry(&retried_detached, 3, 8, 0, 8, 4080);
    let returning_child = [slot(1, 128, 0) as u32, 128, 0,
        slot(2, 32, 0) as u32, 32, 0, slot(3, 8, 0) as u32, 8, 0];
    gpu.queue.write_buffer(&patches, 0, bytemuck::cast_slice(&returning_child));
    let returned_child = run(3);
    assert_entry(&returned_child, 1, 128, 0, 123, 16);
    assert_entry(&returned_child, 1, 129, 0, 8, 16);
    assert_entry(&returned_child, 2, 32, 0, 123, 256);
    assert_entry(&returned_child, 3, 8, 0, 123, 4096);
    // Invalidation matches the published metadata cleared by production evict;
    // the remaining fifteen exact columns and all siblings retain positive counts.
    records[17][0] = NONE;
    records[17][3] = 0;
    gpu.queue.write_buffer(&record_buffer, 0, bytemuck::cast_slice(&records));
    let evicted = run(3);
    assert_entry(&evicted, 1, 128, 0, 123, 15);
    assert_entry(&evicted, 1, 129, 0, 8, 16);
    assert_entry(&evicted, 2, 32, 0, 123, 255);
    assert_entry(&evicted, 3, 8, 0, 123, 4095);
    assert_eq!(read_buffer(&gpu, &record_buffer, record_buffer.size()), bytemuck::cast_slice::<_, u8>(&records));
    records[17] = canonical[17];
    gpu.queue.write_buffer(&record_buffer, 0, bytemuck::cast_slice(&records));
    let restored = run(3);
    assert_entry(&restored, 3, 8, 0, 123, 4096);
    gpu.queue.write_buffer(&patches, 0, bytemuck::cast_slice(&descriptors));
    // Failed publication cannot certify a complete block; retained edit stays.
    records[17][3] |= OVERFLOW;
    gpu.queue.write_buffer(&record_buffer, 0, bytemuck::cast_slice(&records));
    let partial = run(descriptors.len() / 3);
    assert_entry(&partial, 1, 128, 0, 123, 15);
    assert_entry(&partial, 2, 32, 0, 123, 255);
    assert_entry(&partial, 3, 8, 0, 123, 4095);
    // Returning old owner reads its existing records, not aliased new data.
    let returning = [slot(1, 0, 0) as u32, 0, 0, slot(2, 0, 0) as u32, 0, 0, slot(3, 0, 0) as u32, 0, 0];
    gpu.queue.write_buffer(&patches, 0, bytemuck::cast_slice(&returning));
    let returned = run(3);
    assert_entry(&returned, 1, 0, 0, 8, 16);
    assert_entry(&returned, 2, 0, 0, 8, 16);
    assert_entry(&returned, 3, 0, 0, 8, 16);
    assert_eq!(read_buffer(&gpu, &record_buffer, record_buffer.size()), bytemuck::cast_slice::<_, u8>(&records));
    // A negative partial child is retained as incomplete, never promoted.
    let child = slot(2, 0, 0) as u64 * 16;
    gpu.queue.write_buffer(&blocks, child, bytemuck::cast_slice(&[0i32, 0, 8, -7]));
    gpu.queue.write_buffer(&patches, 0, bytemuck::cast_slice(&[slot(3, 0, 0) as u32, 0, 0]));
    let negative = run(1);
    assert_entry(&negative, 3, 0, 0, 8, -7);
}
