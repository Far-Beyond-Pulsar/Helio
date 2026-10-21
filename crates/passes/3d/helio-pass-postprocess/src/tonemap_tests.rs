use super::PostProcessPass;

/// `tonemap_agx` (its punchy look when `punchy`) on scene-linear `inputs`,
/// read back display-linear.
fn agx(inputs: &[[f32; 3]], punchy: bool) -> Vec<[f32; 3]> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        let probe = format!(
            "{}\n@group(1) @binding(0) var<storage, read_write> probe: array<vec4<f32>>;\n\
             @compute @workgroup_size(1) fn probe_agx(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
                 probe[id.x] = vec4<f32>(tonemap_agx(probe[id.x].rgb, {punchy}), 1.0);\n\
             }}",
            helio_core::shader::resolve(&PostProcessPass::build_shader_source(&[])),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("AgX probe"), source: wgpu::ShaderSource::Wgsl(probe.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AgX probe"),
            layout: None,
            module: &shader,
            entry_point: Some("probe_agx"),
            compilation_options: Default::default(),
            cache: None,
        });
        let words: Vec<[f32; 4]> = inputs.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let size = (words.len() * 16) as u64;
        let data = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&data, 0, bytemuck::cast_slice(&words));
        let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups(words.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&data, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let out: Vec<[f32; 4]> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
        out.iter().map(|v| [v[0], v[1], v[2]]).collect()
    })
}

/// Hue in degrees of display-linear `c`, measured on its gamma-encoded
/// value as a viewer sees it.
fn hue(c: [f32; 3]) -> f32 {
    let [r, g, b] = c.map(|v| v.max(0.0).powf(1.0 / 2.2));
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let d = max - min;
    let h = if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h * 60.0
}

/// AgX keeps middle grey near middle grey, rises monotonically to white,
/// and a sunlit saturated green brightening 64 times stays green (hue
/// within 25 degrees) while it rolls toward white, where ACES turns it
/// lime.
#[test]
#[ignore = "requires a GPU"]
fn agx_keeps_grey_and_the_hue_of_bright_colours() {
    let greys = [0.0f32, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0, 16.0, 64.0];
    let green = [0.2f32, 1.0, 0.05];
    let scales = [0.25f32, 1.0, 4.0, 16.0];
    let inputs: Vec<[f32; 3]> = greys.iter().map(|&v| [v; 3]).chain(scales.iter().map(|&k| green.map(|c| c * k))).collect();
    let out = agx(&inputs, false);
    let grey: Vec<f32> = out[..greys.len()].iter().map(|c| c[1]).collect();
    assert!(grey.windows(2).all(|w| w[1] > w[0] || w[1] >= 0.99), "{grey:?}");
    assert!(grey[0] < 1e-3 && (0.15..0.25).contains(&grey[3]) && grey[8] > 0.99, "{grey:?}");
    for c in &out[..greys.len()] {
        assert!((c[0] - c[1]).abs() < 2e-3 && (c[2] - c[1]).abs() < 2e-3, "grey stays neutral: {c:?}");
    }
    let source = hue(green.map(|c| c.powf(2.2)));
    for (k, c) in scales.iter().zip(&out[greys.len()..]) {
        assert!((hue(*c) - source).abs() < 25.0, "x{k}: {c:?} hue {} vs {source}", hue(*c));
    }
}

/// AgX's punchy look keeps greys neutral and rising, and gives a sky blue
/// and a green more saturation than base AgX.
#[test]
#[ignore = "requires a GPU"]
fn agx_punchy_saturates_and_keeps_grey() {
    let saturation = |c: [f32; 3]| {
        let [r, g, b] = c.map(|v| v.max(0.0).powf(1.0 / 2.2));
        let (max, min) = (r.max(g).max(b), r.min(g).min(b));
        if max > 0.0 { (max - min) / max } else { 0.0 }
    };
    let inputs = [[0.05f32; 3], [0.18; 3], [1.0; 3], [0.18, 0.32, 0.62], [0.2, 0.4, 0.09]];
    let (base, punchy) = (agx(&inputs, false), agx(&inputs, true));
    for c in &punchy[..3] {
        assert!((c[0] - c[1]).abs() < 2e-3 && (c[2] - c[1]).abs() < 2e-3, "grey stays neutral: {c:?}");
    }
    assert!(punchy[0][1] < punchy[1][1] && punchy[1][1] < punchy[2][1], "{punchy:?}");
    for n in 3..5 {
        assert!(saturation(punchy[n]) > saturation(base[n]) + 0.1, "{:?} vs {:?}", punchy[n], base[n]);
    }
}

/// AgX punchy keeps a dark colour's hue: grass under sky light from one to
/// five EV under middle grey (shade) keeps the hue it has at one EV under,
/// never clips a channel to zero, still darkens monotonically, and two
/// grass tiles stay as far apart in hue as at the reference.
#[test]
#[ignore = "requires a GPU"]
fn agx_punchy_keeps_the_hue_of_dark_colours() {
    let sky = [0.45f32, 0.58, 0.8];
    let tiles = [[0.34f32, 0.45, 0.07], [0.24, 0.47, 0.08]];
    let evs = [-1.0f32, -2.0, -3.0, -4.0, -5.0];
    let at = |tile: [f32; 3], ev: f32| {
        let c = [tile[0] * sky[0], tile[1] * sky[1], tile[2] * sky[2]];
        let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        c.map(|v| v / y * 0.18 * ev.exp2())
    };
    let inputs: Vec<[f32; 3]> = tiles.iter().flat_map(|&t| evs.iter().map(move |&ev| at(t, ev))).collect();
    let out = agx(&inputs, true);
    let luma = |c: &[f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    for (t, row) in out.chunks(evs.len()).enumerate() {
        let reference = hue(row[0]);
        for (ev, c) in evs.iter().zip(row) {
            assert!(c.iter().all(|&v| v > 0.0), "tile {t} at {ev} EV clipped a channel: {c:?}");
            assert!((hue(*c) - reference).abs() < 2.0, "tile {t} at {ev} EV: hue {} vs {reference}", hue(*c));
        }
        assert!(row.windows(2).all(|w| luma(&w[1]) < luma(&w[0])), "tile {t} darkens: {row:?}");
    }
    let apart = |i: usize| (hue(out[i]) - hue(out[evs.len() + i])).abs();
    assert!(apart(3) > 0.9 * apart(0), "tiles at -4 EV {} degrees apart, at -1 EV {}", apart(3), apart(0));
}
