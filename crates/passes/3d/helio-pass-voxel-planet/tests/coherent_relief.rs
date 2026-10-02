//! Exact shader helper checks; native appearance/performance remain separate gates.
mod common;
use common::*;
use helio_pass_voxel_planet::grid::face_axes;
use wgpu::util::DeviceExt;

#[test]
fn canonical_relief_uses_physical_chart_slope_and_continuous_support() {
    let Some(gpu) = gpu() else { return };
    let surface = include_str!("../shaders/surface.wgsl");
    let helpers = surface
        .split("fn canonical_relief_confidence")
        .nth(1)
        .unwrap()
        .split("// A natural riser")
        .next()
        .unwrap();
    let world = include_str!("../shaders/world.wgsl");
    let axes = world
        .split("fn face_axis")
        .nth(1)
        .unwrap()
        .split("fn mul_q24")
        .next()
        .unwrap();
    let source = format!(
        r#"
        struct Face {{m_a:vec4<f32>,m_b:vec4<f32>}}
        struct Frame {{faces:array<Face,6>,layer:vec4<f32>}}
        struct Probe {{up:vec4<f32>,gradient:vec4<f32>,params:vec4<f32>}}
        @group(0) @binding(0) var<uniform> frame:Frame;
        @group(0) @binding(1) var<storage,read> probes:array<Probe>;
        @group(0) @binding(2) var<storage,read_write> answers:array<vec4<f32>>;
        override PLANE:bool=false;
        fn is_plane()->bool {{return PLANE;}}
        fn face_axis{axes}
        fn canonical_relief_confidence{helpers}
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            answers[id.x*2u]=vec4<f32>(canonical_relief_slope(u32(p.params.x),p.up.xyz,p.gradient.xyz,p.params.y),
                canonical_relief_confidence(p.params.z),canonical_relief_face_weight(0u,p.params.w,false),canonical_relief_face_weight(4u,p.params.w,false));
            let base=detail_filter_weight(p.params.w);
            answers[id.x*2u+1u]=vec4<f32>(base,
                base*canonical_relief_face_weight(0u,p.params.w*2.0,true),
                base*canonical_relief_face_weight(0u,p.params.w*4.0,true),
                base*canonical_relief_face_weight(0u,p.params.w*32.0,false));
        }}
    "#
    );
    let smooth = |lo: f64, hi: f64, x: f64| {
        let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coherent relief helpers"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    for plane in [true, false] {
        for voxel in [0.1f64, 0.3, 1.0] {
            let datum = 6_371_000.0;
            let delta = if plane { voxel } else { voxel / datum };
            let mut frame = [0f32; 52];
            for face in 0u8..6 {
                let [_, a, b] = face_axes(face);
                frame[(face as usize) * 8..(face as usize) * 8 + 3]
                    .copy_from_slice(&a.as_vec3().to_array());
                frame[(face as usize) * 8 + 4..(face as usize) * 8 + 7]
                    .copy_from_slice(&b.as_vec3().to_array());
            }
            frame[49] = voxel as f32;
            frame[50] = delta as f32;
            let mut probes = Vec::<[[f32; 4]; 3]>::new();
            let mut expected = Vec::<[f64; 4]>::new();
            for face in 0u8..6 {
                let [n, a, b] = face_axes(face);
                for (ta, tb) in [(0.0f64, 0.0f64), (0.55, -0.4), (-0.3, 0.61)] {
                    for radius in [datum, datum * 1.01] {
                        let point =
                            |x: f64, y: f64| (n + a * x.tan() + b * y.tan()).normalize() * radius;
                        let up = if plane { n } else { point(ta, tb).normalize() };
                        let g0 = a * 0.7 + b * 0.3;
                        let gradient = g0 - up * g0.dot(up);
                        let eps = 1e-6;
                        // Independent finite differences of the actual cube-sphere
                        // chart, expressed per one index step / radial voxel.
                        let di = if plane {
                            gradient.dot(a) * delta / voxel
                        } else {
                            gradient.dot(point(ta + eps, tb) - point(ta - eps, tb)) / (2.0 * eps)
                                * delta
                                / voxel
                        };
                        let dj = if plane {
                            gradient.dot(b) * delta / voxel
                        } else {
                            gradient.dot(point(ta, tb + eps) - point(ta, tb - eps)) / (2.0 * eps)
                                * delta
                                / voxel
                        };
                        let slope = 8.0 * di.abs().max(dj.abs());
                        for (gradient_squared, projected_cell) in [
                            (3.99, 0.5),
                            (4.0, 0.75),
                            (4.01, 0.7501),
                            (6.5, 1.25),
                            (9.0, 4.0),
                            (4.0, 0.7499),
                            (4.0, 0.9999),
                            (4.0, 1.0),
                            (4.0, 1.0001),
                            (4.0, 1.2499),
                            (4.0, 1.2501),
                        ] {
                            probes.push([
                                up.as_vec3().extend(0.0).to_array(),
                                gradient.as_vec3().extend(0.0).to_array(),
                                [
                                    face as f32,
                                    radius as f32,
                                    gradient_squared as f32,
                                    projected_cell as f32,
                                ],
                            ]);
                            expected.push([
                                slope,
                                1.0 - smooth(4.0, 9.0, gradient_squared),
                                1.0 - smooth(0.75, 1.25, projected_cell),
                                1.0,
                            ]);
                        }
                    }
                }
            }
            let buffer = |label, bytes: &[u8], usage| {
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytes,
                        usage,
                    })
            };
            let frame = buffer(
                "chart frame",
                bytemuck::cast_slice(&frame),
                wgpu::BufferUsages::UNIFORM,
            );
            let input = buffer(
                "chart probes",
                bytemuck::cast_slice(&probes),
                wgpu::BufferUsages::STORAGE,
            );
            let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chart answers"),
                size: (probes.len() * 32) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let pipeline = gpu
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some("probe"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("PLANE", if plane { 1.0 } else { 0.0 })],
                        ..Default::default()
                    },
                    cache: None,
                });
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((probes.len() as u32 + 63) / 64, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
            let bytes = read_buffer(&gpu, &output, (probes.len() * 32) as u64);
            let pairs: &[[[f32; 4]; 2]] = bytemuck::cast_slice(&bytes);
            let actual: Vec<[f32; 4]> = pairs.iter().map(|p| p[0]).collect();
            for (index, pair) in pairs.iter().enumerate() {
                let weight=expected[index][2] as f32;
                for level in 0..3 {
                    assert!((pair[1][level]-weight).abs()<2e-5,
                        "authored filtering changed across selected L0/L1/L2 at probe{index}");
                }
                assert_eq!(pair[1][3],0.0,
                    "oversized streamed fallback wall lost its geometric normal at probe{index}");
            }
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                for component in 0..4 {
                    assert!((a[component] as f64-e[component]).abs()<2e-5,"plane{plane} voxel{voxel} probe{index} component{component}: {:?} expected{:?}",a,e);
                }
            }
            assert_eq!(actual[1][1], 1.0);
            assert!(
                actual[2][1] > 0.9999,
                "normal support jumped at old slope cutoff"
            );
            assert_eq!(
                actual[3][2], 0.0,
                "resolved angular wall received canonical height normal"
            );
            assert_eq!(actual[1][2], 1.0, "subpixel lower boundary must be fully averaged");
            assert_eq!(actual[7][2], 0.5, "one pixel must retain half the authored contrast");
            assert_eq!(actual[10][2], 0.0, "resolvable detail must retain full contrast");
            for (a, b) in [(1usize, 2usize), (6, 7), (7, 8), (9, 10)] {
                assert!((actual[a][2] - actual[b][2]).abs() < 0.001,
                    "filter discontinuity around one-pixel boundaries: {a}/{b}");
            }
            eprintln!(
                "coherent relief plane{plane} voxel{voxel}:{} slope/support cases",
                probes.len()
            );
        }
    }
}
