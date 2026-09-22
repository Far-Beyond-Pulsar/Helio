use super::*;
use glam::{DMat4, DVec4, Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Compare centered GPU history coordinates with f64 matrix algebra. Include
/// translated local origins, where the former world-coordinate round trip lost
/// precision even though the motion between the two cameras remained small.
#[test]
fn clip_reprojection_matches_double_precision_at_rebased_origins() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/tsr_main.wgsl"),
            r#"
// Frozen former implementation, used only as a numerical control.
fn legacy_point(uv:vec2<f32>,depth:f32)->vec3<f32> {
    let dims=vec2<f32>(960.0,540.0);
    let p=uv+tsr.jitter_offset*vec2<f32>(1.0,-1.0)/dims;
    let world=cameras[0].inv_view_proj*vec4<f32>(p.x*2.0-1.0,1.0-p.y*2.0,depth,1.0);
    let old=cameras[0].prev_view_proj*world;
    let ndc=old.xy/old.w;
    return vec3<f32>(vec2<f32>(ndc.x*0.5+0.5,0.5-ndc.y*0.5)
        -tsr.previous_jitter*vec2<f32>(1.0,-1.0)/dims,old.z/old.w);
}
@group(0) @binding(10) var<storage,read_write> results:array<vec4<f32>>;
@compute @workgroup_size(64) fn validate_clip(@builtin(global_invocation_id) id:vec3<u32>) {
    let dims=vec2<f32>(960.0,540.0);
    let uv=(vec2<f32>(f32(id.x%8u),f32(id.x/8u))+0.5)/8.0;
    let depth=array<f32,4>(0.5,0.9,0.99,0.9999)[id.x%4u];
    var current=reproject_history_point(uv,depth,dims);var legacy=legacy_point(uv,depth);
    if tsr.reset!=0u {
        current=reproject_sample_motion(uv,depth,dims);
        let jitter=tsr.jitter_offset*vec2<f32>(1.0,-1.0)/dims;
        let pixel=clamp(floor((uv+jitter)*dims),vec2<f32>(0.0),dims-1.0);
        let center=(pixel+0.5)/dims-jitter;
        let old=legacy_point(center,depth);legacy=vec3<f32>(old.xy+uv-center,old.z);
    }
    results[id.x]=vec4<f32>(current,depth);results[id.x+64u]=vec4<f32>(legacy,depth);
}"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("validate_clip"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 2048,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 2048,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut worst = [0.0_f64; 2];
        let mut depth_worst = [0.0_f64; 2];
        let mut sum_square = [0.0; 2];
        let mut cases = 0;
        for (near, far) in [(0.1, 10000.0), (1000.0, 30_000_000.0)] {
            #[allow(deprecated)]
            let projection = Mat4::perspective_rh(60_f32.to_radians(), 960.0 / 540.0, near, far);
            for origin in [Vec3::ZERO, Vec3::new(1024.0, 512.0, -1024.0)] {
                let previous_view = Mat4::from_translation(-origin);
                for (moving, sampled) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let view = if moving {
                        Mat4::from_rotation_y(0.006)
                            * Mat4::from_translation(Vec3::new(-0.002, 0.001, 0.0))
                            * previous_view
                    } else {
                        previous_view
                    };
                    for frame in [1, 2, 17, 1024] {
                        let jitter = r1_r2_jitter(frame);
                        let old_jitter = r1_r2_jitter(frame - 1);
                        let translate = |j: [f32; 2]| {
                            Mat4::from_translation(Vec3::new(
                                j[0] * 2.0 / 960.0,
                                j[1] * 2.0 / 540.0,
                                0.0,
                            ))
                        };
                        let camera = helio_core::GpuCameraUniforms::new(
                            view,
                            translate(jitter) * projection,
                            origin,
                            near,
                            far,
                            frame as u32,
                            [jitter[0] * 2.0 / 960.0, jitter[1] * 2.0 / 540.0],
                            (translate(old_jitter) * projection) * previous_view,
                        );
                        let uniforms = TsrUniform {
                            jitter_offset: jitter,
                            previous_jitter: old_jitter,
                            reactivity: 0.0,
                            reset: u32::from(sampled),
                            time_delta: 1.0 / 60.0,
                            tap_radius: 1,
                            clip_to_previous: clip_to_previous(&camera).unwrap(),
                        };
                        let cameras =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: None,
                                contents: bytemuck::cast_slice(&[camera; 2]),
                                usage: wgpu::BufferUsages::STORAGE,
                            });
                        let uniform =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: None,
                                contents: bytemuck::bytes_of(&uniforms),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });
                        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &pipeline.get_bind_group_layout(0),
                            entries: &[(5, &cameras), (6, &uniform), (10, &output)].map(
                                |(binding, buffer)| wgpu::BindGroupEntry {
                                    binding,
                                    resource: buffer.as_entire_binding(),
                                },
                            ),
                        });
                        let mut encoder = device.create_command_encoder(&Default::default());
                        {
                            let mut pass = encoder.begin_compute_pass(&Default::default());
                            pass.set_pipeline(&pipeline);
                            pass.set_bind_group(0, &group, &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 2048);
                        queue.submit(Some(encoder.finish()));
                        let (tx, rx) = std::sync::mpsc::channel();
                        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                            tx.send(r).unwrap();
                        });
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                        rx.recv().unwrap().unwrap();
                        {
                            let mapped = staging.slice(..).get_mapped_range().unwrap();
                            let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
                            let current = DMat4::from_cols_array(&camera.view_proj.map(f64::from));
                            let previous =
                                DMat4::from_cols_array(&camera.prev_view_proj.map(f64::from));
                            for i in 0..64 {
                                let uv =
                                    [(i % 8) as f64 / 8.0 + 0.0625, (i / 8) as f64 / 8.0 + 0.0625];
                                let sample = if sampled {
                                    std::array::from_fn(|axis| {
                                        let j = f64::from(jitter[axis]) * [1.0, -1.0][axis];
                                        ((uv[axis] * [960.0, 540.0][axis] + j).floor() + 0.5 - j)
                                            / [960.0, 540.0][axis]
                                    })
                                } else {
                                    uv
                                };
                                let ray_uv: [f64; 2] = std::array::from_fn(|axis| {
                                    sample[axis]
                                        + f64::from(jitter[axis]) * [1.0, -1.0][axis]
                                            / [960.0, 540.0][axis]
                                });
                                let p = previous
                                    * (current.inverse()
                                        * DVec4::new(
                                            ray_uv[0] * 2.0 - 1.0,
                                            1.0 - ray_uv[1] * 2.0,
                                            f64::from(values[i][3]),
                                            1.0,
                                        ));
                                let expected = [
                                    p.x / p.w * 0.5 + 0.5 - f64::from(old_jitter[0]) / 960.0
                                        + uv[0]
                                        - sample[0],
                                    0.5 - p.y / p.w * 0.5
                                        + f64::from(old_jitter[1]) / 540.0
                                        + uv[1]
                                        - sample[1],
                                    p.z / p.w,
                                ];
                                for variant in 0..2 {
                                    let value = values[i + variant * 64];
                                    for axis in 0..2 {
                                        let error = (f64::from(value[axis]) - expected[axis]).abs()
                                            * [960.0, 540.0][axis];
                                        assert!(error.is_finite());
                                        worst[variant] = worst[variant].max(error);
                                        sum_square[variant] += error * error;
                                    }
                                    depth_worst[variant] = depth_worst[variant]
                                        .max((f64::from(value[2]) - expected[2]).abs());
                                }
                                cases += 1;
                            }
                        }
                        staging.unmap();
                    }
                }
            }
        }
        println!("REPROJECTION_PRECISION cases={cases} current_max_px={} legacy_max_px={} current_rms_px={} legacy_rms_px={} current_max_depth_error={} legacy_max_depth_error={}",
            worst[0],worst[1],(sum_square[0]/(cases*2) as f64).sqrt(),(sum_square[1]/(cases*2) as f64).sqrt(),depth_worst[0],depth_worst[1]);
        assert!(
            worst[0] < 0.005,
            "history coordinate error {} pixels",
            worst[0]
        );
        assert!(
            depth_worst[0] < 0.000001,
            "history depth error {}",
            depth_worst[0]
        );
    });
}

#[test]
fn invalid_camera_matrices_do_not_create_nonfinite_uniforms() {
    let mut camera = helio_core::GpuCameraUniforms::zeroed();
    assert!(clip_to_previous(&camera).is_none());
    camera.view_proj = Mat4::IDENTITY.to_cols_array();
    camera.prev_view_proj = Mat4::IDENTITY.to_cols_array();
    assert_eq!(
        clip_to_previous(&camera),
        Some(Mat4::IDENTITY.to_cols_array())
    );
    camera.prev_view_proj[0] = f32::NAN;
    assert!(clip_to_previous(&camera).is_none());
    camera.prev_view_proj = Mat4::IDENTITY.to_cols_array();
    camera.view_proj[0] = f32::INFINITY;
    assert!(clip_to_previous(&camera).is_none());
}
