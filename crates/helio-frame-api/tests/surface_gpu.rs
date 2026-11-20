//! The surface at a lit point goes through both of its encodings and back:
//! a ray stream's `SceneSurface` row within each field's precision, and the
//! G-buffer's texels exactly (before the targets' formats round them). Runs
//! on a real device; skipped without one.

use helio_frame_api::gbuffer::SURFACE_SNIPPET;
use helio_frame_api::rays::stream::{SNIPPET as STREAM, SURFACE_SNIPPET as ROWS};

const SURFACES: u32 = 4096;
/// `SurfaceData` as floats: albedo 4, normal 3, ao, roughness, metallic,
/// F0 3, emissive 3, alpha, flags, subsurface 3, radius, aniso x, y,
/// rotation.
const FIELDS: usize = 25;

const KERNEL: &str = r#"//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

fn h(x: u32) -> f32 {
    var v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return f32(((v >> 22u) ^ v) >> 8u) / 16777216.0;
}

fn write(at: u32, s: SurfaceData) {
    let f = array<f32, 25>(s.albedo.r, s.albedo.g, s.albedo.b, s.albedo.a,
        s.normal.x, s.normal.y, s.normal.z, s.ao, s.roughness, s.metallic,
        s.specular_f0.r, s.specular_f0.g, s.specular_f0.b,
        s.emissive.r, s.emissive.g, s.emissive.b, s.alpha, f32(s.flags),
        s.subsurface_color.r, s.subsurface_color.g, s.subsurface_color.b,
        s.subsurface_radius, s.roughness_aniso_x, s.roughness_aniso_y, s.aniso_rotation);
    for (var k = 0u; k < 25u; k++) { out[at + k] = f[k]; }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let r = i * 32u;
    let z = h(r) * 2.0 - 1.0;
    let phi = h(r + 1u) * 6.2831853;
    let n = vec3<f32>(sqrt(1.0 - z * z) * cos(phi), sqrt(1.0 - z * z) * sin(phi), z);
    var s = surface_data(vec3<f32>(h(r + 2u), h(r + 3u), h(r + 4u)), n, h(r + 5u), h(r + 6u),
        vec3<f32>(h(r + 7u), h(r + 8u), h(r + 9u)) * 100.0);
    s.ao = h(r + 10u);
    s.specular_f0 = vec3<f32>(h(r + 11u), h(r + 12u), h(r + 13u));
    s.flags = u32(h(r + 14u) * 16.0) & 15u;
    s.subsurface_color = vec3<f32>(h(r + 15u), h(r + 16u), h(r + 17u)) * 2.0;
    s.subsurface_radius = h(r + 18u) * 10.0;
    s.roughness_aniso_x = h(r + 19u);
    s.roughness_aniso_y = h(r + 20u);
    s.aniso_rotation = h(r + 21u) * 20.0 - 10.0;
    let base = i * 3u * 25u;
    write(base, s);
    write(base + 25u, scene_surface_unpack(scene_surface_pack(s, n)));
    write(base + 50u, gbuffer_surface_at(gbuffer_surface_texels(s)));
}
"#;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        adapter.request_device(&Default::default()).await.ok()
    })
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
    let source = helio_core::shader::resolve_with(KERNEL, &[STREAM, ROWS, SURFACE_SNIPPET]);
    #[allow(clippy::disallowed_methods)] // a test's one-off module, not a pass's
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let size = u64::from(SURFACES) * 3 * FIELDS as u64 * 4;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(SURFACES / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &read, 0, size);
    queue.submit([encoder.finish()]);
    read.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = read.slice(..).get_mapped_range().expect("mapped");
    bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
}

#[test]
fn a_surface_survives_both_encodings() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let values = run(&device, &queue);
    let mut worst_normal = 0.0f64;
    let unorm8 = 0.5 / 255.0 + 1e-6;
    for i in 0..SURFACES as usize {
        let at = i * 3 * FIELDS;
        let s = &values[at..at + FIELDS];
        let row = &values[at + FIELDS..at + 2 * FIELDS];
        let texels = &values[at + 2 * FIELDS..at + 3 * FIELDS];
        // The G-buffer's encode and decode are exact (the normal normalized).
        for k in 0..FIELDS {
            assert!(
                (s[k] - texels[k]).abs() <= 1e-5 * s[k].abs().max(1.0),
                "surface {i} field {k}: {} -> {}",
                s[k],
                texels[k]
            );
        }
        // The row: each field within its packing's precision.
        for k in [0, 1, 2, 7, 8, 9, 22, 23] {
            assert!(
                (s[k] - row[k]).abs() <= unorm8,
                "surface {i} field {k}: {} -> {}",
                s[k],
                row[k]
            );
        }
        assert_eq!(row[3], 1.0, "a hit is covered");
        assert_eq!(row[16], 1.0, "a hit is covered");
        assert_eq!(s[17], row[17], "flags are exact");
        for k in 10..13 {
            assert!(
                (s[k] - row[k]).abs() <= 0.5 / 1023.0 + 1e-6,
                "F0 {k}: {} -> {}",
                s[k],
                row[k]
            );
        }
        for range in [13..16, 18..21] {
            let most = s[range.clone()].iter().copied().fold(0.0, f32::max);
            for k in range {
                assert!(
                    (s[k] - row[k]).abs() <= most / 256.0 + 1e-6,
                    "radiance {k}: {} -> {}",
                    s[k],
                    row[k]
                );
            }
        }
        // The radius as r / (1 + r) in 8 bits: within half a step of that.
        let (r, back) = (s[21], row[21]);
        assert!(
            (r / (1.0 + r) - back / (1.0 + back)).abs() <= unorm8,
            "radius {r} -> {back}"
        );
        // The rotation: the same direction (a fraction of a turn).
        let (a, b) = (s[24], row[24]);
        assert!(
            (a.cos() - b.cos()).abs() < 2e-4 && (a.sin() - b.sin()).abs() < 2e-4,
            "rotation {a} -> {b}"
        );
        let n = [s[4], s[5], s[6]].map(f64::from);
        let m = [row[4], row[5], row[6]].map(f64::from);
        let cross = [
            n[1] * m[2] - n[2] * m[1],
            n[2] * m[0] - n[0] * m[2],
            n[0] * m[1] - n[1] * m[0],
        ];
        worst_normal = worst_normal.max(cross.iter().map(|v| v * v).sum::<f64>().sqrt().asin());
    }
    assert!(worst_normal < 1e-4, "{worst_normal} rad");
}
