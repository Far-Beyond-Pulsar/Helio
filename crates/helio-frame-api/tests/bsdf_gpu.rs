//! The frame's BSDF (`frame_bsdf`) on a real device, by Monte Carlo over
//! many surfaces and view angles (skipped without a device):
//!
//! - **White furnace**: under a uniform environment of radiance 1, a
//!   surface of albedo 1 reflects `integral of bsdf_eval` toward the eye:
//!   exactly 1 for a white Lambertian surface (F0 0), at most 1 for white
//!   dielectrics (F0 2% and 4%, the roughnesses where the old coupling
//!   `1 - F(wo.h)` gained most), a white wrapped-diffuse (subsurface)
//!   surface and white metals, at every view down to n.wo 0.05
//!   (microfacet multiple scattering is not modelled, so rough surfaces
//!   return less). The function under test is the one the deferred lighting
//!   calls (`frame_bsdf`'s `bsdf_eval`).
//! - **The sampler matches the evaluation**: the mean of `bsdf_sample`'s
//!   weights is that same integral (the estimator a path tracer uses agrees
//!   with the evaluation a light uses), and `bsdf_pdf` integrates to at most
//!   one.
//! - **A specular sample never gains energy**: for a pure specular surface
//!   each weight is `F G2 / G1`, at most 1.
//!
//! The integral is estimated with uniformly distributed directions over the
//! sphere (independent of the BSDF's own sampling). Tolerances: five
//! standard errors of each estimate, from the samples' spread.

use helio_frame_api::gbuffer::{BSDF_SNIPPET, SURFACE_SNIPPET};

const THREADS: u32 = 64 * 64;
const SAMPLES: u32 = 512;
/// Configurations (`config` in the kernel) and view angles.
const CONFIGS: u32 = 12;
const VIEWS: u32 = 4;
/// Per thread, configuration and view: sum and sum of squares of the
/// uniform estimate (luminance), of the sampled weight, the pdf's uniform
/// estimate, the largest weight, the pdf estimate's squares.
const STATS: usize = 7;

const KERNEL: &str = r#"//!use frame_surface
//!use frame_bsdf
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

fn h(x: u32) -> f32 {
    var v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return f32(((v >> 22u) ^ v) >> 8u) / 16777216.0;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The configurations: a white rough dielectric, a white Lambertian
// surface (F0 0), white metal at three roughnesses, a white anisotropic
// metal, a glossy dielectric, a thin two-sided leaf, white dielectrics
// where the old coupling gained most (F0 4% and 2% at roughness 0.3, 2% at
// 0.45), a white subsurface surface.
fn surface(config: u32) -> SurfaceData {
    let n = vec3<f32>(0.0, 0.0, 1.0);
    switch config {
        case 0u: { return surface_data(vec3<f32>(1.0), n, 0.8, 0.0, vec3<f32>(0.0)); }
        case 1u: {
            var s = surface_data(vec3<f32>(1.0), n, 0.5, 0.0, vec3<f32>(0.0));
            s.specular_f0 = vec3<f32>(0.0);
            return s;
        }
        case 2u: { return surface_data(vec3<f32>(1.0), n, 0.3, 1.0, vec3<f32>(0.0)); }
        case 3u: { return surface_data(vec3<f32>(1.0), n, 0.6, 1.0, vec3<f32>(0.0)); }
        case 4u: { return surface_data(vec3<f32>(1.0), n, 1.0, 1.0, vec3<f32>(0.0)); }
        case 5u: {
            var s = surface_data(vec3<f32>(1.0), n, 0.5, 1.0, vec3<f32>(0.0));
            s.flags = SURFACE_FLAG_ANISOTROPIC;
            s.roughness_aniso_x = 0.09;
            s.roughness_aniso_y = 0.4;
            s.aniso_rotation = 0.7;
            return s;
        }
        case 6u: { return surface_data(vec3<f32>(0.6, 0.3, 0.1), n, 0.45, 0.0, vec3<f32>(0.0)); }
        case 7u: {
            return surface_with_transmission(surface_data(vec3<f32>(0.5, 0.7, 0.3), n, 0.7, 0.0, vec3<f32>(0.0)), vec3<f32>(0.3, 0.45, 0.2));
        }
        case 8u: { return surface_data(vec3<f32>(1.0), n, 0.3, 0.0, vec3<f32>(0.0)); }
        case 9u: {
            var s = surface_data(vec3<f32>(1.0), n, 0.3, 0.0, vec3<f32>(0.0));
            s.specular_f0 = vec3<f32>(0.02);
            return s;
        }
        case 10u: {
            var s = surface_data(vec3<f32>(1.0), n, 0.45, 0.0, vec3<f32>(0.0));
            s.specular_f0 = vec3<f32>(0.02);
            return s;
        }
        default: {
            var s = surface_data(vec3<f32>(1.0), n, 0.5, 0.0, vec3<f32>(0.0));
            s.flags = SURFACE_FLAG_SUBSURFACE;
            s.subsurface_color = vec3<f32>(1.0);
            return s;
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    for (var config = 0u; config < 12u; config++) {
        let s = surface(config);
        for (var view = 0u; view < 4u; view++) {
            let c = array<f32, 4>(1.0, 0.5, 0.2, 0.05)[view];
            let wo = vec3<f32>(sqrt(1.0 - c * c), 0.0, c);
            var stats = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for (var k = 0u; k < 512u; k++) {
                let r = ((id.x * 512u + k) * 16u + config) * 4u + view;
                // A uniform direction over the sphere.
                let z = h(r * 8u) * 2.0 - 1.0;
                let phi = h(r * 8u + 1u) * 6.2831853;
                let wi = vec3<f32>(sqrt(max(1.0 - z * z, 0.0)) * cos(phi), sqrt(max(1.0 - z * z, 0.0)) * sin(phi), z);
                let uniform = luminance(bsdf_eval(s, wo, wi)) * 12.566371;
                stats[0] += uniform;
                stats[1] += uniform * uniform;
                let sample = bsdf_sample(s, wo, vec3<f32>(h(r * 8u + 2u), h(r * 8u + 3u), h(r * 8u + 4u)));
                let w = luminance(sample.weight);
                stats[2] += w;
                stats[3] += w * w;
                let density = bsdf_pdf(s, wo, wi) * 12.566371;
                stats[4] += density;
                stats[6] += density * density;
                stats[5] = max(stats[5], max(sample.weight.r, max(sample.weight.g, sample.weight.b)));
            }
            let at = ((id.x * 12u + config) * 4u + view) * 7u;
            for (var j = 0u; j < 7u; j++) { out[at + j] = stats[j]; }
        }
    }
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
    let source = helio_core::shader::resolve_with(KERNEL, &[SURFACE_SNIPPET, BSDF_SNIPPET]);
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
    let size = u64::from(THREADS * CONFIGS * VIEWS) * STATS as u64 * 4;
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
        pass.dispatch_workgroups(THREADS / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &read, 0, size);
    queue.submit([encoder.finish()]);
    read.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = read.slice(..).get_mapped_range().expect("mapped");
    bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
}

/// A configuration and view's totals: (mean, standard error) of the uniform
/// estimate and of the sampled weight, (mean, standard error) of the pdf's
/// integral, the largest weight.
fn totals(values: &[f32], config: u32, view: u32) -> ([f64; 2], [f64; 2], [f64; 2], f64) {
    let mut sums = [0.0f64; STATS];
    let mut most = 0.0f64;
    for thread in 0..THREADS {
        let at = (((thread * CONFIGS + config) * VIEWS + view) as usize) * STATS;
        for (k, sum) in sums.iter_mut().enumerate().take(5) {
            *sum += f64::from(values[at + k]);
        }
        most = most.max(f64::from(values[at + 5]));
        sums[6] += f64::from(values[at + 6]);
    }
    let n = f64::from(THREADS * SAMPLES);
    let stat = |sum: f64, squares: f64| {
        let mean = sum / n;
        [mean, ((squares / n - mean * mean).max(0.0) / n).sqrt()]
    };
    (
        stat(sums[0], sums[1]),
        stat(sums[2], sums[3]),
        stat(sums[4], sums[6]),
        most,
    )
}

#[test]
fn the_bsdf_conserves_energy_and_its_sampler_matches_it() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let values = run(&device, &queue);
    for config in 0..CONFIGS {
        for view in 0..VIEWS {
            let (uniform, sampled, pdf, most) = totals(&values, config, view);
            let label = format!("config {config}, view {view}");
            println!(
                "{label}: integral {:.4} (+-{:.4}), sampled {:.4} (+-{:.4}), pdf mass {:.4} (+-{:.4}), largest weight {most:.3}",
                uniform[0], uniform[1], sampled[0], sampled[1], pdf[0], pdf[1]
            );
            // The two estimates of the same integral agree.
            let tolerance = 5.0 * (uniform[1] * uniform[1] + sampled[1] * sampled[1]).sqrt() + 1e-4;
            assert!(
                (uniform[0] - sampled[0]).abs() <= tolerance,
                "{label}: the evaluated integral {} and the sampled estimate {} disagree",
                uniform[0],
                sampled[0]
            );
            assert!(
                pdf[0] <= 1.0 + 5.0 * pdf[1] + 1e-3,
                "{label}: pdf mass {} (+-{})",
                pdf[0],
                pdf[1]
            );
            // A white Lambertian surface reflects all it receives; white
            // dielectrics, metals and a wrapped diffuse surface no more.
            if config == 1 {
                assert!((sampled[0] - 1.0).abs() <= 1e-3, "{label}: {}", sampled[0]);
            }
            if config <= 5 || config >= 8 {
                assert!(
                    sampled[0] <= 1.0 + 5.0 * sampled[1] + 1e-4,
                    "{label}: a white surface reflects {} of a uniform environment",
                    sampled[0]
                );
            }
            // White metals: every specular sample's weight is F G2 / G1.
            if (2..=5).contains(&config) {
                assert!(most <= 1.0 + 1e-3, "{label}: a specular weight of {most}");
            }
        }
    }
}
