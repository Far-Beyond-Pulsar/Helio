//! Pipelines built off the frame (`helio_core::compile`): a bundle is ready
//! once its job ends, a shader the backend rejects is the bundle's failure
//! (not a panic on the asking thread), and the in-flight count returns to
//! zero. Skips without a GPU adapter, like this crate's other hardware tests.

use helio_core::compile::{in_flight, Compiling};

const VALID: &str = "@compute @workgroup_size(1) fn main() {}";
const INVALID: &str = "@compute @workgroup_size(1) fn main() { let x: u32 = 1.5; }";

fn device() -> Option<wgpu::Device> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .ok()?;
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;
        Some(device)
    })
}

fn pipeline(device: &wgpu::Device, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compile job test"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    helio_core::shader::create_compute_pipeline(
        device,
        wgpu::ComputePipelineDescriptor {
            label: Some("compile job test"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        },
    )
}

#[test]
fn a_bundle_compiles_off_the_asking_thread_and_failures_are_its_own() {
    let Some(device) = device() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: compile jobs");
        return;
    };
    let asking = std::thread::current().id();
    let built = Compiling::start(&device, "valid test pipeline", move |device| {
        (pipeline(device, VALID), std::thread::current().id())
    });
    let rejected = Compiling::start(&device, "invalid test pipeline", |device| {
        pipeline(device, INVALID)
    });

    let (_, on) = built.wait().expect("a valid shader compiles");
    assert_ne!(*on, asking, "the bundle was built on a worker");
    assert!(built.ready().is_some() && built.failed().is_none());
    assert!(!built.is_pending());

    let reason = rejected
        .wait()
        .expect_err("an invalid shader is the bundle's failure");
    assert!(!reason.is_empty());
    assert!(rejected.ready().is_none());
    assert_eq!(rejected.failed(), Some(reason));

    assert_eq!(in_flight(), 0, "every started job ended");
}
