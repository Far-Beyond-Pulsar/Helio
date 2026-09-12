//! Regression test for `RenderGraph::declare_external_input` (Phase 1 of
//! `docs/helio_3_0_spec.md`, §6).
//!
//! Before this phase, `RenderGraph::validate_dependencies()` hardcoded a
//! literal list of "always available" resource names (`"main_scene"`,
//! `"vg"`, `"billboards"`, `"corona_emitters"`, `"depth_texture"`) with no
//! registered link to where those values actually come from (V3 in the
//! spec's audit). This test proves the replacement mechanism actually
//! *validates* something rather than just relocating the same hardcoded
//! exception: a resource that is neither written by any pass in the graph
//! nor registered via `declare_external_input` must fail validation, and
//! registering it must make that same graph validate successfully.

use helio_core::{PassContext, RenderGraph, RenderPass, Result as HelioResult};
use std::sync::Arc;

/// A pass that reads a resource no pass in the graph writes. Used to prove
/// `validate_dependencies()` treats an unregistered, unwritten resource as a
/// real error rather than silently allowing it.
struct OrphanReadPass;

impl RenderPass for OrphanReadPass {
    fn name(&self) -> &'static str {
        "OrphanReadPass"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["orphan_resource"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

#[test]
fn declare_external_input_is_the_source_of_truth_for_validation() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: external input declaration");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("External Input Declaration Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create an external-input-declaration device");

        let device = Arc::new(device);

        // No pass writes "orphan_resource" and nothing has registered it as
        // an external input yet — this must be a real validation error, not
        // a silent pass (the exact bug hardcoding a literal list caused: a
        // resource with no known writer being treated as available anyway).
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(OrphanReadPass));
        let err = graph.validate_dependencies().expect_err(
            "a read with no writer and no declared external input must fail validation",
        );
        assert!(
            err.contains("orphan_resource"),
            "validation error should name the unresolved resource, got: {err}"
        );

        // Registering it as an external input is the one legitimate way to
        // satisfy the read — mirrors how the host `Renderer` supplies
        // main_scene/vg/billboards/corona_emitters today.
        graph.declare_external_input("orphan_resource");
        graph
            .validate_dependencies()
            .expect("a declared external input must satisfy a pass's read of that resource");
    });
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}
