//! Regression test for `ResourceBuilder::write_group` / `RenderPass::publish_group`
//! (Phase 2 of `docs/helio_3_0_spec.md`, §5 — "View groups (replaces the
//! GBuffer special case)").
//!
//! Before this phase, GBuffer's 4-view bundle was special-cased three times
//! inside `helio-core` by exact string matching (`resource_lifetime.rs`'s
//! allocator, `execution.rs`'s per-frame loop, and `execution.rs`'s `lock()`
//! canon-building), and `GBufferPass::publish()` was an empty no-op — all of
//! the bundling happened in core, not in the pass that owns the data. This
//! test proves the generic replacement actually delivers correctly-ordered,
//! correctly-identified views end-to-end:
//!
//!   declare_resources' `write_group` call
//!     -> the allocator's generic grouping (no name pattern-matching)
//!     -> `RenderPass::publish_group` (owned by the producing pass, not core)
//!     -> a downstream consumer reading the bundled `FrameResources` field
//!
//! using a GBufferPass-*shaped* stand-in pass (same group name, member names,
//! and `publish_group` body GBufferPass itself uses) rather than the real
//! `GBufferPass`, to avoid pulling in its material/template-registry/mesh
//! dependencies that are irrelevant to what this phase changed.

use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{PassContext, RenderGraph, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};
mod support;

const GROUP_NAMES: [&str; 4] = [
    "gbuffer_albedo",
    "gbuffer_normal",
    "gbuffer_orm",
    "gbuffer_emissive",
];

/// Stand-in for `GBufferPass`: declares the same 4-view "gbuffer" write_group
/// and publishes it the same way `GBufferPass::publish_group` does, without
/// any of `GBufferPass`'s unrelated machinery.
struct StandInGBufferPass {
    /// Pool views captured during `execute()` (where `ctx.resource_pool` is
    /// available), keyed by declared member order. Used to prove
    /// `publish_group`'s `views` really are the pool-allocated views for
    /// this group's declared names, not just *some* four views.
    ///
    /// Compared with `wgpu::TextureView`'s own `PartialEq` (proxying to the
    /// underlying resource's identity) rather than by the Rust struct's
    /// address — `wgpu::TextureView::clone()` (used internally when a view
    /// is threaded through `PrePassAction`) legitimately produces a new
    /// struct instance at a new address for the *same* underlying view, so
    /// an address comparison would fail even on fully correct behavior.
    pool_views: Arc<Mutex<Option<[wgpu::TextureView; 4]>>>,
    /// Views actually handed to `publish_group`, in the order received.
    publish_views: Arc<Mutex<Option<[wgpu::TextureView; 4]>>>,
}

impl RenderPass for StandInGBufferPass {
    fn name(&self) -> &'static str {
        "StandInGBuffer"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_group(
            "gbuffer",
            [
                ("gbuffer_albedo", ResourceFormat::Rgba8UnormSrgb),
                ("gbuffer_normal", ResourceFormat::Rgba8UnormSrgb),
                ("gbuffer_orm", ResourceFormat::Rgba8UnormSrgb),
                ("gbuffer_emissive", ResourceFormat::Rgba8UnormSrgb),
            ],
            ResourceSize::Output,
        );
    }

    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let views = GROUP_NAMES.map(|n| {
            ctx.resource_pool
                .get_view(n)
                .unwrap_or_else(|| {
                    panic!("declared write_group member '{n}' must be pool-allocated")
                })
                .clone()
        });
        *self.pool_views.lock().unwrap() = Some(views);
        Ok(())
    }

    /// Exactly what `GBufferPass::publish_group` does: turn the generically-
    /// resolved "gbuffer" group into the stable bundled contract downstream
    /// passes read as `frame.gbuffer`.
    fn publish_group<'a>(
        &self,
        group_name: &'static str,
        views: &[&'a wgpu::TextureView],
        frame: &mut libhelio::FrameResources<'a>,
    ) {
        if group_name != "gbuffer" {
            return;
        }
        let [albedo, normal, orm, emissive] = *views else {
            panic!(
                "expected exactly 4 views for the \"gbuffer\" group, got {}",
                views.len()
            );
        };
        *self.publish_views.lock().unwrap() = Some([
            albedo.clone(),
            normal.clone(),
            orm.clone(),
            emissive.clone(),
        ]);
        frame.gbuffer.write(
            libhelio::GBufferViews {
                albedo,
                normal,
                orm,
                emissive,
            },
            "StandInGBuffer",
        );
    }
}

/// Downstream consumer reading the bundled contract, exactly as
/// `DeferredLightPass`/`SsaoPass`/`SsrPass`/etc. do in the real graph.
struct ConsumerPass {
    seen_views: Arc<Mutex<Option<[wgpu::TextureView; 4]>>>,
}

impl RenderPass for ConsumerPass {
    fn name(&self) -> &'static str {
        "Consumer"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gb = ctx
            .resources
            .gbuffer
            .read("Consumer")
            .expect("gbuffer must already be published by the time Consumer executes");
        *self.seen_views.lock().unwrap() = Some([
            gb.albedo.clone(),
            gb.normal.clone(),
            gb.orm.clone(),
            gb.emissive.clone(),
        ]);
        Ok(())
    }
}

#[test]
fn write_group_delivers_correctly_ordered_views_to_a_downstream_consumer() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: write_group bundle");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Write Group Bundle Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let pool_views = Arc::new(Mutex::new(None));
        let publish_views = Arc::new(Mutex::new(None));
        let seen_views = Arc::new(Mutex::new(None));

        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(StandInGBufferPass {
            pool_views: pool_views.clone(),
            publish_views: publish_views.clone(),
        }));
        graph.add_pass(Box::new(ConsumerPass {
            seen_views: seen_views.clone(),
        }));
        graph
            .validate_dependencies()
            .expect("Consumer's read of \"gbuffer\" must be satisfied by StandInGBuffer's write");
        graph.lock(64, 64);

        let scene = helio_core::GpuScene::new(device.clone(), queue.clone());
        let scene_input = support::SceneInputAdapter(&scene);
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Write Group Swapchain Stand-in"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Write Group Depth Stand-in"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        graph
            .execute(&scene_input, &target_view, &depth_view)
            .expect("frame should execute");

        let pool = pool_views
            .lock()
            .unwrap()
            .clone()
            .expect("StandInGBuffer's execute() must run and see its own pool-allocated views");
        let published = publish_views
            .lock()
            .unwrap()
            .clone()
            .expect("publish_group must be called for the declared \"gbuffer\" group");
        let seen = seen_views
            .lock()
            .unwrap()
            .clone()
            .expect("Consumer must read a populated gbuffer bundle, not an unwritten slot");

        assert_eq!(
            published, pool,
            "publish_group's views must be exactly the pool-allocated views for \
             [\"gbuffer_albedo\", \"gbuffer_normal\", \"gbuffer_orm\", \"gbuffer_emissive\"], \
             in that declared order — proving write_group's grouping is order-preserving \
             and not a coincidental match"
        );
        assert_eq!(
            seen, pool,
            "the downstream consumer must receive the exact same views the allocator \
             resolved for this group through frame.gbuffer — not merely a non-panicking read"
        );
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
