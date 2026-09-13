use helio_core::{PassContext, RenderGraph, RenderPass, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Original(u32);
struct Replacement;
impl RenderPass for Original {
    fn name(&self) -> &'static str {
        "original"
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn execute(&mut self, _: &mut PassContext) -> Result<()> {
        Ok(())
    }
}
impl RenderPass for Replacement {
    fn name(&self) -> &'static str {
        "replacement"
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn execute(&mut self, _: &mut PassContext) -> Result<()> {
        Ok(())
    }
}

#[test]
fn replacement_updates_first_instance_lookup_before_and_after_lock() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut graph = RenderGraph::new(&Arc::new(device), &queue);
    graph.add_pass(Box::new(Original(1)));
    graph.add_pass(Box::new(Original(2)));
    graph.replace_pass_at(0, Box::new(Replacement));
    assert!(graph.find_pass::<Replacement>().is_some());
    assert_eq!(graph.find_pass::<Original>().unwrap().0, 2);
    graph.lock(8, 8);
    graph.replace_pass_at(1, Box::new(Replacement));
    assert!(graph.find_pass::<Original>().is_none());
    assert_eq!(graph.pass_index_of::<Replacement>(), Some(0));
    assert!(graph.find_pass_mut::<Replacement>().is_some());
}

struct BundleObserver {
    builds: Arc<AtomicUsize>,
    expect_resize: bool,
    size: Option<[u32; 2]>,
}

impl RenderPass for BundleObserver {
    fn name(&self) -> &'static str {
        "bundle observer"
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn execute(&mut self, _: &mut PassContext) -> Result<()> {
        Ok(())
    }
    fn on_resize(&mut self, _: &wgpu::Device, width: u32, height: u32) {
        self.size = Some([width, height]);
    }
    fn build_gpu_render_bundle(
        &mut self,
        _: &wgpu::Device,
        _: &libhelio::FrameResources<'_>,
    ) -> Option<wgpu::RenderBundle> {
        if self.expect_resize {
            assert_eq!(
                self.size,
                Some([8, 16]),
                "replacement must receive its size before bundles are built"
            );
        }
        self.builds.fetch_add(1, Ordering::SeqCst);
        None
    }
}

#[test]
fn replacement_rebuilds_dependent_bundles_even_when_chains_do_not_change() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let device = Arc::new(device);
    for locked in [false, true] {
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(Original(1)));
        let dependent_builds = Arc::new(AtomicUsize::new(0));
        graph.add_pass(Box::new(BundleObserver {
            builds: dependent_builds.clone(),
            expect_resize: false,
            size: None,
        }));
        if locked {
            graph.lock(8, 16);
        } else {
            graph.init_transients(8, 16);
        }
        let prior_builds = dependent_builds.load(Ordering::SeqCst);
        assert!(prior_builds > 0);
        let replacement_builds = Arc::new(AtomicUsize::new(0));
        graph.replace_pass_at(
            0,
            Box::new(BundleObserver {
                builds: replacement_builds.clone(),
                expect_resize: true,
                size: None,
            }),
        );
        assert!(replacement_builds.load(Ordering::SeqCst) > 0);
        assert!(
            dependent_builds.load(Ordering::SeqCst) > prior_builds,
            "a later bundle must be rebuilt when an earlier resource publisher changes"
        );
        assert_eq!(
            graph.find_pass::<BundleObserver>().unwrap().size,
            Some([8, 16])
        );
    }
}
