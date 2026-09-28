//! Pass-owned SceneDB settings resolution, including empty scenes.
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use pulsar_scenedb::gpu::BufferKey;
use wgpu::util::DeviceExt;

pub struct PostProcessVolumeBlendPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    defaults: wgpu::Buffer,
    blend_output_buf: wgpu::Buffer,
    resolved: wgpu::Buffer,
    fallback_pp_volumes: wgpu::Buffer,
    fallback_cameras: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<[wgpu::Buffer; 3]>,
    /// Published as `"dof_maybe_active"`; see [`DOF_MAYBE_ACTIVE`].
    dof: SettingActivity,
    /// Published as `"bloom_maybe_active"`; see [`BLOOM_MAYBE_ACTIVE`].
    bloom: SettingActivity,
    /// Published as `"auto_exposure_maybe_active"`; see
    /// [`AUTO_EXPOSURE_MAYBE_ACTIVE`].
    auto_exposure: SettingActivity,
}

/// Whether any source the resolver blends from can turn one setting on this
/// frame: its defaults, an enabled camera row, or a weighted volume row that
/// overrides the setting. Hosts register the columns up front, so the rows
/// are read back when SceneDB reports new contents (`SceneBufferLiveness`)
/// and count as "maybe" until read.
struct SettingActivity {
    defaults: bool,
    camera: helio_core::SceneBufferLiveness,
    volume: helio_core::SceneBufferLiveness,
    maybe_active: bool,
}

impl SettingActivity {
    fn new(defaults: bool, camera_row: fn(&[u8]) -> bool, volume_row: fn(&[u8]) -> bool) -> Self {
        Self {
            defaults,
            camera: helio_core::SceneBufferLiveness::with_row_predicate(camera_row),
            volume: helio_core::SceneBufferLiveness::with_row_predicate(volume_row),
            maybe_active: true,
        }
    }

    fn update(
        &mut self,
        ctx: &PrepareContext,
        cameras: Option<&helio_core::BufferHandle>,
        volumes: Option<&helio_core::BufferHandle>,
    ) {
        self.camera.update(ctx.device, ctx.queue, cameras);
        self.volume.update(ctx.device, ctx.queue, volumes);
        self.maybe_active = self.defaults
            || cameras.is_some_and(|handle| self.camera.maybe_live(handle))
            || volumes.is_some_and(|handle| self.volume.maybe_live(handle));
    }
}

/// `bool` registry key: false only when no source the resolver blends from
/// (defaults, camera rows, volumes) can enable depth of field, so the
/// resolved `dof_aperture_shape` is certainly negative this frame. Hosts
/// register these columns up front, so the check reads the rows back when
/// they change (`SceneBufferLiveness`) and is true until they have been read.
pub const DOF_MAYBE_ACTIVE: &str = "dof_maybe_active";

/// `bool` registry key: false only when no source the resolver blends from
/// can enable bloom, so the resolved `bloom_enabled` is certainly 0 this
/// frame and nothing samples the bloom mips. Tracked like
/// [`DOF_MAYBE_ACTIVE`].
pub const BLOOM_MAYBE_ACTIVE: &str = "bloom_maybe_active";

/// `bool` registry key: false only when no source the resolver blends from
/// can select auto exposure, so the resolved `exposure_mode` is certainly
/// manual this frame and nothing reads the metered luminance. Tracked like
/// [`DOF_MAYBE_ACTIVE`].
pub const AUTO_EXPOSURE_MAYBE_ACTIVE: &str = "auto_exposure_maybe_active";

/// The resolver treats any shape that is not negative (including NaN) as DOF.
fn shape_enables_dof(shape: f32) -> bool {
    !(shape < 0.0)
}

fn read_f32(row: &[u8], offset: usize) -> Option<f32> {
    row.get(offset..offset + 4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))
}

fn read_u32(row: &[u8], offset: usize) -> Option<u32> {
    row.get(offset..offset + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

const DOF_SHAPE: usize = std::mem::offset_of!(crate::GpuPostProcessUniforms, dof_aperture_shape);
const BLOOM_ENABLED: usize = std::mem::offset_of!(crate::GpuPostProcessUniforms, bloom_enabled);
const EXPOSURE_MODE: usize = std::mem::offset_of!(crate::GpuPostProcessUniforms, exposure_mode);

/// An enabled camera row whose settings enable DOF. The resolver also
/// matches `view_id`; ignoring it here only errs toward "maybe".
fn camera_row_enables_dof(row: &[u8]) -> bool {
    use crate::CameraPostProcessComponent as C;
    let enabled = read_u32(row, std::mem::offset_of!(C, enabled));
    let shape = read_f32(row, std::mem::offset_of!(C, settings) + DOF_SHAPE);
    match (enabled, shape) {
        (Some(enabled), Some(shape)) => enabled != 0 && shape_enables_dof(shape),
        _ => true,
    }
}

/// A weighted volume row that overrides property 37 (the aperture shape,
/// which encodes DOF on/off) with a value that enables DOF.
fn volume_row_enables_dof(row: &[u8]) -> bool {
    use crate::GpuPostProcessVolume as V;
    const PROPERTY: usize = 37;
    let weight = read_f32(row, std::mem::offset_of!(V, blend_weight));
    let mask = read_u32(row, std::mem::offset_of!(V, override_mask) + PROPERTY / 32 * 4);
    let shape = read_f32(row, std::mem::offset_of!(V, settings) + DOF_SHAPE);
    match (weight, mask, shape) {
        (Some(weight), Some(mask), Some(shape)) => {
            !(weight <= 0.0) && mask & (1 << (PROPERTY % 32)) != 0 && shape_enables_dof(shape)
        }
        _ => true,
    }
}
/// An enabled camera row whose settings enable bloom. Like
/// [`camera_row_enables_dof`], ignoring `view_id` errs toward "maybe".
fn camera_row_enables_bloom(row: &[u8]) -> bool {
    use crate::CameraPostProcessComponent as C;
    let enabled = read_u32(row, std::mem::offset_of!(C, enabled));
    let bloom = read_u32(row, std::mem::offset_of!(C, settings) + BLOOM_ENABLED);
    match (enabled, bloom) {
        (Some(enabled), Some(bloom)) => enabled != 0 && bloom != 0,
        _ => true,
    }
}

/// A weighted volume row that overrides property 9 (`bloom_enabled`) on. A
/// volume can only select between its own value and the baseline's, so one
/// that overrides bloom off never enables it.
fn volume_row_enables_bloom(row: &[u8]) -> bool {
    use crate::GpuPostProcessVolume as V;
    const PROPERTY: usize = 9;
    let weight = read_f32(row, std::mem::offset_of!(V, blend_weight));
    let mask = read_u32(row, std::mem::offset_of!(V, override_mask) + PROPERTY / 32 * 4);
    let bloom = read_u32(row, std::mem::offset_of!(V, settings) + BLOOM_ENABLED);
    match (weight, mask, bloom) {
        (Some(weight), Some(mask), Some(bloom)) => {
            !(weight <= 0.0) && mask & (1 << (PROPERTY % 32)) != 0 && bloom != 0
        }
        _ => true,
    }
}

/// An enabled camera row whose exposure mode is not manual (0). Like
/// [`camera_row_enables_dof`], ignoring `view_id` errs toward "maybe".
fn camera_row_enables_auto_exposure(row: &[u8]) -> bool {
    use crate::CameraPostProcessComponent as C;
    let enabled = read_u32(row, std::mem::offset_of!(C, enabled));
    let mode = read_u32(row, std::mem::offset_of!(C, settings) + EXPOSURE_MODE);
    match (enabled, mode) {
        (Some(enabled), Some(mode)) => enabled != 0 && mode != 0,
        _ => true,
    }
}

/// A weighted volume row that overrides property 0 (`exposure_mode`) with a
/// mode other than manual. Like bloom, a volume only selects between its own
/// mode and the baseline's.
fn volume_row_enables_auto_exposure(row: &[u8]) -> bool {
    use crate::GpuPostProcessVolume as V;
    const PROPERTY: usize = 0;
    let weight = read_f32(row, std::mem::offset_of!(V, blend_weight));
    let mask = read_u32(row, std::mem::offset_of!(V, override_mask) + PROPERTY / 32 * 4);
    let mode = read_u32(row, std::mem::offset_of!(V, settings) + EXPOSURE_MODE);
    match (weight, mask, mode) {
        (Some(weight), Some(mask), Some(mode)) => {
            !(weight <= 0.0) && mask & (1 << (PROPERTY % 32)) != 0 && mode != 0
        }
        _ => true,
    }
}

impl PostProcessVolumeBlendPass {
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_defaults(device, &crate::PostProcessSettings::default())
    }
    pub fn with_defaults(device: &wgpu::Device, settings: &crate::PostProcessSettings) -> Self {
        let shader = helio_core::shader::module(device, "PostProcess Resolver", include_str!("../shaders/postprocess.wgsl"));
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Resolver BGL"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(15, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(16, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(20, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Resolver PL"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PostProcess Resolver"), layout: Some(&layout), module: &shader,
            entry_point: Some("cs_volume_blend"), compilation_options: Default::default(), cache: None,
        });
        let defaults = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PostProcess Defaults"), contents: bytemuck::bytes_of(&settings.to_gpu()), usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let size = std::mem::size_of::<crate::GpuPostProcessUniforms>() as u64;
        Self {
            pipeline, bgl, defaults,
            blend_output_buf: buffer("PostProcess Resolve Storage", size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            resolved: buffer("PostProcess Resolved Uniforms", size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC),
            fallback_pp_volumes: buffer("PostProcess Empty Volumes", std::mem::size_of::<crate::GpuPostProcessVolume>() as u64, wgpu::BufferUsages::STORAGE),
            fallback_cameras: buffer("PostProcess Empty Cameras", std::mem::size_of::<crate::CameraPostProcessComponent>() as u64, wgpu::BufferUsages::STORAGE),
            bind_group: None, bind_group_key: None,
            dof: SettingActivity::new(
                shape_enables_dof(settings.to_gpu().dof_aperture_shape),
                camera_row_enables_dof,
                volume_row_enables_dof,
            ),
            bloom: SettingActivity::new(
                settings.to_gpu().bloom_enabled != 0,
                camera_row_enables_bloom,
                volume_row_enables_bloom,
            ),
            auto_exposure: SettingActivity::new(
                settings.to_gpu().exposure_mode != 0,
                camera_row_enables_auto_exposure,
                volume_row_enables_auto_exposure,
            ),
        }
    }
    /// GPU-derived settings, valid after the resolver dispatch and copy.
    pub fn resolved_uniforms(&self) -> &wgpu::Buffer { &self.resolved }
}
impl RenderPass for PostProcessVolumeBlendPass {
    fn name(&self) -> &'static str { "PostProcessVolumeBlendPass" }
    fn declare_resources(&self, builder: &mut ResourceBuilder) { builder.write_buffer("postprocess_uniforms"); }
    fn writes(&self) -> &'static [&'static str] { &["postprocess_uniforms"] }
    fn render_pass_descriptor<'a>(&'a self, _: &'a wgpu::TextureView, _: &'a wgpu::TextureView, _: &'a helio_core::ResourceRegistry<'a>) -> Option<wgpu::RenderPassDescriptor<'a>> { None }
    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let cameras = ctx.scene_buffers.get(BufferKey::of("camera_postprocess"));
        let volumes = ctx.scene_buffers.get(BufferKey::of("post_process_volumes"));
        self.dof.update(ctx, cameras, volumes);
        self.bloom.update(ctx, cameras, volumes);
        self.auto_exposure.update(ctx, cameras, volumes);
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let volumes = ctx.scene_buffers.get(BufferKey::of("post_process_volumes")).map(|h| &h.buffer).unwrap_or(&self.fallback_pp_volumes);
        let cameras = ctx.scene_buffers.get(BufferKey::of("camera_postprocess")).map(|h| &h.buffer).unwrap_or(&self.fallback_cameras);
        let key = [ctx.camera.clone(), volumes.clone(), cameras.clone()];
        if self.bind_group_key.as_ref() != Some(&key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PostProcess Resolver BG"), layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.defaults.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: volumes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 16, resource: self.blend_output_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 20, resource: cameras.as_entire_binding() },
                ],
            }));
            self.bind_group_key = Some(key);
        }
        // Record with the fog consumers to preserve producer/copy/consumer order.
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PostProcess Resolve"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.blend_output_buf, 0, &self.resolved, 0, std::mem::size_of::<crate::GpuPostProcessUniforms>() as u64);
        Ok(())
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let buffer: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.resolved) };
        frame.write(ResourceKey::new("postprocess_uniforms"), buffer, self.name());
        frame.write(ResourceKey::new(DOF_MAYBE_ACTIVE), self.dof.maybe_active, self.name());
        frame.write(ResourceKey::new(BLOOM_MAYBE_ACTIVE), self.bloom.maybe_active, self.name());
        frame.write(
            ResourceKey::new(AUTO_EXPOSURE_MAYBE_ACTIVE),
            self.auto_exposure.maybe_active,
            self.name(),
        );
    }
}

#[cfg(test)]
mod activity_tests {
    use super::*;
    use crate::{CameraPostProcessComponent, GpuPostProcessVolume, PostProcessSettings};

    fn settings(dof: bool) -> PostProcessSettings {
        let mut settings = PostProcessSettings::default();
        settings.dof_enabled = dof;
        settings
    }

    #[test]
    fn camera_rows() {
        let row = |dof, enabled| {
            let mut row = CameraPostProcessComponent::new(0, &settings(dof));
            row.enabled = enabled;
            row
        };
        assert!(camera_row_enables_dof(bytemuck::bytes_of(&row(true, 1))));
        assert!(!camera_row_enables_dof(bytemuck::bytes_of(&row(false, 1))));
        assert!(!camera_row_enables_dof(bytemuck::bytes_of(&row(true, 0))), "disabled rows are not blended");
        assert!(!camera_row_enables_dof(&[0u8; std::mem::size_of::<CameraPostProcessComponent>()]), "empty row");
        assert!(camera_row_enables_dof(&[0u8; 8]), "a short row errs toward maybe");
    }

    #[test]
    fn volume_rows() {
        let row = |dof, weight, overrides| {
            let mut volume: GpuPostProcessVolume = bytemuck::Zeroable::zeroed();
            volume.settings = settings(dof).to_gpu();
            volume.blend_weight = weight;
            if overrides {
                volume.override_mask[1] |= 1 << 5; // property 37
            }
            volume
        };
        assert!(volume_row_enables_dof(bytemuck::bytes_of(&row(true, 1.0, true))));
        assert!(!volume_row_enables_dof(bytemuck::bytes_of(&row(true, 1.0, false))), "does not override DOF");
        assert!(!volume_row_enables_dof(bytemuck::bytes_of(&row(true, 0.0, true))), "zero weight is inactive");
        assert!(!volume_row_enables_dof(bytemuck::bytes_of(&row(false, 1.0, true))), "overrides DOF off");
        assert!(!volume_row_enables_dof(&[0u8; std::mem::size_of::<GpuPostProcessVolume>()]), "empty row");
    }

    #[test]
    fn bloom_camera_rows() {
        let row = |bloom, enabled| {
            let mut settings = PostProcessSettings::default();
            settings.bloom_enabled = bloom;
            let mut row = CameraPostProcessComponent::new(0, &settings);
            row.enabled = enabled;
            row
        };
        assert!(camera_row_enables_bloom(bytemuck::bytes_of(&row(true, 1))));
        assert!(!camera_row_enables_bloom(bytemuck::bytes_of(&row(false, 1))));
        assert!(!camera_row_enables_bloom(bytemuck::bytes_of(&row(true, 0))), "disabled rows are not blended");
        assert!(!camera_row_enables_bloom(&[0u8; std::mem::size_of::<CameraPostProcessComponent>()]), "empty row");
        assert!(camera_row_enables_bloom(&[0u8; 8]), "a short row errs toward maybe");
    }

    #[test]
    fn bloom_volume_rows() {
        let row = |bloom, weight, overrides| {
            let mut settings = PostProcessSettings::default();
            settings.bloom_enabled = bloom;
            let mut volume: GpuPostProcessVolume = bytemuck::Zeroable::zeroed();
            volume.settings = settings.to_gpu();
            volume.blend_weight = weight;
            if overrides {
                volume.override_mask[0] |= 1 << 9;
            }
            volume
        };
        assert!(volume_row_enables_bloom(bytemuck::bytes_of(&row(true, 1.0, true))));
        assert!(!volume_row_enables_bloom(bytemuck::bytes_of(&row(true, 1.0, false))), "does not override bloom");
        assert!(!volume_row_enables_bloom(bytemuck::bytes_of(&row(true, 0.0, true))), "zero weight is inactive");
        assert!(!volume_row_enables_bloom(bytemuck::bytes_of(&row(false, 1.0, true))), "overrides bloom off");
        assert!(!volume_row_enables_bloom(&[0u8; std::mem::size_of::<GpuPostProcessVolume>()]), "empty row");
    }

    #[test]
    fn auto_exposure_rows() {
        let camera = |auto, enabled| {
            let mut settings = PostProcessSettings::default();
            if auto {
                settings.exposure_mode = crate::ExposureMode::Auto;
            }
            let mut row = CameraPostProcessComponent::new(0, &settings);
            row.enabled = enabled;
            row
        };
        assert!(camera_row_enables_auto_exposure(bytemuck::bytes_of(&camera(true, 1))));
        assert!(!camera_row_enables_auto_exposure(bytemuck::bytes_of(&camera(false, 1))));
        assert!(!camera_row_enables_auto_exposure(bytemuck::bytes_of(&camera(true, 0))));
        assert!(camera_row_enables_auto_exposure(&[0u8; 8]), "a short row errs toward maybe");

        let volume = |auto, weight, overrides| {
            let mut settings = PostProcessSettings::default();
            if auto {
                settings.exposure_mode = crate::ExposureMode::Auto;
            }
            let mut volume: GpuPostProcessVolume = bytemuck::Zeroable::zeroed();
            volume.settings = settings.to_gpu();
            volume.blend_weight = weight;
            if overrides {
                volume.override_mask[0] |= 1;
            }
            volume
        };
        assert!(volume_row_enables_auto_exposure(bytemuck::bytes_of(&volume(true, 1.0, true))));
        assert!(!volume_row_enables_auto_exposure(bytemuck::bytes_of(&volume(true, 1.0, false))));
        assert!(!volume_row_enables_auto_exposure(bytemuck::bytes_of(&volume(true, 0.0, true))));
        assert!(!volume_row_enables_auto_exposure(bytemuck::bytes_of(&volume(false, 1.0, true))));
        assert_eq!(PostProcessSettings::default().to_gpu().exposure_mode, 0, "manual by default");
    }

    #[test]
    fn default_settings_leave_bloom_off() {
        assert_eq!(PostProcessSettings::default().to_gpu().bloom_enabled, 0);
    }

    #[test]
    fn default_settings_leave_dof_off() {
        assert!(!shape_enables_dof(PostProcessSettings::default().to_gpu().dof_aperture_shape));
        assert!(shape_enables_dof(f32::NAN), "the resolver treats NaN as enabled");
    }
}
