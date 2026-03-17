//! Texture creation and sky ambient estimation helpers.

use crate::passes::GBufferTargets;

// On wasm/webgpu some half‑float formats are not supported for sampling.  fall
// back to 8‑bit UNORM variants when compiling to wasm to avoid validation
// errors.  The shader side just reads them as `vec4<f32>` so the difference is
// transparent.
#[cfg(target_arch = "wasm32")]
const GBUF_FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
#[cfg(not(target_arch = "wasm32"))]
const GBUF_FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Create a Depth32Float texture + two views (write + depth-only-sample) at the given resolution.
/// Both RENDER_ATTACHMENT and TEXTURE_BINDING are set so the deferred lighting pass can read depth.
pub(super) fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    // Full-aspect view for render pass attachment
    let write_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    // Depth-only view for texture binding in the deferred lighting bind group
    let sample_view = tex.create_view(&wgpu::TextureViewDescriptor {
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    (tex, write_view, sample_view)
}

/// Create all G-buffer textures and return their views packaged as `GBufferTargets`.
pub(super) fn create_gbuffer_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (
    wgpu::Texture,
    wgpu::Texture,
    wgpu::Texture,
    wgpu::Texture,
    GBufferTargets,
) {
    let make = |label: &str, format: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let albedo_tex  = make("GBuf Albedo",   wgpu::TextureFormat::Rgba8Unorm);
    let normal_tex  = make("GBuf Normal",   GBUF_FLOAT_FORMAT);
    let orm_tex     = make("GBuf ORM",      wgpu::TextureFormat::Rgba8Unorm);
    let emissive_tex = make("GBuf Emissive", GBUF_FLOAT_FORMAT);
    let albedo_view   = albedo_tex.create_view(&Default::default());
    let normal_view   = normal_tex.create_view(&Default::default());
    let orm_view      = orm_tex.create_view(&Default::default());
    let emissive_view = emissive_tex.create_view(&Default::default());
    let targets = GBufferTargets { albedo_view, normal_view, orm_view, emissive_view };
    (albedo_tex, normal_tex, orm_tex, emissive_tex, targets)
}

/// Build the G-buffer read bind group used by the deferred lighting pass (group 1).
pub(super) fn create_gbuffer_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    targets: &GBufferTargets,
    depth_sample_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GBuffer Read Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&targets.albedo_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&targets.normal_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&targets.orm_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&targets.emissive_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth_sample_view) },
        ],
    })
}

/// Approximate sky zenith ambient colour from sun elevation (-1=night, 0=horizon, 1=zenith).
pub(super) fn estimate_sky_ambient(sun_elev: f32, rayleigh: &[f32; 3]) -> [f32; 3] {
    if sun_elev < -0.05 {
        // Night — deep twilight fading to near-black
        let t = ((-sun_elev - 0.05) / 0.95).clamp(0.0, 1.0);
        lerp3([0.04, 0.06, 0.15], [0.01, 0.01, 0.02], t)
    } else if sun_elev < 0.15 {
        // Dawn/dusk — warm golden ambient transitioning to daylight
        let t = ((sun_elev + 0.05) / 0.2).clamp(0.0, 1.0);
        lerp3([0.04, 0.06, 0.15], [0.55, 0.38, 0.20], t)
    } else {
        // Day — derive sky blue from Rayleigh scattering coefficients.
        // Rayleigh coefficients are in km⁻¹ (~0.006–0.033 for Earth) so they
        // need large multipliers to produce visible ambient colours.
        let t = ((sun_elev - 0.15) / 0.85).clamp(0.0, 1.0);
        let day_blue = [rayleigh[0] * 70.0, rayleigh[1] * 45.0, rayleigh[2] * 25.0];
        let noon = [day_blue[0].min(0.7), day_blue[1].min(0.85), day_blue[2].min(1.0)];
        lerp3([0.55, 0.38, 0.20], noon, t)
    }
}

pub(super) fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]
}
