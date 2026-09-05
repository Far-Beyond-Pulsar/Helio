use crate::{
    pipelines::Pipelines,
    resources::{Fallbacks, Targets},
};

pub(crate) fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource<'_>],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, resource)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}
fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}

pub(crate) struct InternalBindings {
    pub grid: wgpu::BindGroup,
    pub sample: [wgpu::BindGroup; 2],
    pub temporal: [wgpu::BindGroup; 2],
    pub composite: [wgpu::BindGroup; 2],
}
impl InternalBindings {
    pub fn new(device: &wgpu::Device, p: &Pipelines, t: &Targets) -> Self {
        let grid = bind_group(
            device,
            "HLFS grid resources",
            &p.grid_bgl,
            &[
                t.coarse.as_entire_binding(),
                t.grid.as_entire_binding(),
                view(&t.depth_bounds.view),
            ],
        );
        let sample = std::array::from_fn(|write| {
            let previous = &t.history[1 - write];
            bind_group(
                device,
                "HLFS sample resources",
                &p.sample_bgl,
                &[
                    t.grid.as_entire_binding(),
                    previous.visible.as_entire_binding(),
                    t.history[write].visible.as_entire_binding(),
                    view(&t.raw_diffuse.view),
                    view(&t.raw_specular.view),
                    view(&previous.geometry.view),
                    view(&t.depth_bounds.view),
                ],
            )
        });
        let temporal = std::array::from_fn(|write| {
            let previous = &t.history[1 - write];
            let next = &t.history[write];
            bind_group(
                device,
                "HLFS temporal resources",
                &p.temporal_bgl,
                &[
                    view(&t.raw_diffuse.view),
                    view(&t.raw_specular.view),
                    view(&previous.diffuse.view),
                    view(&previous.specular.view),
                    view(&previous.geometry.view),
                    view(&next.diffuse.view),
                    view(&next.specular.view),
                    view(&next.geometry.view),
                    t.grid.as_entire_binding(),
                ],
            )
        });
        let composite = std::array::from_fn(|write| {
            let next = &t.history[write];
            bind_group(
                device,
                "HLFS composite resources",
                &p.composite_bgl,
                &[
                    view(&next.diffuse.view),
                    view(&next.specular.view),
                    view(&next.geometry.view),
                    view(&t.depth_bounds.view),
                ],
            )
        });
        Self {
            grid,
            sample,
            temporal,
            composite,
        }
    }
}

pub(crate) struct Inputs<'a> {
    pub camera: &'a wgpu::Buffer,
    pub lights: &'a wgpu::Buffer,
    pub shadow_matrices: &'a wgpu::Buffer,
    pub shadow_atlas: &'a wgpu::TextureView,
    pub shadow_sampler: &'a wgpu::Sampler,
    /// albedo, normal, ORM, emissive, depth, lightmap UV, lightmap, pre-AA, velocity
    pub textures: [&'a wgpu::TextureView; 9],
    pub lightmap_sampler: &'a wgpu::Sampler,
    pub tlas: Option<&'a wgpu::Tlas>,
}
struct CommonKey {
    buffers: [wgpu::Buffer; 3],
    shadow: wgpu::TextureView,
    sampler: wgpu::Sampler,
}
impl CommonKey {
    fn matches(&self, i: &Inputs<'_>) -> bool {
        &self.buffers[0] == i.camera
            && &self.buffers[1] == i.lights
            && &self.buffers[2] == i.shadow_matrices
            && &self.shadow == i.shadow_atlas
            && &self.sampler == i.shadow_sampler
    }
}
struct GBufferKey {
    textures: [wgpu::TextureView; 9],
    sampler: wgpu::Sampler,
}
impl GBufferKey {
    fn matches(&self, i: &Inputs<'_>) -> bool {
        self.textures
            .iter()
            .zip(i.textures)
            .all(|(old, new)| old == new)
            && &self.sampler == i.lightmap_sampler
    }
}
#[derive(Default)]
pub(crate) struct ExternalBindings {
    pub common: Option<wgpu::BindGroup>,
    pub gbuffer: Option<wgpu::BindGroup>,
    pub rt: Option<wgpu::BindGroup>,
    common_key: Option<CommonKey>,
    gbuffer_key: Option<GBufferKey>,
    tlas: Option<wgpu::Tlas>,
}
impl ExternalBindings {
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        p: &Pipelines,
        f: &Fallbacks,
        globals: &wgpu::Buffer,
        shadows: &wgpu::Buffer,
        i: &Inputs<'_>,
    ) {
        // Compare actual wgpu handles. Wrapper addresses can remain unchanged when
        // a growable SceneDB buffer reallocates and must never be cache keys.
        if !self.common_key.as_ref().is_some_and(|k| k.matches(i)) {
            self.common = Some(bind_group(
                device,
                "HLFS scene inputs",
                &p.common_bgl,
                &[
                    globals.as_entire_binding(),
                    i.camera.as_entire_binding(),
                    i.lights.as_entire_binding(),
                    shadows.as_entire_binding(),
                    view(i.shadow_atlas),
                    wgpu::BindingResource::Sampler(i.shadow_sampler),
                    i.shadow_matrices.as_entire_binding(),
                    view(&f.noise_view),
                ],
            ));
            self.common_key = Some(CommonKey {
                buffers: [
                    i.camera.clone(),
                    i.lights.clone(),
                    i.shadow_matrices.clone(),
                ],
                shadow: i.shadow_atlas.clone(),
                sampler: i.shadow_sampler.clone(),
            });
        }
        if !self.gbuffer_key.as_ref().is_some_and(|k| k.matches(i)) {
            self.gbuffer = Some(bind_group(
                device,
                "HLFS GBuffer inputs",
                &p.gbuffer_bgl,
                &[
                    view(i.textures[0]),
                    view(i.textures[1]),
                    view(i.textures[2]),
                    view(i.textures[3]),
                    view(i.textures[4]),
                    view(i.textures[5]),
                    view(i.textures[6]),
                    wgpu::BindingResource::Sampler(i.lightmap_sampler),
                    view(i.textures[7]),
                    view(i.textures[8]),
                ],
            ));
            self.gbuffer_key = Some(GBufferKey {
                textures: i.textures.map(Clone::clone),
                sampler: i.lightmap_sampler.clone(),
            });
        }
        if self.tlas.as_ref() != i.tlas {
            self.rt = p.rt_bgl.as_ref().zip(i.tlas).map(|(bgl, tlas)| {
                bind_group(device, "HLFS current TLAS", bgl, &[tlas.as_binding()])
            });
            self.tlas = i.tlas.cloned();
        }
    }
}
