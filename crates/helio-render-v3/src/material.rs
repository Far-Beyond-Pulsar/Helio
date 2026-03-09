use std::sync::Arc;
use bytemuck::{Pod, Zeroable};

/// GPU-side material uniform — 48 bytes, Pod-safe. One per draw call.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct MaterialUniform {
    pub base_color:      [f32; 4],  // offset  0
    pub metallic:        f32,       // offset 16
    pub roughness:       f32,       // offset 20
    pub emissive_factor: f32,       // offset 24
    pub ao:              f32,       // offset 28
    pub emissive_color:  [f32; 3],  // offset 32
    pub alpha_cutoff:    f32,       // offset 44
}                                   // total: 48 bytes

/// CPU-side material builder.
#[derive(Clone, Debug)]
pub struct Material {
    pub base_color:      [f32; 4],
    pub metallic:        f32,
    pub roughness:       f32,
    pub emissive_factor: f32,
    pub ao:              f32,
    pub emissive_color:  [f32; 3],
    pub alpha_cutoff:    f32,
    pub double_sided:    bool,
    pub blend_mode:      BlendMode,

    // Texture handles — None → white/flat stub used at bind group time.
    pub base_color_texture: Option<Arc<wgpu::Texture>>,
    pub normal_map:         Option<Arc<wgpu::Texture>>,
    pub orm_texture:        Option<Arc<wgpu::Texture>>,
    pub emissive_texture:   Option<Arc<wgpu::Texture>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendMode {
    Opaque,
    Masked,
    Transparent,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color:      [1.0; 4],
            metallic:        0.0,
            roughness:       0.5,
            emissive_factor: 0.0,
            ao:              1.0,
            emissive_color:  [0.0; 3],
            alpha_cutoff:    0.5,
            double_sided:    false,
            blend_mode:      BlendMode::Opaque,
            base_color_texture: None,
            normal_map:         None,
            orm_texture:        None,
            emissive_texture:   None,
        }
    }
}

impl Material {
    pub fn uniform(&self) -> MaterialUniform {
        MaterialUniform {
            base_color:      self.base_color,
            metallic:        self.metallic,
            roughness:       self.roughness,
            emissive_factor: self.emissive_factor,
            ao:              self.ao,
            emissive_color:  self.emissive_color,
            alpha_cutoff:    self.alpha_cutoff,
        }
    }

    pub fn transparent(&self) -> bool {
        self.blend_mode == BlendMode::Transparent
    }
}

/// Fully GPU-resident material plus pre-built bind group.
pub struct GpuMaterial {
    pub uniform_buffer:   Arc<wgpu::Buffer>,
    pub bind_group:       Arc<wgpu::BindGroup>,
    pub transparent_blend: bool,
    pub double_sided:     bool,
}

impl GpuMaterial {
    /// Upload to GPU, resolve stub textures for any missing samplers.
    pub fn upload(
        device:        &wgpu::Device,
        queue:         &wgpu::Queue,
        material:      &Material,
        layout:        &wgpu::BindGroupLayout,
        stubs:         &crate::resources::StubTextures,
        linear_sampler: &wgpu::Sampler,
    ) -> Arc<Self> {
        use wgpu::util::DeviceExt;

        let uniform = material.uniform();
        let ub = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("material_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        let base_color_tv  = resolve_view(material.base_color_texture.as_ref(), &stubs.white_srgb);
        let normal_tv      = resolve_view(material.normal_map.as_ref(),         &stubs.flat_normal);
        let orm_tv         = resolve_view(material.orm_texture.as_ref(),         &stubs.white_linear);
        let emissive_tv    = resolve_view(material.emissive_texture.as_ref(),    &stubs.black_srgb);

        let bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("material_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&base_color_tv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_tv) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(linear_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&orm_tv) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&emissive_tv) },
            ],
        }));

        Arc::new(GpuMaterial {
            uniform_buffer:    ub,
            bind_group:        bg,
            transparent_blend: material.transparent(),
            double_sided:      material.double_sided,
        })
    }
}

fn resolve_view<'a>(opt: Option<&'a Arc<wgpu::Texture>>, stub: &'a wgpu::Texture) -> wgpu::TextureView {
    let tex = opt.map(|a| a.as_ref()).unwrap_or(stub);
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
