//! Standard bind group layouts and builders
//!
//! This is the FOUNDATION of the entire renderer. All pipelines use these
//! standard bind group layouts, enabling proper resource sharing between features.

use std::sync::Arc;
use wgpu;

/// Standard bind group layouts used by all pipelines
///
/// This is the key innovation that enables features to share resources.
/// All pipelines use these same 5 bind group layouts:
///
/// - Group 0: Global (camera, time) - per-frame, shared by all
/// - Group 1: Material (PBR properties, textures) - per-draw
/// - Group 2: Lighting (lights, shadows, env maps) - per-scene
/// - Group 3: Bindless textures (optional) - static
/// - Group 4: Pass-specific storage (compute) - per-pass
#[derive(Clone)]
pub struct BindGroupLayouts {
    pub global: Arc<wgpu::BindGroupLayout>,
    pub material: Arc<wgpu::BindGroupLayout>,
    pub lighting: Arc<wgpu::BindGroupLayout>,
    pub textures: Arc<wgpu::BindGroupLayout>,
    pub storage: Arc<wgpu::BindGroupLayout>,
}

impl BindGroupLayouts {
    /// Create the standard bind group layouts
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            global: Arc::new(Self::create_global_layout(device)),
            material: Arc::new(Self::create_material_layout(device)),
            lighting: Arc::new(Self::create_lighting_layout(device)),
            textures: Arc::new(Self::create_textures_layout(device)),
            storage: Arc::new(Self::create_storage_layout(device)),
        }
    }

    /// Group 0: Global uniforms (camera, time, etc.)
    ///
    /// Updated every frame, shared by all draw calls
    fn create_global_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Global Bind Group Layout"),
            entries: &[
                // Binding 0: Camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Globals uniform (frame, delta_time, etc.)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Group 1: Material properties and textures
    ///
    /// Updated per draw call, contains PBR material data
    fn create_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &[
                // Binding 0: Material data uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Base color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Normal map
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: Material sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Group 2: Lighting data (lights, shadows, environment maps)
    ///
    /// Updated when lights change, shared across lit objects
    fn create_lighting_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting Bind Group Layout"),
            entries: &[
                // Binding 0: Light data storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Shadow atlas (depth texture array)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Shadow sampler (comparison)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // Binding 3: Environment cubemap (for IBL)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: Shadow light-space matrices storage buffer (PCF in geometry shader)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Group 3: Bindless texture array (optional, for advanced rendering)
    ///
    /// Rarely updated, allows indexing into texture array
    fn create_textures_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bindless Textures Bind Group Layout"),
            entries: &[
                // Binding 0: Texture array (bindless textures)
                // Note: This requires texture_binding_array feature
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None, // Non-bindless for now, can be made dynamic later
                },
            ],
        })
    }

    /// Group 4: Pass-specific storage buffers (for compute passes)
    ///
    /// Per-pass specific, used for compute shader data
    fn create_storage_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Bind Group Layout"),
            entries: &[
                // Binding 0: Generic storage buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Get all layouts as a slice (for pipeline creation)
    pub fn as_slice(&self) -> Vec<&wgpu::BindGroupLayout> {
        vec![
            &self.global,
            &self.material,
            &self.lighting,
            &self.textures,
            &self.storage,
        ]
    }
}

/// Helper for building bind groups
///
/// Provides a fluent API for creating bind groups that match the standard layouts
pub struct BindGroupBuilder {
    label: Option<String>,
    entries: Vec<wgpu::BindGroupEntry<'static>>,
}

impl BindGroupBuilder {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            entries: Vec::new(),
        }
    }

    /// Add a buffer binding
    pub fn buffer(mut self, binding: u32, buffer: &'static wgpu::Buffer) -> Self {
        self.entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
        });
        self
    }

    /// Add a buffer binding with offset and size
    pub fn buffer_range(
        mut self,
        binding: u32,
        buffer: &'static wgpu::Buffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) -> Self {
        self.entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset,
                size,
            }),
        });
        self
    }

    /// Add a texture view binding
    pub fn texture(mut self, binding: u32, view: &'static wgpu::TextureView) -> Self {
        self.entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        });
        self
    }

    /// Add a sampler binding
    pub fn sampler(mut self, binding: u32, sampler: &'static wgpu::Sampler) -> Self {
        self.entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
        self
    }

    /// Build the bind group
    pub fn build(self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: self.label.as_deref(),
            layout,
            entries: &self.entries,
        })
    }
}
