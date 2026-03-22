use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::{Mat3, Mat4, Vec3};
use helio_v3::{
    scene::GrowableBuffer,
    DrawIndexedIndirectArgs, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb, GpuInstanceData,
    GpuLight, GpuMaterial, GpuScene,
};
use libhelio::{GpuMeshletEntry, GpuShadowMatrix, VgFrameData};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::arena::{DenseArena, DenseRemove, SparsePool};
use crate::handles::{LightId, MaterialId, MeshId, ObjectId, TextureId, VirtualObjectId};
use crate::material::{
    GpuMaterialTextureSlot, GpuMaterialTextures, MaterialAsset, MaterialTextureRef, MaterialTextures,
    TextureTransform, TextureUpload, MAX_TEXTURES,
};
use crate::mesh::{MeshBuffers, MeshPool, MeshSlice, MeshUpload};
use crate::vg::{VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor, generate_lod_meshes, meshletize, sort_triangles_spatially};

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("invalid {resource} handle")]
    InvalidHandle { resource: &'static str },
    #[error("{resource} is still in use")]
    ResourceInUse { resource: &'static str },
    #[error("scene texture capacity exceeded")]
    TextureCapacityExceeded,
}

pub type Result<T> = std::result::Result<T, SceneError>;

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
    pub near: f32,
    pub far: f32,
    pub jitter: [f32; 2],
}

impl Camera {
    pub fn from_matrices(view: Mat4, proj: Mat4, position: Vec3, near: f32, far: f32) -> Self {
        Self {
            view,
            proj,
            position,
            near,
            far,
            jitter: [0.0, 0.0],
        }
    }

    pub fn perspective_look_at(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let view = Mat4::look_at_rh(position, target, up);
        let proj = Mat4::perspective_rh(fov_y_radians, aspect, near, far);
        Self::from_matrices(view, proj, position, near, far)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ObjectDescriptor {
    pub mesh: MeshId,
    pub material: MaterialId,
    pub transform: Mat4,
    pub bounds: [f32; 4],
    pub flags: u32,
}

#[derive(Debug, Clone)]
struct MaterialRecord {
    gpu: GpuMaterial,
    textures: MaterialTextures,
    ref_count: u32,
}

#[derive(Debug, Clone)]
struct LightRecord {
    gpu: GpuLight,
}

#[derive(Debug, Clone)]
struct ObjectRecord {
    mesh: MeshId,
    material: MaterialId,
    instance: GpuInstanceData,
    aabb: GpuInstanceAabb,
    draw: GpuDrawCall,
}

struct TextureRecord {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    ref_count: u32,
}

fn invalid(resource: &'static str) -> SceneError {
    SceneError::InvalidHandle { resource }
}

// ─── Virtual geometry records ────────────────────────────────────────────────

/// CPU-side record for a virtual mesh: uploaded mesh handles (one per LOD) + precomputed meshlets.
#[derive(Debug, Clone)]
struct VirtualMeshRecord {
    /// Mesh pool handles for each LOD level; index 0 = full detail, 1 = medium, 2 = coarse.
    pub mesh_ids: Vec<MeshId>,
    /// Precomputed meshlet descriptors for all LODs combined.
    /// `lod_error` encodes the LOD level: 0.0 = full, 1.0 = medium, 2.0 = coarse.
    pub meshlets: Vec<GpuMeshletEntry>,
    pub ref_count: u32,
}

/// CPU-side record for one virtual object (a VG instance in the scene).
#[derive(Debug, Clone)]
struct VirtualObjectRecord {
    pub virtual_mesh: VirtualMeshId,
    pub instance: GpuInstanceData,
}

fn tombstone_material() -> GpuMaterial {
    GpuMaterial {
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        ..GpuMaterial::zeroed()
    }
}

fn tombstone_material_textures() -> GpuMaterialTextures {
    GpuMaterialTextures::missing()
}

fn gpu_texture_slot(texture: Option<MaterialTextureRef>) -> GpuMaterialTextureSlot {
    let Some(texture) = texture else {
        return GpuMaterialTextureSlot::missing();
    };
    let uv_channel = texture.uv_channel.min(1);
    let TextureTransform {
        offset,
        scale,
        rotation_radians,
    } = texture.transform;
    GpuMaterialTextureSlot {
        texture_index: texture.texture.slot(),
        uv_channel,
        _pad: [0; 2],
        offset_scale: [offset[0], offset[1], scale[0], scale[1]],
        rotation: [rotation_radians.sin(), rotation_radians.cos(), 0.0, 0.0],
    }
}

fn gpu_material_textures(textures: &MaterialTextures) -> GpuMaterialTextures {
    GpuMaterialTextures {
        base_color: gpu_texture_slot(textures.base_color),
        normal: gpu_texture_slot(textures.normal),
        roughness_metallic: gpu_texture_slot(textures.roughness_metallic),
        emissive: gpu_texture_slot(textures.emissive),
        occlusion: gpu_texture_slot(textures.occlusion),
        specular_color: gpu_texture_slot(textures.specular_color),
        specular_weight: gpu_texture_slot(textures.specular_weight),
        params: [
            textures.normal_scale,
            textures.occlusion_strength,
            textures.alpha_cutoff,
            0.0,
        ],
    }
}

fn each_material_texture_ref<F>(textures: &MaterialTextures, mut f: F)
where
    F: FnMut(MaterialTextureRef),
{
    for texture in [
        textures.base_color,
        textures.normal,
        textures.roughness_metallic,
        textures.emissive,
        textures.occlusion,
        textures.specular_color,
        textures.specular_weight,
    ]
    .into_iter()
    .flatten()
    {
        f(texture);
    }
}

fn normal_matrix(transform: Mat4) -> [f32; 12] {
    let mat3 = Mat3::from_mat4(transform).inverse().transpose();
    let cols = mat3.to_cols_array();
    [
        cols[0], cols[1], cols[2], 0.0,
        cols[3], cols[4], cols[5], 0.0,
        cols[6], cols[7], cols[8], 0.0,
    ]
}

fn sphere_to_aabb(bounds: [f32; 4]) -> GpuInstanceAabb {
    let center = Vec3::new(bounds[0], bounds[1], bounds[2]);
    let radius = Vec3::splat(bounds[3]);
    let min = center - radius;
    let max = center + radius;
    GpuInstanceAabb {
        min: min.to_array(),
        _pad0: 0.0,
        max: max.to_array(),
        _pad1: 0.0,
    }
}

fn object_gpu_data(mesh: MeshId, material_slot: usize, desc: ObjectDescriptor, slice: MeshSlice) -> ObjectRecord {
    ObjectRecord {
        mesh,
        material: desc.material,
        instance: GpuInstanceData {
            model: desc.transform.to_cols_array(),
            normal_mat: normal_matrix(desc.transform),
            bounds: desc.bounds,
            mesh_id: mesh.slot(),
            material_id: material_slot as u32,
            flags: desc.flags,
            _pad: 0,
        },
        aabb: sphere_to_aabb(desc.bounds),
        // `first_instance` is set to 0 here; the actual GPU slot is assigned during
        // `rebuild_instance_buffers()` called from `flush()`. `instance_count` is not
        // meaningful per-object — it is computed per-group during the rebuild.
        draw: GpuDrawCall {
            index_count: slice.index_count,
            first_index: slice.first_index,
            vertex_offset: slice.first_vertex as i32,
            first_instance: 0,
            instance_count: 0,
        },
    }
}

pub struct Scene {
    gpu_scene: GpuScene,
    mesh_pool: MeshPool,
    textures: SparsePool<TextureRecord, TextureId>,
    texture_binding_version: u64,
    material_textures: GrowableBuffer<GpuMaterialTextures>,
    _placeholder_texture: wgpu::Texture,
    placeholder_view: wgpu::TextureView,
    placeholder_sampler: wgpu::Sampler,
    materials: SparsePool<MaterialRecord, MaterialId>,
    lights: DenseArena<LightRecord, LightId>,
    objects: DenseArena<ObjectRecord, ObjectId>,
    /// True when the objects list has changed and the GPU instance/draw_call/indirect
    /// buffers need to be rebuilt from scratch (sorted by mesh+material for instancing).
    objects_dirty: bool,
    prev_view_proj: Mat4,

    // ── Virtual geometry ──────────────────────────────────────────────────────
    /// All uploaded virtual meshes keyed by their handle.
    vg_meshes: HashMap<VirtualMeshId, VirtualMeshRecord>,
    /// Next free VirtualMeshId slot counter (monotonically increasing).
    vg_next_mesh_id: u32,
    /// Dense array of virtual objects (one entry per `insert_virtual_object` call).
    vg_objects: DenseArena<VirtualObjectRecord, VirtualObjectId>,
    /// Set when VG topology or transforms change; triggers `rebuild_vg_buffers()`.
    vg_objects_dirty: bool,
    /// Monotonically increasing counter forwarded to `VgFrameData::buffer_version`.
    /// The VG pass re-uploads GPU buffers only when this advances.
    vg_buffer_version: u64,
    /// Flattened meshlet entries for the current VG layout (rebuilt when dirty).
    vg_cpu_meshlets: Vec<GpuMeshletEntry>,
    /// Instance data for all VG objects (one entry per VG object, in order).
    vg_cpu_instances: Vec<GpuInstanceData>,
}

impl Scene {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        helio_v3::upload::record_upload_bytes(4);
        let placeholder_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Helio Placeholder Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let placeholder_view =
            placeholder_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Helio Placeholder Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            mesh_pool: MeshPool::new(device.clone()),
            gpu_scene: GpuScene::new(device.clone(), queue.clone()),
            textures: SparsePool::new(),
            texture_binding_version: 0,
            material_textures: GrowableBuffer::new(
                device,
                256,
                wgpu::BufferUsages::STORAGE,
                "Helio Material Texture Buffer",
            ),
            _placeholder_texture: placeholder_texture,
            placeholder_view,
            placeholder_sampler,
            materials: SparsePool::new(),
            lights: DenseArena::new(),
            objects: DenseArena::new(),
            objects_dirty: true, // rebuild on first flush
            prev_view_proj: Mat4::IDENTITY,
            vg_meshes: HashMap::new(),
            vg_next_mesh_id: 0,
            vg_objects: DenseArena::new(),
            vg_objects_dirty: false,
            vg_buffer_version: 0,
            vg_cpu_meshlets: Vec::new(),
            vg_cpu_instances: Vec::new(),
        }
    }

    pub fn gpu_scene(&self) -> &GpuScene {
        &self.gpu_scene
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.mesh_pool.buffers()
    }

    pub fn material_texture_buffer(&self) -> &wgpu::Buffer {
        self.material_textures.buffer()
    }

    pub fn texture_binding_version(&self) -> u64 {
        self.texture_binding_version
    }

    pub fn texture_view_for_slot(&self, slot: usize) -> &wgpu::TextureView {
        self.textures
            .get_by_slot(slot)
            .map(|texture| &texture.view)
            .unwrap_or(&self.placeholder_view)
    }

    pub fn texture_sampler_for_slot(&self, slot: usize) -> &wgpu::Sampler {
        self.textures
            .get_by_slot(slot)
            .map(|texture| &texture.sampler)
            .unwrap_or(&self.placeholder_sampler)
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.gpu_scene.width = width;
        self.gpu_scene.height = height;
    }

    pub fn update_camera(&mut self, camera: Camera) {
        let uniforms = GpuCameraUniforms::new(
            camera.view,
            camera.proj,
            camera.position,
            camera.near,
            camera.far,
            self.gpu_scene.frame_count as u32,
            camera.jitter,
            self.prev_view_proj,
        );
        self.prev_view_proj = camera.proj * camera.view;
        self.gpu_scene.camera.update(uniforms);
    }

    pub fn insert_mesh(&mut self, mesh: MeshUpload) -> MeshId {
        self.mesh_pool.insert(mesh)
    }

    pub fn insert_texture(&mut self, texture: TextureUpload) -> Result<TextureId> {
        if !self.textures.has_free_slot() && self.textures.slot_len() >= MAX_TEXTURES {
            return Err(SceneError::TextureCapacityExceeded);
        }

        helio_v3::upload::record_upload_bytes(texture.data.len() as u64);
        let gpu_texture = self.gpu_scene.device.create_texture_with_data(
            &self.gpu_scene.queue,
            &wgpu::TextureDescriptor {
                label: texture.label.as_deref(),
                size: wgpu::Extent3d {
                    width: texture.width,
                    height: texture.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &texture.data,
        );
        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.gpu_scene.device.create_sampler(&wgpu::SamplerDescriptor {
            label: texture.label.as_deref(),
            address_mode_u: texture.sampler.address_mode_u,
            address_mode_v: texture.sampler.address_mode_v,
            address_mode_w: texture.sampler.address_mode_w,
            mag_filter: texture.sampler.mag_filter,
            min_filter: texture.sampler.min_filter,
            mipmap_filter: texture.sampler.mipmap_filter,
            ..Default::default()
        });
        let (id, _, _) = self.textures.insert(TextureRecord {
            _texture: gpu_texture,
            view,
            sampler,
            ref_count: 0,
        });
        self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        Ok(id)
    }

    pub fn remove_texture(&mut self, id: TextureId) -> Result<()> {
        let Some(texture) = self.textures.get(id) else {
            return Err(invalid("texture"));
        };
        if texture.ref_count != 0 {
            return Err(SceneError::ResourceInUse { resource: "texture" });
        }
        self.textures.remove(id).ok_or_else(|| invalid("texture"))?;
        self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        Ok(())
    }

    pub fn remove_mesh(&mut self, id: MeshId) -> Result<()> {
        let Some(record) = self.mesh_pool.get(id) else {
            return Err(invalid("mesh"));
        };
        if record.ref_count != 0 {
            return Err(SceneError::ResourceInUse { resource: "mesh" });
        }
        self.mesh_pool.remove(id).ok_or_else(|| invalid("mesh"))?;
        Ok(())
    }

    pub fn insert_material(&mut self, material: GpuMaterial) -> MaterialId {
        self.insert_material_asset(material.into())
            .expect("plain GPU materials must insert without texture validation failures")
    }

    pub fn insert_material_asset(&mut self, material: MaterialAsset) -> Result<MaterialId> {
        self.validate_material_textures(&material.textures)?;
        self.bump_texture_refs(&material.textures, 1)?;

        let gpu_textures = gpu_material_textures(&material.textures);
        let (id, slot, is_new) = self.materials.insert(MaterialRecord {
            gpu: material.gpu,
            textures: material.textures,
            ref_count: 0,
        });
        if is_new {
            let pushed = self.gpu_scene.materials.push(material.gpu);
            debug_assert_eq!(pushed, slot);
            let pushed = self.material_textures.push(gpu_textures);
            debug_assert_eq!(pushed, slot);
        } else {
            let updated_material = self.gpu_scene.materials.update(slot, material.gpu);
            let updated_textures = self.material_textures.update(slot, gpu_textures);
            debug_assert!(updated_material && updated_textures);
        }
        Ok(id)
    }

    pub fn update_material(&mut self, id: MaterialId, material: GpuMaterial) -> Result<()> {
        let Some((slot, record)) = self.materials.get_mut_with_slot(id) else {
            return Err(invalid("material"));
        };
        record.gpu = material;
        let updated = self.gpu_scene.materials.update(slot, material);
        debug_assert!(updated);
        Ok(())
    }

    pub fn update_material_asset(&mut self, id: MaterialId, material: MaterialAsset) -> Result<()> {
        self.validate_material_textures(&material.textures)?;
        let Some(old_textures) = self.materials.get(id).map(|record| record.textures.clone()) else {
            return Err(invalid("material"));
        };
        self.bump_texture_refs(&material.textures, 1)?;
        self.bump_texture_refs(&old_textures, -1)?;

        let Some((slot, record)) = self.materials.get_mut_with_slot(id) else {
            return Err(invalid("material"));
        };
        record.gpu = material.gpu;
        record.textures = material.textures.clone();

        let updated_material = self.gpu_scene.materials.update(slot, material.gpu);
        let updated_textures = self
            .material_textures
            .update(slot, gpu_material_textures(&material.textures));
        debug_assert!(updated_material && updated_textures);
        Ok(())
    }

    pub fn remove_material(&mut self, id: MaterialId) -> Result<()> {
        let Some(record) = self.materials.get(id) else {
            return Err(invalid("material"));
        };
        if record.ref_count != 0 {
            return Err(SceneError::ResourceInUse { resource: "material" });
        }
        let (slot, removed) = self.materials.remove(id).ok_or_else(|| invalid("material"))?;
        self.bump_texture_refs(&removed.textures, -1)?;
        let updated_material = self.gpu_scene.materials.update(slot, tombstone_material());
        let updated_textures = self
            .material_textures
            .update(slot, tombstone_material_textures());
        debug_assert!(updated_material && updated_textures);
        Ok(())
    }

    pub fn insert_light(&mut self, light: GpuLight) -> LightId {
        let (id, dense_index) = self.lights.insert(LightRecord { gpu: light });
        let pushed = self.gpu_scene.lights.push(light);
        debug_assert_eq!(pushed, dense_index);
        id
    }

    pub fn update_light(&mut self, id: LightId, light: GpuLight) -> Result<()> {
        let Some((dense_index, record)) = self.lights.get_mut_with_index(id) else {
            return Err(invalid("light"));
        };
        record.gpu = light;
        let updated = self.gpu_scene.lights.update(dense_index, light);
        debug_assert!(updated);
        Ok(())
    }

    pub fn remove_light(&mut self, id: LightId) -> Result<()> {
        let removed = self.lights.remove(id).ok_or_else(|| invalid("light"))?;
        let gpu_removed = self.gpu_scene.lights.swap_remove(removed.dense_index);
        debug_assert!(gpu_removed.is_some());
        Ok(())
    }

    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> Result<ObjectId> {
        let mesh_slice = {
            let mesh = self.mesh_pool.get(desc.mesh).ok_or_else(|| invalid("mesh"))?;
            mesh.slice
        };
        let material_slot = {
            let (slot, material) = self
                .materials
                .get_mut_with_slot(desc.material)
                .ok_or_else(|| invalid("material"))?;
            material.ref_count += 1;
            slot
        };
        self.mesh_pool
            .get_mut(desc.mesh)
            .ok_or_else(|| invalid("mesh"))?
            .ref_count += 1;

        let record = object_gpu_data(desc.mesh, material_slot, desc, mesh_slice);
        let (id, _) = self.objects.insert(record);
        // Defer GPU buffer upload to flush() which will sort by (mesh,material)
        // and build instanced draw calls automatically.
        self.objects_dirty = true;
        Ok(id)
    }

    pub fn update_object_transform(&mut self, id: ObjectId, transform: Mat4) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        record.instance.model = transform.to_cols_array();
        record.instance.normal_mat = normal_matrix(transform);
        // If the GPU layout is stable (no pending rebuild), update the slot in-place.
        // If a rebuild is pending the new data will be included in it automatically.
        if !self.objects_dirty {
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.instances.update(slot, record.instance);
        }
        Ok(())
    }

    pub fn update_object_material(&mut self, id: ObjectId, material: MaterialId) -> Result<()> {
        let new_slot = {
            let (slot, new_material) = self
                .materials
                .get_mut_with_slot(material)
                .ok_or_else(|| invalid("material"))?;
            new_material.ref_count += 1;
            slot
        };
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        let old_material_id = record.material;
        record.material = material;
        record.instance.material_id = new_slot as u32;
        if let Some((_, old_material)) = self.materials.get_mut_with_slot(old_material_id) {
            old_material.ref_count = old_material.ref_count.saturating_sub(1);
        }
        // Material change may move the object to a different instancing group.
        self.objects_dirty = true;
        Ok(())
    }

    pub fn update_object_bounds(&mut self, id: ObjectId, bounds: [f32; 4]) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        record.instance.bounds = bounds;
        record.aabb = sphere_to_aabb(bounds);
        // Bounds don't affect the instancing group, so update in-place when layout is stable.
        if !self.objects_dirty {
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.instances.update(slot, record.instance);
            self.gpu_scene.aabbs.update(slot, record.aabb);
        }
        Ok(())
    }

    pub fn remove_object(&mut self, id: ObjectId) -> Result<()> {
        let DenseRemove { removed, .. } =
            self.objects.remove(id).ok_or_else(|| invalid("object"))?;

        if let Some(material) = self.materials.get_mut_with_slot(removed.material).map(|(_, m)| m) {
            material.ref_count = material.ref_count.saturating_sub(1);
        }
        if let Some(mesh) = self.mesh_pool.get_mut(removed.mesh) {
            mesh.ref_count = mesh.ref_count.saturating_sub(1);
        }
        // GPU buffers rebuilt on next flush().
        self.objects_dirty = true;
        Ok(())
    }

    pub fn flush(&mut self) {
        // Assign sequential shadow atlas base layers to each shadow-casting light.
        // Convention: shadow_index == u32::MAX  → no shadow.
        // Always 6 slots per light (matches FACES_PER_LIGHT in shadow_matrices.wgsl):
        //   Point:       6 cube-face matrices
        //   Directional: 4 CSM cascades + 2 identity padding slots
        //   Spot:        1 perspective matrix + 5 unused (zeroed) slots
        // Cap at 42 shadow casters (42 × 6 = 252 ≤ 256 atlas layers).
        {
            const MAX_SHADOW_CASTERS: usize = 42;
            const FACES_PER_LIGHT: u32 = 6;
            let light_count = self.gpu_scene.lights.len();
            let mut next_layer: u32 = 0;
            let mut shadow_caster_count = 0usize;
            for i in 0..light_count {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if light.shadow_index == u32::MAX {
                    // Explicitly disabled — leave as-is.
                    continue;
                }
                if shadow_caster_count >= MAX_SHADOW_CASTERS {
                    // Over cap: disable shadow for this light.
                    let mut disabled = light;
                    disabled.shadow_index = u32::MAX;
                    self.gpu_scene.lights.update(i, disabled);
                    continue;
                }
                let mut assigned = light;
                assigned.shadow_index = next_layer;
                self.gpu_scene.lights.update(i, assigned);
                next_layer += FACES_PER_LIGHT;
                shadow_caster_count += 1;
            }
            let needed = (next_layer as usize).max(1);
            if self.gpu_scene.shadow_matrices.len() != needed {
                self.gpu_scene.shadow_matrices.set_data(
                    vec![GpuShadowMatrix::zeroed(); needed],
                );
            }
        }
        let queue = self.gpu_scene.queue.clone();
        self.mesh_pool.flush(&queue);
        self.material_textures.flush(&queue);
        // Rebuild instanced draw lists when the object set has changed.
        if self.objects_dirty {
            self.rebuild_instance_buffers();
            self.objects_dirty = false;
        }
        // Rebuild virtual geometry CPU buffers when VG topology or transforms changed.
        if self.vg_objects_dirty {
            self.rebuild_vg_buffers();
            self.vg_objects_dirty = false;
        }
        self.gpu_scene.flush();
    }

    /// Sorts all registered objects by (mesh_id, material_id) and reconstructs the
    /// GPU instance buffer, draw_call buffer, and indirect buffer so that objects with
    /// the same mesh **and** material share a single `DrawIndexedIndirect` command with
    /// `instance_count > 1`.  This gives the GPU hardware instancing for free.
    ///
    /// Called once from `flush()` whenever `objects_dirty` is true.  For a fully static
    /// scene (no additions/removals after the first frame) this path executes exactly
    /// once and is then skipped on every subsequent frame — O(0) steady-state cost.
    fn rebuild_instance_buffers(&mut self) {
        let n = self.objects.dense_len();
        if n == 0 {
            self.gpu_scene.instances.set_data(Vec::new());
            self.gpu_scene.aabbs.set_data(Vec::new());
            self.gpu_scene.draw_calls.set_data(Vec::new());
            self.gpu_scene.indirect.set_data(Vec::new());
            self.gpu_scene.visibility.set_data(Vec::new());
            return;
        }

        // Build a sort order over the dense array indices, grouped by (mesh_id, material_id).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| {
            let r = self.objects.get_dense(i).unwrap();
            (r.instance.mesh_id, r.instance.material_id)
        });

        let mut instances: Vec<GpuInstanceData> = Vec::with_capacity(n);
        let mut aabbs: Vec<GpuInstanceAabb> = Vec::with_capacity(n);
        let mut draw_calls: Vec<GpuDrawCall> = Vec::new();
        let mut indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();
        // Track the new GPU slot assigned to each dense-array entry.
        let mut gpu_slots: Vec<u32> = vec![0u32; n];

        let mut i = 0;
        while i < order.len() {
            let r0 = self.objects.get_dense(order[i]).unwrap();
            let key = (r0.instance.mesh_id, r0.instance.material_id);
            let group_start = instances.len() as u32;
            let (index_count, first_index, vertex_offset) =
                (r0.draw.index_count, r0.draw.first_index, r0.draw.vertex_offset);

            // Consume all objects in this group.
            while i < order.len() {
                let r = self.objects.get_dense(order[i]).unwrap();
                if (r.instance.mesh_id, r.instance.material_id) != key {
                    break;
                }
                gpu_slots[order[i]] = instances.len() as u32;
                instances.push(r.instance);
                aabbs.push(r.aabb);
                i += 1;
            }

            let instance_count = instances.len() as u32 - group_start;
            draw_calls.push(GpuDrawCall {
                index_count,
                first_index,
                vertex_offset,
                first_instance: group_start,
                instance_count,
            });
            indirect.push(DrawIndexedIndirectArgs {
                index_count,
                instance_count,
                first_index,
                base_vertex: vertex_offset,
                first_instance: group_start,
            });
        }

        // Patch each ObjectRecord with its new GPU slot so that in-frame
        // `update_object_transform` / `update_object_bounds` can update in-place.
        for (di, &slot) in gpu_slots.iter().enumerate() {
            if let Some(r) = self.objects.get_dense_mut(di) {
                r.draw.first_instance = slot;
            }
        }

        log::debug!(
            "rebuild_instance_buffers: {} objects → {} draw groups",
            n,
            draw_calls.len()
        );

        self.gpu_scene.instances.set_data(instances);
        self.gpu_scene.aabbs.set_data(aabbs);
        self.gpu_scene.draw_calls.set_data(draw_calls);
        self.gpu_scene.indirect.set_data(indirect);
        self.gpu_scene.visibility.set_data(vec![1u32; self.gpu_scene.instances.len()]);
    }


    pub fn advance_frame(&mut self) {
        self.gpu_scene.frame_count = self.gpu_scene.frame_count.wrapping_add(1);
    }

    fn validate_material_textures(&self, textures: &MaterialTextures) -> Result<()> {
        let mut validation = Ok(());
        each_material_texture_ref(textures, |texture| {
            if validation.is_err() {
                return;
            }
            if self.textures.get(texture.texture).is_none() {
                validation = Err(invalid("texture"));
            }
        });
        validation
    }

    fn bump_texture_refs(&mut self, textures: &MaterialTextures, delta: i32) -> Result<()> {
        for texture in [
            textures.base_color,
            textures.normal,
            textures.roughness_metallic,
            textures.emissive,
            textures.occlusion,
            textures.specular_color,
            textures.specular_weight,
        ]
        .into_iter()
        .flatten()
        {
            let (_, record) = self
                .textures
                .get_mut_with_slot(texture.texture)
                .ok_or_else(|| invalid("texture"))?;
            if delta >= 0 {
                record.ref_count = record.ref_count.saturating_add(delta as u32);
            } else {
                record.ref_count = record.ref_count.saturating_sub((-delta) as u32);
            }
        }
        Ok(())
    }

    // ── Virtual geometry API ──────────────────────────────────────────────────

    /// Upload a high-resolution mesh and decompose it into GPU meshlets for virtual
    /// geometry rendering.  The mesh is also registered in the normal `MeshPool` so
    /// it shares vertex/index storage with regular meshes.
    ///
    /// Returns a `VirtualMeshId` that you pass to `insert_virtual_object`.
    pub fn insert_virtual_mesh(&mut self, upload: VirtualMeshUpload) -> VirtualMeshId {
        // Generate three LOD levels (full, medium, coarse) via vertex clustering.
        let lod_meshes = generate_lod_meshes(&upload.vertices, &upload.indices);

        let mut all_meshlets: Vec<GpuMeshletEntry> = Vec::new();
        let mut mesh_ids: Vec<MeshId> = Vec::new();

        for (lod_level, (lod_verts, lod_indices)) in lod_meshes.into_iter().enumerate() {
            // Spatially sort triangles before uploading so the mega-buffer index
            // data matches what meshletize expects (sorted = tight cluster bounds).
            let sorted_indices = sort_triangles_spatially(&lod_verts, &lod_indices);
            let mesh_id = self.mesh_pool.insert(MeshUpload {
                vertices: lod_verts.clone(),
                indices:  sorted_indices.clone(),
            });
            let slice = self.mesh_pool.get(mesh_id).unwrap().slice;

            let mut meshlets = meshletize(
                &lod_verts,
                &sorted_indices,
                slice.first_index,
                slice.first_vertex,
            );
            // Tag with LOD level so the cull shader can select by distance.
            for m in &mut meshlets {
                m.lod_error = lod_level as f32;
            }
            all_meshlets.extend(meshlets);
            mesh_ids.push(mesh_id);
        }

        let id = VirtualMeshId(self.vg_next_mesh_id);
        self.vg_next_mesh_id += 1;
        self.vg_meshes.insert(id, VirtualMeshRecord {
            mesh_ids,
            meshlets: all_meshlets,
            ref_count: 0,
        });
        id
    }

    /// Remove a virtual mesh.  Fails if any `VirtualObjectId` still references it.
    pub fn remove_virtual_mesh(&mut self, id: VirtualMeshId) -> Result<()> {
        let record = self.vg_meshes.get(&id).ok_or_else(|| invalid("virtual_mesh"))?;
        if record.ref_count != 0 {
            return Err(SceneError::ResourceInUse { resource: "virtual_mesh" });
        }
        self.vg_meshes.remove(&id);
        Ok(())
    }

    /// Place an instance of a virtual mesh into the scene.
    pub fn insert_virtual_object(&mut self, desc: VirtualObjectDescriptor) -> Result<VirtualObjectId> {
        let record = self.vg_meshes.get_mut(&desc.virtual_mesh)
            .ok_or_else(|| invalid("virtual_mesh"))?;
        record.ref_count += 1;

        let instance = GpuInstanceData {
            model: desc.transform.to_cols_array(),
            normal_mat: normal_matrix(desc.transform),
            bounds: desc.bounds,
            mesh_id: record.mesh_ids[0].slot(),
            material_id: desc.material_id,
            flags: desc.flags,
            _pad: 0,
        };
        let (id, _) = self.vg_objects.insert(VirtualObjectRecord {
            virtual_mesh: desc.virtual_mesh,
            instance,
        });
        self.vg_objects_dirty = true;
        Ok(id)
    }

    /// Update the world transform of a virtual object.  In-place if no topology rebuild
    /// is pending; otherwise the change will be included in the next rebuild automatically.
    pub fn update_virtual_object_transform(
        &mut self,
        id: VirtualObjectId,
        transform: Mat4,
    ) -> Result<()> {
        let Some((_, record)) = self.vg_objects.get_mut_with_index(id) else {
            return Err(invalid("virtual_object"));
        };
        record.instance.model = transform.to_cols_array();
        record.instance.normal_mat = normal_matrix(transform);
        // Mark dirty so vg_frame_data() picks up the new transform.
        self.vg_objects_dirty = true;
        Ok(())
    }

    /// Remove a virtual object from the scene.
    pub fn remove_virtual_object(&mut self, id: VirtualObjectId) -> Result<()> {
        let removed = self.vg_objects.remove(id).ok_or_else(|| invalid("virtual_object"))?;
        if let Some(mesh_record) = self.vg_meshes.get_mut(&removed.removed.virtual_mesh) {
            mesh_record.ref_count = mesh_record.ref_count.saturating_sub(1);
        }
        self.vg_objects_dirty = true;
        Ok(())
    }

    /// Returns a `VgFrameData` view into the CPU-side meshlet/instance buffers, or
    /// `None` if there are no virtual geometry objects in the scene.
    pub fn vg_frame_data(&self) -> Option<VgFrameData<'_>> {
        if self.vg_cpu_meshlets.is_empty() {
            return None;
        }
        Some(VgFrameData {
            meshlets: bytemuck::cast_slice(&self.vg_cpu_meshlets),
            instances: bytemuck::cast_slice(&self.vg_cpu_instances),
            meshlet_count: self.vg_cpu_meshlets.len() as u32,
            instance_count: self.vg_cpu_instances.len() as u32,
            buffer_version: self.vg_buffer_version,
        })
    }

    /// Rebuild `vg_cpu_meshlets` and `vg_cpu_instances` from the current VG object set.
    ///
    /// This assigns each VG object a contiguous `instance_index` slot and patches the
    /// `GpuMeshletEntry::instance_index` field in all meshlets owned by that object.
    fn rebuild_vg_buffers(&mut self) {
        let instance_count = self.vg_objects.dense_len();
        self.vg_cpu_instances.clear();
        self.vg_cpu_meshlets.clear();
        self.vg_cpu_instances.reserve(instance_count);

        for i in 0..instance_count {
            let Some(obj) = self.vg_objects.get_dense(i) else { continue };
            let instance_index = self.vg_cpu_instances.len() as u32;
            self.vg_cpu_instances.push(obj.instance);

            let Some(mesh_record) = self.vg_meshes.get(&obj.virtual_mesh) else { continue };
            for mut meshlet in mesh_record.meshlets.iter().copied() {
                meshlet.instance_index = instance_index;
                self.vg_cpu_meshlets.push(meshlet);
            }
        }

        self.vg_buffer_version = self.vg_buffer_version.wrapping_add(1);
        log::debug!(
            "rebuild_vg_buffers: {} VG objects → {} meshlets",
            instance_count,
            self.vg_cpu_meshlets.len()
        );
    }
}


