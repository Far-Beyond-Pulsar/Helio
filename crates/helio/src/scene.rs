use std::sync::Arc;

use bytemuck::Zeroable;
use glam::{Mat3, Mat4, Vec3};
use helio_v3::{
    scene::GrowableBuffer,
    DrawIndexedIndirectArgs, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb, GpuInstanceData,
    GpuLight, GpuMaterial, GpuScene,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::arena::{DenseArena, DenseRemove, SparsePool};
use crate::handles::{LightId, MaterialId, MeshId, ObjectId, TextureId};
use crate::material::{
    GpuMaterialTextureSlot, GpuMaterialTextures, MaterialAsset, MaterialTextureRef, MaterialTextures,
    TextureTransform, TextureUpload, MAX_TEXTURES,
};
use crate::mesh::{MeshBuffers, MeshPool, MeshSlice, MeshUpload};

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

fn object_gpu_data(mesh: MeshId, material_slot: usize, desc: ObjectDescriptor, slice: MeshSlice, instance_id: usize) -> ObjectRecord {
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
        draw: GpuDrawCall {
            index_count: slice.index_count,
            first_index: slice.first_index,
            vertex_offset: slice.first_vertex as i32,
            instance_id: instance_id as u32,
            _pad: 0,
        },
    }
}

fn compute_cascade_matrix(
    inv_view_proj: Mat4,
    view: Mat4,
    light_dir: Vec3,
    near: f32,
    far: f32,
    cam_near: f32,
    cam_far: f32,
) -> Mat4 {
    // Compute frustum corners for this cascade in world space
    let ndc_corners = [
        Vec3::new(-1.0, -1.0, 0.0),  // near bottom-left
        Vec3::new( 1.0, -1.0, 0.0),  // near bottom-right
        Vec3::new(-1.0,  1.0, 0.0),  // near top-left
        Vec3::new( 1.0,  1.0, 0.0),  // near top-right
        Vec3::new(-1.0, -1.0, 1.0),  // far bottom-left
        Vec3::new( 1.0, -1.0, 1.0),  // far bottom-right
        Vec3::new(-1.0,  1.0, 1.0),  // far top-left
        Vec3::new( 1.0,  1.0, 1.0),  // far top-right
    ];

    // Transform corners to world space
    let mut world_corners = [Vec3::ZERO; 8];
    for (i, ndc) in ndc_corners.iter().enumerate() {
        let w = inv_view_proj * ndc.extend(1.0);
        world_corners[i] = w.truncate() / w.w;
    }

    // Interpolate corners to cascade near/far planes
    let lambda_near = (near - cam_near) / (cam_far - cam_near);
    let lambda_far = (far - cam_near) / (cam_far - cam_near);

    for i in 0..4 {
        let near_pt = world_corners[i];
        let far_pt = world_corners[i + 4];
        world_corners[i] = near_pt + (far_pt - near_pt) * lambda_near;
        world_corners[i + 4] = near_pt + (far_pt - near_pt) * lambda_far;
    }

    // Compute bounding sphere center (simple average)
    let center = world_corners.iter().fold(Vec3::ZERO, |acc, &v| acc + v) / 8.0;

    // Compute light view matrix
    let light_up = if light_dir.y.abs() > 0.9 { Vec3::Z } else { Vec3::Y };
    let light_view = Mat4::look_at_rh(center, center + light_dir, light_up);

    // Transform corners to light space and find bounds
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for corner in world_corners {
        let light_space = (light_view * corner.extend(1.0)).truncate();
        min = min.min(light_space);
        max = max.max(light_space);
    }

    // Expand bounds slightly to avoid edge clipping
    let expand = 2.0;
    min -= Vec3::splat(expand);
    max += Vec3::splat(expand);

    // Create orthographic projection for this cascade
    let light_proj = Mat4::orthographic_rh(min.x, max.x, min.y, max.y, -max.z, -min.z);

    light_proj * light_view
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
    prev_view_proj: Mat4,
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
            prev_view_proj: Mat4::IDENTITY,
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

        let next_instance = self.gpu_scene.instances.len();
        let record = object_gpu_data(desc.mesh, material_slot, desc, mesh_slice, next_instance);
        let (id, dense_index) = self.objects.insert(record.clone());

        let pushed_instance = self.gpu_scene.instances.push(record.instance);
        let pushed_aabb = self.gpu_scene.aabbs.push(record.aabb);
        let pushed_draw = self.gpu_scene.draw_calls.push(record.draw);
        self.gpu_scene.indirect.push(DrawIndexedIndirectArgs::culled(
            0,
            0,
            0,
            0,
        ));
        let visible = DrawIndexedIndirectArgs {
            index_count: record.draw.index_count,
            instance_count: 1,
            first_index: record.draw.first_index,
            base_vertex: record.draw.vertex_offset,
            first_instance: dense_index as u32,
        };
        let updated_indirect = self.gpu_scene.indirect.update(dense_index, visible);
        debug_assert!(updated_indirect);
        self.gpu_scene.visibility.push(1);

        debug_assert_eq!(pushed_instance, dense_index);
        debug_assert_eq!(pushed_aabb, dense_index);
        debug_assert_eq!(pushed_draw, dense_index);

        Ok(id)
    }

    pub fn update_object_transform(&mut self, id: ObjectId, transform: Mat4) -> Result<()> {
        let Some((dense_index, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        record.instance.model = transform.to_cols_array();
        record.instance.normal_mat = normal_matrix(transform);
        let updated = self.gpu_scene.instances.update(dense_index, record.instance);
        debug_assert!(updated);
        Ok(())
    }

    pub fn update_object_material(&mut self, id: ObjectId, material: MaterialId) -> Result<()> {
        let Some((dense_index, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        let new_slot = {
            let (slot, new_material) = self
                .materials
                .get_mut_with_slot(material)
                .ok_or_else(|| invalid("material"))?;
            new_material.ref_count += 1;
            slot
        };
        let (_, old_material) = self
            .materials
            .get_mut_with_slot(record.material)
            .ok_or_else(|| invalid("material"))?;
        old_material.ref_count = old_material.ref_count.saturating_sub(1);

        record.material = material;
        record.instance.material_id = new_slot as u32;
        let updated = self.gpu_scene.instances.update(dense_index, record.instance);
        debug_assert!(updated);
        Ok(())
    }

    pub fn update_object_bounds(&mut self, id: ObjectId, bounds: [f32; 4]) -> Result<()> {
        let Some((dense_index, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        record.instance.bounds = bounds;
        record.aabb = sphere_to_aabb(bounds);
        let updated_instance = self.gpu_scene.instances.update(dense_index, record.instance);
        let updated_aabb = self.gpu_scene.aabbs.update(dense_index, record.aabb);
        debug_assert!(updated_instance && updated_aabb);
        Ok(())
    }

    pub fn remove_object(&mut self, id: ObjectId) -> Result<()> {
        let DenseRemove { removed, dense_index, moved } =
            self.objects.remove(id).ok_or_else(|| invalid("object"))?;

        if let Some(material) = self.materials.get_mut_with_slot(removed.material).map(|(_, material)| material) {
            material.ref_count = material.ref_count.saturating_sub(1);
        }
        if let Some(mesh) = self.mesh_pool.get_mut(removed.mesh) {
            mesh.ref_count = mesh.ref_count.saturating_sub(1);
        }

        let _ = self.gpu_scene.instances.swap_remove(dense_index);
        let _ = self.gpu_scene.aabbs.swap_remove(dense_index);
        let _ = self.gpu_scene.draw_calls.swap_remove(dense_index);
        let _ = self.gpu_scene.indirect.swap_remove(dense_index);
        let _ = self.gpu_scene.visibility.swap_remove(dense_index);

        if let Some((moved_handle, moved_index)) = moved {
            let (_, moved_record) = self
                .objects
                .get_mut_with_index(moved_handle)
                .ok_or_else(|| invalid("object"))?;
            moved_record.draw.instance_id = moved_index as u32;
            let updated_draw = self.gpu_scene.draw_calls.update(moved_index, moved_record.draw);
            let updated_indirect = self.gpu_scene.indirect.update(
                moved_index,
                DrawIndexedIndirectArgs {
                    index_count: moved_record.draw.index_count,
                    instance_count: 1,
                    first_index: moved_record.draw.first_index,
                    base_vertex: moved_record.draw.vertex_offset,
                    first_instance: moved_index as u32,
                },
            );
            debug_assert!(updated_draw && updated_indirect);
        }

        Ok(())
    }

    pub fn flush(&mut self) {
        // Compute shadow matrices for all shadow-casting lights
        self.compute_shadow_matrices();

        let queue = self.gpu_scene.queue.clone();
        self.mesh_pool.flush(&queue);
        self.material_textures.flush(&queue);
        self.gpu_scene.flush();
    }

    fn compute_shadow_matrices(&mut self) {
        use libhelio::GpuShadowMatrix;

        const MAX_SHADOW_LIGHTS: usize = 4;

        // Clear existing matrices
        self.gpu_scene.shadow_matrices.0.clear();

        // Get camera uniforms for CSM computation
        let camera_uniforms = self.gpu_scene.camera.get();
        let view_mat = Mat4::from_cols_array(&camera_uniforms.view);
        let proj_mat = Mat4::from_cols_array(&camera_uniforms.proj);
        let view_proj = proj_mat * view_mat;
        let inv_view_proj = view_proj.inverse();

        // O(1): Only process first MAX_SHADOW_LIGHTS (4) shadow-casting lights
        let mut shadow_lights_processed = 0;
        for (_, light_record) in self.lights.iter() {
            if shadow_lights_processed >= MAX_SHADOW_LIGHTS {
                break;  // O(1) guarantee: stop after 4 lights
            }

            let light = &light_record.gpu;

            // Skip lights without shadows
            if light.shadow_index == u32::MAX {
                continue;
            }

            shadow_lights_processed += 1;

            let light_type = light.light_type;
            let light_pos = Vec3::new(light.position_range[0], light.position_range[1], light.position_range[2]);
            let light_dir = Vec3::new(light.direction_outer[0], light.direction_outer[1], light.direction_outer[2]).normalize();

            if light_type == 0 {  // Directional light - 4 CSM cascades
                let csm_splits = [16.0, 80.0, 300.0, 1400.0];
                let near = camera_uniforms.position_near[3];
                let far = camera_uniforms.projection[3];

                let mut prev_split = near;
                for (cascade_idx, &split_far) in csm_splits.iter().enumerate() {
                    let split_near = prev_split;
                    let split_far = split_far.min(far);

                    // Compute frustum corners for this cascade
                    let cascade_view_proj = compute_cascade_matrix(
                        inv_view_proj, view_mat, light_dir, split_near, split_far, near, far,
                    );

                    let matrix = GpuShadowMatrix {
                        light_view_proj: cascade_view_proj.to_cols_array(),
                        atlas_rect: [0.0, 0.0, 1.0, 1.0],  // Full atlas (no tiling yet)
                        bias_split: [0.0001, 0.0, split_far, 0.0],
                    };
                    self.gpu_scene.shadow_matrices.push(matrix);

                    prev_split = split_far;
                }
            } else if light_type == 1 {  // Point light - 6 cube faces
                let faces = [
                    (Vec3::X,  Vec3::Y),  // +X
                    (Vec3::NEG_X, Vec3::Y),  // -X
                    (Vec3::Y,  Vec3::NEG_Z), // +Y
                    (Vec3::NEG_Y, Vec3::Z),  // -Y
                    (Vec3::Z,  Vec3::Y),  // +Z
                    (Vec3::NEG_Z, Vec3::Y),  // -Z
                ];

                for (forward, up) in faces {
                    let view = Mat4::look_at_rh(light_pos, light_pos + forward, up);
                    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, light.position_range[3]);
                    let matrix = GpuShadowMatrix {
                        light_view_proj: (proj * view).to_cols_array(),
                        atlas_rect: [0.0, 0.0, 1.0, 1.0],
                        bias_split: [0.0001, 0.0, 0.0, 0.0],
                    };
                    self.gpu_scene.shadow_matrices.push(matrix);
                }
            } else if light_type == 2 {  // Spot light - 1 perspective matrix
                let view = Mat4::look_at_rh(light_pos, light_pos + light_dir, Vec3::Y);
                let outer_angle = light.direction_outer[3].acos() * 2.0;
                let proj = Mat4::perspective_rh(outer_angle, 1.0, 0.1, light.position_range[3]);
                let matrix = GpuShadowMatrix {
                    light_view_proj: (proj * view).to_cols_array(),
                    atlas_rect: [0.0, 0.0, 1.0, 1.0],
                    bias_split: [0.0001, 0.0, 0.0, 0.0],
                };
                self.gpu_scene.shadow_matrices.push(matrix);
            }
        }
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
}
