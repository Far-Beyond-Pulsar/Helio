use std::sync::Arc;

use bytemuck::Zeroable;
use glam::{Mat3, Mat4, Vec3};
use helio_v3::{
    DrawIndexedIndirectArgs, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb, GpuInstanceData,
    GpuLight, GpuMaterial, GpuScene,
};
use thiserror::Error;

use crate::arena::{DenseArena, DenseRemove, SparsePool};
use crate::handles::{LightId, MaterialId, MeshId, ObjectId};
use crate::mesh::{MeshBuffers, MeshPool, MeshSlice, MeshUpload};

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("invalid {resource} handle")]
    InvalidHandle { resource: &'static str },
    #[error("{resource} is still in use")]
    ResourceInUse { resource: &'static str },
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
        let proj = Mat4::perspective_rh_gl(fov_y_radians, aspect, near, far);
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

pub struct Scene {
    gpu_scene: GpuScene,
    mesh_pool: MeshPool,
    materials: SparsePool<MaterialRecord, MaterialId>,
    lights: DenseArena<LightRecord, LightId>,
    objects: DenseArena<ObjectRecord, ObjectId>,
    prev_view_proj: Mat4,
}

impl Scene {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            mesh_pool: MeshPool::new(device.clone()),
            gpu_scene: GpuScene::new(device, queue),
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
        let (id, slot, is_new) = self.materials.insert(MaterialRecord {
            gpu: material,
            ref_count: 0,
        });
        if is_new {
            let pushed = self.gpu_scene.materials.push(material);
            debug_assert_eq!(pushed, slot);
        } else {
            let updated = self.gpu_scene.materials.update(slot, material);
            debug_assert!(updated);
        }
        id
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

    pub fn remove_material(&mut self, id: MaterialId) -> Result<()> {
        let Some(record) = self.materials.get(id) else {
            return Err(invalid("material"));
        };
        if record.ref_count != 0 {
            return Err(SceneError::ResourceInUse { resource: "material" });
        }
        let (slot, _) = self.materials.remove(id).ok_or_else(|| invalid("material"))?;
        let updated = self.gpu_scene.materials.update(slot, tombstone_material());
        debug_assert!(updated);
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
        let queue = self.gpu_scene.queue.clone();
        self.mesh_pool.flush(&queue);
        self.gpu_scene.flush();
    }

    pub fn advance_frame(&mut self) {
        self.gpu_scene.frame_count = self.gpu_scene.frame_count.wrapping_add(1);
    }
}
