//! Internal helper functions for scene object management.

use glam::{Mat3, Mat4, Vec3};
use helio_v3::{GpuDrawCall, GpuInstanceAabb, GpuInstanceData};

use crate::groups::GroupMask;
use crate::handles::MeshId;
use crate::material::{GpuMaterialTextureSlot, GpuMaterialTextures, MaterialTextureRef, TextureTransform};
use crate::mesh::MeshSlice;

use super::types::ObjectRecord;
use super::ObjectDescriptor;

/// Compute the normal matrix from a model transform matrix.
///
/// The normal matrix is the inverse-transpose of the model matrix's upper-left 3×3 block.
/// This transformation correctly handles non-uniform scaling when transforming normal vectors
/// from object space to world space.
///
/// # Why precompute this?
///
/// Computing a 3×3 matrix inverse is expensive (requires adjugate calculation). By computing
/// it once on the CPU when the transform changes, we avoid doing this operation per-vertex
/// in the vertex shader.
///
/// # Performance
/// - CPU cost: O(1) per transform update
/// - GPU savings: Eliminates O(vertices) inverse operations per frame
///
/// # Returns
/// A 3×4 padded normal matrix (12 floats) ready for GPU upload.
pub(super) fn normal_matrix(transform: Mat4) -> [f32; 12] {
    let mat3 = Mat3::from_mat4(transform).inverse().transpose();
    let cols = mat3.to_cols_array();
    [
        cols[0], cols[1], cols[2], 0.0,
        cols[3], cols[4], cols[5], 0.0,
        cols[6], cols[7], cols[8], 0.0,
    ]
}

/// Convert a bounding sphere to an axis-aligned bounding box (AABB).
///
/// # Parameters
/// - `bounds`: `[center.x, center.y, center.z, radius]`
///
/// # Returns
/// An AABB that conservatively encloses the sphere.
///
/// # Note
/// This conversion is conservative (the AABB is larger than the sphere) but allows
/// simpler AABB-based culling in some shader passes.
pub(super) fn sphere_to_aabb(bounds: [f32; 4]) -> GpuInstanceAabb {
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

/// Test if an object is visible based on group membership and hidden groups.
///
/// # Visibility Semantics
/// - An object is **hidden** if **any** of its groups are currently hidden
/// - Ungrouped objects (`groups == GroupMask::NONE`) are **always visible**
///
/// # Parameters
/// - `groups`: The object's group membership bitmask
/// - `group_hidden`: Bitmask of currently hidden groups
///
/// # Returns
/// `true` if the object should be rendered, `false` if hidden
///
/// # Example
/// ```ignore
/// let obj_groups = GroupMask::from_id(GroupId(0)); // Object in group 0
/// let hidden = GroupMask::from_id(GroupId(0));     // Group 0 is hidden
/// assert!(!object_is_visible(obj_groups, hidden)); // Object is hidden
/// ```
#[inline(always)]
pub(super) fn object_is_visible(groups: GroupMask, group_hidden: GroupMask) -> bool {
    groups.is_empty() || !groups.intersects(group_hidden)
}

/// Construct a complete `ObjectRecord` from an `ObjectDescriptor`.
///
/// This function builds all GPU-side data structures needed to render an object:
/// - Instance data (model matrix, normal matrix, bounds, mesh/material indices, flags)
/// - AABB (converted from sphere bounds)
/// - Draw call template (index count, offsets, etc.)
///
/// # Parameters
/// - `mesh`: Mesh handle (for slot lookup)
/// - `material_slot`: Material's slot index in the material storage buffer
/// - `desc`: User-provided object descriptor
/// - `slice`: Mesh geometry slice (index count, offsets)
///
/// # Returns
/// A fully initialized `ObjectRecord` ready for insertion into the scene.
///
/// # Note
/// The `gpu_slot` and `draw.first_instance` fields are initialized to 0 and will be
/// patched by `rebuild_instance_buffers_*()` functions.
pub(super) fn object_gpu_data(
    mesh: MeshId,
    material_slot: usize,
    desc: ObjectDescriptor,
    slice: MeshSlice
) -> ObjectRecord {
    ObjectRecord {
        mesh,
        material: desc.material,
        groups: desc.groups,
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
        gpu_slot: 0,
    }
}

/// Convert a material texture reference to a GPU texture slot descriptor.
///
/// # Parameters
/// - `texture`: Optional texture reference with transform
///
/// # Returns
/// A GPU texture slot descriptor with texture index and UV transform parameters.
/// Returns a "missing" slot if `texture` is `None`.
pub(super) fn gpu_texture_slot(texture: Option<MaterialTextureRef>) -> GpuMaterialTextureSlot {
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

/// Convert material textures to GPU material texture descriptor.
///
/// Builds the complete GPU-side material texture data structure from CPU-side
/// material texture references.
///
/// # Parameters
/// - `textures`: Material texture references (base color, normal, roughness, etc.)
///
/// # Returns
/// A GPU material texture descriptor with all texture slots and parameters.
pub(super) fn gpu_material_textures(textures: &crate::material::MaterialTextures) -> GpuMaterialTextures {
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

/// Iterate over all texture references in a material texture set.
///
/// Calls the provided closure for each texture reference that is `Some`.
/// Used for reference counting when materials are inserted or removed.
///
/// # Parameters
/// - `textures`: Material texture set
/// - `f`: Closure to call for each texture reference
pub(super) fn each_material_texture_ref<F>(textures: &crate::material::MaterialTextures, mut f: F)
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
