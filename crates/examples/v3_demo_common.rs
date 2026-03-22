use glam::{Mat4, Vec3};
use helio::{GpuLight, GpuMaterial, LightId, LightType, MaterialId, MeshId, MeshUpload, ObjectDescriptor, PackedVertex, Renderer};

pub fn make_material(
    base_color: [f32; 4],
    roughness: f32,
    metallic: f32,
    emissive: [f32; 3],
    emissive_strength: f32,
) -> GpuMaterial {
    GpuMaterial {
        base_color,
        emissive: [emissive[0], emissive[1], emissive[2], emissive_strength],
        roughness_metallic: [roughness, metallic, 1.5, 0.5],
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        workflow: 0,
        flags: 0,
        _pad: 0,
    }
}

pub fn directional_light(direction: [f32; 3], color: [f32; 3], intensity: f32) -> GpuLight {
    GpuLight {
        position_range: [0.0, 0.0, 0.0, f32::MAX],
        direction_outer: [direction[0], direction[1], direction[2], 0.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0,  // Enable shadows
        light_type: LightType::Directional as u32,
        inner_angle: 0.0,
        _pad: 0,
    }
}

pub fn point_light(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> GpuLight {
    GpuLight {
        position_range: [position[0], position[1], position[2], range],
        direction_outer: [0.0, 0.0, -1.0, 0.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0,  // Enable shadows
        light_type: LightType::Point as u32,
        inner_angle: 0.0,
        _pad: 0,
    }
}

pub fn spot_light(
    position: [f32; 3],
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
) -> GpuLight {
    GpuLight {
        position_range: [position[0], position[1], position[2], range],
        direction_outer: [direction[0], direction[1], direction[2], outer_angle.cos()],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0,  // Enable shadows
        light_type: LightType::Spot as u32,
        inner_angle: inner_angle.cos(),
        _pad: 0,
    }
}

pub fn insert_object(
    renderer: &mut Renderer,
    mesh: MeshId,
    material: MaterialId,
    transform: Mat4,
    radius: f32,
) -> helio::SceneResult<helio::ObjectId> {
    renderer.insert_object(ObjectDescriptor {
        mesh,
        material,
        transform,
        bounds: [transform.w_axis.x, transform.w_axis.y, transform.w_axis.z, radius],
        flags: 0,
        groups: helio::GroupMask::NONE,
    })
}

pub fn cube_mesh(center: [f32; 3], half_extent: f32) -> MeshUpload {
    box_mesh(center, [half_extent, half_extent, half_extent])
}

pub fn box_mesh(center: [f32; 3], half_extents: [f32; 3]) -> MeshUpload {
    let c = Vec3::from_array(center);
    let e = Vec3::from_array(half_extents);
    let corners = [
        c + Vec3::new(-e.x, -e.y,  e.z),
        c + Vec3::new( e.x, -e.y,  e.z),
        c + Vec3::new( e.x,  e.y,  e.z),
        c + Vec3::new(-e.x,  e.y,  e.z),
        c + Vec3::new(-e.x, -e.y, -e.z),
        c + Vec3::new( e.x, -e.y, -e.z),
        c + Vec3::new( e.x,  e.y, -e.z),
        c + Vec3::new(-e.x,  e.y, -e.z),
    ];
    let faces = [
        ([0, 1, 2, 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([5, 4, 7, 6], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]),
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([1, 5, 6, 2], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([3, 2, 6, 7], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([4, 5, 1, 0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face_index, (quad, normal, tangent)) in faces.iter().enumerate() {
        let base = (face_index * 4) as u32;
        let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        for (i, corner_index) in quad.iter().enumerate() {
            vertices.push(PackedVertex::from_components(
                corners[*corner_index].to_array(),
                *normal,
                uv[i],
                *tangent,
                1.0,
            ));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshUpload { vertices, indices }
}

pub fn plane_mesh(center: [f32; 3], half_extent: f32) -> MeshUpload {
    let c = Vec3::from_array(center);
    let e = half_extent;
    let normal = [0.0, 1.0, 0.0];
    let tangent = [1.0, 0.0, 0.0];
    let positions = [
        c + Vec3::new(-e, 0.0, -e),
        c + Vec3::new(e, 0.0, -e),
        c + Vec3::new(e, 0.0, e),
        c + Vec3::new(-e, 0.0, e),
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let vertices = positions
        .into_iter()
        .zip(uvs)
        .map(|(position, uv)| PackedVertex::from_components(position.to_array(), normal, uv, tangent, 1.0))
        .collect();
    let indices = vec![0, 1, 2, 0, 2, 3];
    MeshUpload { vertices, indices }
}

pub fn update_point_light(renderer: &mut Renderer, id: LightId, position: Vec3, color: [f32; 3], intensity: f32, range: f32) {
    let _ = renderer.update_light(id, point_light(position.to_array(), color, intensity, range));
}
