use nebula_core::scene::{
    BakeMesh, MaterialDesc, SceneGeometry, Transform,
};
use glam::Mat4;

// ── Transform ─────────────────────────────────────────────────────────────────

#[test]
fn transform_identity_is_identity_matrix() {
    let t = Transform::IDENTITY;
    assert_eq!(t.0, Mat4::IDENTITY);
}

#[test]
fn transform_default_is_identity() {
    let t = Transform::default();
    assert_eq!(t.0, Mat4::IDENTITY);
}

#[test]
fn transform_clone_and_copy() {
    let t = Transform::IDENTITY;
    let t2 = t;          // Copy
    let t3 = t.clone();  // Clone
    assert_eq!(t.0, t2.0);
    assert_eq!(t.0, t3.0);
}

// ── MaterialDesc ──────────────────────────────────────────────────────────────

#[test]
fn material_desc_default_albedo_is_light_grey() {
    let m = MaterialDesc::default();
    assert!((m.albedo[0] - 0.8).abs() < 1e-6);
    assert!((m.albedo[1] - 0.8).abs() < 1e-6);
    assert!((m.albedo[2] - 0.8).abs() < 1e-6);
    assert!((m.albedo[3] - 1.0).abs() < 1e-6);
}

#[test]
fn material_desc_default_roughness() {
    let m = MaterialDesc::default();
    assert!((m.roughness - 0.5).abs() < 1e-6);
}

#[test]
fn material_desc_default_metallic_is_zero() {
    let m = MaterialDesc::default();
    assert!((m.metallic - 0.0).abs() < 1e-6);
}

#[test]
fn material_desc_casts_shadows_by_default() {
    assert!(MaterialDesc::default().casts_shadows);
}

#[test]
fn material_desc_audio_absorption_in_range() {
    let m = MaterialDesc::default();
    assert!(m.audio_absorption >= 0.0 && m.audio_absorption <= 1.0);
    assert!(m.audio_scattering >= 0.0 && m.audio_scattering <= 1.0);
}

// ── BakeMesh ──────────────────────────────────────────────────────────────────

fn triangle_mesh() -> BakeMesh {
    BakeMesh {
        id: uuid::Uuid::new_v4(),
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        lightmap_uvs: None,
        indices: vec![0, 1, 2],
        material_ids: vec![0, 0, 0],
        world_transform: Transform::IDENTITY,
    }
}

#[test]
fn bake_mesh_has_expected_vertex_count() {
    let m = triangle_mesh();
    assert_eq!(m.positions.len(), 3);
    assert_eq!(m.normals.len(), 3);
    assert_eq!(m.uvs.len(), 3);
}

#[test]
fn bake_mesh_index_count_is_multiple_of_three() {
    let m = triangle_mesh();
    assert_eq!(m.indices.len() % 3, 0);
}

#[test]
fn bake_mesh_no_lightmap_uvs_by_default() {
    let m = triangle_mesh();
    assert!(m.lightmap_uvs.is_none());
}

#[test]
fn bake_mesh_with_lightmap_uvs() {
    let mut m = triangle_mesh();
    m.lightmap_uvs = Some(vec![[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]);
    assert!(m.lightmap_uvs.is_some());
    assert_eq!(m.lightmap_uvs.unwrap().len(), 3);
}

#[test]
fn bake_mesh_material_ids_length() {
    let m = triangle_mesh();
    // material_ids is per-vertex in this impl
    assert_eq!(m.material_ids.len(), m.positions.len());
}

// ── SceneGeometry ─────────────────────────────────────────────────────────────

#[test]
fn scene_geometry_default_is_empty() {
    let s = SceneGeometry::default();
    assert!(s.meshes.is_empty());
    assert!(s.lights.is_empty());
}

#[test]
fn scene_geometry_push_mesh() {
    let mut s = SceneGeometry::default();
    s.meshes.push(triangle_mesh());
    assert_eq!(s.meshes.len(), 1);
}

#[test]
fn scene_geometry_multiple_meshes() {
    let mut s = SceneGeometry::default();
    for _ in 0..5 {
        s.meshes.push(triangle_mesh());
    }
    assert_eq!(s.meshes.len(), 5);
}

#[test]
fn scene_geometry_mesh_vertices_accessible() {
    let mut s = SceneGeometry::default();
    s.meshes.push(triangle_mesh());
    let m = &s.meshes[0];
    assert_eq!(m.positions[0], [0.0, 0.0, 0.0]);
    assert_eq!(m.positions[1], [1.0, 0.0, 0.0]);
    assert_eq!(m.positions[2], [0.0, 1.0, 0.0]);
}

#[test]
fn scene_geometry_total_triangle_count() {
    let mut s = SceneGeometry::default();
    s.meshes.push(triangle_mesh());
    let total_tris: usize = s.meshes.iter().map(|m| m.indices.len() / 3).sum();
    assert_eq!(total_tris, 1);
}

#[test]
fn bake_mesh_unique_ids() {
    let a = triangle_mesh();
    let b = triangle_mesh();
    // Each new_v4 should produce a unique ID
    assert_ne!(a.id, b.id);
}
