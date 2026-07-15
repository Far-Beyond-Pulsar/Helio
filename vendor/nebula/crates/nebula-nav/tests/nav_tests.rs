use nebula_nav::{NavConfig, NavOutput, NavVertex, NavPolygon, CHUNK_TAG};
use nebula_core::traits::BakeOutput;

// ── Chunk tag ─────────────────────────────────────────────────────────────────

#[test]
fn nav_chunk_tag_is_navm() {
    assert_eq!(CHUNK_TAG.to_bytes(), *b"NAVM");
}

// ── NavConfig defaults ────────────────────────────────────────────────────────

#[test]
fn nav_config_default_agent_radius() {
    assert!((NavConfig::default().agent_radius - 0.4).abs() < f32::EPSILON);
}

#[test]
fn nav_config_default_agent_height() {
    assert!((NavConfig::default().agent_height - 1.8).abs() < f32::EPSILON);
}

#[test]
fn nav_config_default_max_step_height() {
    assert!((NavConfig::default().max_step_height - 0.4).abs() < f32::EPSILON);
}

#[test]
fn nav_config_default_max_slope_deg_is_45() {
    assert!((NavConfig::default().max_slope_deg - 45.0).abs() < f32::EPSILON);
}

#[test]
fn nav_config_default_cell_size() {
    assert!((NavConfig::default().cell_size - 0.3).abs() < f32::EPSILON);
}

#[test]
fn nav_config_default_cell_height() {
    assert!((NavConfig::default().cell_height - 0.2).abs() < f32::EPSILON);
}

#[test]
fn nav_config_default_min_region_area_is_8() {
    assert_eq!(NavConfig::default().min_region_area, 8);
}

#[test]
fn nav_config_default_merge_region_area_is_20() {
    assert_eq!(NavConfig::default().merge_region_area, 20);
}

#[test]
fn nav_config_default_bake_aabb_is_none() {
    assert!(NavConfig::default().bake_aabb.is_none());
}

// ── Fast preset ───────────────────────────────────────────────────────────────

#[test]
fn nav_config_fast_cell_size_is_1() {
    assert!((NavConfig::fast().cell_size - 1.0).abs() < f32::EPSILON);
}

#[test]
fn nav_config_fast_cell_height_is_0_5() {
    assert!((NavConfig::fast().cell_height - 0.5).abs() < f32::EPSILON);
}

#[test]
fn nav_config_fast_min_region_area_is_4() {
    assert_eq!(NavConfig::fast().min_region_area, 4);
}

// ── Ultra preset ──────────────────────────────────────────────────────────────

#[test]
fn nav_config_ultra_cell_size_is_0_15() {
    assert!((NavConfig::ultra().cell_size - 0.15).abs() < 1e-5);
}

#[test]
fn nav_config_ultra_cell_height_is_0_1() {
    assert!((NavConfig::ultra().cell_height - 0.1).abs() < 1e-5);
}

#[test]
fn nav_config_ultra_min_region_area_is_16() {
    assert_eq!(NavConfig::ultra().min_region_area, 16);
}

// ── Preset ordering ───────────────────────────────────────────────────────────

#[test]
fn nav_config_cell_size_ordering_fast_coarser() {
    // Fast has larger voxels (less detail), ultra has smallest voxels.
    assert!(NavConfig::fast().cell_size > NavConfig::default().cell_size);
    assert!(NavConfig::default().cell_size > NavConfig::ultra().cell_size);
}

// ── BakeOutput trait ──────────────────────────────────────────────────────────

#[test]
fn nav_output_kind_name_is_navmesh() {
    assert_eq!(NavOutput::kind_name(), "navmesh");
}

// ── NavOutput construction ────────────────────────────────────────────────────

fn make_triangle_navmesh() -> NavOutput {
    NavOutput {
        vertices: vec![
            NavVertex { position: [0.0, 0.0, 0.0] },
            NavVertex { position: [1.0, 0.0, 0.0] },
            NavVertex { position: [0.5, 0.0, 1.0] },
        ],
        polygons: vec![
            NavPolygon {
                vertex_indices: vec![0, 1, 2],
                neighbour_indices: vec![u32::MAX, u32::MAX, u32::MAX],
                area_flags: 0,
            }
        ],
        aabb_min: [0.0, 0.0, 0.0],
        aabb_max: [1.0, 0.0, 1.0],
        walkable_area: 0.5,
        config_json: "{}".to_string(),
    }
}

#[test]
fn nav_output_vertex_count() {
    let nav = make_triangle_navmesh();
    assert_eq!(nav.vertices.len(), 3);
}

#[test]
fn nav_output_polygon_count() {
    let nav = make_triangle_navmesh();
    assert_eq!(nav.polygons.len(), 1);
}

#[test]
fn nav_output_polygon_has_3_vertices() {
    let nav = make_triangle_navmesh();
    assert_eq!(nav.polygons[0].vertex_indices.len(), 3);
}

#[test]
fn nav_output_polygon_neighbour_count_matches_vertex_count() {
    let nav = make_triangle_navmesh();
    let poly = &nav.polygons[0];
    assert_eq!(poly.vertex_indices.len(), poly.neighbour_indices.len());
}

#[test]
fn nav_output_border_edges_use_u32_max_sentinel() {
    let nav = make_triangle_navmesh();
    for &n in &nav.polygons[0].neighbour_indices {
        assert_eq!(n, u32::MAX, "a lone triangle has no neighbours");
    }
}

#[test]
fn nav_output_walkable_area_is_positive() {
    let nav = make_triangle_navmesh();
    assert!(nav.walkable_area > 0.0);
}

// ── NavVertex ─────────────────────────────────────────────────────────────────

#[test]
fn nav_vertex_position_accessible() {
    let v = NavVertex { position: [1.0, 2.0, 3.0] };
    assert_eq!(v.position, [1.0_f32, 2.0, 3.0]);
}

#[test]
fn nav_vertex_clone() {
    let v = NavVertex { position: [1.0, 0.0, 0.0] };
    let c = v;
    assert_eq!(c.position[0], 1.0);
}

// ── Serialize / deserialize ───────────────────────────────────────────────────

#[test]
fn nav_output_serialize_deserialize_roundtrip() {
    let nav = make_triangle_navmesh();
    let bytes = nav.serialize_to_bytes().expect("serialize");
    let back = NavOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.vertices.len(), 3);
    assert_eq!(back.polygons.len(), 1);
    assert_eq!(back.polygons[0].vertex_indices, vec![0u32, 1, 2]);
    assert!((back.walkable_area - 0.5).abs() < 1e-6);
}

#[test]
fn nav_output_serialize_produces_nonempty_bytes() {
    let nav = make_triangle_navmesh();
    let bytes = nav.serialize_to_bytes().expect("serialize");
    assert!(!bytes.is_empty());
}

#[test]
fn nav_output_deserialize_corrupt_returns_error() {
    let result = NavOutput::deserialize_from_bytes(b"\xDE\xAD\xBE\xEF");
    assert!(result.is_err());
}

#[test]
fn nav_output_empty_mesh_roundtrip() {
    let nav = NavOutput {
        vertices: vec![],
        polygons: vec![],
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
        walkable_area: 0.0,
        config_json: "{}".to_string(),
    };
    let bytes = nav.serialize_to_bytes().expect("serialize");
    let back = NavOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.vertices.len(), 0);
    assert_eq!(back.polygons.len(), 0);
    assert_eq!(back.walkable_area, 0.0);
}

// ── JSON serde ────────────────────────────────────────────────────────────────

#[test]
fn nav_config_json_roundtrip() {
    let cfg = NavConfig::ultra();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: NavConfig = serde_json::from_str(&json).expect("deserialize");
    assert!((back.cell_size - cfg.cell_size).abs() < 1e-5);
    assert!((back.agent_height - cfg.agent_height).abs() < 1e-5);
    assert_eq!(back.min_region_area, cfg.min_region_area);
}

#[test]
fn nav_config_bake_aabb_roundtrip() {
    let mut cfg = NavConfig::default();
    cfg.bake_aabb = Some(([0.0, 0.0, 0.0], [10.0, 5.0, 10.0]));
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: NavConfig = serde_json::from_str(&json).expect("deserialize");
    let aabb = back.bake_aabb.expect("bake_aabb should be Some");
    assert_eq!(aabb.0, [0.0, 0.0, 0.0]);
    assert_eq!(aabb.1, [10.0, 5.0, 10.0]);
}

// ── Baker (CPU + BakeContext gated) ───────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter for BakeContext"]
fn nav_baker_produces_output_for_empty_scene() {
    use nebula_core::{context::BakeContext, progress::NullReporter, scene::SceneGeometry};
    use nebula_nav::NavBaker;
    use nebula_core::traits::BakePass;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let scene = SceneGeometry::default();
        let cfg = NavConfig::fast();
        let out = NavBaker.execute(&scene, &cfg, &ctx, &NullReporter).await
            .expect("bake");
        // Empty scene may produce an empty or minimal mesh.
        assert!(out.polygons.len() < 1_000_000); // sanity bound
    });
}
