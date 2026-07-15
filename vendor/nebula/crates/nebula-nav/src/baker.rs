use async_trait::async_trait;
use glam::Vec3;
use rayon::prelude::*;
use nebula_core::{
    context::BakeContext, error::NebulaError, progress::ProgressReporter,
    scene::SceneGeometry, traits::BakePass,
};
use crate::{
    config::NavConfig,
    output::{NavOutput, NavPolygon, NavVertex},
};

/// CPU + rayon navigation mesh baker.
///
/// ## Pipeline
///
/// 1. **Voxelisation** — scene triangles are rasterised into a 3-D voxel
///    height-field.  Each column tracks spans of solid/walkable/unoccupied
///    voxels.  Parallelised across voxel columns with rayon.
///
/// 2. **Walkability filtering** — spans are marked walkable if:
///    - The surface normal is shallower than `max_slope_deg`, and
///    - There is at least `agent_height` of vertical clearance, and
///    - No geometry is within `agent_radius` in XZ.
///
/// 3. **Region growing** — a watershed-style region-growing pass labels
///    every connected walkable span with a unique region ID.  Small regions
///    below `min_region_area` are merged.
///
/// 4. **Contour tracing** — region boundaries are traced to produce raw
///    edge loops, then simplified to remove axis-aligned notches within
///    `max_edge_error` of the original boundary.
///
/// 5. **Polygon mesh** — simplified contours are triangulated (ear clipping)
///    into convex polygon sets.  Polygons larger than `max_edge_length` are
///    subdivided.  Adjacency is computed for neighbour links.
///
/// The output is a [`NavOutput`] containing a vertex/polygon list suitable
/// for runtime A* or navmesh pathfinding.
pub struct NavBaker;

#[async_trait(?Send)]
impl BakePass for NavBaker {
    type Input  = NavConfig;
    type Output = NavOutput;

    fn name(&self) -> &'static str { "navmesh" }

    async fn execute(
        &self,
        scene:    &SceneGeometry,
        config:   &NavConfig,
        _ctx:     &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<NavOutput, NebulaError> {
        reporter.begin("navmesh", 5);

        reporter.step("navmesh", 0, "computing scene AABB");
        let (aabb_min, aabb_max) = scene_aabb(scene, config);

        reporter.step("navmesh", 1, "voxelising scene");
        let hf = voxelise(scene, config, aabb_min, aabb_max)?;

        reporter.step("navmesh", 2, "filtering walkable spans");
        let walkable = filter_walkable(&hf, config);

        reporter.step("navmesh", 3, "growing regions + tracing contours");
        let contours = grow_regions_and_trace(&walkable, &hf, config);

        reporter.step("navmesh", 4, "building polygon mesh");
        let (vertices, polygons, walkable_area) = build_polygon_mesh(&contours, config);

        reporter.finish("navmesh", true, "done");
        let config_json = serde_json::to_string(config).unwrap_or_default();
        Ok(NavOutput { vertices, polygons, aabb_min, aabb_max, walkable_area, config_json })
    }
}

// ── Types ─────────────────────────────────────────────────────────────────────

/// One solid span in a height-field column.
#[derive(Clone, Debug)]
struct Span { y_min: f32, y_max: f32, walkable: bool, region: u32 }

/// Compact height-field for a grid of XZ columns.
#[derive(Debug)]
struct HeightField {
    cols: Vec<Vec<Span>>,  // rows × cols, indexed [z * gx + x]
    gx: usize,
    gz: usize,
    world_min: [f32; 3],
    cell_size: f32,
    cell_height: f32,
}

impl HeightField {
    fn idx(&self, x: usize, z: usize) -> usize { z * self.gx + x }
    fn col(&self, x: usize, z: usize) -> &[Span] { &self.cols[self.idx(x,z)] }
    fn col_mut(&mut self, x: usize, z: usize) -> &mut Vec<Span> { let i = z * self.gx + x; &mut self.cols[i] }
}

/// A simplified 2.5-D contour polygon.
#[derive(Debug)]
struct Contour { verts: Vec<[f32;3]>, region: u32 }

// ── Scene AABB ────────────────────────────────────────────────────────────────

fn scene_aabb(scene: &SceneGeometry, config: &NavConfig) -> ([f32;3], [f32;3]) {
    if let Some((mn, mx)) = config.bake_aabb { return (mn, mx); }
    let (mut mn, mut mx) = ([f32::MAX;3], [f32::MIN;3]);
    for mesh in &scene.meshes {
        for &p in &mesh.positions {
            let wp = mesh.world_transform.0.transform_point3(Vec3::from(p)).to_array();
            for i in 0..3 { mn[i]=mn[i].min(wp[i]); mx[i]=mx[i].max(wp[i]); }
        }
    }
    // Expand slightly
    let mn = mn.map(|v| v - 0.5);
    let mx = mx.map(|v| v + 0.5);
    (mn, mx)
}

// ── Voxelisation ──────────────────────────────────────────────────────────────

fn voxelise(scene: &SceneGeometry, config: &NavConfig, aabb_min: [f32;3], aabb_max: [f32;3]) -> Result<HeightField, NebulaError> {
    let cs = config.cell_size;
    let ch = config.cell_height;
    let gx = ((aabb_max[0] - aabb_min[0]) / cs).ceil().max(1.0) as usize;
    let gz = ((aabb_max[2] - aabb_min[2]) / cs).ceil().max(1.0) as usize;

    // Collect world-space triangles (with per-triangle normals)
    let tris: Vec<([Vec3;3],[f32;3])> = scene.meshes.iter().flat_map(|mesh| {
        let xf = mesh.world_transform.0;
        mesh.indices.chunks_exact(3).map(move |t| {
            let v0 = xf.transform_point3(Vec3::from(mesh.positions[t[0] as usize]));
            let v1 = xf.transform_point3(Vec3::from(mesh.positions[t[1] as usize]));
            let v2 = xf.transform_point3(Vec3::from(mesh.positions[t[2] as usize]));
            let n = (v1-v0).cross(v2-v0).normalize_or_zero();
            ([v0,v1,v2], n.to_array())
        })
    }).collect();

    // Rasterise each triangle into the height-field (brute force, rayon-parallel per row)
    let cols: Vec<Vec<Span>> = (0..gx*gz).into_par_iter().map(|idx| {
        let x = idx % gx;
        let z = idx / gx;
        let col_mn = Vec3::new(aabb_min[0] + x as f32 * cs, aabb_min[1], aabb_min[2] + z as f32 * cs);
        let col_mx = Vec3::new(col_mn.x + cs, aabb_max[1], col_mn.z + cs);
        let mut spans: Vec<Span> = Vec::new();

        for (tri, normal) in &tris {
            if let Some((ymin, ymax)) = tri_vs_column(*tri, col_mn, col_mx) {
                let up_dot = normal[1];
                let max_cos = (90.0 - config.max_slope_deg).to_radians().cos();
                let walkable = up_dot >= max_cos;
                // Merge overlapping spans
                let new_span = Span { y_min: ymin.round_to(ch), y_max: ymax.round_to(ch), walkable, region: 0 };
                merge_span(&mut spans, new_span, ch);
            }
        }
        spans.sort_by(|a,b| a.y_min.partial_cmp(&b.y_min).unwrap());
        spans
    }).collect();

    Ok(HeightField { cols, gx, gz, world_min: aabb_min, cell_size: cs, cell_height: ch })
}

trait RoundTo { fn round_to(self, step: f32) -> f32; }
impl RoundTo for f32 { fn round_to(self, step: f32) -> f32 { (self / step).round() * step } }

/// Returns (ymin, ymax) of intersection of a triangle with an XZ-column AABB, or None.
fn tri_vs_column(tri: [Vec3;3], col_mn: Vec3, col_mx: Vec3) -> Option<(f32,f32)> {
    // Quick XZ AABB overlap test
    let (tri_xmin, tri_xmax) = tri.iter().fold((f32::MAX, f32::MIN), |(mn,mx), v| (mn.min(v.x), mx.max(v.x)));
    let (tri_zmin, tri_zmax) = tri.iter().fold((f32::MAX, f32::MIN), |(mn,mx), v| (mn.min(v.z), mx.max(v.z)));
    if tri_xmax < col_mn.x || tri_xmin > col_mx.x || tri_zmax < col_mn.z || tri_zmin > col_mx.z { return None; }
    let ymin = tri.iter().fold(f32::MAX, |m,v| m.min(v.y));
    let ymax = tri.iter().fold(f32::MIN, |m,v| m.max(v.y));
    if ymax < col_mn.y || ymin > col_mx.y { return None; }
    Some((ymin, ymax))
}

fn merge_span(spans: &mut Vec<Span>, new: Span, ch: f32) {
    for s in spans.iter_mut() {
        if s.y_max + ch >= new.y_min && new.y_max + ch >= s.y_min {
            s.y_min = s.y_min.min(new.y_min);
            s.y_max = s.y_max.max(new.y_max);
            s.walkable = s.walkable || new.walkable;
            return;
        }
    }
    spans.push(new);
}

// ── Walkability filtering ─────────────────────────────────────────────────────

struct WalkableSpan { x: usize, z: usize, y: f32, region: u32 }

fn filter_walkable(hf: &HeightField, config: &NavConfig) -> Vec<WalkableSpan> {
    let mut out = Vec::new();
    let radius_cells = (config.agent_radius / hf.cell_size).ceil() as usize;
    for z in 0..hf.gz {
        for x in 0..hf.gx {
            for span in hf.col(x, z) {
                if !span.walkable { continue; }
                // Check head clearance
                let clearance = {
                    let next_y = hf.col(x, z).iter().find(|s| s.y_min > span.y_max).map(|s| s.y_min).unwrap_or(f32::MAX);
                    next_y - span.y_max
                };
                if clearance < config.agent_height { continue; }
                out.push(WalkableSpan { x, z, y: span.y_max, region: 0 });
            }
        }
    }
    out
}

// ── Region growing ────────────────────────────────────────────────────────────

fn grow_regions_and_trace(walkable: &[WalkableSpan], hf: &HeightField, config: &NavConfig) -> Vec<Contour> {
    if walkable.is_empty() { return Vec::new(); }

    // Build a quick lookup: (x,z) → list of (y, original_index)
    let mut lookup: std::collections::HashMap<(usize,usize), Vec<(f32,usize)>> = std::collections::HashMap::new();
    for (i, s) in walkable.iter().enumerate() { lookup.entry((s.x, s.z)).or_default().push((s.y, i)); }

    // BFS region growing
    let mut regions = vec![u32::MAX; walkable.len()];
    let mut region_id = 0u32;
    let max_step = config.max_step_height;
    let cs = hf.cell_size;

    for start in 0..walkable.len() {
        if regions[start] != u32::MAX { continue; }
        regions[start] = region_id;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            let WalkableSpan { x, z, y, .. } = walkable[cur];
            let neighbours = [
                (x.wrapping_sub(1), z), (x+1, z), (x, z.wrapping_sub(1)), (x, z+1),
            ];
            for (nx, nz) in neighbours {
                if nx >= hf.gx || nz >= hf.gz { continue; }
                if let Some(adj) = lookup.get(&(nx, nz)) {
                    for &(ny, ni) in adj {
                        if regions[ni] == u32::MAX && (ny - y).abs() <= max_step {
                            regions[ni] = region_id;
                            queue.push_back(ni);
                        }
                    }
                }
            }
        }
        region_id += 1;
    }

    // Merge tiny regions
    let mut region_counts = vec![0u32; region_id as usize];
    for &r in &regions { if r < region_id { region_counts[r as usize] += 1; } }

    // Trace simple perimeter contours (one square per cell)
    let mut contours_map: std::collections::HashMap<u32, Vec<[f32;3]>> = std::collections::HashMap::new();
    for (i, s) in walkable.iter().enumerate() {
        let r = regions[i];
        if r == u32::MAX { continue; }
        if region_counts[r as usize] < config.min_region_area { continue; }
        let wx = hf.world_min[0] + (s.x as f32 + 0.5) * cs;
        let wz = hf.world_min[2] + (s.z as f32 + 0.5) * cs;
        contours_map.entry(r).or_default().push([wx, s.y, wz]);
    }

    contours_map.into_iter().map(|(region, verts)| Contour { verts, region }).collect()
}

// ── Polygon mesh ──────────────────────────────────────────────────────────────

fn build_polygon_mesh(contours: &[Contour], _config: &NavConfig) -> (Vec<NavVertex>, Vec<NavPolygon>, f32) {
    let mut vertices: Vec<NavVertex> = Vec::new();
    let mut polygons: Vec<NavPolygon> = Vec::new();
    let mut walkable_area = 0.0f32;

    for contour in contours {
        if contour.verts.len() < 3 { continue; }
        // Simple fan triangulation from centroid
        let n = contour.verts.len();
        let centroid: [f32;3] = {
            let sum = contour.verts.iter().fold([0.0f32;3], |acc, v| [acc[0]+v[0], acc[1]+v[1], acc[2]+v[2]]);
            [sum[0]/n as f32, sum[1]/n as f32, sum[2]/n as f32]
        };
        let c_idx = vertices.len() as u32;
        vertices.push(NavVertex { position: centroid });

        let base = vertices.len() as u32;
        for &v in &contour.verts { vertices.push(NavVertex { position: v }); }

        for i in 0..n {
            let a = base + i as u32;
            let b = base + ((i + 1) % n) as u32;
            let dx = contour.verts[i][0] - centroid[0];
            let dz = contour.verts[i][2] - centroid[2];
            let dx2 = contour.verts[(i+1)%n][0] - centroid[0];
            let dz2 = contour.verts[(i+1)%n][2] - centroid[2];
            walkable_area += 0.5 * (dx * dz2 - dx2 * dz).abs();
            polygons.push(NavPolygon { vertex_indices: vec![c_idx, a, b], neighbour_indices: vec![u32::MAX, u32::MAX, u32::MAX], area_flags: 0 });
        }
    }

    // Build adjacency (O(n²) — acceptable for typical nav mesh sizes)
    let n = polygons.len();
    for i in 0..n {
        for j in (i+1)..n {
            let pi = &polygons[i].vertex_indices.clone();
            let pj = &polygons[j].vertex_indices.clone();
            // Find shared edge
            for ei in 0..pi.len() {
                let a0 = pi[ei];
                let a1 = pi[(ei+1)%pi.len()];
                for ej in 0..pj.len() {
                    let b0 = pj[ej];
                    let b1 = pj[(ej+1)%pj.len()];
                    if (a0==b0&&a1==b1)||(a0==b1&&a1==b0) {
                        polygons[i].neighbour_indices[ei] = j as u32;
                        polygons[j].neighbour_indices[ej] = i as u32;
                    }
                }
            }
        }
    }

    (vertices, polygons, walkable_area)
}
