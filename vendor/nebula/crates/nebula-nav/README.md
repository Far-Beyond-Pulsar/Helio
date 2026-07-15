# nebula-nav

**Navigation mesh construction and pathfinding pre-computation** — CPU + rayon voxelisation pipeline.

---

## What is a navigation mesh?

A **navigation mesh (navmesh)** is a simplified polygon mesh that represents the walkable surfaces of a scene. Instead of performing pathfinding over raw triangle geometry (millions of triangles) the engine runs A* or flow-field pathfinding over the navmesh (typically a few hundred to a few thousand convex polygons). The navmesh abstracts away the geometric complexity of the level while preserving its connectivity.

Every game engine with characters that navigate a level has a navmesh. It is one of the most fundamental baked assets in game development.

---

## How the bake works in practice

The pipeline follows the same voxelisation + region-growing approach as Recast Navigation (used by Unreal Engine's NavMesh system, Unity's NavMesh, and many others), implemented here in pure Rust with `rayon` parallelism.

### Step 1 — Voxelisation

The scene geometry is rasterised into a 3-D **height-field** — a 2-D grid of columns, each column containing a sorted list of vertical solid spans. The grid resolution is controlled by `cell_size` (XZ plane) and `cell_height` (Y axis).

Each mesh triangle is tested against every XZ column it overlaps (AABB rejection first, then exact intersection). Overlapping spans in the same column are merged to keep the height-field compact.

This step runs **in parallel across all columns** using `rayon::into_par_iter` — a 300 × 300 column grid with complex geometry bakes in milliseconds.

### Step 2 — Walkability filtering

Each span is marked walkable based on three criteria:
1. **Surface normal** — the triangle's upward component must exceed `cos(90° − max_slope_deg)`. Vertical walls and steep ramps are not walkable.
2. **Head clearance** — there must be at least `agent_height` of vertical space above the span top before the next solid span or the top of the world.
3. **Lateral erosion** — walkable spans within `agent_radius` of a wall edge are discarded, ensuring the agent capsule never clips through geometry.

### Step 3 — Region growing

All walkable spans form a connectivity graph. A BFS flood-fill labels each connected component of walkable spans with a unique **region ID**. Vertical steps ≤ `max_step_height` are treated as connected (the agent can step up or down). Regions smaller than `min_region_area` voxels are pruned or merged into neighbours.

This step is conceptually similar to the "fill + select" tool in image editing: contiguous walkable areas become distinct regions.

### Step 4 — Contour tracing

The boundary of each region is traced to produce a simplified polygon outline. Raw voxel boundaries are staircase-shaped (axis-aligned); the tracer simplifies edges that deviate less than `max_edge_error` world units from the underlying geometry, producing smooth diagonal edges that better follow curved walls and slopes.

### Step 5 — Polygon mesh

Simplified contours are triangulated (fan from centroid) and collected into a final polygon mesh. Adjacency between polygons that share an edge is computed and stored in `NavPolygon::neighbour_indices`, enabling the runtime pathfinder to walk between polygons in O(1) per step.

The output stores walkable area in square world units, which is useful for progress reporting and level validation ("the agent can walk over 2 340 m² of this level").

---

## Configuration

```rust
let config = NavConfig {
    agent_radius:      0.4,  // Capsule radius (world units)
    agent_height:      1.8,  // Capsule height (world units)
    max_step_height:   0.4,  // Max climbable step
    max_slope_deg:     45.0, // Max walkable slope angle
    cell_size:         0.3,  // XZ voxel resolution
    cell_height:       0.2,  // Y voxel resolution
    min_region_area:   8,    // Min walkable island (voxels)
    max_edge_length:   12.0, // Contour simplification
    max_edge_error:    1.3,
    ..Default::default()
};
```

**Presets:**

| Preset | Cell size | Cell height | Use case |
|---|---|---|---|
| `fast()` | 1.0 m | 0.5 m | Quick editor preview |
| `default()` | 0.3 m | 0.2 m | Standard production |
| `ultra()` | 0.15 m | 0.1 m | High-detail narrow spaces |

---

## How baking improves performance

A real-time navmesh build (Recast's dynamic navmesh) on a complex level with 500 000 triangles can take 50–500 ms on CPU — far too long to perform during gameplay. The result would also change every frame if geometry were dynamic, invalidating cached paths.

Baking the navmesh at content-creation time means:
- **Zero cost at runtime** to build the mesh — it is loaded directly from the `.nebula` file.
- **A* pathfinding runs on hundreds of polygons**, not millions of triangles — orders of magnitude faster.
- **Path queries typically complete in < 1 ms** for levels with thousands of nav polygons.

The only limitation is that baked navmeshes cannot react to dynamic geometry changes (e.g., a door that opens to create a new passage). For those cases the runtime would need to re-bake or patch the affected region — but the static majority of the level remains baked and cheap.

---

## Output

```rust
pub struct NavOutput {
    pub vertices:      Vec<NavVertex>,   // World-space positions
    pub polygons:      Vec<NavPolygon>,  // Triangles with neighbour links
    pub aabb_min:      [f32; 3],
    pub aabb_max:      [f32; 3],
    pub walkable_area: f32,              // Total walkable area (m²)
    pub config_json:   String,
}

pub struct NavPolygon {
    pub vertex_indices:   Vec<u32>,  // Indices into NavOutput::vertices
    pub neighbour_indices: Vec<u32>, // Adjacent polygon index, or u32::MAX for boundary edges
    pub area_flags:       u32,       // User-defined area type (water, road, …)
}
```

---

## License

MIT OR Apache-2.0
