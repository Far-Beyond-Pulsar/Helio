// The flat terrain program; mirror of FlatField in src/landform.rs.
struct TerrainConstants {
    a: vec4<i32>, // height (mm), surface, soil, soil depth (cells)
    b: vec4<i32>, // rock, pad
}

fn terrain_height(p: vec3<i32>, level: u32) -> i32 {
    return terrain.a.x;
}

fn ground_material(p: vec3<i32>, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    if depth == 0 { return u32(terrain.a.y); }
    if depth <= terrain.a.w { return u32(terrain.a.z); }
    return u32(terrain.b.x);
}
