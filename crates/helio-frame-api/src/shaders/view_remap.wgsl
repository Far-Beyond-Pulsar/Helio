// The view-remap chain a pixel belongs to, or -1 (`helio_frame_api::lighting::
// VIEW_REMAPS`). A drawer of remapped content tags its pixels' lightmap UV
// (-1, -3 - chain); (-1, -2) is the stored voxel tag and (-2, -2) the VG
// sentinel.
fn view_remap_of(lightmap_uv: vec2<f32>) -> i32 {
    if lightmap_uv.x > -1.5 && lightmap_uv.x < -0.5 && lightmap_uv.y < -2.5 {
        return min(i32(round(-3.0 - lightmap_uv.y)), 255);
    }
    return -1;
}
