struct PatchSettings { low:vec3<i32>, side:u32, step:u32, pad0:u32,disable_skip:u32,pad2:u32 }
@group(0) @binding(31) var<uniform> patch_settings:PatchSettings;
@group(0) @binding(32) var<storage,read> patch_directory:array<u32>;
@group(0) @binding(33) var<storage,read> surface_words:array<u32>;

struct PatchPage { base:u32, low:vec3<i32> }
fn stored_cached_page(cell:vec3<i32>)->PatchPage {
    let step=i32(patch_settings.step);
    if step<=0 {return PatchPage(0xffffffffu,vec3<i32>(0));}
    let tile_low=voxel_low(cell,32*step);
    let key=tile_low/(32*step);
    let index=key-patch_settings.low;
    if any(index<vec3<i32>(0)) || any(index>=vec3<i32>(i32(patch_settings.side))) {return PatchPage(0xffffffffu,tile_low);}
    let side=i32(patch_settings.side);
    let base=patch_directory[u32(index.x+index.y*side+index.z*side*side)];
    return PatchPage(base,tile_low);
}
fn stored_cached_material(page:PatchPage,cell:vec3<i32>)->u32 {
    if page.base==0xffffffffu {return 0xffffffffu;}
    var q=vec3<u32>(cell-page.low);
    if patch_settings.step!=1u {q/=patch_settings.step;}
    return surface_material(page.base,q);
}
