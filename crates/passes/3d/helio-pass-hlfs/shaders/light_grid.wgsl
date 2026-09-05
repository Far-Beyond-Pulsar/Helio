// Two-level conservative culling. Overflow is explicit, never a truncated light set.
@group(2) @binding(0) var<storage, read_write> coarse_grid: array<CoarseTile>;
@group(2) @binding(1) var<storage, read_write> fine_grid: array<LightTile>;
@group(2) @binding(2) var depth_bounds: texture_storage_2d<r32float,write>;
var<workgroup> accepted: atomic<u32>;
var<workgroup> min_depth: atomic<u32>;
var<workgroup> max_depth: atomic<u32>;
var<workgroup> packed: array<u32, 256>;

fn sphere_in_tile(light: GpuLight, lo: vec2<u32>, hi: vec2<u32>, zlo: f32, zhi: f32) -> bool {
    if light.light_type == 0u { return true; }
    if light.position_range.w <= 0.0 { return false; }
    let m = cameras[0].view_proj;
    let r0 = vec4<f32>(m[0][0],m[1][0],m[2][0],m[3][0]);
    let r1 = vec4<f32>(m[0][1],m[1][1],m[2][1],m[3][1]);
    let r2 = vec4<f32>(m[0][2],m[1][2],m[2][2],m[3][2]);
    let r3 = vec4<f32>(m[0][3],m[1][3],m[2][3],m[3][3]);
    let lower = vec2<f32>(lo) / vec2<f32>(globals.screen_size);
    let upper = vec2<f32>(min(hi, globals.screen_size)) / vec2<f32>(globals.screen_size);
    let planes = array<vec4<f32>, 6>(
        r0 - (2.0*lower.x-1.0)*r3, (2.0*upper.x-1.0)*r3-r0,
        (1.0-2.0*lower.y)*r3-r1, r1-(1.0-2.0*upper.y)*r3,
        r2-zlo*r3, zhi*r3-r2);
    for (var i=0u; i<6u; i++) {
        let plane=planes[i];
        if dot(plane,vec4<f32>(light.position_range.xyz,1.0)) < -light.position_range.w*length(plane.xyz) - 1e-4 {
            return false;
        }
    }
    return true;
}
@compute @workgroup_size(64)
fn coarse(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane == 0u { atomicStore(&accepted,0u); }
    workgroupBarrier();
    let lo=group.xy*COARSE_TILE_SIZE;
    if globals.light_count>65535u {
        if lane==0u { coarse_grid[group.y*div_ceil(globals.screen_size,COARSE_TILE_SIZE).x+group.x].count=INVALID_LIGHT; }
        return;
    }
    for (var i=lane; i<globals.light_count; i+=64u) {
        if sphere_in_tile(lights[i],lo,lo+COARSE_TILE_SIZE,0.0,1.0) {
            let slot=atomicAdd(&accepted,1u);
            if slot<COARSE_CAPACITY { packed[slot]=i; }
        }
    }
    workgroupBarrier();
    let index=group.y*div_ceil(globals.screen_size,COARSE_TILE_SIZE).x+group.x;
    let count=atomicLoad(&accepted);
    if lane==0u { coarse_grid[index].count=count; }
    for(var i=lane;i<(min(count,COARSE_CAPACITY)+1u)/2u;i+=64u) {
        coarse_grid[index].indices[i]=packed[2u*i]|(select(65535u,packed[2u*i+1u],2u*i+1u<count)<<16u);
    }
}
@compute @workgroup_size(8,8)
fn fine(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane==0u { atomicStore(&accepted,0u); atomicStore(&min_depth,bitcast<u32>(1.0)); atomicStore(&max_depth,0u); }
    workgroupBarrier();
    let lo=group.xy*TILE_SIZE;
    let p=lo+local.xy;
    if all(p<globals.screen_size) {
        let d=textureLoad(gbuf_depth,vec2<i32>(p),0);
        if d<1.0 { atomicMin(&min_depth,bitcast<u32>(d)); atomicMax(&max_depth,bitcast<u32>(d)); }
    }
    workgroupBarrier();
    let zlo=bitcast<f32>(atomicLoad(&min_depth));
    let zhi=bitcast<f32>(atomicLoad(&max_depth));
    if lane==0u { textureStore(depth_bounds,vec2<i32>(group.xy),vec4<f32>(zlo)); }
    let ci=(lo.y/COARSE_TILE_SIZE)*div_ceil(globals.screen_size,COARSE_TILE_SIZE).x+lo.x/COARSE_TILE_SIZE;
    let coarse_count=coarse_grid[ci].count;
    let ti=group.y*div_ceil(globals.screen_size,TILE_SIZE).x+group.x;
    if coarse_count>COARSE_CAPACITY {
        // The sampler switches to the complete global set. All lights retain support.
        if lane==0u { fine_grid[ti].count=INVALID_LIGHT; }
        return;
    }
    if zlo<=zhi {
        for(var i=lane;i<coarse_count;i+=64u) {
            let id=(coarse_grid[ci].indices[i/2u]>>(16u*(i&1u)))&65535u;
            if sphere_in_tile(lights[id],lo,lo+TILE_SIZE,zlo,zhi) {
                let slot=atomicAdd(&accepted,1u);
                if slot<GRID_CAPACITY { packed[slot]=id; }
            }
        }
    }
    workgroupBarrier();
    let count=atomicLoad(&accepted);
    if lane==0u { fine_grid[ti].count=count; }
    if lane<(min(count,GRID_CAPACITY)+1u)/2u {
        fine_grid[ti].indices[lane]=packed[2u*lane]|(select(65535u,packed[2u*lane+1u],2u*lane+1u<count)<<16u);
    }
}
