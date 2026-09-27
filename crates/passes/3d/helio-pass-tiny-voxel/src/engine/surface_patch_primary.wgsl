// Only complete a ray after its entire prefix was read from current pages.
// Missing pages, an outside camera, or leaving the patch preserve the prepared
// ray for the canonical renderer. Exhaustion remains an explicit diagnostic.
@compute @workgroup_size(8,8)
fn surface_patch_primary(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=vec2<u32>(p.screen.xy)) || p.settings.z<=0.0 {return;}
    let step=i32(patch_settings.step);
    if step<=0 {return;}
    let index=id.x+id.y*u32(p.screen.x);
    let rd=primary_hits[index].normal;
    if all(rd==vec3<f32>(0.0)) {return;}
    let bits=bitcast<vec3<u32>>(rd);
    let negative=((bits&vec3<u32>(0x80000000u))!=vec3<u32>(0u))
        & ((bits&vec3<u32>(0x7fffffffu))!=vec3<u32>(0u));
    let direction=select(vec3<i32>(1),vec3<i32>(-1),negative);
    let absolute=abs(rd);
    var cell=voxel_low(p.origin.xyz,step);
    // A ray starting exactly on a plane owns the interval in its direction.
    cell-=select(vec3<i32>(0),vec3<i32>(step),
        negative & (cell==p.origin.xyz) & (p.fraction.xyz==vec3<f32>(0.0)));
    var face=0u;
    var distance=0.0;
    for(var visited=0u;visited<800u;visited++) {
        let page=stored_cached_page(cell);
        if page.base==0xffffffffu {return;}
        let material=stored_cached_material(page,cell);
        if material!=0u {
            // An inside-solid start has no entry face; keep existing handling.
            if face==0u {return;}
            let sampled=voxel_sample(cell,step);
            primary_hits[index]=Hit(sampled,0x88000001u|(material<<8u)|(face<<28u),rd,distance);
            return;
        }
        let boundary=cell+select(vec3<i32>(0),vec3<i32>(step),direction>vec3<i32>(0));
        let times=array<vec2<f32>,3>(
            patch_delta(boundary.x,p.origin.x,p.fraction.x,direction.x),
            patch_delta(boundary.y,p.origin.y,p.fraction.y,direction.y),
            patch_delta(boundary.z,p.origin.z,p.fraction.z,direction.z));
        var axis=0u;
        for(var a=1u;a<3u;a++) {
            if patch_compare(times[a],absolute[a],times[axis],absolute[axis])<0 {axis=a;}
        }
        distance=patch_distance(times[axis],absolute[axis]);
        if distance>p.settings.x {return;}
        face=1u+axis*2u+u32(direction[axis]>0);
        // Cross all genuinely coincident planes together; a zero-width edge
        // contact is not an occupied interval.
        for(var a=0u;a<3u;a++) {
            if patch_compare(times[a],absolute[a],times[axis],absolute[axis])==0 {
                cell[a]+=direction[a]*step;
            }
        }
    }
    primary_hits[index]=Hit(vec3<i32>(0),0x08000002u,rd,0.0);
}
