// Only complete a ray after its entire prefix was read from current pages.
// Missing pages, an outside camera, or leaving the patch preserve the prepared
// ray for the canonical renderer. Exhaustion remains an explicit diagnostic.
@compute @workgroup_size(8,8)
fn surface_patch_primary(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=vec2<u32>(p.screen.xy)) || p.settings.z<=0.0 {return;}
    let index=id.x+id.y*u32(p.screen.x);
    if (primary_hits[index].status&0x08000000u)!=0u {return;}
    let step=i32(patch_settings.step);
    if step<=0 {return;}
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
        negative & (cell==p.origin.xyz)
        & ((bitcast<vec3<u32>>(p.fraction.xyz)&vec3<u32>(0x7fffffffu))==vec3<u32>(0u)));
    var face=0u;
    var distance=0.0;
    for(var visited=0u;visited<800u;visited++) {
        let page=stored_cached_page(cell);
        if page.base==0xffffffffu {return;}
        var q=vec3<u32>(cell-page.low);
        if step!=1 {q/=u32(step);}
        let descriptor=surface_descriptor(page.base,q);
        var material=descriptor;
        if descriptor>=4u {material=surface_material(page.base,q);}
        if material!=0u {
            // An inside-solid start has no entry face; keep existing handling.
            if face==0u {return;}
            let sampled=voxel_sample(cell,step);
            primary_hits[index]=Hit(sampled,0x88000001u|(material<<8u)|(face<<28u),rd,distance);
            return;
        }
        var low=cell;var span=step;
        if patch_settings.disable_skip==0u && descriptor==0u {
            low=page.low+vec3<i32>(q&vec3<u32>(0xfffffffcu))*step;span=4*step;
            if surface_words[page.base]==0u {low=page.low;span=32*step;}
        }
        let boundary=low+select(vec3<i32>(0),vec3<i32>(span),direction>vec3<i32>(0));
        let times=array<vec2<f32>,3>(
            patch_delta(boundary.x,p.origin.x,p.fraction.x,direction.x),
            patch_delta(boundary.y,p.origin.y,p.fraction.y,direction.y),
            patch_delta(boundary.z,p.origin.z,p.fraction.z,direction.z));
        var axis=0u;
        for(var a=1u;a<3u;a++) {
            if patch_compare(times[a],absolute[a],times[axis],absolute[axis])<0 {axis=a;}
        }
        distance=patch_distance(times[axis],absolute[axis]);
        var face_axis=axis;
        // Cross all genuinely coincident planes together; a zero-width edge
        // contact is not an occupied interval.
        if span==step {
            for(var a=0u;a<3u;a++) {
                if a==axis || patch_compare(times[a],absolute[a],times[axis],absolute[axis])==0 {
                    cell[a]+=direction[a]*step;
                }
            }
        } else {
            // A certified empty box may skip many grid planes. Recover the
            // other cell coordinates by monotone exact comparisons, not by
            // flooring a rounded ray position at the box's exit.
            for(var a=0u;a<3u;a++) {
                if a==axis {
                    cell[a]=boundary[a]-select(step,0,direction[a]>0);
                    continue;
                }
                if (bits[a]&0x7fffffffu)==0u {continue;}
                var crossed=0;
                var maximum=(boundary[a]-cell[a])*direction[a]/step+select(1,0,direction[a]>0);
                let first=cell[a]+select(0,step,direction[a]>0);
                while crossed<maximum {
                    let middle=(crossed+maximum+1)/2;
                    let plane=first+(middle-1)*direction[a]*step;
                    let numerator=patch_delta(plane,p.origin[a],p.fraction[a],direction[a]);
                    let comparison=patch_compare(numerator,absolute[a],times[axis],absolute[axis]);
                    if comparison<=0 {
                        crossed=middle;
                        if comparison==0 {face_axis=min(face_axis,a);}
                    } else {maximum=middle-1;}
                }
                cell[a]+=crossed*direction[a]*step;
            }
            // Re-evaluate using the same tie-winning voxel plane as plain DDA.
            let plane=cell[face_axis]+select(step,0,direction[face_axis]>0);
            distance=patch_distance(patch_delta(plane,p.origin[face_axis],
                p.fraction[face_axis],direction[face_axis]),absolute[face_axis]);
        }
        // Apply the range cutoff to the same tie-winning plane as plain DDA.
        if distance>p.settings.x {return;}
        face=1u+face_axis*2u+u32(direction[face_axis]>0);
    }
    primary_hits[index]=Hit(vec3<i32>(0),0x08000002u,rd,0.0);
}
