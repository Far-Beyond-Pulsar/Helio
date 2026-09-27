// Source-faithful occupancy trial, not a filtered far representation.
// Camera arithmetic remains f32; compare actual rays with the f64 CPU oracle.
// Edits are immutable for the published cut. Linear edit scans are a measured
// prototype limitation, not the intended scalable edit acceleration structure.
struct CanonicalSettings { clearance:vec4<f32>, counts:vec4<u32> }
@group(0) @binding(30) var<uniform> canonical_settings:CanonicalSettings;

fn stored_far_hit(n:StoredNode,ro:vec3<f32>,rd:vec3<f32>,start:f32,end:f32)->Hit {
    let base_step=i32(max(p.settings.w,1.0));
    let entry=canonical_position(p.origin.xyz,p.fraction.xyz,ro,rd,start);
    let high=n.low+vec3<i32>(i32(32u<<n.level));
    let requested_anchor=entry.cell;
    let anchor=clamp(requested_anchor,n.low,high-1);
    let fraction=entry.fraction+vec3<f32>(requested_anchor-anchor);
    let step=select(vec3<i32>(-1),vec3<i32>(1),rd>=vec3<f32>(0.0));
    let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
    let guard=1.0+f32(base_step)*0.1;
    var cell=clamp(anchor+vec3<i32>(floor(fraction+rd*0.00002)),n.low,high-1);
    var t=0.0;
    for(var iteration=0u;iteration<32768u;iteration++) {
        if STORED_TRACE_WORK {stored_work.z+=1u;}
        let sampled=voxel_sample(cell,base_step);
        var latest=canonical_settings.counts.x;
        var found=false;
        var material=0u;
        while latest>0u {
            latest-=1u;
            if edit_contains(sampled,edits[latest]) {
                material=edits[latest].material;found=true;break;
            }
        }
        var safe=0.0;
        if found {
            let edit=edits[latest];
            let distance=length(vec3<f32>(sampled-edit.cell))*0.1;
            safe=f32(edit.radius_units)*0.05*0.999999-distance*1.000001-guard;
        } else {
            let sample=base_sample(sampled,vec3<f32>(0.0));
            material=u32(sample.solid);
            let bounds=canonical_settings.clearance;
            safe=clamp((-sample.density*0.99999-bounds.y)/bounds.x*0.99999,0.0,bounds.z)-guard;
        }
        if material!=0u {
            let low=(vec3<f32>(voxel_low(cell,base_step)-anchor)-fraction)*0.1;
            let box=stored_box(low,f32(base_step)*0.1,rd);
            return Hit(sampled,0x80000001u|(n.level<<2u)|(material<<8u)|(stored_face(box.normal)<<28u),rd,start+max(0.0,box.near));
        }
        let first=select(0u,latest+1u,found);
        for(var i=first;i<canonical_settings.counts.x;i++) {
            let edit=edits[i];
            if edit.material!=0u {
                let distance=length(vec3<f32>(sampled-edit.cell))*0.1;
                safe=min(safe,distance*0.999999-f32(edit.radius_units)*0.05*1.000001-guard);
            }
        }
        // Cover rounding of a local far-brick position before making a
        // clearance jump. An exhausted ray stays explicit, never becomes air.
        safe-=abs(t)*0.000001;
        let next_t=t+min(safe,max(0.0,end-start-t));
        let next_cell=anchor+vec3<i32>(floor(fraction+rd*next_t*10.0));
        if safe>0.1 && next_t>t && any(next_cell!=cell) && all((next_cell-cell)*step>=vec3<i32>(0)) {
            t=next_t;cell=next_cell;
        } else {
            let low=voxel_low(cell,base_step);
            let boundary=low+select(vec3<i32>(0),vec3<i32>(base_step),rd>=vec3<f32>(0.0));
            let next=(vec3<f32>(boundary-anchor)-fraction)*0.1*inverse;
            var axis=0u;if next.y<next.x {axis=1u;}if next.z<next[axis] {axis=2u;}
            t=max(t,next[axis]);
            cell[axis]=boundary[axis]-select(1,0,step[axis]>0);
        }
        if t>=end-start || any(cell<n.low) || any(cell>=high) {
            return Hit(vec3<i32>(0),0u,rd,end);
        }
    }
    return Hit(cell,2u|(n.level<<2u),rd,start+t);
}
