// Caller declares surface_words: array<u32> in read-only storage.
// Coordinates and ray distances use authored voxel units, local to this brick.
// The caller must preserve planetary precision when constructing these small
// local coordinates; subtracting large f32 world positions is not sufficient.
fn surface_descriptor(base:u32, q:vec3<u32>)->u32 {
    let root=surface_words[base];
    if root<4u {return root;}
    let micro=q/4u;
    return surface_words[base+1u+micro.x+micro.y*8u+micro.z*64u];
}
fn surface_material(base:u32,q:vec3<u32>)->u32 {
    if any(q>=vec3<u32>(32u)) {return 0xffffffffu;}
    let descriptor=surface_descriptor(base,q);
    if descriptor<4u {return descriptor;}
    let local=q&vec3<u32>(3u);
    let bit=local.x+local.y*4u+local.z*16u;
    let offset=base+513u+(descriptor-4u)*4u+bit/32u;
    return ((surface_words[offset]>>(bit&31u))&1u)
        |(((surface_words[offset+2u]>>(bit&31u))&1u)<<1u);
}
struct SurfaceHit {
    cell:vec3<i32>, material:u32,
    face:u32, iterations:u32, distance:f32, pad:u32,
}
fn surface_trace(base:u32,ro:vec3<f32>,rd:vec3<f32>,skip_uniform:bool)->SurfaceHit {
    var miss=SurfaceHit(vec3<i32>(0),0u,0u,0u,0.0,0u);
    if all(rd==vec3<f32>(0.0)) {return miss;}
    for(var a=0u;a<3u;a++) {
        if rd[a]==0.0 && (ro[a]<0.0 || ro[a]>=32.0) {return miss;}
    }
    let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
    let first=min(-ro*inverse,(vec3<f32>(32.0)-ro)*inverse);
    let last=max(-ro*inverse,(vec3<f32>(32.0)-ro)*inverse);
    var axis=0u;if first.y>first.x {axis=1u;}if first.z>first[axis] {axis=2u;}
    var t=max(0.0,first[axis]);
    let end=min(last.x,min(last.y,last.z));
    if end<=t {return miss;}
    let sign=select(vec3<i32>(-1),vec3<i32>(1),rd>=vec3<f32>(0.0));
    let point=fma(rd,vec3<f32>(t),ro);
    var cell=vec3<i32>(floor(point));
    cell-=vec3<i32>(select(vec3<u32>(0u),vec3<u32>(1u),(rd<vec3<f32>(0.0)) & (point==floor(point))));
    // The entry face owns its half-open boundary even if multiplication rounded.
    if first[axis]>=0.0 {cell[axis]=select(31,0,sign[axis]>0);}
    cell=clamp(cell,vec3<i32>(0),vec3<i32>(31));
    for(var iteration=0u;iteration<100u;iteration++) {
        miss.iterations=iteration+1u;
        let q=vec3<u32>(cell);
        let descriptor=surface_descriptor(base,q);
        let material=surface_material(base,q);
        if material!=0u {
            return SurfaceHit(cell,material,1u+axis*2u+u32(sign[axis]>0),iteration+1u,t,0u);
        }
        var side=1;
        if skip_uniform && descriptor==0u {side=select(4,32,surface_words[base]==0u);}
        let low=(cell/side)*side;
        let boundary=low+select(vec3<i32>(0),vec3<i32>(side),sign>vec3<i32>(0));
        let next=(vec3<f32>(boundary)-ro)*inverse;
        axis=0u;if next.y<next.x {axis=1u;}if next.z<next[axis] {axis=2u;}
        let crossing=next[axis];
        if crossing>=end {miss.distance=end;return miss;}
        t=max(t,crossing);
        if side==1 {
            // Cross coincident planes together; zero-area edge contacts do not
            // become opaque cells merely because one axis happened to win a tie.
            cell+=select(vec3<i32>(0),sign,next==vec3<f32>(crossing));
        } else {
            let p=fma(rd,vec3<f32>(t),ro);
            cell=vec3<i32>(floor(p));
            cell-=vec3<i32>(select(vec3<u32>(0u),vec3<u32>(1u),(rd<vec3<f32>(0.0)) & (p==floor(p))));
            for(var a=0u;a<3u;a++) {
                if next[a]==crossing {cell[a]=boundary[a]-select(0,1,sign[a]<0);}
            }
        }
        if any(cell<vec3<i32>(0)) || any(cell>=vec3<i32>(32)) {miss.distance=t;return miss;}
    }
    miss.material=0xfffffffeu; // Explicit exhaustion, never air.
    miss.distance=t;
    return miss;
}
