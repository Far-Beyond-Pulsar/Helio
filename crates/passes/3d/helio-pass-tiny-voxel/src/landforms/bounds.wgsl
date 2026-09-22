// Append after sampling.wgsl. Upload the hierarchy belonging to exactly the
// same BoundedLandforms snapshot as lf_heights and lf_settings.resolution.
@group(0) @binding(4) var<storage, read> lf_bounds_tree: array<vec2<i32>>;
struct LFRegionBounds { height: vec2<i32>, classification: u32, valid: u32 }

fn lf_patch_sample(face:u32, atlas_cell:vec2<u32>, t:vec2<u32>) -> i32 {
    let side = lf_settings.resolution + 1u;
    let i = face*side*side + atlas_cell.y*side + atlas_cell.x;
    return lf_lerp(lf_lerp(lf_heights[i],lf_heights[i+1u],t.x),
        lf_lerp(lf_heights[i+side],lf_heights[i+side+1u],t.x),t.y);
}

fn lf_rectangle(face:u32, low:vec2<u32>, high:vec2<u32>) -> vec2<i32> {
    let n = lf_settings.resolution;
    let first = min(low >> vec2<u32>(16u),vec2<u32>(n-1u));
    let last = min(high >> vec2<u32>(16u),vec2<u32>(n-1u));
    var result = vec2<i32>(40000000,-40000000);
    if all(last-first <= vec2<u32>(1u)) {
        for(var v=first.y;v<=last.y;v++) {for(var u=first.x;u<=last.x;u++) {
            let origin = vec2<u32>(u,v)*65536u;
            let a = min(max(low,origin)-origin,vec2<u32>(65536u));
            let b = min(max(high,origin)-origin,vec2<u32>(65536u));
            for(var corner=0u;corner<4u;corner++) {
                let t = vec2<u32>(select(a.x,b.x,(corner&1u)!=0u),select(a.y,b.y,(corner&2u)!=0u));
                let value = lf_patch_sample(face,vec2<u32>(u,v),t);
                result = vec2<i32>(min(result.x,value-2),max(result.y,value+2));
            }
        }}
        result = clamp(result,vec2<i32>(-40000000),vec2<i32>(40000000));
    } else {
        let span = max(last.x-first.x+1u,last.y-first.y+1u);
        var level = 0u;
        var tile = 1u;
        while tile<span {tile*=2u;level++;}
        let width = n>>level;
        let stride = (4u*n*n-1u)/3u;
        let offset = face*stride+(4u*n*n-4u*width*width)/3u;
        for(var v=first.y>>level;v<=last.y>>level;v++) {
            for(var u=first.x>>level;u<=last.x>>level;u++) {
                let range = lf_bounds_tree[offset+v*width+u];
                result = vec2<i32>(min(result.x,range.x),max(result.y,range.y));
            }
        }
    }
    return result;
}

fn lf_projection_interval(low:i32, high:i32, d0:u32, d1:u32) -> vec2<u32> {
    var result = vec2<u32>(0xffffffffu,0u);
    for(var corner=0u;corner<4u;corner++) {
        let d = select(d0,d1,(corner&1u)!=0u);
        let c = clamp(select(low,high,(corner&2u)!=0u),-i32(d),i32(d));
        let projected = lf_project(c,d,lf_settings.resolution);
        let q = projected.x*65536u+projected.y;
        result = vec2<u32>(min(result.x,q),max(result.y,q));
    }
    return result;
}

// Inclusive cell AABB, valid only within the canonical domain. Classification:
// 0 mixed, 1 all air, 2 all solid. Invalid regions have four zero words.
fn lf_classify_region(minimum:vec3<i32>, maximum:vec3<i32>) -> LFRegionBounds {
    let n = lf_settings.resolution;
    if any(minimum>maximum) || any(minimum<vec3<i32>(-100000000))
        || any(maximum>vec3<i32>(100000000)) || n==0u || n>256u || (n&(n-1u))!=0u {
        return LFRegionBounds(vec2<i32>(0),0u,0u);
    }
    let low = minimum*2+vec3<i32>(1);
    let high = maximum*2+vec3<i32>(1);
    let closest = select(min(abs(low),abs(high)),vec3<i32>(1),
        (low<vec3<i32>(0))&(high>vec3<i32>(0)));
    let farthest = max(abs(low),abs(high));
    var heights = vec2<i32>(40000000,-40000000);
    for(var face=0u;face<6u;face++) {
        let axis = face/2u;
        var other = vec2<u32>(1u,2u);
        if axis==1u {other=vec2<u32>(0u,2u);}
        if axis==2u {other=vec2<u32>(0u,1u);}
        var d0 = max(low[axis],1);
        var d1 = high[axis];
        if (face&1u)!=0u {d0=max(-high[axis],1);d1=-low[axis];}
        d0 = max(d0,max(closest[other.x],closest[other.y]));
        if d1<d0 {continue;}
        let u = lf_projection_interval(low[other.x],high[other.x],u32(d0),u32(d1));
        let v = lf_projection_interval(low[other.y],high[other.y],u32(d0),u32(d1));
        let range = lf_rectangle(face,vec2<u32>(u.x,v.x),vec2<u32>(u.y,v.y));
        heights = vec2<i32>(min(heights.x,range.x),max(heights.y,range.y));
    }
    let near = vec3<u32>(closest);
    let far = vec3<u32>(farthest);
    let near_squared = lf_add(lf_add(lf_mul(near.x,near.x),lf_mul(near.y,near.y)),lf_mul(near.z,near.z));
    let far_squared = lf_add(lf_add(lf_mul(far.x,far.x),lf_mul(far.y,far.y)),lf_mul(far.z,far.z));
    let inner = u32(127420000+heights.x);
    let outer = u32(127420000+heights.y);
    var classification = 0u;
    if lf_greater(near_squared,lf_mul(outer,outer)) {classification=1u;}
    else if !lf_greater(far_squared,lf_mul(inner,inner)) {classification=2u;}
    return LFRegionBounds(heights,classification,1u);
}
