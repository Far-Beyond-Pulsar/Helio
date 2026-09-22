// Append after sampling.wgsl and bounds.wgsl. Settings words are resolution,
// detail seed, edit count, volume revision (1). All input edits are validated.
struct VFEdit {
    cell:vec3<i32>, radius_units:u32,
    material:u32, pad0:u32, pad1:u32, pad2:u32,
}
@group(0) @binding(5) var<storage,read> vf_edits:array<VFEdit>;
struct VFSample { material:u32, source:u32, valid:u32, pad:u32 }

fn vf_hash(c:vec3<i32>,seed:u32)->u32 {
    let v=bitcast<vec3<u32>>(c);
    var h=v.x*0x8da6b343u ^ v.y*0xd8163841u ^ v.z*0xcb1ab31fu ^ seed;
    h ^= h>>16u;h*=0x7feb352du;h^=h>>15u;h*=0x846ca68bu;return h^(h>>16u);
}
fn vf_smooth(t:u32)->u32 {
    if t>=65536u {return 65536u;}
    return lf_mul(t*t,196608u-2u*t).y;
}
// Noise endpoints stay in [0,65535], and the fraction is at most 65536.
// Their unsigned product fits u32, unlike general atlas-height interpolation.
fn vf_lerp(a:i32,b:i32,fraction:u32)->i32 {
    let delta=b-a;
    let product=u32(abs(delta))*fraction;
    let whole=product>>16u;
    if delta<0 {return a-i32(whole+select(0u,1u,(product&65535u)!=0u));}
    return a+i32(whole);
}
fn vf_noise(c:vec3<i32>,shift:u32,seed:u32)->i32 {
    let size=1u<<shift;
    let offsets=vec3<i32>((vec3<u32>(seed)*vec3<u32>(0x9e3779b9u,0x85ebca6bu,0xc2b2ae35u)
        ^vec3<u32>(0xa341316cu,0xc8013ea4u,0xad90777du))&vec3<u32>(size-1u));
    let translated=c+offsets;
    let base=translated>>vec3<u32>(shift);
    let raw=(vec3<u32>(translated-base*i32(size))*2u+vec3<u32>(1u))<<vec3<u32>(15u-shift);
    let t=vec3<u32>(vf_smooth(raw.x),vf_smooth(raw.y),vf_smooth(raw.z));
    var values:array<i32,8>;
    for(var i=0u;i<8u;i++) {
        values[i]=i32(vf_hash(base+vec3<i32>(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u)),seed)&65535u);
    }
    return vf_lerp(vf_lerp(vf_lerp(values[0],values[1],t.x),vf_lerp(values[2],values[3],t.x),t.y),
        vf_lerp(vf_lerp(values[4],values[5],t.x),vf_lerp(values[6],values[7],t.x),t.y),t.z);
}
fn vf_scale(value:i32,amplitude:i32)->i32 {return (value*amplitude)>>15u;}
fn vf_detail_values(values:vec3<i32>)->i32 {
    return vf_scale(values.x-32768,8000)+vf_scale(16384-abs(values.y-32768),480)+vf_scale(values.z-32768,40);
}
fn vf_detail(c:vec3<i32>)->i32 {
    let seed=lf_settings.pad0;
    return vf_detail_values(vec3<i32>(vf_noise(c,14u,seed^73u),vf_noise(c,10u,seed^191u),vf_noise(c,7u,seed^311u)));
}
fn vf_height(c:vec3<i32>)->i32 {return bitcast<i32>(lf_sample_cell(c).value.y)+vf_detail(c);}
fn vf_valid(c:vec3<i32>)->bool {
    return lf_settings.pad2==1u && all(c>=vec3<i32>(-100000000)) && all(c<=vec3<i32>(100000000));
}
fn vf_within(delta:vec3<u32>,radius:u32)->bool {
    if any(delta>vec3<u32>(radius/2u)) {return false;}
    let sum=lf_add(lf_add(lf_mul(delta.x,delta.x),lf_mul(delta.y,delta.y)),lf_mul(delta.z,delta.z));
    let squared=vec2<u32>(sum.x<<2u,(sum.y<<2u)|(sum.x>>30u));
    return !lf_greater(squared,lf_mul(radius,radius));
}
fn vf_sample(c:vec3<i32>)->VFSample {
    if !vf_valid(c) {return VFSample(0u,0u,0u,0u);}
    for(var count=lf_settings.pad1;count>0u;count--) {
        let e=vf_edits[count-1u];
        if vf_within(vec3<u32>(abs(c-e.cell)),e.radius_units) {return VFSample(e.material,count,1u,0u);}
    }
    let radius=u32(127420000+vf_height(c));
    let center=vec3<u32>(abs(c*2+vec3<i32>(1)));
    let squared=lf_add(lf_add(lf_mul(center.x,center.x),lf_mul(center.y,center.y)),lf_mul(center.z,center.z));
    return VFSample(select(0u,1u,!lf_greater(squared,lf_mul(radius,radius))),0u,1u,0u);
}

fn vf_noise_range(low:vec3<i32>,high:vec3<i32>,shift:u32,seed:u32)->vec2<i32> {
    let middle=low+(high-low)/2;
    let distance=max(middle-low,high-middle);
    let displacement=u32(distance.x+distance.y+distance.z);
    let size=1u<<shift;
    if 3u*displacement>=2u*size {return vec2<i32>(0,65535);}
    let value=vf_noise(middle,shift,seed);
    if displacement==0u {return vec2<i32>(value);}
    let error=i32(((3u*65535u*displacement+2u*size-1u)>>(shift+1u))+12u);
    return vec2<i32>(max(value-error,0),min(value+error,65535));
}
fn vf_detail_range(low:vec3<i32>,high:vec3<i32>)->vec2<i32> {
    let seed=lf_settings.pad0;
    let a=vf_noise_range(low,high,14u,seed^73u)-vec2<i32>(32768);
    let ridge=vf_noise_range(low,high,10u,seed^191u)-vec2<i32>(32768);
    let nearest=select(min(abs(ridge.x),abs(ridge.y)),0,ridge.x<=0 && ridge.y>=0);
    let farthest=max(abs(ridge.x),abs(ridge.y));
    let c=vf_noise_range(low,high,7u,seed^311u)-vec2<i32>(32768);
    return vec2<i32>(vf_scale(a.x,8000)+vf_scale(16384-farthest,480)+vf_scale(c.x,40),
        vf_scale(a.y,8000)+vf_scale(16384-nearest,480)+vf_scale(c.y,40));
}
fn vf_procedural_region(low:vec3<i32>,high:vec3<i32>)->LFRegionBounds {
    let base=lf_classify_region(low,high);
    if base.valid==0u || lf_settings.pad2!=1u {return LFRegionBounds(vec2<i32>(0),0u,0u);}
    let heights=base.height+vf_detail_range(low,high);
    let a=low*2+vec3<i32>(1);let b=high*2+vec3<i32>(1);
    let near=vec3<u32>(select(min(abs(a),abs(b)),vec3<i32>(1),(a<vec3<i32>(0))&(b>vec3<i32>(0))));
    let far=vec3<u32>(max(abs(a),abs(b)));
    let near_squared=lf_add(lf_add(lf_mul(near.x,near.x),lf_mul(near.y,near.y)),lf_mul(near.z,near.z));
    let far_squared=lf_add(lf_add(lf_mul(far.x,far.x),lf_mul(far.y,far.y)),lf_mul(far.z,far.z));
    let inner=u32(127420000+heights.x);let outer=u32(127420000+heights.y);
    var classification=0u;
    if lf_greater(near_squared,lf_mul(outer,outer)) {classification=1u;}
    else if !lf_greater(far_squared,lf_mul(inner,inner)) {classification=2u;}
    return LFRegionBounds(heights,classification,1u);
}
fn vf_classify_region(low:vec3<i32>,high:vec3<i32>)->LFRegionBounds {
    let procedural=vf_procedural_region(low,high);
    if procedural.valid==0u {return procedural;}
    var classification=procedural.classification;
    for(var i=0u;i<lf_settings.pad1;i++) {
        let e=vf_edits[i];
        let nearest=vec3<u32>(max(max(low-e.cell,e.cell-high),vec3<i32>(0)));
        if !vf_within(nearest,e.radius_units) {continue;}
        let farthest=vec3<u32>(max(abs(low-e.cell),abs(high-e.cell)));
        let replacement=select(2u,1u,e.material==0u);
        if vf_within(farthest,e.radius_units) {classification=replacement;}
        else if classification!=replacement {classification=0u;}
    }
    return LFRegionBounds(procedural.height,classification,1u);
}
