// Shared voxel recipe adapter and material shading. Rendering traversal lives in engine/stored.wgsl.
struct Params {
    origin: vec4<i32>, fraction: vec4<f32>, radial: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>, screen: vec4<f32>,
    lighting: vec4<f32>, settings: vec4<f32>,

}
struct Edit { cell: vec3<i32>, material: u32, radius: f32, radius_units: u32, pad1: f32, pad2: f32 }
struct BaseSample {density:f32,solid:bool}
struct TerrainMaterial {source:u32,grass:f32,moisture:f32,strata:f32}
struct Hit { cell:vec3<i32>, status:u32, normal:vec3<f32>, distance:f32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> edits:array<Edit>;
@group(0) @binding(9) var<storage,read_write> primary_hits:array<Hit>;
fn hash(c:vec3<i32>, seed:u32)->u32 {
    let q=bitcast<vec3<u32>>(c);
    var h=q.x*0x8da6b343u ^ q.y*0xd8163841u ^ q.z*0xcb1ab31fu ^ seed;
    h ^= h>>16u; h *= 0x7feb352du; h ^= h>>15u; h *= 0x846ca68bu; return h ^ (h>>16u);
}

fn noise_offset(shift:u32,seed:u32)->vec3<i32> {
    return vec3<i32>((vec3<u32>(seed)*vec3<u32>(0x9e3779b9u,0x85ebca6bu,0xc2b2ae35u)
        ^vec3<u32>(0xa341316cu,0xc8013ea4u,0xad90777du))&vec3<u32>((1u<<shift)-1u));
}

fn noise(c:vec3<i32>, f:vec3<f32>, shift:u32, seed:u32)->f32 {
    let shifted=c+noise_offset(shift,seed);
    let size=i32(1u<<shift); let base=shifted>>vec3<u32>(shift);
    var t=(vec3<f32>(shifted-base*size)+f)/f32(size); t=t*t*(3.0-2.0*t);
    var v:array<f32,8>;
    for(var i=0u;i<8u;i++) { v[i]=f32(hash(base+vec3<i32>(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u)),seed)&65535u)/65535.0; }
    return mix(mix(mix(v[0],v[1],t.x),mix(v[2],v[3],t.x),t.y),
        mix(mix(v[4],v[5],t.x),mix(v[6],v[7],t.x),t.y),t.z)*2.0-1.0;
}

fn terrain_units(c:vec3<i32>)->i32 {
    if !vf_valid(c) {return 0;}
    return vf_height(c);
}

fn base_sample(c:vec3<i32>,local:vec3<f32>)->BaseSample {
    if !vf_valid(c) {return BaseSample(-1.0,false);}
    let radius=127420000+terrain_units(c);
    let density=radial_depth(c,radius);
    return BaseSample(density,radius>0 && density>=0.0);
}

fn multiply_wide(a:u32,b:u32)->vec2<u32> {
    let a0=a&65535u;let a1=a>>16u;let b0=b&65535u;let b1=b>>16u;
    let lo0=a0*b0;let cross0=a1*b0+(lo0>>16u);let cross1=a0*b1+(cross0&65535u);
    return vec2<u32>((cross1<<16u)|(lo0&65535u),a1*b1+(cross0>>16u)+(cross1>>16u));
}

fn add_wide(a:vec2<u32>,b:vec2<u32>)->vec2<u32> {let lo=a.x+b.x;return vec2<u32>(lo,a.y+b.y+select(0u,1u,lo<a.x));}

fn edit_contains(c:vec3<i32>,e:Edit)->bool {
    let delta=abs(c-e.cell);
    let radius2=e.radius_units;
    if any(delta>vec3<i32>(i32((radius2+1u)/2u))) {return false;}
    let d=vec3<u32>(delta);
    if radius2>20000u {
        let sum=add_wide(add_wide(multiply_wide(d.x,d.x),multiply_wide(d.y,d.y)),multiply_wide(d.z,d.z));
        let squared=vec2<u32>(sum.x<<2u,(sum.y<<2u)|(sum.x>>30u));
        let radius_squared=multiply_wide(radius2,radius2);
        return squared.y<radius_squared.y || (squared.y==radius_squared.y && squared.x<=radius_squared.x);
    }
    return (d.x*d.x+d.y*d.y+d.z*d.z)*4u<=radius2*radius2;
}

fn terrain_material(c:vec3<i32>,altitude:f32,footprint:f32,grass:f32,source:u32)->TerrainMaterial {
    // Coherent patches remain fixed to the volume. Unresolved pigment tends
    // to its mean instead of turning into independent per-voxel noise.
    let pigment=noise(c,vec3<f32>(0.5),9u,1217u)*(1.0-smoothstep(10.0,51.2,footprint));
    let region=noise(c,vec3<f32>(0.5),13u,911u)*(1.0-smoothstep(160.0,819.2,footprint));
    let moisture=clamp(0.5+pigment*0.3+region*0.3,0.0,1.0);
    let strata=sin(altitude*1.15+region*2.0)*(1.0-smoothstep(0.5,5.5,footprint));
    return TerrainMaterial(source,grass,moisture,strata);
}

fn face_albedo(normal:vec3<f32>,radial:vec3<f32>,depth:f32,altitude:f32,material:TerrainMaterial)->vec3<f32> {
    let upward=max(dot(normal,radial),0.0);
    let vegetation=mix(vec3<f32>(0.16,0.29,0.055),vec3<f32>(0.105,0.245,0.045),material.moisture);
    let rock=vec3<f32>(0.34,0.32,0.28)*(1.0+material.strata*0.04);
    let soil=vec3<f32>(0.19,0.125,0.065)*(1.0+material.strata*0.04);
    // Turf is a thin layer of the natural volume, including shallow risers.
    // Its assignment is fixed to the canonical cell, never screen coverage.
    // Deeper and excavated faces expose soil/rock; cube normals stay intact.
    var albedo=rock;
    if material.grass>=0.5 && depth<1.2 {
        albedo=soil;
        if dot(normal,radial)>-0.25 && depth<0.16 && altitude<=90000.0 {albedo=vegetation;}
    }
    if altitude < -180000.0 {albedo=vec3<f32>(0.12,0.15,0.15);}
    if altitude > 90000.0 && upward>=0.5 {albedo=vec3<f32>(0.76,0.80,0.79);}
    if material.source==2u {albedo=select(rock,soil,depth<1.2 && material.grass>=0.5);}
    if material.source==3u {albedo=vec3<f32>(0.35,0.36,0.36);}
    return albedo;
}

fn material_gradient(c:vec3<i32>)->vec3<f32> {
    // Fixed cell-centred material slope; visible normals remain voxel faces.
    let dx=terrain_units(c+vec3<i32>(8,0,0))-terrain_units(c-vec3<i32>(8,0,0));
    let dy=terrain_units(c+vec3<i32>(0,8,0))-terrain_units(c-vec3<i32>(0,8,0));
    let dz=terrain_units(c+vec3<i32>(0,0,8))-terrain_units(c-vec3<i32>(0,0,8));
    return vec3<f32>(f32(dx),f32(dy),f32(dz))*0.03125-normalize(vec3<f32>(c)+0.5);
}

fn radial_depth(c:vec3<i32>,radius:i32)->f32 {
    // Subtract squared integer radii before conversion to float. Subtracting
    // Earth-sized float heights made shallow material layers camera-dependent.
    if any(abs(c)>vec3<i32>(100000000)){return 0.0;}
    if radius<=0{return 0.0;}
    let v=vec3<u32>(abs(c*2+1));
    let squared=add_wide(add_wide(multiply_wide(v.x,v.x),multiply_wide(v.y,v.y)),multiply_wide(v.z,v.z));
    let outer=multiply_wide(u32(radius),u32(radius));
    let inside=squared.y<outer.y || (squared.y==outer.y && squared.x<=outer.x);
    let larger=select(squared,outer,inside);let smaller=select(outer,squared,inside);
    let difference=vec2<u32>(larger.x-smaller.x,larger.y-smaller.y-select(0u,1u,larger.x<smaller.x));
    let numerator=f32(difference.y)*4294967296.0+f32(difference.x);
    let denominator=f32(radius)+sqrt(f32(squared.y)*4294967296.0+f32(squared.x));
    return select(-1.0,1.0,inside)*0.05*numerator/max(denominator,1.0);
}

fn surface_depth(c:vec3<i32>,local:vec3<f32>)->f32 {
    return radial_depth(c,127420000+terrain_units(c));
}

fn cell_altitude(c:vec3<i32>)->f32 {
    return -radial_depth(c,127420000);
}
