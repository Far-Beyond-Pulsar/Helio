// Stochastic ray-tracing acoustic simulator.
// Each GPU thread fires one ray bundle and accumulates energy into the RIR.

struct Params {
    listener_pos: vec3<f32>, _p0: f32,
    n_samples: u32, n_rays: u32, num_meshes: u32, num_indices: u32,
    speed_of_sound: f32, time_res: f32, max_duration: f32, _p1: f32,
    air_abs: array<f32, 8>,
    seed: u32, _p2: vec3<u32>,
}
struct Vertex { pos: vec3<f32>, _p: f32, normal: vec3<f32>, _p2: f32 }
struct Mesh   { idx_off: u32, idx_cnt: u32, vert_off: u32, mat_idx: u32, xform: mat4x4<f32> }
struct Material { absorption: array<f32,8>, scattering: array<f32,8> }

@group(0) @binding(0) var<uniform>        params:  Params;
@group(0) @binding(1) var<storage, read>  verts:   array<Vertex>;
@group(0) @binding(2) var<storage, read>  indices: array<u32>;
@group(0) @binding(3) var<storage, read>  meshes:  array<Mesh>;
@group(0) @binding(4) var<storage, read>  mats:    array<Material>;
@group(0) @binding(5) var<storage, read_write> rir: array<atomic<i32>>; // Q8.23 fixed-point floats

var<private> rng: u32;
fn pcg() -> u32 { rng=rng*747796405u+2891336453u; let w=(((rng>>((rng>>28u)+4u))^rng)*277803737u); return (w>>22u)^w; }
fn rf() -> f32 { return f32(pcg())*(1.0/4294967296.0); }
fn uniform_sphere() -> vec3<f32> {
    let u1=rf();let u2=rf();
    let z=1.0-2.0*u1;let r=sqrt(max(0.0,1.0-z*z));let phi=6.28318530718*u2;
    return vec3<f32>(r*cos(phi),r*sin(phi),z);
}

struct Hit { t: f32, normal: vec3<f32>, mat_idx: u32, hit: bool }
fn scene_intersect(ro:vec3<f32>,rd:vec3<f32>)->Hit{
    var best=Hit(1e30,vec3<f32>(0.0,1.0,0.0),0u,false);
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
            let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
            let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
            let e1=p1-p0;let e2=p2-p0;let h=cross(rd,e2);let det=dot(e1,h);
            if abs(det)<1e-7{continue;}
            let inv=1.0/det;let s=ro-p0;
            let u=inv*dot(s,h);if u<0.0||u>1.0{continue;}
            let q=cross(s,e1);let v=inv*dot(rd,q);if v<0.0||(u+v)>1.0{continue;}
            let t=inv*dot(e2,q);
            if t>1e-4 && t<best.t{
                let nw=(m.xform*vec4<f32>(normalize((verts[i0].normal+verts[i1].normal+verts[i2].normal)/3.0),0.0)).xyz;
                best=Hit(t,normalize(nw),m.mat_idx,true);
            }
        }
    }
    return best;
}

fn hemisphere_around(n:vec3<f32>)->vec3<f32>{
    var t=vec3<f32>(1.0,0.0,0.0);if abs(n.x)>0.9{t=vec3<f32>(0.0,1.0,0.0);}
    let b=normalize(cross(n,t));let tt=cross(b,n);
    let u1=rf();let u2=rf();
    let r=sqrt(max(0.0,1.0-u1*u1));let phi=6.28318530718*u2;
    return normalize(r*cos(phi)*tt+r*sin(phi)*b+u1*n);
}

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    if gid.x >= params.n_rays { return; }
    rng = gid.x * 1664525u + params.seed;
    // Fire one stochastic ray from the listener outward
    var ro=params.listener_pos;
    var rd=uniform_sphere();
    var energy=array<f32,8>(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0);
    var travel_time=0.0;
    for(var bounce=0u;bounce<32u;bounce++){
        if travel_time >= params.max_duration { break; }
        let h=scene_intersect(ro,rd);
        if !h.hit { break; }
        let dist=h.t;
        travel_time += dist / params.speed_of_sound;
        if travel_time >= params.max_duration { break; }
        let sample_idx=u32(travel_time / params.time_res);
        if sample_idx >= params.n_samples { break; }
        let mat=mats[h.mat_idx];
        // Accumulate energy arrival into the RIR buffer (atomic add via fixed-point)
        for(var band=0u;band<8u;band++){
            let air=exp(-params.air_abs[band]*dist);
            energy[band]*=(1.0-mat.absorption[band])*air;
            let fixed_val=i32(energy[band]*8388608.0); // Q8.23
            let buf_idx=band*params.n_samples+sample_idx;
            atomicAdd(&rir[buf_idx],fixed_val);
        }
        // Scatter direction
        let scatter=rf();
        if scatter < mat.scattering[0] {
            rd=normalize(hemisphere_around(h.normal)+uniform_sphere()*0.3);
        } else {
            rd=reflect(rd,h.normal);
        }
        ro = ro+rd*(h.t+0.001);
    }
}
