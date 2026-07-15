struct Params {
    world_min: vec3<f32>, cell_size: f32,
    grid_dims: vec3<u32>, cell_count: u32,
    words_per_cell: u32, ray_budget: u32,
    max_dist: f32, vis_threshold: u32,
    num_meshes: u32, num_indices: u32,
    seed: u32, _p: u32,
}
struct Vertex { pos: vec3<f32>, _p: f32 }
struct Mesh   { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: mat4x4<f32> }

@group(0) @binding(0) var<uniform>        params:  Params;
@group(0) @binding(1) var<storage, read>  verts:   array<Vertex>;
@group(0) @binding(2) var<storage, read>  indices: array<u32>;
@group(0) @binding(3) var<storage, read>  meshes:  array<Mesh>;
@group(0) @binding(4) var<storage, read_write> pvs_bits: array<atomic<u32>>;

var<private> rng: u32;
fn pcg() -> u32 { rng=rng*747796405u+2891336453u; let w=(((rng>>((rng>>28u)+4u))^rng)*277803737u); return (w>>22u)^w; }
fn rf() -> f32 { return f32(pcg())*(1.0/4294967296.0); }

fn uniform_sphere() -> vec3<f32> {
    let u1=rf(); let u2=rf();
    let z=1.0-2.0*u1; let r=sqrt(max(0.0,1.0-z*z)); let phi=6.28318*u2;
    return vec3<f32>(r*cos(phi),r*sin(phi),z);
}

fn ray_hits_geom(ro:vec3<f32>,rd:vec3<f32>,max_t:f32)->bool{
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
            let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
            let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
            let e1=p1-p0; let e2=p2-p0; let h=cross(rd,e2); let det=dot(e1,h);
            if abs(det)<1e-7{continue;}
            let inv=1.0/det; let s=ro-p0;
            let u=inv*dot(s,h); if u<0.0||u>1.0{continue;}
            let q=cross(s,e1); let v=inv*dot(rd,q); if v<0.0||(u+v)>1.0{continue;}
            let t=inv*dot(e2,q); if t>1e-4&&t<max_t{return true;}
        }
    }
    return false;
}

fn cell_centre(flat_idx:u32)->vec3<f32>{
    let gx=params.grid_dims.x; let gy=params.grid_dims.y;
    let z=flat_idx/(gy*gx); let tmp=flat_idx%(gy*gx); let y=tmp/gx; let x=tmp%gx;
    return params.world_min+vec3<f32>(f32(x)+0.5,f32(y)+0.5,f32(z)+0.5)*params.cell_size;
}

fn flat_cell_at(p:vec3<f32>)->u32{
    let rel=(p-params.world_min)/params.cell_size;
    let ix=u32(rel.x); let iy=u32(rel.y); let iz=u32(rel.z);
    if ix>=params.grid_dims.x||iy>=params.grid_dims.y||iz>=params.grid_dims.z{return 0xFFFFFFFFu;}
    return iz*params.grid_dims.y*params.grid_dims.x+iy*params.grid_dims.x+ix;
}

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let src=gid.x; if src>=params.cell_count{return;}
    rng=src*1664525u+params.seed;
    let src_pos=cell_centre(src);
    for(var r=0u;r<params.ray_budget;r++){
        let dir=uniform_sphere();
        if ray_hits_geom(src_pos,dir,params.max_dist){continue;}
        // Find which target cell the ray reached
        let end_pos=src_pos+dir*params.max_dist;
        let dst=flat_cell_at(end_pos);
        if dst==0xFFFFFFFFu{continue;}
        // Set visibility bit: pvs_bits uses u32 halves of u64 words
        // (WGSL doesn't have i64; we split words_per_cell×u64 as 2×words_per_cell×u32)
        let bit_idx=dst;
        let word=src*(params.words_per_cell*2u)+(bit_idx/32u);
        let bit =bit_idx%32u;
        atomicOr(&pvs_bits[word],1u<<bit);
    }
}
