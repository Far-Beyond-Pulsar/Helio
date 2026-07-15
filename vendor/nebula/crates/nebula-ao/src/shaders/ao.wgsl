struct Params { res: u32, ray_count: u32, max_dist: f32, bias: f32, num_meshes: u32, num_indices: u32, seed: u32, _p: u32 }
struct Vertex { pos: vec3<f32>, _p: f32, normal: vec3<f32>, _p2: f32, lm_uv: vec2<f32>, _p3: vec2<f32> }
struct Mesh   { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: mat4x4<f32> }

@group(0) @binding(0) var<uniform>        params:  Params;
@group(0) @binding(1) var<storage, read>  verts:   array<Vertex>;
@group(0) @binding(2) var<storage, read>  indices: array<u32>;
@group(0) @binding(3) var<storage, read>  meshes:  array<Mesh>;
@group(0) @binding(4) var                 out_ao:  texture_storage_2d<r32float, write>;

var<private> rng: u32;
fn pcg() -> u32 { rng = rng*747796405u+2891336453u; let w=(((rng>>((rng>>28u)+4u))^rng)*277803737u); return (w>>22u)^w; }
fn rf() -> f32 { return f32(pcg())*(1.0/4294967296.0); }

fn hemisphere(n: vec3<f32>) -> vec3<f32> {
    let u1=rf(); let u2=rf();
    let r=sqrt(max(0.0,1.0-u1*u1)); let phi=6.28318530718*u2;
    var t=vec3<f32>(1.0,0.0,0.0); if abs(n.x)>0.9{t=vec3<f32>(0.0,1.0,0.0);}
    let b=normalize(cross(n,t)); let tt=cross(b,n);
    return normalize(r*cos(phi)*tt+r*sin(phi)*b+u1*n);
}

fn ray_tri(ro:vec3<f32>,rd:vec3<f32>,v0:vec3<f32>,v1:vec3<f32>,v2:vec3<f32>)->f32{
    let e1=v1-v0; let e2=v2-v0; let h=cross(rd,e2); let det=dot(e1,h);
    if abs(det)<1e-7{return -1.0;} let inv=1.0/det; let s=ro-v0;
    let u=inv*dot(s,h); if u<0.0||u>1.0{return -1.0;}
    let q=cross(s,e1); let v=inv*dot(rd,q); if v<0.0||(u+v)>1.0{return -1.0;}
    let t=inv*dot(e2,q); if t>1e-4{return t;} return -1.0;
}

fn scene_hit(ro:vec3<f32>,rd:vec3<f32>,max_t:f32)->bool{
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
            let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
            let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
            if ray_tri(ro,rd,p0,p1,p2) < max_t { return true; }
        }
    }
    return false;
}

fn texel_info(lm_uv:vec2<f32>)->vec4<f32>{// xyz=pos w=valid(1=yes)
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let uv0=verts[i0].lm_uv; let uv1=verts[i1].lm_uv; let uv2=verts[i2].lm_uv;
            let d1=uv1-uv0; let d2=uv2-uv0; let dp=lm_uv-uv0;
            let inv=1.0/(d1.x*d2.y-d1.y*d2.x);
            let u=(dp.x*d2.y-dp.y*d2.x)*inv; let v=(d1.x*dp.y-d1.y*dp.x)*inv;
            if u>=0.0&&v>=0.0&&(u+v)<=1.0{
                let w=1.0-u-v;
                let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
                let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
                let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
                return vec4<f32>(p0*w+p1*u+p2*v,1.0);
            }
        }
    }
    return vec4<f32>(0.0,0.0,0.0,0.0);
}

fn texel_normal(lm_uv:vec2<f32>)->vec3<f32>{
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let uv0=verts[i0].lm_uv;let uv1=verts[i1].lm_uv;let uv2=verts[i2].lm_uv;
            let d1=uv1-uv0;let d2=uv2-uv0;let dp=lm_uv-uv0;
            let inv=1.0/(d1.x*d2.y-d1.y*d2.x);
            let u=(dp.x*d2.y-dp.y*d2.x)*inv;let v=(d1.x*dp.y-d1.y*dp.x)*inv;
            if u>=0.0&&v>=0.0&&(u+v)<=1.0{
                let w=1.0-u-v;
                let n0=(m.xform*vec4<f32>(verts[i0].normal,0.0)).xyz;
                let n1=(m.xform*vec4<f32>(verts[i1].normal,0.0)).xyz;
                let n2=(m.xform*vec4<f32>(verts[i2].normal,0.0)).xyz;
                return normalize(n0*w+n1*u+n2*v);
            }
        }
    }
    return vec3<f32>(0.0,1.0,0.0);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let res=params.res;
    if gid.x>=res||gid.y>=res{return;}
    rng = gid.x + gid.y*res + params.seed*res*res;
    let lm_uv=(vec2<f32>(gid.xy)+0.5)/f32(res);
    let ti=texel_info(lm_uv);
    if ti.w<0.5{textureStore(out_ao,vec2<i32>(gid.xy),vec4<f32>(0.0));return;}
    let pos=ti.xyz; let n=texel_normal(lm_uv);
    var unoccluded=0u;
    for(var s=0u;s<params.ray_count;s++){
        let dir=hemisphere(n);
        if !scene_hit(pos+n*params.bias,dir,params.max_dist){unoccluded+=1u;}
    }
    let ao=f32(unoccluded)/f32(params.ray_count);
    textureStore(out_ao,vec2<i32>(gid.xy),vec4<f32>(ao,0.0,0.0,0.0));
}
