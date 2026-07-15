struct Params { probe_pos: vec3<f32>, face: u32, res: u32, samples: u32, num_meshes: u32, num_indices: u32, view: mat4x4<f32> }
struct Vertex { pos: vec3<f32>, _p: f32, normal: vec3<f32>, _p2: f32, uv: vec2<f32>, _p3: vec2<f32> }
struct Mesh   { idx_off: u32, idx_cnt: u32, vert_off: u32, _p: u32, xform: mat4x4<f32> }

@group(0) @binding(0) var<uniform>        params:  Params;
@group(0) @binding(1) var<storage, read>  verts:   array<Vertex>;
@group(0) @binding(2) var<storage, read>  indices: array<u32>;
@group(0) @binding(3) var<storage, read>  meshes:  array<Mesh>;
@group(0) @binding(4) var                 out_tex: texture_storage_2d<rgba32float, write>;

var<private> rng: u32;
fn pcg() -> u32 { rng=rng*747796405u+2891336453u; let w=(((rng>>((rng>>28u)+4u))^rng)*277803737u); return (w>>22u)^w; }
fn rf() -> f32 { return f32(pcg())*(1.0/4294967296.0); }

fn hemisphere(n:vec3<f32>)->vec3<f32>{
    let u1=rf();let u2=rf();let r=sqrt(max(0.0,1.0-u1*u1));let phi=6.28318530718*u2;
    var t=vec3<f32>(1.0,0.0,0.0);if abs(n.x)>0.9{t=vec3<f32>(0.0,1.0,0.0);}
    let b=normalize(cross(n,t));let tt=cross(b,n);
    return normalize(r*cos(phi)*tt+r*sin(phi)*b+u1*n);
}

fn ray_tri(ro:vec3<f32>,rd:vec3<f32>,v0:vec3<f32>,v1:vec3<f32>,v2:vec3<f32>)->f32{
    let e1=v1-v0;let e2=v2-v0;let h=cross(rd,e2);let det=dot(e1,h);
    if abs(det)<1e-7{return 1e30;}let inv=1.0/det;let s=ro-v0;
    let u=inv*dot(s,h);if u<0.0||u>1.0{return 1e30;}
    let q=cross(s,e1);let v=inv*dot(rd,q);if v<0.0||(u+v)>1.0{return 1e30;}
    let t=inv*dot(e2,q);if t>1e-4{return t;}return 1e30;
}

struct Hit{t:f32,normal:vec3<f32>,albedo:vec3<f32>}
fn scene_hit(ro:vec3<f32>,rd:vec3<f32>)->Hit{
    var best=Hit(1e30,vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.5,0.5,0.5));
    for(var mi=0u;mi<params.num_meshes;mi++){
        let m=meshes[mi];
        for(var ii=m.idx_off;ii<m.idx_off+m.idx_cnt;ii+=3u){
            let i0=indices[ii];let i1=indices[ii+1u];let i2=indices[ii+2u];
            let p0=(m.xform*vec4<f32>(verts[i0].pos,1.0)).xyz;
            let p1=(m.xform*vec4<f32>(verts[i1].pos,1.0)).xyz;
            let p2=(m.xform*vec4<f32>(verts[i2].pos,1.0)).xyz;
            let t=ray_tri(ro,rd,p0,p1,p2);
            if t<best.t{
                let n0=(m.xform*vec4<f32>(verts[i0].normal,0.0)).xyz;
                let n1=(m.xform*vec4<f32>(verts[i1].normal,0.0)).xyz;
                let n2=(m.xform*vec4<f32>(verts[i2].normal,0.0)).xyz;
                best=Hit(t,normalize((n0+n1+n2)/3.0),vec3<f32>(0.5,0.5,0.5));
            }
        }
    }
    return best;
}

fn face_dir(face:u32,u:f32,v:f32)->vec3<f32>{
    switch face {
        case 0u:{return normalize(vec3<f32>(1.0,v,-u));}
        case 1u:{return normalize(vec3<f32>(-1.0,v,u));}
        case 2u:{return normalize(vec3<f32>(u,1.0,-v));}
        case 3u:{return normalize(vec3<f32>(u,-1.0,v));}
        case 4u:{return normalize(vec3<f32>(u,v,1.0));}
        default:{return normalize(vec3<f32>(-u,v,-1.0));}
    }
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let res=params.res;
    if gid.x>=res||gid.y>=res{return;}
    rng=gid.x+gid.y*res+(params.face+1u)*res*res;
    let uf=(f32(gid.x)+0.5)/f32(res)*2.0-1.0;
    let vf=(f32(gid.y)+0.5)/f32(res)*2.0-1.0;
    let dir=face_dir(params.face,uf,vf);
    var col=vec3<f32>(0.0);
    var throughput=vec3<f32>(1.0);
    var ro=params.probe_pos;var rd=dir;
    for(var bounce=0u;bounce<4u;bounce++){
        let h=scene_hit(ro,rd);
        if h.t>=1e29{col+=throughput*vec3<f32>(0.05,0.07,0.12);break;}
        ro=ro+rd*h.t+h.normal*0.001;
        throughput*=h.albedo;
        rd=hemisphere(h.normal);
    }
    textureStore(out_tex,vec2<i32>(gid.xy),vec4<f32>(col,1.0));
}
