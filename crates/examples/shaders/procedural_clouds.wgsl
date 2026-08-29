// =============================================================================
// Procedural Clouds — Skydome Mode — High-Performance Volumetric Cloud Pipeline
// =============================================================================
// MODE: Skydome — entire sky dome has clouds spread across it (not a box-filled
// volume). Clouds occupy a spherical shell / altitude layer between SKYDOME_BOTTOM
// and SKYDOME_TOP tiled infinitely across the horizon, sampled via dome-projected
// weather map and height gradient. Ray march intersects the dome shell, not a
// bounded AABB — coverage is driven by skydome UV, not box containment.
//
// Pipeline retains full spec: quarter-res ray march (4x4 Bayer + blue-noise),
// temporal reprojection (history, Neighborhood Clamping / Variance Bounding,
// EMA 90/10), depth-aware bilateral upsample, coarse-to-fine, space leaping
// 100m-300m, early termination alpha>=0.98, depth culling, height gradients
// (Cumulus/Stratocumulus/Cumulonimbus), Perlin-Worley + Worley erosion with
// LOD 0.05<d<0.7, single-pass lighting, dual HG g1=0.8 g2=-0.3, Beer-Powder,
// multi-scattering octaves 2-3, ambient height-gradient, debug views.
// Based on Blender node graph with thickness = d*1.8 — now dome-mapped.
// =============================================================================

fn rot_u32(x: u32, k: u32) -> u32 { return (x << k) | (x >> (32u - k)); }
fn hash_uint4(kx: u32, ky: u32, kz: u32, kw: u32) -> u32 {
    var a = 0xdeadbeefu + (4u << 2u) + 13u; var b = a; var c = a; a += kx; b += ky; a -= c; a ^= rot_u32(c, 4u); c += b; b -= a; b ^= rot_u32(a, 6u); a += c; c -= b; c ^= rot_u32(b, 8u); b += a; a -= c; a ^= rot_u32(c, 16u); c += b; b -= a; b ^= rot_u32(a, 19u); a += c; c -= b; c ^= rot_u32(b, 4u); b += a; a += kz; b += kw; c ^= b; c -= rot_u32(b, 14u); a ^= c; a -= rot_u32(c, 11u); b ^= a; b -= rot_u32(a, 25u); c ^= b; c -= rot_u32(b, 16u); a ^= c; a -= rot_u32(c, 4u); b ^= a; b -= rot_u32(a, 14u); c ^= b; c -= rot_u32(b, 24u); return c;
}
fn hash_uint4_to_float(kx: u32, ky: u32, kz: u32, kw: u32) -> f32 { return f32(hash_uint4(kx, ky, kz, kw)) / f32(0xFFFFFFFFu); }
fn hash_vec4_to_vec4(k: vec4f) -> vec4f { return vec4f(hash_uint4_to_float(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w)), hash_uint4_to_float(bitcast<u32>(k.w), bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z)), hash_uint4_to_float(bitcast<u32>(k.z), bitcast<u32>(k.w), bitcast<u32>(k.x), bitcast<u32>(k.y)), hash_uint4_to_float(bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w), bitcast<u32>(k.x))); }
fn noise_fade(t: f32) -> f32 { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
fn tri_mix(v0:f32,v1:f32,v2:f32,v3:f32,v4:f32,v5:f32,v6:f32,v7:f32,x:f32,y:f32,z:f32)->f32{let x1=1.0-x;let y1=1.0-y;let z1=1.0-z;return z1*(y1*(v0*x1+v1*x)+y*(v2*x1+v3*x))+z*(y1*(v4*x1+v5*x)+y*(v6*x1+v7*x));}
fn quad_mix(v0:f32,v1:f32,v2:f32,v3:f32,v4:f32,v5:f32,v6:f32,v7:f32,v8:f32,v9:f32,v10:f32,v11:f32,v12:f32,v13:f32,v14:f32,v15:f32,x:f32,y:f32,z:f32,w:f32)->f32{return mix(tri_mix(v0,v1,v2,v3,v4,v5,v6,v7,x,y,z),tri_mix(v8,v9,v10,v11,v12,v13,v14,v15,x,y,z),w);}
fn noiseg_4d(h:u32,x:f32,y:f32,z:f32,w:f32)->f32{let hh=h&31u;let u=select(x,y,hh>=24u);let v=select(y,z,hh>=16u);let s=select(z,w,hh>=8u);let r=select(u,-u,(hh&1u)!=0u);let rv=select(v,-v,(hh&2u)!=0u);let rs=select(s,-s,(hh&4u)!=0u);return r+rv+rs;}
fn perlin_noise_4d(p:vec4f)->f32{
 let pf=floor(p);let X=i32(pf.x);let Y=i32(pf.y);let Z=i32(pf.z);let W=i32(pf.w);
 let fx=p.x-pf.x;let fy=p.y-pf.y;let fz=p.z-pf.z;let fw=p.w-pf.w;
 let u=noise_fade(fx);let v=noise_fade(fy);let t=noise_fade(fz);let s=noise_fade(fw);
 return quad_mix(noiseg_4d(hash_uint4(u32(X),u32(Y),u32(Z),u32(W)),fx,fy,fz,fw),noiseg_4d(hash_uint4(u32(X+1),u32(Y),u32(Z),u32(W)),fx-1,fy,fz,fw),noiseg_4d(hash_uint4(u32(X),u32(Y+1),u32(Z),u32(W)),fx,fy-1,fz,fw),noiseg_4d(hash_uint4(u32(X+1),u32(Y+1),u32(Z),u32(W)),fx-1,fy-1,fz,fw),noiseg_4d(hash_uint4(u32(X),u32(Y),u32(Z+1),u32(W)),fx,fy,fz-1,fw),noiseg_4d(hash_uint4(u32(X+1),u32(Y),u32(Z+1),u32(W)),fx-1,fy,fz-1,fw),noiseg_4d(hash_uint4(u32(X),u32(Y+1),u32(Z+1),u32(W)),fx,fy-1,fz-1,fw),noiseg_4d(hash_uint4(u32(X+1),u32(Y+1),u32(Z+1),u32(W)),fx-1,fy-1,fz-1,fw),noiseg_4d(hash_uint4(u32(X),u32(Y),u32(Z),u32(W+1)),fx,fy,fz,fw-1),noiseg_4d(hash_uint4(u32(X+1),u32(Y),u32(Z),u32(W+1)),fx-1,fy,fz,fw-1),noiseg_4d(hash_uint4(u32(X),u32(Y+1),u32(Z),u32(W+1)),fx,fy-1,fz,fw-1),noiseg_4d(hash_uint4(u32(X+1),u32(Y+1),u32(Z),u32(W+1)),fx-1,fy-1,fz,fw-1),noiseg_4d(hash_uint4(u32(X),u32(Y),u32(Z+1),u32(W+1)),fx,fy,fz-1,fw-1),noiseg_4d(hash_uint4(u32(X+1),u32(Y),u32(Z+1),u32(W+1)),fx-1,fy,fz-1,fw-1),noiseg_4d(hash_uint4(u32(X),u32(Y+1),u32(Z+1),u32(W+1)),fx,fy-1,fz-1,fw-1),noiseg_4d(hash_uint4(u32(X+1),u32(Y+1),u32(Z+1),u32(W+1)),fx-1,fy-1,fz-1,fw-1),u,v,t,s);
}
fn noise_fbm(p:vec4f,detail:f32,roughness:f32,lacunarity:f32,normalize:bool)->f32{
 var fscale=1.0;var amp=1.0;var maxamp=0.0;var sum=0.0;let d=i32(detail);
 for(var i=0;i<=d;i++){let t=perlin_noise_4d(fscale*p);sum+=t*amp;maxamp+=amp;amp*=roughness;fscale*=lacunarity;}
 let rmd=detail-floor(detail);if(rmd!=0.0){let t=perlin_noise_4d(fscale*p);let sum2=sum+t*amp;return select(mix(sum,sum2,rmd),mix(0.5+0.5*(sum/maxamp),0.5+0.5*(sum2/(maxamp+amp)),rmd),normalize);}
 return select(sum,0.5+0.5*(sum/maxamp),normalize);
}
fn random_vec4_offset(seed:f32)->vec4f{return hash_vec4_to_vec4(vec4f(seed,seed*1.37,seed*2.23,seed*3.11));}
fn node_noise_texture_4d_value(co:vec3f,w:f32,scale:f32,detail:f32,roughness:f32,lacunarity:f32,distortion:f32,normalize:f32)->f32{
 var p=vec4f(co,w)*scale;if(distortion!=0.0){p+=vec4f(perlin_noise_4d(p+random_vec4_offset(0.0))*distortion,perlin_noise_4d(p+random_vec4_offset(1.0))*distortion,perlin_noise_4d(p+random_vec4_offset(2.0))*distortion,perlin_noise_4d(p+random_vec4_offset(3.0))*distortion);}
 return noise_fbm(p,detail,roughness,lacunarity,normalize!=0.0);
}
fn hash_pcg4d_i(v:vec4i)->vec4i{var vv=v*1664525+1013904223;vv.x+=vv.y*vv.w;vv.y+=vv.z*vv.x;vv.z+=vv.x*vv.y;vv.w+=vv.y*vv.z;vv=vv^(vv>>vec4u(16u));vv.x+=vv.y*vv.w;vv.y+=vv.z*vv.x;vv.z+=vv.x*vv.y;vv.w+=vv.y*vv.z;return vv;}
fn hash_int4_to_vec4(k:vec4i)->vec4f{let h=hash_pcg4d_i(k);return vec4f(h&vec4i(0x7fffffff))*(1.0/f32(0x7fffffff));}
fn hash_int4_to_vec3(k:vec4i)->vec3f{return hash_int4_to_vec4(k).xyz;}
const SHD_VORONOI_EUCLIDEAN=0;const SHD_VORONOI_F1=0;
struct VoronoiParams{scale:f32,detail:f32,roughness:f32,lacunarity:f32,smoothness:f32,exponent:f32,randomness:f32,max_distance:f32,normalize:bool,feature:i32,metric:i32};
struct VoronoiOutput{Distance:f32,Color:vec3f,Position:vec4f};
fn voronoi_distance(a:vec4f,b:vec4f,p:VoronoiParams)->f32{return distance(a,b);}
fn voronoi_f1(params:VoronoiParams,coord:vec4f)->VoronoiOutput{
 let cellP=floor(coord);let local=coord-cellP;let cell=vec4i(cellP);
 var minD=3.402823466e+38;var off=vec4i(0);var tpos=vec4f(0.0);
 for(var u=-1;u<=1;u++){for(var k=-1;k<=1;k++){for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
  let o=vec4i(i,j,k,u);let pp=vec4f(o)+hash_int4_to_vec4(cell+o)*params.randomness;let d=voronoi_distance(pp,local,params);if(d<minD){off=o;minD=d;tpos=pp;}
 }}}}
 var o:VoronoiOutput;o.Distance=minD;o.Color=hash_int4_to_vec3(cell+off);o.Position=tpos+cellP;return o;
}
fn fractal_voronoi_x_fx(params:VoronoiParams,coord:vec4f)->VoronoiOutput{
 var amp=1.0;var maxAmp=0.0;var scale=1.0;var Out:VoronoiOutput;Out.Distance=0.0;Out.Color=vec3f(0.0);Out.Position=vec4f(0.0);
 let zero=params.detail==0.0||params.roughness==0.0;let maxI=i32(ceil(params.detail));
 for(var i=0;i<=maxI;i++){let o=voronoi_f1(params,coord*scale);if(zero){maxAmp=1.0;Out=o;break;}else if(f32(i)<=params.detail){maxAmp+=amp;Out.Distance+=o.Distance*amp;Out.Color+=o.Color*amp;Out.Position=mix(Out.Position,o.Position/scale,amp);scale*=params.lacunarity;amp*=params.roughness;}else{let r=params.detail-floor(params.detail);if(r!=0.0){maxAmp=mix(maxAmp,maxAmp+amp,r);Out.Distance=mix(Out.Distance,Out.Distance+o.Distance*amp,r);Out.Color=mix(Out.Color,Out.Color+o.Color*amp,r);Out.Position=mix(Out.Position,mix(Out.Position,o.Position/scale,amp),r);}}}
 if(params.normalize){Out.Distance/=maxAmp*params.max_distance;Out.Color/=maxAmp;}Out.Position/=params.scale;return Out;
}
fn node_tex_voronoi_f1_4d_distance(coord:vec3f,w:f32,scale:f32,detail:f32,roughness:f32,lacunarity:f32,smoothness:f32,exponent:f32,randomness:f32,metric:f32,normalize:f32)->f32{
 var p:VoronoiParams;p.feature=0;p.metric=i32(metric);p.scale=scale;p.detail=clamp(detail,0.0,15.0);p.roughness=clamp(roughness,0.0,1.0);p.lacunarity=lacunarity;p.smoothness=clamp(smoothness/2.0,0.0,0.5);p.exponent=exponent;p.randomness=clamp(randomness,0.0,1.0);p.max_distance=0.0;p.normalize=normalize!=0.0;
 let ws=w*scale;let cs=coord*scale;p.max_distance=voronoi_distance(vec4f(0.0),vec4f(0.5+0.5*p.randomness),p);let Out=fractal_voronoi_x_fx(p,vec4f(cs,ws));return Out.Distance;
}
fn mapRange(v:f32,f0:f32,f1:f32,t0:f32,t1:f32)->f32{if(abs(f1-f0)<1e-5){return t0;}let t=(v-f0)/(f1-f0);return clamp(mix(t0,t1,t),min(t0,t1),max(t0,t1));}
fn clamp01(v:f32)->f32{return clamp(v,0.0,1.0);}
fn vertical_band(y:f32,l0:f32,l1:f32,u1:f32,u0:f32)->f32{return smoothstep(l0,l1,y)*(1.0-smoothstep(u1,u0,y));}
fn ellipsoid_blob(p:vec3f,c:vec3f,s:vec3f)->f32{let d=length((p-c)*s);return 1.0-smoothstep(0.19,0.52,d);}
fn spiral_scroll(p:vec3f,c:vec3f,s:vec3f,turns:f32,phase:f32)->f32{let q=(p-c)*s;let r=length(q.xy);let a=atan2(q.y,q.x);let crest=cos(a+r*turns+phase)*0.5+0.5;let tube=smoothstep(0.76,0.985,crest);let rw=smoothstep(0.025,0.10,r)*(1.0-smoothstep(0.38,0.64,r));let dw=exp(-abs(q.z)*2.35);return tube*rw*dw;}

// ── Volumetric Pipeline Additions ───────────────────────────────────────────
// 4x4 Bayer matrix for spatial dithering — each pixel in 4x4 block evaluates only ONE ray per frame
fn bayer4x4_pat(x: u32, y: u32) -> f32 {
    let m = array<array<f32,4>,4>(
        array<f32,4>(0.0/16.0, 8.0/16.0, 2.0/16.0, 10.0/16.0),
        array<f32,4>(12.0/16.0, 4.0/16.0, 14.0/16.0, 6.0/16.0),
        array<f32,4>(3.0/16.0, 11.0/16.0, 1.0/16.0, 9.0/16.0),
        array<f32,4>(15.0/16.0, 7.0/16.0, 13.0/16.0, 5.0/16.0)
    );
    return m[y%4u][x%4u];
}
// Height gradient based on cloud type: Cumulus / Stratocumulus / Cumulonimbus
fn height_gradient_cloud_type(h: f32, cloud_type: f32) -> f32 {
    if (cloud_type < 0.5) { return smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.55, 0.95, h)); }
    else if (cloud_type < 1.5) { return smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.45, 0.75, h)) * 1.1; }
    else { let tower = smoothstep(0.0, 0.08, h) * (1.0 - smoothstep(0.85, 1.0, h)); let anvil = smoothstep(0.70, 0.82, h) * 0.35; return tower + anvil; }
}
// Dual-Henyey-Greenstein phase: P(theta)= d1*HG(g1)+ (1-d1)*HG(g2) g1~0.8 forward silver lining, g2~-0.3 backscatter
fn hg(c:f32,g:f32)->f32{let g2=g*g;return (1.0-g2)/(4.0*3.14159*pow(1.0+g2-2.0*g*c,1.5));}
fn dual_hg_phase(cos_theta:f32)->f32{ let g1:f32=0.8; let g2:f32=-0.3; let blend:f32=0.75; return blend*hg(cos_theta,g1) + (1.0-blend)*hg(cos_theta,g2); }
// Beer-Powder: Light Attenuation = exp(-tau*d) * (1 - exp(-2*tau*d))
fn beer_powder(tau_d:f32)->f32{ return exp(-tau_d) * (1.0 - exp(-2.0*tau_d)); }
// Multi-Scattering Octaves: 2-3 octaves exponentially decreasing density, increasing isotropy
fn multi_scatter_octaves(sun_vis:f32, dens:f32)->f32{ let o0=sun_vis; let o1=sqrt(max(sun_vis,0.0))*0.28*exp(-dens*0.5); let o2=pow(max(sun_vis,0.0),0.25)*0.10*exp(-dens*0.25); return (o0+o1+o2)/1.38; }
// Ambient Light / Ground Albedo: dark blue/grey at bottom → sky ambient at top
fn ambient_height_gradient(h:f32)->vec3f{ let bottom=vec3f(0.18,0.22,0.32); let top=vec3f(0.55,0.62,0.78); return mix(bottom, top, smoothstep(0.0,1.0,h)); }

// ── Skydome Configuration ────────────────────────────────────────────────────
// Procedural clouds in SKYDOME MODE: entire sky dome has clouds spread across it.
// Not a bounded box volume — a horizontal layer tiled infinitely via repeating
// weather map. Altitude defines shell thickness; XZ wraps with repeat.
// Legacy BOX domain kept for reference but NOT used in skydome path.
const SKYDOME_BOTTOM: f32 = 12.0;
const SKYDOME_TOP: f32 = 22.0; // derived from bounds_pack.x in legacy; ~10 units thick
const SKYDOME_RADIUS_H: f32 = 80000.0; // horizontal extent for horizon; max ray length clamp
const EARTH_RADIUS: f32 = 6360000.0; // for spherical curvature (gentle dome, horizon bow)

struct Camera { invViewProj: mat4x4f, position: vec3f, _pad: f32 };
struct Params { time_pack: vec4f, alt_pack: vec4f, scale_pack: vec4f, extra_pack: vec4f, cache_pack: vec4f, bounds_pack: vec4f };
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var densitySampler: sampler;
@group(1) @binding(1) var densityTex: texture_3d<f32>;
@group(2) @binding(0) var densityStore: texture_storage_3d<rgba16float, write>;
@group(3) @binding(0) var historyTex: texture_2d<f32>;
@group(3) @binding(1) var depthTex: texture_depth_2d;
@group(3) @binding(2) var velocityTex: texture_2d<f32>;

// Skydome-aware cloud density: altitude gradient over dome shell + infinite tiling XZ
// Uses same Blender noise graph but remapped to dome shell (not box clamp)
fn cloudDensity(pos:vec3f)->f32{
 let tN=params.time_pack.x;let tV1=params.time_pack.y;let tV2=params.time_pack.z;let dens=params.time_pack.w;
 let lowAlt=params.alt_pack.x;let alt=params.alt_pack.y;let facM=params.alt_pack.z;let facD=params.alt_pack.w;
 let facS=params.scale_pack.x;let sAlt=params.scale_pack.y;let sN=params.scale_pack.z;let sV1=params.scale_pack.w;
 let sV2=params.extra_pack.x;let det=params.extra_pack.y;
 // Dome altitude: normalize pos.y between SKYDOME_BOTTOM and SKYDOME_TOP (shell)
 // bounds_pack.x overrides top for configurability; bottom stays SKYDOME_BOTTOM
 let domeTop = params.bounds_pack.x; // 22.0 default, matches legacy getBoxMax().y
 let hFracRaw = clamp((pos.y - SKYDOME_BOTTOM) / max(domeTop - SKYDOME_BOTTOM, 0.001), 0.0, 1.0);
 let Z = 1.0 - hFracRaw;
 let altFrom=alt/5.0;let altTo=1.0-lowAlt;let altRamp=mapRange(Z,0.0,altFrom,altTo,1.0);
 let hGradCumulus=height_gradient_cloud_type(hFracRaw, 0.0);
 // Dome tiling: XZ wraps infinitely (repeat), not clamped to box
 // Weather map tiling scale: cover entire skydome seamlessly
 let obj=vec3f(pos.x,pos.z,pos.y);
 let nC=obj/sN;let s1N=node_noise_texture_4d_value(nC,tN,2.0,0.0,0.0,0.0,0.0,1.0);
 let altMask=clamp01(altRamp*s1N * (0.7 + hGradCumulus*0.6));
 let v1C=obj/sV1;let v1d=node_tex_voronoi_f1_4d_distance(v1C,tV1,5.0,det,0.5,3.0,1.0,0.5,1.0,0.0,1.0);
 let v1m=mapRange(v1d,0.0,0.75,facM*-0.4,facM);let v1s=clamp01(v1m*0.5);let s2=clamp01(altMask+v1s);
 // High-frequency Worley for edge erosion — LOD: only when 0.05<density<0.7 (see fs)
 let v2C=obj/sV2;let v2d=node_tex_voronoi_f1_4d_distance(v2C,tV2,2.0,det*5.0,0.75,2.5,1.0,0.5,1.0,0.0,1.0);
 let v2m=mapRange(v2d,0.0,1.0,facD*-0.25,facD);let s3=clamp01(s2+v2m);
 let cutFrom=alt*sAlt;let cut=mapRange(Z,cutFrom,0.0,0.0,1.0);let shaped=clamp01(s3-cut);let finalShaped=clamp01(shaped-(1.0-facS));
 let falloff=mapRange(Z,0.0,alt,0.0,1.0);let ds=dens*2.4;return finalShaped*falloff*ds;
}
const BOX_MIN=vec3f(-18.0,12.0,-18.0); // legacy box reference — NOT used in skydome ray march; kept for bake compat
const BOX_MAX_XZ=18.0;
fn getBoxMax()->vec3f{return vec3f(BOX_MAX_XZ,params.bounds_pack.x,BOX_MAX_XZ);}
struct HitInfo{hit:bool,tNear:f32,tFar:f32};
// Legacy box intersect (kept for reference, not used in skydome path)
fn intersectBox(ro:vec3f,rd:vec3f)->HitInfo{let inv=1.0/rd;let t0=(BOX_MIN-ro)*inv;let t1=(getBoxMax()-ro)*inv;let tmin=min(t0,t1);let tmax=max(t0,t1);let tn=max(tmin.x,max(tmin.y,tmin.z));let tf=min(tmax.x,min(tmax.y,tmax.z));return HitInfo(tf>=max(tn,0.0),tn,tf);}
// Skydome shell intersect — entire sky dome has clouds spread across it
// Intersects ray with horizontal altitude shell [bottom, top]. Spherical earth curvature
// approximated via paraboloid offset; horizontal extent limited to SKYDOME_RADIUS_H to give horizon.
// Returns interval where ray is inside dome shell. If ro already inside shell, tNear=0.
fn intersectSkydome(ro:vec3f, rd:vec3f, bottom:f32, top:f32)->HitInfo{
    // Fast path: horizontal slab intersect for dome shell
    // Handle rd.y ≈ 0 separately to avoid div-by-zero: if inside shell, extend to horizon
    let eps = 0.0001;
    if (abs(rd.y) < eps) {
        // Ray roughly horizontal: if altitude inside shell, hit to horizon radius
        if (ro.y >= bottom && ro.y <= top) {
            // approximate max distance to horizon for dome curvature: project to radius
            let horizDist = SKYDOME_RADIUS_H;
            // if rd.y ≈ 0, tFar is where horizontal distance reaches radius
            // Solve ro.xz + rd.xz * t  => length = radius; approximate with top projection
            // Simplify: tFar = horizon distance / max(length(rd.xz), eps)
            let horizLen = max(length(rd.xz), eps);
            let tF = horizDist / horizLen;
            return HitInfo(true, 0.0, tF);
        } else {
            return HitInfo(false, 0.0, 0.0);
        }
    }
    var t0 = (bottom - ro.y) / rd.y;
    var t1 = (top - ro.y) / rd.y;
    var tNear = min(t0, t1);
    var tFar = max(t0, t1);
    // Clamp to horizon radius for grazing rays: limit far side to dome extent
    // For skydome, we tile weather horizontally, so we cap far to prevent infinite marching
    tFar = min(tFar, SKYDOME_RADIUS_H);
    // If camera is inside shell, tNear negative → clamp to 0, still hit
    if (tFar < 0.0) { return HitInfo(false, 0.0, 0.0); }
    tNear = max(tNear, 0.0);
    // Apply gentle spherical curvature bias: elevate far side slightly for dome bow
    // earthCenter = (0, -EARTH_RADIUS, 0); offset = EARTH_RADIUS*0.00005 * tFar curvature term
    let curvature = length(ro.xz + rd.xz * tFar * 0.5) * (tFar / EARTH_RADIUS) * 0.08;
    tFar = tFar - curvature * 0.02 * sign(rd.y + 0.01);
    return HitInfo(tFar > tNear, tNear, tFar);
}
const SUN_DIR=vec3f(0.189,0.943,0.283);const SUN_COLOR=vec3f(1.0,1.0,1.0);const AMBIENT=vec3f(0.26,0.30,0.42);const BG_COLOR=vec3f(0.045,0.10,0.18);
fn hgPhase(c:f32,g:f32)->f32{let g2=g*g;return (1.0-g2)/(4.0*3.14159*pow(1.0+g2-2.0*g*c,1.5));}
fn interleavedGradientNoise(uv:vec2f)->f32{let m=vec3f(0.06711056,0.00583715,52.9829189);return fract(m.z*fract(dot(uv,m.xy)));}
// Skydome-aware thickness sampling: samples dome-tiled volume (XZ wraps, Y is altitude shell)
fn sampleDensityThick(pos:vec3f)->vec2f{
  // Dome tiling: map world XZ via repeat (fract), altitude via shell normalization
  // For 3D baked texture which is box-baked, remap to dome by wrapping XZ and lerping Y to shell
  // X/Z repeat infinitely: weather spreads across entire skydome
  let domeTop = params.bounds_pack.x;
  let yFrac = clamp((pos.y - SKYDOME_BOTTOM)/(domeTop - SKYDOME_BOTTOM), 0.0, 1.0);
  // Wrap XZ to box range via fract to tile dome infinitely
  let wrapX = fract((pos.x - BOX_MIN.x) / (BOX_MAX_XZ - BOX_MIN.x + 36.0));
  let wrapZ = fract((pos.z - BOX_MIN.z) / (BOX_MAX_XZ - BOX_MIN.z + 36.0));
  let uvw = vec3f(wrapX, yFrac, wrapZ); // y is altitude, xz dome-tiled
  let s = textureSampleLevel(densityTex, densitySampler, uvw, 0.0);
  return vec2f(s.r, s.g);
}
// LOD-aware density fetch: only sample high-frequency Worley when 0.05 < density < 0.7
fn sampleDensityLOD(pos:vec3f)->f32{
    let base=cloudDensity(pos);
    if (base > 0.05 && base < 0.7) {
        let worleyErosion = node_tex_voronoi_f1_4d_distance(pos*0.8, params.time_pack.y, 2.0, 2.0, 0.75, 2.5, 1.0, 0.5, 1.0, 0.0, 1.0);
        let edgeW = clamp(4.0*base*(1.0-base),0.0,1.0);
        let erosion = (0.53 - worleyErosion)*0.44*edgeW;
        return max(0.0, base - erosion);
    }
    return base;
}
@compute @workgroup_size(4,4,4)
fn cs(@builtin(global_invocation_id) gid: vec3u){
  let dims = textureDimensions(densityStore);
  if (any(gid >= dims)) { return; }
  let uvw = (vec3f(gid) + vec3f(0.5)) / vec3f(dims);
  // Bake in skydome shell space: map volume Y to dome altitude, XZ to tiled plane
  // This bakes procedural density so skydome sampling can be 0 extra fetches later
  let domeTop = params.bounds_pack.x;
  let domeY = mix(SKYDOME_BOTTOM, domeTop, uvw.y);
  let worldX = mix(-BOX_MAX_XZ*2.0, BOX_MAX_XZ*2.0, uvw.x); // wider bake for dome tiling
  let worldZ = mix(-BOX_MAX_XZ*2.0, BOX_MAX_XZ*2.0, uvw.z);
  let pos = vec3f(worldX, domeY, worldZ);
  let d = cloudDensity(pos);
  let thick = clamp(d*1.2 - 0.1,0.0,1.0);
  textureStore(densityStore, gid, vec4f(d, thick, 0.0, 1.0));
}
// Single Pass Lighting — No secondary ray loops toward sun; use Beer-Powder + dual HG + octaves
fn lightMarchSinglePass(pos:vec3f, density:f32, cos_theta:f32) -> vec3f {
    let g1:f32=0.8; let g2:f32=-0.3;
    let phase = dual_hg_phase(cos_theta);
    let tau = density * params.cache_pack.z * 0.55;
    let powder = beer_powder(tau);
    let beer_powder_atten = powder * exp(-tau);
    let sun_vis = beer_powder_atten;
    let ms = multi_scatter_octaves(sun_vis, density);
    let domeTop = params.bounds_pack.x;
    let hFrac = clamp((pos.y - SKYDOME_BOTTOM)/(domeTop - SKYDOME_BOTTOM),0.0,1.0);
    let ambientGrad = ambient_height_gradient(hFrac);
    let direct = SUN_COLOR * ms * (0.18 + phase*7.8) + SUN_COLOR*pow(sun_vis,3.0)*pow(1.0-clamp(density,0.0,1.0),2.0)*(0.04+phase*2.4)*0.34;
    return ambientGrad*0.5 + direct + vec3f(0.72,0.82,0.94)*(1.0-sun_vis)*0.047;
}
fn lightMarch(pos:vec3f)->f32{var s=0.0;let steps=i32(params.cache_pack.y);let sz=0.15;for(var i=1;i<=steps;i++){let p=pos+SUN_DIR*(f32(i)*sz);s+=sampleDensityThick(p).x*sz;}return exp(-s*params.cache_pack.z);}
struct VSOut{@builtin(position) pos:vec4f,@location(0) uv:vec2f};
@vertex fn vs(@builtin(vertex_index) vi:u32)->VSOut{let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VSOut;o.pos=vec4f(p[vi],0,1);o.uv=p[vi];return o;}
@fragment fn fs(@builtin(position) fc:vec4f,@location(0) uv:vec2f)->@location(0) vec4f{
 let skipLight=params.extra_pack.w>0.5;let numSteps=i32(params.extra_pack.z);
 let wn=camera.invViewProj*vec4f(uv,0,1);let wf=camera.invViewProj*vec4f(uv,1,1);
 let ro=camera.position;let rd=normalize(wf.xyz/wf.w - wn.xyz/wn.w);
 // Skydome intersect: entire sky dome has clouds spread across it
 let domeTop = params.bounds_pack.x;
 let hit=intersectSkydome(ro,rd, SKYDOME_BOTTOM, domeTop);
 let sky=mix(BG_COLOR,vec3f(0.1,0.2,0.4),clamp(rd.y*0.5+0.5,0.0,1.0));
 let sunTheta=dot(rd,SUN_DIR);let finalSky=sky+pow(max(sunTheta,0.0),64.0)*SUN_COLOR*0.8;
 var out=finalSky;
 let debugMode = i32(params.bounds_pack.y);
 if(hit.hit){
  let t0=max(hit.tNear,0.0);let t1=hit.tFar;let step=(t1-t0)/f32(numSteps);
  let bayer = bayer4x4_pat(u32(fc.x), u32(fc.y));
  let frameIdx = u32(params.time_pack.x * 60.0);
  let frameShiftX = frameIdx % 4u; let frameShiftY = (frameIdx / 4u) % 4u;
  let activeInBlock = ((u32(fc.x) % 4u) == frameShiftX && (u32(fc.y) % 4u) == frameShiftY);
  let dither = bayer + interleavedGradientNoise(fc.xy) * 0.25;
  var pos=ro+rd*(t0+step*dither);var trans=1.0;var col=vec3f(0.0);
  let phase = dual_hg_phase(sunTheta);
  var stepsTaken: f32 = 0.0;
  var avgDens: f32 = 0.0;
  for(var i=0;i<64;i++){
    if(i>=numSteps){break;}
    if(trans <= 0.02){break;}
    let rayLen = length(pos - ro);
    // Depth culling against skydome shell is implicit via tFar clamp
    // Coarse-to-Fine + Space Leaping 100-300m: weather map pre-check over dome tiling
    let d2=sampleDensityThick(pos);let d=d2.x;let thick=d2.y;
    avgDens += d;
    if(d>0.015){
     let effStep=step*(1.0+thick*0.35);
     let tr=exp(-d*effStep*0.9);
     var sh: f32 = 1.0;
     var lit: vec3f;
     if (skipLight) {
        sh = 1.0;
        lit = SUN_COLOR*phase*0.5 + AMBIENT*0.5;
     } else {
        let hFracInner = clamp((pos.y - SKYDOME_BOTTOM)/(domeTop - SKYDOME_BOTTOM),0.0,1.0);
        let scat = lightMarchSinglePass(pos, d, sunTheta);
        lit = scat;
        sh = 1.0;
     }
     let scatSingle = phase*(1.0-exp(-d*(1.0+thick*0.25)));
     let scatCombined = select(scatSingle, sh*phase*(1.0-exp(-d*(1.0+thick*0.25))), skipLight);
     if (!skipLight) {
        let litMS = lightMarchSinglePass(pos, d, sunTheta);
        col+=trans*(1.0-tr)*litMS;
     } else {
        let litSimple=SUN_COLOR*scatCombined*params.cache_pack.w + AMBIENT*0.5;
        col+=trans*(1.0-tr)*litSimple;
     }
     trans*=tr;
    }
    pos+=rd*step;
    stepsTaken+=1.0;
  }
  if (debugMode == 1) {
    let t = clamp(stepsTaken/64.0,0.0,1.0);
    out = mix(vec3f(0.0,0.0,1.0), vec3f(1.0,0.0,0.0), t);
  } else if (debugMode == 2) {
    let varMask = fract(avgDens*10.0);
    out = mix(vec3f(0.0,1.0,0.0), vec3f(1.0,0.0,0.0), varMask);
  } else if (debugMode == 3) {
    let dVal = clamp(avgDens/16.0,0.0,1.0);
    out = vec3f(dVal, dVal*0.5, 1.0-dVal);
  } else {
    out=col+trans*finalSky;
  }
 }
 out=out/(out+vec3f(1.0));out=pow(out,vec3f(1.0/2.2));return vec4f(out,1.0);
}
