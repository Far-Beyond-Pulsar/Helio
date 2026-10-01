// GPU view scoring, bounded selection, persistent tile allocation and commit.
struct GpuLight {
    position_range:   vec4f,
    direction_outer:  vec4f,
    color_intensity:  vec4f,
    shadow_index:     u32,
    light_type:       u32,
    inner_angle:      f32,
    _pad:             u32,
    god_rays_enabled:  u32,
    god_rays_density:  f32,
    god_rays_weight:   f32,
    god_rays_decay:    f32,
    god_rays_exposure: f32,
    flare_enabled:      u32,
    flare_type:         u32,
    flare_intensity:    f32,
    flare_scale:        f32,
    flare_tint_r:       f32,
    flare_tint_g:       f32,
    flare_tint_b:       f32,
    ies_profile_index:    i32,
    light_function_index: i32,
    ies_angle_scale:      f32,
    ies_angle_offset:     f32,
}


struct Params {
    rows:u32, capacity:u32, nonce:u32, atlas_size:u32,
    view_proj:mat4x4f, inv_view_proj:mat4x4f,
    camera:vec4f, // xyz position, w max shadow distance
    tuning:vec4f, // viewport height, max resolution, hysteresis, unused
}
struct Candidate { key:u32, resolution:u32, mask:u32, fade:f32 }
struct Resident {
    owner:u32, kind:u32, flags:u32, target:u32,
    score:f32, strength:f32, resolution:u32, hash:u32,
    tiles:array<vec4u,6>,
}
struct Table { header:vec4u, slots:array<Resident,256> }
@group(0) @binding(0) var<storage,read_write> lights:array<GpuLight>;
@group(0) @binding(1) var<uniform> params:Params;
@group(0) @binding(2) var<storage,read_write> proposed:Table;
@group(0) @binding(3) var<storage,read> active:Table;
@group(0) @binding(4) var<storage,read_write> candidates:array<Candidate>;
const NONE:u32=0xffffffffu;
fn requested(l:GpuLight)->bool {
    return select(l.shadow_index!=NONE,(l._pad&4u)!=0u,(l._pad&8u)!=0u);
}
fn plane_outside(p:vec4f,c:vec3f,r:f32)->bool { return dot(p,vec4f(c,1.0)) < -r*length(p.xyz); }
fn visible(l:GpuLight)->bool {
    let m=transpose(params.view_proj);
    let c=l.position_range.xyz;
    let r=max(l.position_range.w,0.0);
    return !(plane_outside(m[3]+m[0],c,r)||plane_outside(m[3]-m[0],c,r)
        ||plane_outside(m[3]+m[1],c,r)||plane_outside(m[3]-m[1],c,r)
        ||plane_outside(m[2],c,r)||plane_outside(m[3]-m[2],c,r));
}
fn point_mask(pos:vec3f)->u32 {
    // Conservative SAT: reject a cube-face pyramid only when a plane
    // separates all eight receiver-frustum vertices from that pyramid.
    var dirs=array<vec3f,6>(vec3f(1,0,0),vec3f(-1,0,0),vec3f(0,1,0),vec3f(0,-1,0),vec3f(0,0,1),vec3f(0,0,-1));
    var mask=0u;
    for(var f=0u;f<6u;f++) {
        let d=dirs[f];
        let a=select(vec3f(1,0,0),vec3f(0,1,0),f<2u);
        let b=cross(d,a);
        var planes=array<vec3f,5>(d,d+a,d-a,d+b,d-b);
        var keep=true;
        for(var p=0u;p<5u;p++) {
            var maximum=-3.0e38;
            for(var c=0u;c<8u;c++) {
                let clip=vec4f(select(-1.0,1.0,(c&1u)!=0u),select(-1.0,1.0,(c&2u)!=0u),select(0.0,1.0,(c&4u)!=0u),1.0);
                let h=params.inv_view_proj*clip;
                let world=h.xyz/select(1e-6,h.w,abs(h.w)>1e-6);
                maximum=max(maximum,dot(planes[p],world-pos));
            }
            if maximum < -0.001 { keep=false; }
        }
        if keep { mask|=1u<<f; }
    }
    return mask;
}
@compute @workgroup_size(64)
fn score_lights(@builtin(global_invocation_id) gid:vec3u) {
    let i=gid.x;
    if i>=min(params.rows,arrayLength(&lights)) { return; }
    candidates[i]=Candidate(0u,0u,0u,0.0);
    let l=lights[i];
    if !(l.color_intensity.w>0.0)||!requested(l)||(l._pad&48u)==48u { return; }
    if l.light_type!=0u&&!visible(l) { return; }
    let distance=length(l.position_range.xyz-params.camera.xyz);
    let fade=select(1.0-smoothstep(params.camera.w*0.8,params.camera.w,max(0.0,distance-l.position_range.w)),1.0,l.light_type==0u);
    if fade<=0.0 { return; }
    let clip=params.view_proj*vec4f(l.position_range.xyz,1);
    let rows=transpose(params.view_proj);
    let projection_scale=max(length(rows[0].xyz),length(rows[1].xyz));
    let radius=projection_scale*max(l.position_range.w,0.0)/max(abs(clip.w),0.01);
    let coverage=select(clamp(radius*radius,0.000001,1.0),1.0,l.light_type==0u);
    let priority=select(1.0,f32((l._pad>>8u)&255u)/16.0,((l._pad>>8u)&255u)!=0u);
    var score=coverage*fade*priority;
    let incumbent=(l._pad&8u)!=0u && l.shadow_index!=NONE;
    score*=select(1.0,1.0+params.tuning.z,incumbent);
    let max_log=(l._pad>>16u)&15u;
    let limit=min(u32(params.tuning.y),select(2048u,1u<<max_log,max_log!=0u));
    let pixels=sqrt(coverage)*params.tuning.x;
    var res=128u;
    for(var t=0u;t<4u;t++) { if f32(res)<pixels && res*2u<=limit { res*=2u; } }
    var mask=1u;
    if l.light_type==0u { mask=15u; }
    if l.light_type==1u { mask=point_mask(l.position_range.xyz); }
    if mask==0u { return; }
    candidates[i]=Candidate(bitcast<u32>(score),res,mask,fade);
}
var<workgroup> histogram:array<atomic<u32>,256>;
var<workgroup> offsets:array<u32,256>;
var<workgroup> threshold:u32;
var<workgroup> threshold_mask:u32;
var<workgroup> needed:u32;
var<workgroup> take_all:u32;
fn scan(lid:u32)->u32 {
    workgroupBarrier();
    if lid==0u { var sum=0u; for(var j=0u;j<256u;j++){let n=offsets[j];offsets[j]=sum;sum+=n;} }
    workgroupBarrier();
    return offsets[lid];
}
@compute @workgroup_size(256)
fn select_lights(@builtin(local_invocation_index) lid:u32) {
    let n=min(params.rows,arrayLength(&candidates));
    let chunk=(n+255u)/256u;
    let begin=min(lid*chunk,n); let end=min(begin+chunk,n);
    if lid==0u { threshold=0u;threshold_mask=0u;needed=params.capacity;take_all=0u; }
    for(var digit=0u;digit<4u;digit++) {
        let shift=24u-8u*digit;
        atomicStore(&histogram[lid],0u);workgroupBarrier();
        let prefix=threshold;let mask=threshold_mask;
        for(var i=begin;i<end;i++) { let k=candidates[i].key;if k!=0u&&(k&mask)==prefix { atomicAdd(&histogram[(k>>shift)&255u],1u); } }
        workgroupBarrier();
        if lid==0u&&take_all==0u {
            if digit==0u { var total=0u;for(var b=0u;b<256u;b++){total+=atomicLoad(&histogram[b]);} if total<=needed {take_all=1u;} }
            if take_all==0u {
                var above=0u;var bucket=255u;
                for(var b=0u;b<256u;b++){bucket=255u-b;let count=atomicLoad(&histogram[bucket]);if above+count>=needed {break;} above+=count;}
                needed-=min(above,needed);threshold|=bucket<<shift;threshold_mask|=255u<<shift;
            }
        }
        workgroupBarrier();
    }
    var ties=0u;
    for(var i=begin;i<end;i++){if candidates[i].key!=0u&&candidates[i].key==threshold{ties++;}}
    offsets[lid]=ties;let tie_base=scan(lid);
    var tie_rank=tie_base;var winners=0u;
    for(var i=begin;i<end;i++) {
        let k=candidates[i].key;if k==0u{continue;}
        var win=take_all!=0u||k>threshold;
        if !win&&k==threshold {win=tie_rank<needed;tie_rank++;}
        if win {winners++;}
    }
    workgroupBarrier();offsets[lid]=winners;var slot=scan(lid);tie_rank=tie_base;
    for(var i=begin;i<end;i++) {
        let k=candidates[i].key;if k==0u{continue;}
        var win=take_all!=0u||k>threshold;
        if !win&&k==threshold {win=tie_rank<needed;tie_rank++;}
        if win&&slot<params.capacity {proposed.slots[slot].owner=i+1u;slot++;}
    }
    if lid==255u {proposed.header=vec4u(min(slot,params.capacity),params.nonce,0u,0u);}
}
var<private> occupied:array<u32,4096>;
var<private> selected:array<u32,256>;
fn mark(tile:vec4u) {
    let cells=params.atlas_size/128u;
    for(var y=tile.y/128u;y<(tile.y+tile.z)/128u;y++) {
        for(var x=tile.x/128u;x<(tile.x+tile.z)/128u;x++) {occupied[y*cells+x]=1u;}
    }
}
fn allocate_tile(wanted:u32)->vec4u {
    let cells=params.atlas_size/128u;
    var size=min(wanted,params.atlas_size);
    for(var tier=0u;tier<5u;tier++) {
        let step=max(size/128u,1u);
        for(var y=0u;y+step<=cells;y+=step) { for(var x=0u;x+step<=cells;x+=step) {
            var free=true;
            for(var dy=0u;dy<step;dy++){for(var dx=0u;dx<step;dx++){if occupied[(y+dy)*cells+x+dx]!=0u {free=false;}}}
            if free {let tile=vec4u(x*128u,y*128u,size,0u);mark(tile);return tile;}
        }}
        if size<=128u {break;} size/=2u;
    }
    return vec4u(0u);
}
@compute @workgroup_size(1)
fn pack_tiles() {
    let count=proposed.header.x;
    for(var s=0u;s<count;s++){selected[s]=proposed.slots[s].owner;}
    // Stable deterministic priority order, bounded by the resident capacity.
    for(var i=1u;i<count;i++){let owner=selected[i];var j=i;
        for(var k=0u;k<256u;k++){if j==0u{break;}let prev=selected[j-1u];
            let a=candidates[owner-1u].key;let b=candidates[prev-1u].key;
            if a<b||(a==b&&owner>prev){break;}selected[j]=prev;j--;}
        selected[j]=owner;
    }
    var empty:Resident;
    for(var s=0u;s<params.capacity;s++) {
        var r=active.slots[s];var keep=false;
        for(var i=0u;i<count;i++){if r.owner!=0u&&selected[i]==r.owner {keep=true;}}
        r.target=0u;
        if r.owner>0u&&r.owner<=params.rows&&keep {
            let c=candidates[r.owner-1u];r.score=bitcast<f32>(c.key);r.flags=lights[r.owner-1u]._pad;
            // Fade before reallocating tiers; the old tile remains valid until zero.
            if c.resolution==r.resolution || abs(f32(c.resolution)-f32(r.resolution))<=f32(r.resolution)*params.tuning.z {
                r.target=u32(c.fade*65535.0);
            }
        }
        if r.target==0u&&r.strength<=0.0 {r=empty;}
        proposed.slots[s]=r;
        if r.owner!=0u {for(var f=0u;f<6u;f++){mark(r.tiles[f]);}}
    }
    for(var i=0u;i<count;i++) {
        let owner=selected[i];var slot=NONE;var found=false;
        for(var s=0u;s<params.capacity;s++) {
            if proposed.slots[s].owner==owner {slot=s;found=true;break;}
            if slot==NONE&&proposed.slots[s].owner==0u {slot=s;}
        }
        if slot==NONE {continue;}
        let c=candidates[owner-1u];let l=lights[owner-1u];
        if !found {proposed.slots[slot]=empty;proposed.slots[slot].owner=owner;
            proposed.slots[slot].kind=l.light_type;proposed.slots[slot].flags=l._pad;
            proposed.slots[slot].target=u32(c.fade*65535.0);proposed.slots[slot].resolution=c.resolution;
            proposed.slots[slot].score=bitcast<f32>(c.key);}
        for(var f=0u;f<6u;f++) {
            if (c.mask&(1u<<f))!=0u && proposed.slots[slot].tiles[f].z==0u {
                proposed.slots[slot].tiles[f]=allocate_tile(proposed.slots[slot].resolution);
            }
        }
    }
    proposed.header.y=params.nonce;
}
@compute @workgroup_size(64)
fn commit_lights(@builtin(global_invocation_id) gid:vec3u) {
    let i=gid.x;if i>=min(params.rows,arrayLength(&lights)){return;}
    let l=lights[i];if !(l.color_intensity.w>0.0){return;}
    let wants=requested(l);var flags=(l._pad&~4u)|8u|select(0u,4u,wants);
    if (flags&1u)==0u {flags|=1u|select(0u,2u,wants);}
    lights[i]._pad=flags;
    var index=NONE;
    if wants {for(var s=0u;s<params.capacity;s++){if active.slots[s].owner==i+1u {index=s*6u;break;}}}
    lights[i].shadow_index=index;
}
