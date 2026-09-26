// Generated bricks are immutable while a published tree references them.
// Occupancy generation is separate from traversal. Material shading is bounded
// per visible pixel; neither camera nor sunlight traversal evaluates the recipe.
struct StoredNode { low:vec3<i32>,level:u32,child:u32 }
struct BrickJob { low:vec3<i32>,level:u32,slot:u32,pad0:u32,pad1:u32,pad2:u32 }
// Four-word root header (low.xyz, level), followed by one child link per
// node. Bounds and levels are implicit in the complete octree topology.
@group(0) @binding(24) var<storage,read> stored_nodes:array<u32>;
@group(0) @binding(25) var<storage,read_write> stored_materials:array<u32>;
@group(0) @binding(26) var<storage,read> brick_jobs:array<BrickJob>;
@group(0) @binding(27) var<storage,read> brick_edits:array<u32>;

@compute @workgroup_size(64)
fn generate_bricks(@builtin(global_invocation_id) id:vec3<u32>) {
    let job=brick_jobs[id.y];let stride=i32(1u<<job.level);
    if job.level>0u {
        if id.x>=729u {return;}
        let q=vec3<i32>(i32(id.x%9u),i32((id.x/9u)%9u),i32(id.x/81u));
        let c=voxel_sample(job.low+q*(stride*4),i32(max(job.pad2,1u)));
        var density=base_sample(c,vec3<f32>(0.0)).density;var material=1u;
        for(var e=0u;e<job.pad1;e++) {
            let edit=edits[brick_edits[job.pad0+e]];
            let brush=f32(edit.radius_units)*0.05-length(vec3<f32>(c-edit.cell))*0.1;
            if edit.material==0u {density=min(density,-brush);}
            else {density=max(density,brush);if brush>=0.0 {material=edit.material;}}
        }
        stored_materials[job.slot*2048u+id.x]=(bitcast<u32>(density)&0xfffffffcu)|material;
        return;
    }
    var word=0u;
    for(var v=0u;v<16u;v++) {
        let i=id.x*16u+v;
        let q=vec3<i32>(i32(i%32u),i32((i/32u)%32u),i32(i/1024u));
        let c=voxel_sample(job.low+q*stride+vec3<i32>((stride-1)/2),i32(max(job.pad2,1u)));
        var material=0u;var replaced=false;var remaining=job.pad1;
        while remaining>0u {
            remaining-=1u;let edit=edits[brick_edits[job.pad0+remaining]];
            if edit_contains(c,edit) {material=edit.material;replaced=true;break;}
        }
        if !replaced {material=u32(base_sample(c,vec3<f32>(0.0)).solid);}
        word|=(material&3u)<<(v*2u);
    }
    stored_materials[job.slot*2048u+id.x]=word;
}

// After generation, cache one maximum per interpolation cell and per brick.
// Negative maxima certify empty space independently of ray direction.
var<workgroup> brick_maxima:array<f32,64>;
@compute @workgroup_size(64)
fn bound_bricks(@builtin(global_invocation_id) id:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let job=brick_jobs[id.y];
    if group.x==0u {
        var maximum=-3.402823e38;
        if job.level>0u {
            for(var i=lane;i<729u;i+=64u) {
                maximum=max(maximum,bitcast<f32>(stored_materials[job.slot*2048u+i]&0xfffffffcu));
            }
        }
        brick_maxima[lane]=maximum;workgroupBarrier();
        for(var offset=32u;offset>0u;offset/=2u) {
            if lane<offset {brick_maxima[lane]=max(brick_maxima[lane],brick_maxima[lane+offset]);}
            workgroupBarrier();
        }
        if lane==0u && job.level>0u {stored_materials[job.slot*2048u+1241u]=bitcast<u32>(brick_maxima[0]);}
    }
    if job.level==0u {return;}
    let q=vec3<i32>(i32(id.x%8u),i32((id.x/8u)%8u),i32(id.x/64u));
    let offset=job.slot*2048u+u32(q.x+q.y*9+q.z*81);
    let a=vec4<u32>(stored_materials[offset],stored_materials[offset+1u],stored_materials[offset+9u],stored_materials[offset+10u]);
    let b=vec4<u32>(stored_materials[offset+81u],stored_materials[offset+82u],stored_materials[offset+90u],stored_materials[offset+91u]);
    let v0=bitcast<vec4<f32>>(a&vec4<u32>(0xfffffffcu));
    let v1=bitcast<vec4<f32>>(b&vec4<u32>(0xfffffffcu));
    let maxima=max(v0,v1);
    stored_materials[job.slot*2048u+729u+id.x]=bitcast<u32>(max(max(maxima.x,maxima.y),max(maxima.z,maxima.w)));
}

struct StoredCamera {
    view:mat4x4<f32>,proj:mat4x4<f32>,view_proj:mat4x4<f32>,inv_view_proj:mat4x4<f32>,
    position_near:vec4<f32>,forward_far:vec4<f32>,jitter_frame:vec4<f32>,prev_view_proj:mat4x4<f32>,
}
@group(1) @binding(0) var<storage,read> stored_cameras:array<StoredCamera>;
fn stored_ray(pixel:vec2<u32>)->vec3<f32> {
    let camera=stored_cameras[0];let uv=(vec2<f32>(pixel)+0.5)/p.screen.xy;
    let ndc=vec2<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0)-camera.jitter_frame.xy;
    let rotation=mat3x3<f32>(camera.view[0].xyz,camera.view[1].xyz,camera.view[2].xyz);
    return normalize(transpose(rotation)*vec3<f32>(ndc.x/camera.proj[0][0],ndc.y/camera.proj[1][1],-1.0));
}
fn stored_face(normal:vec3<f32>)->u32 {
    var face=1u;for(var a=0u;a<3u;a++){if abs(normal[a])>0.5 {face=1u+a*2u+u32(normal[a]<0.0);}}return face;
}
fn stored_normal(status:u32)->vec3<f32> {
    let face=(status>>28u)&7u;var normal=vec3<f32>(0.0);
    if face>0u && face<=6u {normal[(face-1u)/2u]=select(1.0,-1.0,(face&1u)==0u);}return normal;
}
struct BoxInterval { near:f32,far:f32,normal:vec3<f32> }
fn stored_box(lo:vec3<f32>,size:f32,rd:vec3<f32>)->BoxInterval {
    let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
    let a=lo*inverse;let b=(lo+size)*inverse;
    let first=min(a,b);let last=max(a,b);
    var axis=0u;if first.y>first.x {axis=1u;}if first.z>first[axis] {axis=2u;}
    var normal=vec3<f32>(0.0);normal[axis]=select(1.0,-1.0,rd[axis]>=0.0);
    return BoxInterval(first[axis],min(last.x,min(last.y,last.z)),normal);
}
fn stored_material(slot:u32,q:vec3<i32>)->u32 {
    let i=u32(q.x+q.y*32+q.z*1024);
    return (stored_materials[slot*2048u+i/16u]>>((i%16u)*2u))&3u;
}
// Cache one interpolation cell in registers while walking its 10 cm voxels.
struct DensityGrid { low:vec4<f32>, high:vec4<f32>, gradient:vec3<f32>, rate:f32, q:vec3<i32> }
fn stored_density_grid(n:StoredNode,q:vec3<i32>,rd:vec3<f32>)->DensityGrid {
    let offset=(n.child&0x7fffffffu)*2048u+u32(q.x+q.y*9+q.z*81);
    let a=vec4<u32>(stored_materials[offset],stored_materials[offset+1u],stored_materials[offset+9u],stored_materials[offset+10u]);
    let b=vec4<u32>(stored_materials[offset+81u],stored_materials[offset+82u],stored_materials[offset+90u],stored_materials[offset+91u]);
    let v0=bitcast<vec4<f32>>(a&vec4<u32>(0xfffffffcu));
    let v1=bitcast<vec4<f32>>(b&vec4<u32>(0xfffffffcu));
    let dx=vec4<f32>(v0.y-v0.x,v0.w-v0.z,v1.y-v1.x,v1.w-v1.z);
    let dy=vec4<f32>(v0.z-v0.x,v0.w-v0.y,v1.z-v1.x,v1.w-v1.y);
    let dz=v1-v0;
    let lower=vec3<f32>(min(min(dx.x,dx.y),min(dx.z,dx.w)),min(min(dy.x,dy.y),min(dy.z,dy.w)),min(min(dz.x,dz.y),min(dz.z,dz.w)));
    let upper=vec3<f32>(max(max(dx.x,dx.y),max(dx.z,dx.w)),max(max(dy.x,dy.y),max(dy.z,dy.w)),max(max(dz.x,dz.y),max(dz.z,dz.w)));
    let scale=f32(4u<<n.level)*0.1;
    let gradient=max(abs(lower),abs(upper))/scale*1.00001;
    // Each partial derivative is bilinear and bounded by its four corners.
    // Preserve signs before summing: tangential rays can cancel large partials.
    let rate=dot(max(lower*rd,upper*rd),vec3<f32>(1.0))/scale+dot(gradient,abs(rd))*0.00001;
    return DensityGrid(v0,v1,gradient,max(rate,0.000001),q);
}
fn stored_density_value(grid:DensityGrid,f:vec3<f32>)->f32 {
    let x=mix(vec4<f32>(grid.low.xz,grid.high.xz),vec4<f32>(grid.low.yw,grid.high.yw),f.x);
    let y=mix(x.xz,x.yw,f.y);
    return mix(y.x,y.y,f.z);
}
fn stored_density_hit(n:StoredNode,ro:vec3<f32>,rd:vec3<f32>,start:f32,end:f32)->Hit {
    let base_step=i32(max(p.settings.w,1.0));
    // The traversal parameter is local to this brick. At orbital distances,
    // adding 10 cm to a camera-distance f32 can be a no-op.
    let entry=p.fraction.xyz+(ro+rd*start)*10.0;
    let stride=i32(1u<<n.level);let high=n.low+vec3<i32>(32*stride);
    let requested_anchor=p.origin.xyz+vec3<i32>(floor(entry));
    let anchor=clamp(requested_anchor,n.low,high-1);
    // Orbital f32 rounding may put the entry one cell outside its selected
    // brick. Project that approximate entry inside, before local traversal.
    let fraction=select(fract(entry),vec3<f32>(0.5),anchor!=requested_anchor);
    var t=0.0;
    var cell=anchor+vec3<i32>(floor(fraction+rd*0.00002));
    cell=clamp(cell,n.low,high-1);
    let step=select(vec3<i32>(-1),vec3<i32>(1),rd>=vec3<f32>(0.0));
    let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
    let spacing=f32(stride*4);
    var grid:DensityGrid;grid.q=vec3<i32>(-1);var grid_exit=0.0;
    for(var iteration=0u;iteration<16384u;iteration++) {
        let sampled=voxel_sample(cell,base_step);
        let position=vec3<f32>(sampled-n.low)/spacing;
        // Integer ownership matters at a negative crossing: converting a
        // multi-million-cell offset to f32 can round boundary-1 back up.
        let q=(cell-n.low)>>vec3<u32>(n.level+2u);
        if any(grid.q!=q) {
            let grid_low=n.low+q*(stride*4);
            let grid_lo=(vec3<f32>(grid_low-anchor)-fraction)*0.1;
            let grid_size=f32(stride*4)*0.1;
            let metadata=(n.child&0x7fffffffu)*2048u+729u+u32(q.x+q.y*8+q.z*64);
            // Trilinear weights are nonnegative: eight negative corners certify
            // every 10 cm sample inside this interpolation cell as empty.
            if bitcast<f32>(stored_materials[metadata])<0.0 {
                let exits=max(grid_lo*inverse,(grid_lo+grid_size)*inverse);
                var axis=0u;if exits.y<exits.x {axis=1u;}if exits.z<exits[axis] {axis=2u;}
                t=max(t,exits[axis]);
                let next_cell=anchor+vec3<i32>(floor(fraction+rd*t*10.0));
                cell=select(min(next_cell,cell),max(next_cell,cell),step>vec3<i32>(0));
                cell[axis]=grid_low[axis]+select(-1,stride*4,step[axis]>0);
                if t>end-start || any(cell<n.low) || any(cell>=high) {return Hit(vec3<i32>(0),0u,rd,end);}
                continue;
            }
            grid=stored_density_grid(n,q,rd);
            grid_exit=stored_box(grid_lo,grid_size,rd).far;
        }
        let f=clamp(position-vec3<f32>(q),vec3<f32>(0.0),vec3<f32>(1.0));
        let value=stored_density_value(grid,f);
        if value>=0.0 {
            let material_q=q+vec3<i32>(f>=vec3<f32>(0.5));
            let material=stored_materials[(n.child&0x7fffffffu)*2048u+u32(material_q.x+material_q.y*9+material_q.z*81)]&3u;
            // Enter a real 10 cm cube, even when its occupancy comes from the
            // distant density approximation. Coarse sample spacing is never a
            // rendered cube size.
            let low=(vec3<f32>(voxel_low(cell,base_step)-anchor)-fraction)*0.1;
            let bounds=stored_box(low,f32(base_step)*0.1,rd);
            return Hit(sampled,0x80000001u|(n.level<<2u)|(material<<8u)|(stored_face(bounds.normal)<<28u),rd,start+max(0.0,bounds.near));
        }
        let sample_position=(vec3<f32>(cell-anchor)-fraction)*0.1;
        // Bound movement along this ray, including the displacement from the
        // sampled voxel corner and the next corner's quantization error.
        let offset=rd*t-sample_position;
        let error=dot(grid.gradient,max(abs(offset),abs(offset-vec3<f32>(f32(base_step)*0.1+0.000001))));
        let safe=((-value-error)/grid.rate)*0.95;
        // The derivative bound applies only inside this interpolation cell.
        let exit_margin=max(0.09,abs(grid_exit)*0.0000002);
        var advance=min(safe,max(0.0,grid_exit-t-exit_margin));
        var next_t=t+advance;
        var next_cell=anchor+vec3<i32>(floor(fraction+rd*next_t*10.0));
        // If rounding crosses the certified region, shorten the same safe
        // jump. Rejecting it outright can force a long voxel-by-voxel walk.
        if any(((next_cell-n.low)>>vec3<u32>(n.level+2u))!=q) {
            advance*=0.5;next_t=t+advance;
            next_cell=anchor+vec3<i32>(floor(fraction+rd*next_t*10.0));
        }
        // Even a brick can span millions of metres. A rounded floating step
        // must never retry the same state; integer DDA guarantees progress.
        let next_q=(next_cell-n.low)>>vec3<u32>(n.level+2u);
        if advance>0.1 && next_t>t && any(next_cell!=cell) && all(next_q==q) && all((next_cell-cell)*step>=vec3<i32>(0)) {
            t=next_t;
            cell=next_cell;
        } else {
            let boundary=cell+select(vec3<i32>(0),vec3<i32>(1),rd>=vec3<f32>(0.0));
            let next=((vec3<f32>(boundary-anchor)-fraction)*0.1)*inverse;
            var axis=0u;if next.y<next.x {axis=1u;}if next.z<next[axis] {axis=2u;}
            t=max(t,next[axis]);cell[axis]+=step[axis];
        }
        if t>end-start || any(cell<n.low) || any(cell>=high) {return Hit(vec3<i32>(0),0u,rd,end);}
    }
    return Hit(cell,2u,rd,start+t);
}
// Camera-relative origin is in metres, keeping exact nearby addresses out of
// Earth-sized floats. Coarse distant cells have a bounded projected size.
fn stored_trace(ro:vec3<f32>,rd:vec3<f32>,maximum:f32)->Hit {
    let root=StoredNode(vec3<i32>(bitcast<i32>(stored_nodes[0]),bitcast<i32>(stored_nodes[1]),bitcast<i32>(stored_nodes[2])),stored_nodes[3],stored_nodes[4]);
    let root_lo=(vec3<f32>(root.low-p.origin.xyz)-p.fraction.xyz)*0.1-ro;
    let root_hit=stored_box(root_lo,f32(32u<<root.level)*0.1,rd);
    let limit=min(maximum,root_hit.far);var t=max(0.0,root_hit.near);
    var normal=root_hit.normal;
    if limit<t {return Hit(vec3<i32>(0),0u,rd,t);}
    var cell=p.origin.xyz+vec3<i32>(floor(p.fraction.xyz+(ro+rd*(t+max(0.000002,abs(t)*0.0000002)))*10.0));
    var node_index=0u;var node_level=root.level;
    var ancestors:array<u32,28>;
    for(var visited=0u;visited<2048u;visited++) {
        let epsilon=max(0.000002,abs(t)*0.0000002);
        if any(cell<root.low) || any(cell>=root.low+vec3<i32>(i32(32u<<root.level))) {return Hit(vec3<i32>(0),0u,rd,t);}
        // Retain the descent path. Adjacent bricks usually share almost all
        // ancestors; restarting at the planet root would repeat those reads.
        let relative=vec3<u32>(cell-root.low);
        var child=stored_nodes[4u+node_index];
        for(var depth=0u;depth<28u;depth++) {
            ancestors[node_level]=node_index;
            if (child&0x80000000u)!=0u {break;}
            let octant=(relative>>vec3<u32>(node_level+4u))&vec3<u32>(1u);
            node_index=child+octant.x+octant.y*2u+octant.z*4u;
            node_level-=1u;child=stored_nodes[4u+node_index];
        }
        let shift=vec3<u32>(node_level+5u);
        let low=root.low+vec3<i32>((relative>>shift)<<shift);
        let n=StoredNode(low,node_level,child);
        let lo=(vec3<f32>(n.low-p.origin.xyz)-p.fraction.xyz)*0.1-ro;
        let voxel=f32(1u<<n.level)*0.1;
        let box=stored_box(lo,32.0*voxel,rd);
        if n.child==0xfffffffeu {
            return Hit(cell,0x80000101u|(stored_face(normal)<<28u),rd,t);
        }
        if n.child!=0xffffffffu && n.level>0u {
            // The recipe's conservative bounds may select an empty brick.
            // Once generated, its corner maximum certifies the whole field.
            if bitcast<f32>(stored_materials[(n.child&0x7fffffffu)*2048u+1241u])>=0.0 {
                let hit=stored_density_hit(n,ro,rd,t,min(box.far,limit));
                if (hit.status&3u)!=0u {return hit;}
            }
            t=box.far;
        } else if n.child!=0xffffffffu {
            var q=clamp(vec3<i32>(floor((rd*(t+epsilon)-lo)/voxel)),vec3<i32>(0),vec3<i32>(31));
            let step=select(vec3<i32>(-1),vec3<i32>(1),rd>=vec3<f32>(0.0));
            let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
            var next=(lo+(vec3<f32>(q)+select(vec3<f32>(0.0),vec3<f32>(1.0),rd>=vec3<f32>(0.0)))*voxel)*inverse;
            let stride=voxel*abs(inverse);
            for(var crossing=0u;crossing<97u;crossing++) {
                let material=stored_material(n.child&0x7fffffffu,q);
                if material!=0u {
                    let hit_cell=voxel_sample(n.low+q*i32(1u<<n.level),i32(max(p.settings.w,1.0)));
                    return Hit(hit_cell,0x80000001u|(n.level<<2u)|(material<<8u)|(stored_face(normal)<<28u),rd,t);
                }
                var axis=0u;if next.y<next.x {axis=1u;}if next.z<next[axis] {axis=2u;}
                t=next[axis];next[axis]+=stride[axis];q[axis]+=step[axis];
                normal=vec3<f32>(0.0);normal[axis]=-f32(step[axis]);
                if any(q<vec3<i32>(0)) || any(q>=vec3<i32>(32)) || t>limit {break;}
            }
        }
        if box.far>=limit {return Hit(vec3<i32>(0),0u,rd,limit);}
        // Compute the outgoing face without accumulating distance increments.
        let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
        let exits=max(lo*inverse,(lo+32.0*voxel)*inverse);
        var axis=0u;if exits.y<exits.x {axis=1u;}if exits.z<exits[axis] {axis=2u;}
        normal=vec3<f32>(0.0);normal[axis]=select(1.0,-1.0,rd[axis]>=0.0);
        t=max(box.far,t);
        if t>=limit {return Hit(vec3<i32>(0),0u,rd,limit);}
        let previous_cell=cell;
        cell=p.origin.xyz+vec3<i32>(floor(p.fraction.xyz+(ro+rd*(t+max(0.000002,abs(t)*0.0000002)))*10.0));
        // Crossing ownership is integer state. A rounded ray position must not
        // select the same leaf repeatedly at a boundary.
        cell[axis]=n.low[axis]+select(-1,i32(32u<<n.level),rd[axis]>=0.0);
        if any(cell<root.low) || any(cell>=root.low+vec3<i32>(i32(32u<<root.level))) {return Hit(vec3<i32>(0),0u,rd,t);}
        let changed=vec3<u32>(cell-root.low)^vec3<u32>(previous_cell-root.low);
        let parent_level=firstLeadingBit(changed.x|changed.y|changed.z)-4u;
        node_level=min(root.level,max(n.level+1u,parent_level));
        node_index=ancestors[node_level];
    }
    return Hit(vec3<i32>(0),2u,rd,t);
}
@compute @workgroup_size(8,8)
fn stored_primary(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=vec2<u32>(p.screen.xy)) {return;}
    let rd=stored_ray(id.xy);var hit=Hit(vec3<i32>(0),3u,rd,0.0);
    if p.settings.z>0.0 {hit=stored_trace(vec3<f32>(0.0),rd,p.settings.x);}
    primary_hits[id.x+id.y*u32(p.screen.x)]=hit;
}
@vertex fn stored_fullscreen(@builtin(vertex_index) v:u32)->@builtin(position) vec4<f32> {
    return vec4<f32>(f32((v<<1u)&2u)*2.0-1.0,1.0-f32(v&2u)*2.0,0.0,1.0);
}
struct StoredSurface {
    @location(0) albedo:vec4<f32>,@location(1) normal:vec4<f32>,
    @location(2) orm:vec4<f32>,@location(3) emissive:vec4<f32>,
    @location(4) lightmap:vec2<f32>,@location(5) sss:vec4<f32>,
    @location(6) extra:vec4<f32>,@location(7) velocity:vec4<f32>,@builtin(frag_depth) depth:f32,
}
override STORED_REVERSE_DEPTH:bool=false;
@fragment fn stored_surface(@builtin(position) pixel:vec4<f32>)->StoredSurface {
    let hit=primary_hits[u32(pixel.x)+u32(pixel.y)*u32(p.screen.x)];
    if (hit.status&3u)==0u {discard;}
    if (hit.status&3u)!=1u {
        let loading=(hit.status&3u)==3u;
        let color=select(vec3<f32>(4.0,0.0,2.6),vec3<f32>(0.018,0.023,0.03),loading);
        return StoredSurface(vec4<f32>(0.0),vec4<f32>(0.0,1.0,0.0,0.04),vec4<f32>(1.0),vec4<f32>(color,0.04),vec2<f32>(-1.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),select(0.0,1.0,STORED_REVERSE_DEPTH));
    }
    let normal=stored_normal(hit.status);let camera=stored_cameras[0];
    let position=hit.normal*hit.distance;
    let view_position=(camera.view*vec4<f32>(position,0.0)).xyz;
    let clip=camera.proj*vec4<f32>(view_position,1.0);let depth=clip.z/clip.w;
    if clip.w<=0.0 || depth<0.0 || depth>1.0 {discard;}
    let level=(hit.status>>2u)&31u;let material=(hit.status>>8u)&3u;
    let radial=normalize(vec3<f32>(hit.cell)+0.5);let altitude=cell_altitude(hit.cell);
    let width=max(hit.distance*p.up.w*2.0/p.screen.y,0.0001);
    let gradient=material_gradient(hit.cell);let grass=smoothstep(0.55,0.9,dot(-normalize(gradient),radial));
    let pigment=terrain_material(hit.cell,altitude,width/max(0.08,abs(dot(normalize(gradient),hit.normal))),grass,material);
    let center=(vec3<f32>(hit.cell-p.origin.xyz)+0.5-p.fraction.xyz)*0.1;
    // A distant reconstructed surface has its own zero depth; using the
    // canonical height here would turn interpolation error into soil bands.
    let depth_in_material=select(surface_depth(hit.cell,center),0.0,level>0u && material==1u);
    let albedo=face_albedo(normal,radial,depth_in_material,altitude,pigment);
    let previous=camera.prev_view_proj*vec4<f32>(camera.position_near.xyz+position,1.0);
    var velocity=vec2<f32>(0.0);
    if previous.w>0.0 {let ndc=previous.xy/previous.w;velocity=pixel.xy-(vec2<f32>(ndc.x,-ndc.y)*0.5+0.5)*p.screen.xy;}
    return StoredSurface(vec4<f32>(albedo,1.0),vec4<f32>(normal,0.04),vec4<f32>(1.0,0.92,0.0,0.04),vec4<f32>(0.0,0.0,0.0,0.04),vec2<f32>(-1.0,-2.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(velocity,0.0,0.0),depth);
}
@group(1) @binding(1) var stored_receiver:texture_2d<f32>;
@group(1) @binding(2) var stored_sun:texture_storage_2d<rgba16float,write>;
@group(1) @binding(3) var stored_sun_direction:texture_storage_2d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn stored_visibility(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=vec2<u32>(p.screen.xy)) {return;}
    let sun=normalize(p.lighting.xyz);if all(id.xy==vec2<u32>(0u)) {textureStore(stored_sun_direction,vec2<i32>(0),vec4<f32>(sun,0.0));}
    let hit=primary_hits[id.x+id.y*u32(p.screen.x)];var visibility=1.0;
    if (hit.status&3u)==1u && all(textureLoad(stored_receiver,vec2<i32>(id.xy),0).xy==vec2<f32>(-1.0,-2.0)) {
        let normal=stored_normal(hit.status);visibility=0.0;
        if dot(normal,sun)>0.0 {
            // Shadow detail beyond exact residency is filtered to a quarter
            // camera pixel. Avoid marching centimetre contact shadows from
            // orbit; the offset is along the queried ray, never a larger cube.
            let level=(hit.status>>2u)&31u;
            let footprint=hit.distance*p.up.w*2.0/p.screen.y;
            let offset=select(0.0,footprint*0.25,level>0u);
            let origin=hit.normal*hit.distance+normal*max(0.0001,hit.distance*0.00000012)+sun*offset;
            let blocker=stored_trace(origin,sun,p.settings.x);
            visibility=select(0.0,1.0,(blocker.status&3u)==0u);
            if (blocker.status&3u)==2u {visibility=-1.0;}
        }
    }
    textureStore(stored_sun,vec2<i32>(id.xy),vec4<f32>(visibility,sun));
}
