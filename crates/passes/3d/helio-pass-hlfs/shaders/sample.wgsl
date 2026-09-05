@group(2) @binding(0) var<storage, read> grid: array<LightTile>;
@group(2) @binding(1) var<storage, read> previous_visible: array<VisibleTile>;
@group(2) @binding(2) var<storage, read_write> next_visible: array<VisibleTile>;
@group(2) @binding(3) var raw_diffuse: texture_storage_2d<r32uint,write>;
@group(2) @binding(4) var raw_specular: texture_storage_2d<r32uint,write>;
@group(2) @binding(5) var previous_geometry: texture_2d<u32>;
@group(2) @binding(6) var screen_depth_bounds: texture_2d<f32>;
var<workgroup> seen_ids: array<u32,256>;
var<workgroup> seen_weights: array<f32,256>;
var<workgroup> slot_ids: array<u32,64>;
var<workgroup> slot_weights: array<f32,64>;
var<workgroup> sorted_ids: array<u32,16>;

struct Reservoir { selected: u32, importance_value: f32, weight_sum: f32, random_value: f32, }
fn reservoir_add(r: ptr<function, Reservoir>, id: u32, importance_value: f32, weight: f32) {
    if weight<=0.0 { return; }
    (*r).weight_sum+=weight;
    let probability=weight/(*r).weight_sum;
    // Warp the selected/rejected interval back to [0,1), preserving STBN.
    if (*r).random_value<probability {
        (*r).selected=id; (*r).importance_value=importance_value;
        (*r).random_value/=max(probability,1e-20);
    } else {
        (*r).random_value=((*r).random_value-probability)/max(1.0-probability,1e-20);
    }
}

fn trace_visibility(id: u32, surface: Surface, pixel: vec2<u32>) -> f32 {
    return shadow_factor(id,surface.position,surface.normal,vec2<f32>(pixel)+0.5,globals.frame);
}

@compute @workgroup_size(8,8)
fn sample_lights(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    for(var i=0u;i<4u;i++) { seen_ids[lane*4u+i]=INVALID_LIGHT; seen_weights[lane*4u+i]=0.0; }
    workgroupBarrier();
    if all(gid.xy<globals.sample_size) {
        let pixel=sample_pixel(gid.xy,globals.frame);
        var result=Lighting(vec3<f32>(0.0),vec3<f32>(0.0));
        var confidence=1.0;
        if textureLoad(gbuf_depth,vec2<i32>(pixel),0)<1.0 && globals.light_count>0u {
            let s=surface_at(pixel);
            let tile=(pixel.y/TILE_SIZE)*div_ceil(globals.screen_size,TILE_SIZE).x+pixel.x/TILE_SIZE;
            let grid_count=grid[tile].count;
            let overflow=grid_count>GRID_CAPACITY;
            let population=select(grid_count,globals.light_count,overflow);
            if population<=globals.sample_count || globals.debug_mode==1u {
                // Exact path for small sets, and an uncapped oracle for GPU regression tests.
                let n=select(population,globals.light_count,globals.debug_mode==1u);
                for(var i=0u;i<n;i++) {
                    var id=i;
                    if !overflow && globals.debug_mode!=1u { id=grid[tile].indices[i]; }
                    if importance(id,s)<=0.0 { continue; }
                    let vis=trace_visibility(id,s,pixel);
                    let light=evaluate_light(id,s,vis);
                    result.diffuse+=light.diffuse; result.specular+=light.specular;
                    if vis>0.0 && i<4u { seen_ids[lane*4u+i]=id; seen_weights[lane*4u+i]=importance(id,s); }
                }
            } else {
                confidence=0.0;
                let prev_uv=previous_uv(pixel,s.position);
                let valid_uv=all(prev_uv>=vec2<f32>(0.0)) && all(prev_uv<vec2<f32>(1.0));
                let prev_pixel=vec2<i32>(prev_uv*vec2<f32>(globals.sample_size));
                let previous_z=-(globals.previous_view*vec4<f32>(s.position,1.0)).z;
                var valid_history=false;
                if globals.history_valid!=0u && valid_uv {
                    valid_history=geometry_matches(load_geometry(previous_geometry,prev_pixel),s.normal,previous_z);
                }
                // Stochastic bilinear tile lookup. Adjacent tiles share support at borders.
                let tile_dims=div_ceil(globals.sample_size,TILE_SIZE);
                let tile_f=prev_uv*vec2<f32>(globals.sample_size)/f32(TILE_SIZE)-0.5;
                let tile_jitter=vec2<f32>(stbn(gid.xy,0u),stbn(gid.xy,1u));
                let guide_xy=vec2<u32>(clamp(floor(tile_f+tile_jitter),vec2<f32>(0.0),vec2<f32>(tile_dims-1u)));
                let guide_tile=guide_xy.y*tile_dims.x+guide_xy.x;
                var guide: array<u32,16>;
                var guide_count=0u;
                if globals.history_valid!=0u {
                    for(var i=0u;i<VISIBLE_CAPACITY;i++) {
                        let id=previous_visible[guide_tile].indices[i];
                        if id<globals.light_count { guide[guide_count]=id; guide_count++; }
                    }
                }
                let hidden_fraction=select(0.5,globals.discovery_fraction,valid_history);
                let candidate_count=select(min(globals.candidate_count*2u,16u),globals.candidate_count,valid_history);
                var rng=hash_u32(pixel.x+pixel.y*globals.screen_size.x+globals.frame*0x9e3779b9u);
                var traced_ids: array<u32,4>;
                var traced_visibility: array<f32,4>;
                var guide_target: array<f32,16>;
                for(var i=0u;i<guide_count;i++) { guide_target[i]=importance(guide[i],s); }
                for(var sample=0u;sample<globals.sample_count;sample++) {
                    // Evaluate the visible list in full and a bounded unbiased
                    // estimate of its complement from the conservative light grid.
                    var candidate_ids: array<u32,32>;
                    var candidate_targets: array<f32,32>;
                    var directional_weight=0.0; var local_weight=0.0;
                    for(var i=0u;i<guide_count;i++) {
                        candidate_ids[i]=guide[i]; candidate_targets[i]=guide_target[i];
                        if lights[guide[i]].light_type==0u { directional_weight+=guide_target[i]; }
                        else { local_weight+=guide_target[i]; }
                    }
                    let inverse_proposal=f32(population)/f32(candidate_count);
                    for(var candidate=0u;candidate<candidate_count;candidate++) {
                        let pick=min(u32((f32(candidate)+random(&rng))*inverse_proposal),population-1u);
                        var id=pick; if !overflow { id=grid[tile].indices[pick]; }
                        var importance_value=importance(id,s);
                        // Binary search the sorted guide to sample its complement.
                        var lo=0u; var hi=guide_count;
                        while lo<hi { let mid=(lo+hi)/2u; if guide[mid]<id { lo=mid+1u; } else { hi=mid; } }
                        if lo<guide_count && guide[lo]==id { importance_value=0.0; }
                        candidate_ids[guide_count+candidate]=id; candidate_targets[guide_count+candidate]=importance_value;
                        if lights[id].light_type==0u { directional_weight+=importance_value*inverse_proposal; }
                        else { local_weight+=importance_value*inverse_proposal; }
                    }
                    var directional_scale=1.0;
                    if local_weight>1e-5 && directional_weight>0.0 { directional_scale=min(1.0,0.5*local_weight/directional_weight); }
                    var visible_reservoir=Reservoir(INVALID_LIGHT,0.0,0.0,(f32(sample)+stbn(gid.xy,2u))/f32(globals.sample_count));
                    var hidden_reservoir=Reservoir(INVALID_LIGHT,0.0,0.0,(f32(sample)+stbn(gid.xy,3u))/f32(globals.sample_count));
                    for(var i=0u;i<guide_count+candidate_count;i++) {
                        let id=candidate_ids[i];
                        var importance_value=candidate_targets[i];
                        if lights[id].light_type==0u { importance_value*=directional_scale; }
                        if i<guide_count { reservoir_add(&visible_reservoir,id,importance_value,importance_value); }
                        else { reservoir_add(&hidden_reservoir,id,importance_value,importance_value*inverse_proposal); }
                    }
                    let v=visible_reservoir.weight_sum; let h=hidden_reservoir.weight_sum;
                    var hidden_budget=h;
                    // Cap hidden selection weight to 20% (50% on disocclusion),
                    // relaxing when there is no useful visible history.
                    if v>1e-5 && hidden_fraction<1.0 { hidden_budget=min(h,v*hidden_fraction/(1.0-hidden_fraction)); }
                    let p_hidden=hidden_budget/max(v+hidden_budget,1e-20);
                    var chosen=visible_reservoir; var group_probability=1.0-p_hidden;
                    if visible_reservoir.random_value<p_hidden { chosen=hidden_reservoir; group_probability=p_hidden; }
                    let selected=chosen.selected;
                    traced_ids[sample]=selected;
                    if selected==INVALID_LIGHT { continue; }
                    var vis=-1.0;
                    for(var i=0u;i<sample;i++) {
                        if traced_ids[i]==selected { vis=traced_visibility[i]; break; }
                    }
                    if vis<0.0 { vis=trace_visibility(selected,s,pixel); }
                    traced_visibility[sample]=vis;
                    // Correct BOTH reservoir weights and the clamped group PDF.
                    // A hidden/directional budget must not remove lighting energy.
                    let normalization=chosen.weight_sum/max(chosen.importance_value*group_probability*f32(globals.sample_count),1e-20);
                    let light=evaluate_light(selected,s,vis);
                    result.diffuse+=light.diffuse*normalization;
                    result.specular+=light.specular*normalization;
                    if vis>0.0 { seen_ids[lane*4u+sample]=selected; seen_weights[lane*4u+sample]=importance(selected,s); }
                }
            }
        }
        textureStore(raw_diffuse,vec2<i32>(gid.xy),pack_radiance(result.diffuse,gid.xy,4u));
        textureStore(raw_specular,vec2<i32>(gid.xy),pack_radiance(result.specular,gid.xy,7u));
    }
    workgroupBarrier();
    let output_tile=group.y*div_ceil(globals.sample_size,TILE_SIZE).x+group.x;
    // Parallel deduplication keeps one explicit ID in each of 64 scratch slots.
    // Visibility is binary: penumbra hits rank by unoccluded importance too.
    var best=INVALID_LIGHT; var weight=0.0;
    for(var i=0u;i<256u;i++) {
        let id=seen_ids[i]; let w=seen_weights[i];
        if id!=INVALID_LIGHT && (hash_u32(id)&63u)==lane && (w>weight || (w==weight && id<best)) { best=id; weight=w; }
    }
    slot_ids[lane]=best; slot_weights[lane]=weight;
    if lane<VISIBLE_CAPACITY { sorted_ids[lane]=INVALID_LIGHT; }
    workgroupBarrier();
    // Rank scratch slots by importance, retaining at most 16. Collisions and
    // capacity eviction leave lights discoverable through the hidden reservoir.
    var rank=0u;
    for(var i=0u;i<64u;i++) {
        if slot_weights[i]>weight || (slot_weights[i]==weight && slot_ids[i]<best) { rank++; }
    }
    workgroupBarrier();
    if rank>=VISIBLE_CAPACITY { best=INVALID_LIGHT; }
    slot_ids[lane]=best;
    workgroupBarrier();
    // Rank the retained unique IDs numerically for sorted-list membership tests.
    var sorted_rank=0u; var count=0u;
    for(var i=0u;i<64u;i++) {
        if slot_ids[i]<best { sorted_rank++; }
        if slot_ids[i]!=INVALID_LIGHT { count++; }
    }
    if best!=INVALID_LIGHT { sorted_ids[sorted_rank]=best; }
    workgroupBarrier();
    if lane<VISIBLE_CAPACITY { next_visible[output_tile].indices[lane]=sorted_ids[lane]; }
    if lane==0u { next_visible[output_tile].count=count; }
}
