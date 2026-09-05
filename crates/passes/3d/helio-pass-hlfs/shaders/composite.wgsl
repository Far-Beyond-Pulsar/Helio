@group(2) @binding(0) var filtered_lighting: texture_2d<u32>;
@group(2) @binding(1) var filtered_geometry: texture_2d<u32>;
@group(2) @binding(2) var screen_depth_bounds: texture_2d<f32>;
@group(2) @binding(3) var previous_lighting: texture_2d<u32>;
@group(2) @binding(4) var previous_geometry: texture_2d<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p=array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
    return vec4<f32>(p[vi],0.0,1.0);
}
@fragment
fn fs_main(@builtin(position) fragment: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel=vec2<i32>(fragment.xy);
    let depth=textureLoad(gbuf_depth,pixel,0);
    if depth>=1.0 { return textureLoad(pre_aa_texture,pixel,0); }
    let s=surface_at(vec2<u32>(pixel));
    let z=-(cameras[0].view*vec4<f32>(s.position,1.0)).z;
    let sample_pos=fragment.xy/f32(globals.sample_scale)-0.5;
    let base=vec2<i32>(floor(sample_pos));
    let fraction=fract(sample_pos);
    var diffuse=vec3<f32>(0.0); var specular=vec3<f32>(0.0); var weight_sum=0.0;
    var selected=vec2<i32>(-1);
    let roll=stbn(vec2<u32>(pixel),21u);
    var neighbor_weights: array<f32,4>;
    // Select one bilinear neighbor after excluding unrelated surfaces.
    for(var i=0u;i<4u;i++) {
        let offset=vec2<i32>(vec2<u32>(i&1u,i>>1u)); let p=base+offset;
        if any(p<vec2<i32>(0)) || any(p>=vec2<i32>(globals.sample_size)) { continue; }
        let geo=load_geometry(filtered_geometry,p);
        if !geometry_matches(geo,s.normal,z) { continue; }
        let weights=select(1.0-fraction,fraction,offset==vec2<i32>(1));
        let weight=weights.x*weights.y*pow(max(dot(oct_decode(geo.xy),s.normal),0.0),32.0);
        weight_sum+=weight;
        neighbor_weights[i]=weight;
    }
    var remaining=roll*weight_sum;
    for(var i=0u;i<4u;i++) {
        if neighbor_weights[i]>0.0 && remaining<neighbor_weights[i] { selected=base+vec2<i32>(vec2<u32>(i&1u,i>>1u)); break; }
        remaining-=neighbor_weights[i];
    }
    if selected.x>=0 {
        diffuse=load_radiance(filtered_lighting,selected,0u);
        specular=load_radiance(filtered_lighting,selected,1u);
        if globals.debug_mode==3u {
            let confidence=(textureLoad(filtered_geometry,selected,0).y&(1u<<28u))!=0u;
            return vec4<f32>(vec3<f32>(select(0.0,1.0,confidence)),1.0);
        }
    } else if globals.history_valid!=0u && (globals.surface_flags&2u)!=0u {
        let uv=previous_uv(vec2<u32>(pixel),s.position);
        let previous_z=-(globals.previous_view*vec4<f32>(s.position,1.0)).z;
        let previous_base=vec2<i32>(floor(uv*vec2<f32>(globals.sample_size)-0.5));
        for(var i=0u;i<4u;i++) {
            let p=previous_base+vec2<i32>(vec2<u32>(i&1u,i>>1u));
            if any(p<vec2<i32>(0)) || any(p>=vec2<i32>(globals.sample_size)) { continue; }
            if geometry_matches(load_geometry(previous_geometry,p),s.normal,previous_z) {
                diffuse=load_radiance(previous_lighting,p,0u);
                specular=load_radiance(previous_lighting,p,1u); weight_sum=1.0; break;
            }
        }
    }
    if weight_sum==0.0 && globals.light_count>0u {
        // Thin surfaces may have no matching half-resolution sample. Shade
        // those pixels directly with the same bounded candidate/shadow budget.
        var rng=hash_u32(u32(pixel.x)+u32(pixel.y)*globals.screen_size.x+globals.frame*0x9e3779b9u);
        for(var sample=0u;sample<globals.sample_count;sample++) {
            var selected=INVALID_LIGHT; var selected_target=0.0; var total=0.0;
            for(var candidate=0u;candidate<globals.candidate_count;candidate++) {
                let id=min(u32(random(&rng)*f32(globals.light_count)),globals.light_count-1u);
                let proxy=importance(id,s); total+=proxy;
                if proxy>0.0 && random(&rng)*total<proxy { selected=id; selected_target=proxy; }
            }
            if selected!=INVALID_LIGHT {
                let visibility=shadow_factor(selected,s.position,s.normal,fragment.xy,globals.frame);
                let light=evaluate_light(selected,s,visibility);
                let normalization=total*f32(globals.light_count)/(selected_target*f32(globals.candidate_count*globals.sample_count));
                diffuse+=light.diffuse*normalization; specular+=light.specular*normalization;
            }
        }
    }
    // Ambient supplies indirect diffuse when no baked lighting is available.
    // Lightmapped surfaces use the bake directly, without an additional AO term.
    let orm=textureLoad(gbuf_orm,pixel,0).rgb;
    let f=s.f0+(max(vec3<f32>(1.0-s.roughness),s.f0)-s.f0)*pow5(1.0-max(dot(s.normal,s.view),0.0));
    var indirect=(1.0-f)*(1.0-s.metallic)*globals.ambient.rgb*s.albedo*clamp(orm.r,0.0,1.0);
    let lm_uv=textureLoad(gbuf_lightmap_uv,pixel,0).xy;
    if (globals.surface_flags&1u)!=0u && lm_uv.x>=0.0 {
        indirect=textureSampleLevel(baked_lightmap,lightmap_sampler,lm_uv,0.0).rgb*s.albedo;
    }
    let direct=(diffuse*s.albedo+specular*s.specular_factor)/globals.exposure;
    let emissive=textureLoad(gbuf_emissive,pixel,0).rgb;
    return vec4<f32>(finite_color(direct+indirect+emissive),1.0);
}
