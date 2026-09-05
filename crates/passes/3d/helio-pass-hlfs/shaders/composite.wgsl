@group(2) @binding(0) var filtered_diffuse: texture_2d<u32>;
@group(2) @binding(1) var filtered_specular: texture_2d<u32>;
@group(2) @binding(2) var filtered_geometry: texture_2d<u32>;
@group(2) @binding(3) var screen_depth_bounds: texture_2d<f32>;

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
    let center=clamp(vec2<i32>(round(sample_pos)),vec2<i32>(0),vec2<i32>(globals.sample_size)-1);
    let moments=load_moments(filtered_geometry,center);
    let cd=vec4<f32>(load_radiance(filtered_diffuse,center),moments.x);
    let cs=vec4<f32>(load_radiance(filtered_specular,center),moments.y);
    let mean_d=luminance(cd.rgb); let mean_s=luminance(cs.rgb);
    let variance=max(max(cd.a-mean_d*mean_d,0.0)/max(mean_d*mean_d,0.0001),
        max(cs.a-mean_s*mean_s,0.0)/max(mean_s*mean_s,0.0001));
    let age=load_geometry(filtered_geometry,center).w;
    var diffuse=vec3<f32>(0.0); var specular=vec3<f32>(0.0); var weight_sum=0.0;
    if globals.debug_mode!=0u || (globals.sample_scale==1u && (age>globals.max_history || (age>=4.0 && variance<=0.002))) {
        diffuse=cd.rgb; specular=cs.rgb; weight_sum=1.0;
    } else {
        // One sparse rotated filter, with a narrow footprint for stable signals.
        let radius=select(1,2,variance>0.02 || age<4.0);
        let phase=globals.frame&3u;
        for(var y=-2;y<=2;y++) { for(var x=-2;x<=2;x++) {
            if abs(x)>radius || abs(y)>radius { continue; }
            if radius==2 && abs(x)+abs(y)>2 && ((u32(x+2)+u32(y+2)+phase)&1u)==0u { continue; }
            let p=center+vec2<i32>(x,y);
            if any(p<vec2<i32>(0)) || any(p>=vec2<i32>(globals.sample_size)) { continue; }
            let geo=load_geometry(filtered_geometry,p);
            if !geometry_matches(geo,s.normal,z) { continue; }
            let offset=vec2<f32>(p)-sample_pos;
            let spatial=exp(-dot(offset,offset)/f32(radius*radius));
            let weight=spatial*pow(max(dot(oct_decode(geo.xy),s.normal),0.0),32.0);
            let d=load_radiance(filtered_diffuse,p); let sp=load_radiance(filtered_specular,p);
            // Tonemapped accumulation for disocclusions suppresses sparse fireflies.
            if age<4.0 {
                diffuse+=d/(1.0+luminance(d))*weight; specular+=sp/(1.0+luminance(sp))*weight;
            } else { diffuse+=d*weight; specular+=sp*weight; }
            weight_sum+=weight;
        }}
    }
    if weight_sum>0.0 {
        diffuse/=weight_sum; specular/=weight_sum;
        if age<4.0 && globals.debug_mode==0u {
            diffuse/=max(1.0-luminance(diffuse),1e-4); specular/=max(1.0-luminance(specular),1e-4);
        }
    } else if globals.light_count>0u {
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
    if globals.has_lightmap!=0u && lm_uv.x>=0.0 {
        indirect=textureSampleLevel(baked_lightmap,lightmap_sampler,lm_uv,0.0).rgb*s.albedo;
    }
    let direct=(diffuse*s.albedo+specular*s.specular_factor)/globals.exposure;
    let emissive=textureLoad(gbuf_emissive,pixel,0).rgb;
    return vec4<f32>(finite_color(direct+indirect+emissive),1.0);
}
