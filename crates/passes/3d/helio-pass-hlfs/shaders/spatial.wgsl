@group(2) @binding(0) var filtered_lighting: texture_2d<u32>;
@group(2) @binding(1) var filtered_geometry: texture_2d<u32>;
@group(2) @binding(2) var spatial_lighting: texture_storage_2d<rg32uint,write>;
@compute @workgroup_size(8,8)
fn spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=globals.sample_size) { return; }
    let pixel=sample_pixel(id.xy,globals.frame);
    if textureLoad(gbuf_depth,vec2<i32>(pixel),0)>=1.0 {
        textureStore(spatial_lighting,vec2<i32>(id.xy),vec4<u32>(0u)); return;
    }
    let s=surface_at(pixel);
    let z=-(cameras[0].view*vec4<f32>(s.position,1.0)).z;
    let sample_pos=vec2<f32>(id.xy);
    let center=clamp(vec2<i32>(round(sample_pos)),vec2<i32>(0),vec2<i32>(globals.sample_size)-1);
    let moments=load_moments(filtered_geometry,center);
    let cd=vec4<f32>(load_radiance(filtered_lighting,center,0u),moments.x);
    let cs=vec4<f32>(load_radiance(filtered_lighting,center,1u),moments.y);
    let mean_d=luminance(cd.rgb); let mean_s=luminance(cs.rgb);
    let variance=max(max(cd.a-mean_d*mean_d,0.0)/max(mean_d*mean_d,0.0001),
        max(cs.a-mean_s*mean_s,0.0)/max(mean_s*mean_s,0.0001));
    let age=load_geometry(filtered_geometry,center).w;
    let confidence=(textureLoad(filtered_geometry,center,0).y&(1u<<28u))!=0u;
    var diffuse=vec3<f32>(0.0); var specular=vec3<f32>(0.0); var weight_sum=0.0;
    if globals.debug_mode!=0u || age>globals.max_history || (age>=4.0 && variance<=0.002) {
        textureStore(spatial_lighting,center,textureLoad(filtered_lighting,center,0)); return;
    } else {
        // One sparse rotated filter, with a narrow footprint for stable signals.
        let radius=select(1,2,variance>select(0.02,0.05,confidence) || age<4.0);
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
            let d=load_radiance(filtered_lighting,p,0u); let sp=load_radiance(filtered_lighting,p,1u);
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

    }
    textureStore(spatial_lighting,center,vec4<u32>(pack_radiance(diffuse,id.xy,22u).x,pack_radiance(specular,id.xy,25u).x,0u,0u));
}
