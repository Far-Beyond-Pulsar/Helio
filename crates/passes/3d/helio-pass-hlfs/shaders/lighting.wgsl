struct Incident { direction: vec3<f32>, radiance: vec3<f32>, }
struct Lighting { diffuse: vec3<f32>, specular: vec3<f32>, }
struct Surface {
    position: vec3<f32>, normal: vec3<f32>, view: vec3<f32>,
    albedo: vec3<f32>, f0: vec3<f32>, roughness: f32, metallic: f32,
}
fn surface_at(pixel: vec2<u32>) -> Surface {
    let p=vec2<i32>(pixel);
    let position=world_position(vec2<f32>(pixel)+0.5,textureLoad(gbuf_depth,p,0));
    let albedo=max(textureLoad(gbuf_albedo,p,0).rgb,vec3<f32>(0.0));
    let orm=textureLoad(gbuf_orm,p,0).rgb;
    let metallic=clamp(orm.b,0.0,1.0);
    return Surface(position,safe_normalize(textureLoad(gbuf_normal,p,0).xyz),
        safe_normalize(cameras[0].position_near.xyz-position),albedo,
        mix(vec3<f32>(0.04),albedo,metallic),clamp(orm.g,0.02,1.0),metallic);
}
fn incident(light: GpuLight, position: vec3<f32>) -> Incident {
    if light.light_type==0u {
        return Incident(safe_normalize(-light.direction_outer.xyz),max(light.color_intensity.rgb*light.color_intensity.w,vec3<f32>(0.0)));
    }
    let delta=light.position_range.xyz-position;
    let d2=dot(delta,delta); let range=light.position_range.w;
    if range<=0.0 || d2>=range*range { return Incident(vec3<f32>(0.0),vec3<f32>(0.0)); }
    let direction=safe_normalize(delta);
    let normalized_d2=d2/(range*range);
    var attenuation=max(0.0,1.0-normalized_d2*normalized_d2)/max(d2,0.0001);
    if light.light_type==2u {
        let cosine=dot(-direction,light.direction_outer.xyz);
        // Equal inner/outer angles are a hard cone, not smoothstep(0,0,x).
        let cone=clamp((cosine-light.direction_outer.w)/max(light.inner_angle-light.direction_outer.w,1e-5),0.0,1.0);
        attenuation*=cone*cone*(3.0-2.0*cone);
    }
    return Incident(direction,max(light.color_intensity.rgb*light.color_intensity.w*attenuation,vec3<f32>(0.0)));
}
fn importance(id: u32, s: Surface) -> f32 {
    let inc=incident(lights[id],s.position);
    let ndl=max(dot(s.normal,inc.direction),0.0);
    // Cheap, positive proxy. The PDF correction below does not require an exact BRDF.
    // Log weighting prevents one very bright/occluded light monopolizing candidates.
    return log2(1.0+max(luminance(inc.radiance)*ndl*globals.exposure,0.0));
}
fn evaluate_light(id: u32, s: Surface, visibility: f32) -> Lighting {
    let inc=incident(lights[id],s.position);
    let ndl=max(dot(s.normal,inc.direction),0.0);
    let ndv=max(dot(s.normal,s.view),0.0);
    let h=safe_normalize(s.view+inc.direction);
    let ndh=max(dot(s.normal,h),0.0);
    let a=s.roughness*s.roughness; let a2=a*a;
    let denominator=ndh*ndh*(a2-1.0)+1.0;
    let d=a2/(PI*denominator*denominator+0.0001);
    let k=(s.roughness+1.0)*(s.roughness+1.0)/8.0;
    let g=(ndv/(ndv*(1.0-k)+k+0.0001))*(ndl/(ndl*(1.0-k)+k+0.0001));
    let f=s.f0+(1.0-s.f0)*pow5(1.0-clamp(dot(h,s.view),0.0,1.0));
    let energy=inc.radiance*(ndl*visibility*globals.exposure);
    // Demodulate before filtering. Albedo and F0 are restored at full resolution.
    return Lighting((1.0-f)*(1.0-s.metallic)*energy/PI,
        d*g*f*energy/((4.0*ndv*ndl+0.0001)*max(s.f0,vec3<f32>(0.04))));
}
