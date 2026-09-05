struct LightMatrix {
    mat: mat4x4<f32>,
}

struct CascadeConfig {
    split_distance: f32,
    depth_bias: f32,
    filter_radius: f32,
    pcss_light_size: f32,
}

struct ShadowConfig {
    cascades: array<CascadeConfig, 4>,
    enable_pcss: u32,
    pcss_blocker_samples: u32,
    pcss_filter_samples: u32,
    pcf_sample_count: u32,
}

@group(0) @binding(3) var<uniform> shadow_config: ShadowConfig;
@group(0) @binding(4) var shadow_atlas: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(0) @binding(6) var <storage, read> shadow_matrices: array<LightMatrix>;

// Vogel disk sampling - blue-noise-like spiral pattern for high-quality PCF
fn vogel_disk_sample(sample_idx: u32, sample_count: u32, theta: f32) -> vec2<f32> {
    let GOLDEN_ANGLE = 2.39996323;
    let r = sqrt(f32(sample_idx) + 0.5) / sqrt(f32(sample_count));
    let angle = f32(sample_idx) * GOLDEN_ANGLE + theta;
    return vec2<f32>(cos(angle), sin(angle)) * r;
}

// Per-pixel hash for PCF rotation (reduces banding artifacts)
fn hash22(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    let d = dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * d);
}

fn point_light_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(0u, 1u, dir.x < 0.0);
    } else if a.y >= a.x && a.y >= a.z {
        return select(2u, 3u, dir.y < 0.0);
    } else {
        return select(4u, 5u, dir.z < 0.0);
    }
}

fn pcss_blocker_search(
    layer: u32,
    shadow_uv: vec2<f32>,
    receiver_depth: f32,
    search_radius: f32,
    blocker_samples: u32,
    theta: f32
) -> vec2<f32> {
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    for (var i = 0u; i < blocker_samples; i++) {
        let offset = vogel_disk_sample(i, blocker_samples, theta) * search_radius;
        let sample_uv = shadow_uv + offset;
        let pixel_coord = vec2<i32>(sample_uv * ATLAS_SIZE);

        if any(pixel_coord < vec2<i32>(0)) || any(pixel_coord >= vec2<i32>(i32(ATLAS_SIZE))) {
            continue;
        }

        let occluder_depth = textureLoad(shadow_atlas, pixel_coord, i32(layer), 0);
        if occluder_depth < receiver_depth - 0.0001 {
            blocker_sum += occluder_depth;
            blocker_count += 1.0;
        }
    }

    if blocker_count < 0.5 {
        return vec2<f32>(0.0, 0.0);
    }

    return vec2<f32>(blocker_sum / blocker_count, blocker_count);
}

fn pcss_penumbra_size(receiver_depth: f32, avg_blocker_depth: f32, light_size: f32) -> f32 {
    return (receiver_depth - avg_blocker_depth) / max(avg_blocker_depth, 0.001) * light_size;
}

fn sample_cascade_shadow(layer: u32, cascade_idx: u32, cascade_scale: f32, world_pos: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    if layer >= arrayLength(&shadow_matrices) || layer >= textureNumLayers(shadow_atlas) { return 1.0; }
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let theta = hash22(frag_coord) * 6.28318530718;

    // OPTIMIZATION: Adaptive PCF sample count based on cascade distance
    let base_count = shadow_config.pcf_sample_count;
    var pcf_count: u32;
    switch cascade_idx {
        case 0u: { pcf_count = base_count; }
        case 1u: { pcf_count = max(base_count * 3u / 4u, 4u); }
        case 2u: { pcf_count = max(base_count / 2u, 4u); }
        default: { pcf_count = max(base_count / 4u, 4u); }
    }

    var lit_sum = 0.0;
    for (var i = 0u; i < pcf_count; i++) {
        let offset = vogel_disk_sample(i, pcf_count, theta) * (cascade_scale / f32(textureDimensions(shadow_atlas).x));
        lit_sum += textureSampleCompareLevel(shadow_atlas, shadow_sampler, shadow_uv + offset, i32(layer), ndc.z);
    }

    return lit_sum / f32(pcf_count);
}

fn sample_cascade_shadow_pcss(layer: u32, cascade_idx: u32, world_pos: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    let config = shadow_config.cascades[cascade_idx];
    if layer >= arrayLength(&shadow_matrices) || layer >= textureNumLayers(shadow_atlas) { return 1.0; }
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let receiver_depth = ndc.z;
    let theta = hash22(frag_coord) * 6.28318530718;

    // Blocker search uses unbiased depth so nearby occluders are correctly identified.
    let search_radius = config.pcss_light_size / ATLAS_SIZE;
    let blocker = pcss_blocker_search(layer, shadow_uv, receiver_depth, search_radius, shadow_config.pcss_blocker_samples, theta);

    if blocker.y < 0.5 {
        return 1.0;
    }

    let penumbra = pcss_penumbra_size(receiver_depth, blocker.x, config.pcss_light_size);
    let filter_radius = clamp(penumbra / ATLAS_SIZE, config.filter_radius / ATLAS_SIZE, config.filter_radius * 3.0 / ATLAS_SIZE);

    var lit_sum = 0.0;

    for (var i = 0u; i < shadow_config.pcss_filter_samples; i++) {
        let offset = vogel_disk_sample(i, shadow_config.pcss_filter_samples, theta) * filter_radius;
        lit_sum += textureSampleCompareLevel(shadow_atlas, shadow_sampler, shadow_uv + offset, i32(layer), receiver_depth);
    }

    return lit_sum / f32(shadow_config.pcss_filter_samples);
}

fn screen_occluded(light: GpuLight, position: vec3<f32>, normal: vec3<f32>) -> bool {
    let max_distance=globals.ambient.w;
    if max_distance<=0.0 { return false; }
    let origin=position+normal*0.02;
    let inc=incident(light,origin);
    var ray_length=max_distance;
    if light.light_type!=0u { ray_length=min(ray_length,length(light.position_range.xyz-origin)); }
    if ray_length<0.025 || dot(normal,inc.direction)<=0.0 { return false; }
    // Traverse current-frame minimum depths from large cells down to pixels.
    // NDC is linear along a projected segment, including perspective cameras.
    let clip0=cameras[0].view_proj*vec4<f32>(origin,1.0);
    let clip1=cameras[0].view_proj*vec4<f32>(origin+inc.direction*ray_length,1.0);
    if clip0.w<=0.0 || clip1.w<=0.0 { return false; }
    let ndc0=clip0.xyz/clip0.w; let ndc1=clip1.xyz/clip1.w;
    let screen=vec2<f32>(globals.screen_size);
    let start=(ndc0.xy*vec2<f32>(0.5,-0.5)+0.5)*screen;
    let finish=(ndc1.xy*vec2<f32>(0.5,-0.5)+0.5)*screen;
    let delta=finish-start;
    let span=max(abs(delta.x),abs(delta.y));
    if span<0.5 { return false; }
    let top=i32(textureNumLevels(screen_depth_bounds))-1;
    var level=min(top,max(0,i32(floor(log2(max(span/8.0,1.0))))));
    var t=min(0.5/span,0.5);
    for(var iteration=0u;iteration<96u;iteration++) {
        if t>=1.0 { break; }
        let point=start+delta*t;
        if any(point<vec2<f32>(0.0)) || any(point>=screen) { break; }
        let z=mix(ndc0.z,ndc1.z,t);
        if z<0.0 || z>=1.0 { break; }
        let cell_size=f32(TILE_SIZE)*exp2(f32(level));
        let cell=floor(point/cell_size);
        let boundary=(cell+select(vec2<f32>(0.0),vec2<f32>(1.0),delta>=vec2<f32>(0.0)))*cell_size;
        let exits=select(vec2<f32>(1e20),(boundary-point)/select(vec2<f32>(1.0),delta,abs(delta)>vec2<f32>(1e-6)),abs(delta)>vec2<f32>(1e-6));
        let cell_end=min(1.0,t+max(min(exits.x,exits.y),0.0));
        let minimum=textureLoad(screen_depth_bounds,vec2<i32>(cell),level).r;
        if max(z,mix(ndc0.z,ndc1.z,cell_end))<=minimum {
            t=cell_end+0.001/span; level=min(level+1,top); continue;
        }
        if level>0 { level-=1; continue; }
        let p=vec2<i32>(point);
        let pixel_boundary=floor(point)+select(vec2<f32>(0.0),vec2<f32>(1.0),delta>=vec2<f32>(0.0));
        let pixel_exits=select(vec2<f32>(1e20),(pixel_boundary-point)/select(vec2<f32>(1.0),delta,abs(delta)>vec2<f32>(1e-6)),abs(delta)>vec2<f32>(1e-6));
        let pixel_end=min(1.0,t+max(min(pixel_exits.x,pixel_exits.y),0.0));
        let depth=textureLoad(gbuf_depth,p,0);
        if depth<1.0 {
            let surface=world_position(vec2<f32>(p)+0.5,depth);
            let ray_start=world_position(point,z);
            let ray_end=world_position(start+delta*pixel_end,mix(ndc0.z,ndc1.z,pixel_end));
            let surface_z=-(cameras[0].view*vec4<f32>(surface,1.0)).z;
            let z0=-(cameras[0].view*vec4<f32>(ray_start,1.0)).z;
            let z1=-(cameras[0].view*vec4<f32>(ray_end,1.0)).z;
            let thickness=max(0.025,max(z0,z1)*0.002);
            if max(z0,z1)-surface_z>0.003 && min(z0,z1)-surface_z<thickness { return true; }
        }
        t=pixel_end+0.001/span;
    }
    return false;
}

fn shadow_factor(light_idx: u32, world_pos: vec3<f32>, N: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }
    if light_idx >= globals.light_count { return 1.0; }

    let light = lights[light_idx];
    if light.shadow_index == 4294967295u { return 1.0; }
    if screen_occluded(light,world_pos,N) { return 0.0; }

    var light_dir: vec3<f32>;
    if light.light_type == 0u {
        light_dir = normalize(-light.direction_outer.xyz);
    } else {
        light_dir = normalize(light.position_range.xyz - world_pos);
    }
    let NdotL = max(dot(N, light_dir), 0.0);
    let normal_offset = N * NORMAL_OFFSET_SCALE * (1.0 - NdotL);
    let biased_pos = world_pos + normal_offset;

    var layer: u32;
    if light.light_type > 0u && light.light_type < 2u {
        let to_frag = biased_pos - light.position_range.xyz;
        layer = light.shadow_index + point_light_face(to_frag);
        return sample_cascade_shadow(layer, 0u, 1.0, biased_pos, frag_coord, frame);
    } else if light.light_type == 0u {
        let dist = length(world_pos - cameras[0].position_near.xyz);
        let splits = globals.csm_splits;

        var cascade_a = 3u;
        var cascade_b = 3u;
        var blend = 0.0;
        const BLEND_ZONE = 0.1;

        if dist < splits.x * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 0u;
        } else if dist < splits.x * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 0u;
            cascade_b = 1u;
            blend = smoothstep(splits.x * (1.0 - BLEND_ZONE / 2.0), splits.x * (1.0 + BLEND_ZONE / 2.0), dist);
        } else if dist < splits.y * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 1u;
        } else if dist < splits.y * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 1u;
            cascade_b = 2u;
            blend = smoothstep(splits.y * (1.0 - BLEND_ZONE / 2.0), splits.y * (1.0 + BLEND_ZONE / 2.0), dist);
        } else if dist < splits.z * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 2u;
        } else if dist < splits.z * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 2u;
            cascade_b = 3u;
            blend = smoothstep(splits.z * (1.0 - BLEND_ZONE / 2.0), splits.z * (1.0 + BLEND_ZONE / 2.0), dist);
        } else {
            cascade_a = 3u;
        }

        let use_pcss = shadow_config.enable_pcss != 0u && shadow_config.cascades[cascade_a].pcss_light_size > 0.0;

        let layer_a = light.shadow_index + cascade_a;
        var shadow_a: f32;
        if use_pcss {
            shadow_a = sample_cascade_shadow_pcss(layer_a, cascade_a, biased_pos, frag_coord, frame);
        } else {
            let cascade_scale_a = 1.0 + f32(cascade_a) * 1.5;
            shadow_a = sample_cascade_shadow(layer_a, cascade_a, cascade_scale_a, biased_pos, frag_coord, frame);
        }

        if blend <= 0.001 { return shadow_a; }

        if cascade_b != cascade_a && blend > 0.001 {
            let use_pcss_b = shadow_config.enable_pcss != 0u && shadow_config.cascades[cascade_b].pcss_light_size > 0.0;
            let layer_b = light.shadow_index + cascade_b;
            var shadow_b: f32;
            if use_pcss_b {
                shadow_b = sample_cascade_shadow_pcss(layer_b, cascade_b, biased_pos, frag_coord, frame);
            } else {
                let cascade_scale_b = 1.0 + f32(cascade_b) * 1.5;
                shadow_b = sample_cascade_shadow(layer_b, cascade_b, cascade_scale_b, biased_pos, frag_coord, frame);
            }
            return mix(shadow_a, shadow_b, blend);
        }

        return shadow_a;
    } else {
        layer = light.shadow_index;
        return sample_cascade_shadow(layer, 0u, 1.0, biased_pos, frag_coord, frame);
    }
}
