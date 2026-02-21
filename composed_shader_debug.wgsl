// Base Geometry Shader

struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Transform {
    model: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> transform: Transform;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal: u32,
    @location(4) tangent: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

// INJECT_VERTEXPREAMBLE

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    // INJECT_VERTEXMAIN

    let world_pos = transform.model * vec4<f32>(vertex.position, 1.0);
    let world_normal = normalize((transform.model * vec4<f32>(decode_normal(vertex.normal), 0.0)).xyz);

    var output: VertexOutput;
    output.position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.tex_coords = vertex.tex_coords;

    // INJECT_VERTEXPOSTPROCESS

    return output;
}

// Material data structures and bindings

struct MaterialData {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive_strength: f32,
    ao: f32,
};

// ===== Cloud and Sky Constants =====
const CLOUD_HEIGHT_MIN: f32 = 200.0;   // World-space altitude of cloud base
const CLOUD_HEIGHT_MAX: f32 = 400.0;   // World-space altitude of cloud top
const CLOUD_COVERAGE: f32  = 0.58;     // Fraction of sky covered (0=clear, 1=overcast)

// ===== Sun Constants =====
// Sun angular size: controls the hard edge of the sun disc
// Higher values = smaller sun (0.9985 = large, 0.9992 = medium, 0.9995 = small)
const SUN_DISC_SIZE: f32 = 0.9995 ;

// Sun glow size: controls the soft corona/bloom around the sun
// Lower values = larger glow (0.96 = huge, 0.98 = large, 0.99 = small)
const SUN_GLOW_SIZE: f32 = 0.99;

// Sun brightness at midday (multiplier for bloom effect)
const SUN_BRIGHTNESS_MAX: f32 = 45.0;  // Maximum at noon
const SUN_BRIGHTNESS_MIN: f32 = 6.0;   // Minimum at horizon

// Helper function: rendering expects BGR, so swap R and B channels
fn rgb(r: f32, g: f32, b: f32) -> vec3<f32> {
    return vec3<f32>(b, g, r);  // Swapped to BGR
}

// ===== 3D Noise =====

fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth quintic-interpolated 3D value noise
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic smoothstep for less grid-aliasing than cubic
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    return mix(
        mix(
            mix(hash(i + vec3<f32>(0.0, 0.0, 0.0)), hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)), hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)), hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)), hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value     = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos       = p;

    for (var i = 0; i < octaves; i++) {
        value     += amplitude * noise3d(pos * frequency);
        frequency *= 2.1;
        amplitude *= 0.5;
    }

    return value;
}

// ===== Volumetric Cloud Density =====

// Returns cloud density [0,1] at a world-space position.
// Only nonzero inside the cloud slab [CLOUD_HEIGHT_MIN, CLOUD_HEIGHT_MAX].
fn get_cloud_density(world_pos: vec3<f32>, time: f32) -> f32 {
    let cloud_thickness = CLOUD_HEIGHT_MAX - CLOUD_HEIGHT_MIN;

    // Height normalised within the cloud slab
    let height_frac = (world_pos.y - CLOUD_HEIGHT_MIN) / cloud_thickness;
    if (height_frac < 0.0 || height_frac > 1.0) {
        return 0.0;
    }

    // Height gradient: puffy in the lower two-thirds, wispy at the top
    let height_fade = smoothstep(0.0, 0.12, height_frac)
                    * smoothstep(1.0, 0.55, height_frac);

    // Animate by translating the noise domain over time
    let cloud_speed  = vec3<f32>(0.8, 0.0, 0.45);
    let anim_pos     = world_pos * 0.0025 + cloud_speed * time * 0.00025;

    // Large-scale base shape
    let base = fbm(anim_pos, 4);

    // Coverage threshold: erode below the threshold so we get distinct cloud masses
    let threshold    = 1.0 - CLOUD_COVERAGE;
    let coverage_raw = max(0.0, base - threshold) / CLOUD_COVERAGE;
    if (coverage_raw < 0.001) {
        return 0.0;
    }

    // Fine detail erosion for fluffy, billowing edges
    let detail_pos = anim_pos * 2.7 + vec3<f32>(0.1, 0.2, 0.3);
    let detail     = fbm(detail_pos, 3) * 0.35;
    let shaped     = max(0.0, coverage_raw - detail * (1.0 - coverage_raw));

    return clamp(shaped * height_fade, 0.0, 1.0);
}

// Integrate cloud density along the view ray through the cloud slab.
// Uses more samples for proper 3D depth when viewing from the side.
fn get_cloud_coverage_for_ray(view_dir: vec3<f32>, camera_pos: vec3<f32>, time: f32) -> f32 {
    // Camera is always below the clouds; ray must be pointing upward
    if (view_dir.y < 0.02) {
        return 0.0;
    }

    // Ray-slab intersection
    let t_min = (CLOUD_HEIGHT_MIN - camera_pos.y) / view_dir.y;
    let t_max = (CLOUD_HEIGHT_MAX - camera_pos.y) / view_dir.y;
    if (t_min < 0.0 && t_max < 0.0) {
        return 0.0;
    }

    let t_enter   = max(0.0, t_min);
    let t_exit    = t_max;
    
    // More samples for side views (better 3D appearance)
    let num_steps = 8;
    let step_size = (t_exit - t_enter) / f32(num_steps);

    var total = 0.0;
    var transmittance = 1.0;
    
    for (var i = 0; i < num_steps; i++) {
        let t          = t_enter + (f32(i) + 0.5) * step_size;
        let sample_pos = camera_pos + view_dir * t;
        let density    = get_cloud_density(sample_pos, time);
        
        // Accumulate density with transmittance for depth
        total += density * transmittance;
        transmittance *= exp(-density * 0.5);
        
        if (transmittance < 0.01) {
            break;
        }
    }

    return clamp(total / f32(num_steps) * 1.5, 0.0, 1.0);
}

// Cheap self-shadow: 2 density samples toward the sun from the cloud midpoint.
// Returns [0,1] where 1 = fully lit, 0 = fully shadowed.  Costs ~2 noise evals.
fn get_cloud_self_shadow(view_dir: vec3<f32>, camera_pos: vec3<f32>, sun_dir: vec3<f32>, time: f32) -> f32 {
    if (view_dir.y < 0.02) { return 1.0; }
    let t_mid   = ((CLOUD_HEIGHT_MIN + CLOUD_HEIGHT_MAX) * 0.5 - camera_pos.y) / view_dir.y;
    let mid_pos = camera_pos + view_dir * t_mid;
    let d1      = get_cloud_density(mid_pos + sun_dir * 50.0, time);
    let d2      = get_cloud_density(mid_pos + sun_dir * 100.0, time);
    return exp(-(d1 + d2) * 2.5);
}

// ===== Time-of-Day Sky Gradient =====
// sun_height: -1 = midnight below horizon, 0 = on the horizon, +1 = zenith noon

fn get_sky_zenith_color(sun_height: f32) -> vec3<f32> {
    // Define colors in RGB and convert to BGR for rendering
    let night    = rgb(0.003, 0.006, 0.022);   // Dark navy
    let twilight = rgb(0.10,  0.14,  0.44);    // Dusky purple-blue
    let day      = rgb(0.07,  0.26,  0.78);    // Rich azure

    if (sun_height < -0.15) {
        return night;
    } else if (sun_height < 0.12) {
        return mix(night, twilight, smoothstep(-0.15, 0.12, sun_height));
    } else {
        return mix(twilight, day, smoothstep(0.12, 0.6, sun_height));
    }
}

fn get_sky_horizon_color(sun_height: f32) -> vec3<f32> {
    // Define colors in RGB and convert to BGR for rendering
    let night    = rgb(0.005, 0.008, 0.026);  // Near-black blue
    let twilight = rgb(1.00,  0.42,  0.08);   // Burning orange
    let day      = rgb(0.52,  0.72,  0.96);   // Pale sky blue

    if (sun_height < -0.15) {
        return night;
    } else if (sun_height < 0.12) {
        return mix(night, twilight, smoothstep(-0.15, 0.12, sun_height));
    } else {
        return mix(twilight, day, smoothstep(0.12, 0.5, sun_height));
    }
}

fn get_sun_disc_color(sun_height: f32) -> vec3<f32> {
    // Define colors in RGB and convert to BGR for rendering
    let sunset = rgb(1.0, 0.45, 0.05);  // Orange-red
    let noon   = rgb(1.0, 0.96, 0.88);  // Warm white
    return mix(sunset, noon, smoothstep(0.0, 0.4, sun_height));
}

// ===== Star Field =====
fn get_stars(view_dir: vec3<f32>, sun_height: f32) -> vec3<f32> {
    if (sun_height > 0.15) {
        return rgb(0.0, 0.0, 0.0);  // No stars in daylight
    }
    let visibility = smoothstep(0.15, -0.10, sun_height);

    // Two layers of stars at different angular densities for depth
    let v1    = floor(view_dir * 180.0);
    let h1    = hash(v1);
    let d1    = length(fract(view_dir * 180.0) - 0.5);
    let star1 = smoothstep(0.07, 0.0, d1) * select(0.0, h1 * 1.5, h1 > 0.97);

    let v2    = floor(view_dir * 320.0 + vec3<f32>(17.3, 31.7, 5.1));
    let h2    = hash(v2);
    let d2    = length(fract(view_dir * 320.0 + vec3<f32>(17.3, 31.7, 5.1)) - 0.5);
    let star2 = smoothstep(0.04, 0.0, d2) * select(0.0, h2 * 0.9, h2 > 0.985);

    // Slight blue-orange variation like real stars
    let star_col = mix(rgb(0.80, 0.85, 1.00), rgb(1.00, 0.95, 0.80), h1);
    return star_col * (star1 + star2) * visibility;
}

// ===== Main Sky Colour Calculation =====

fn calculate_sky_color(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    let view_dir = normalize(world_pos - camera_pos);
    let time     = camera.time;  // Elapsed seconds from CameraUniforms

    // --- Sun direction ---
    // Static midday position; swap for animated rotation once a time-of-day
    // system passes a cycle speed:
    //   let angle = time * 0.0002;
    //   let sun_dir = normalize(vec3(cos(angle), sin(angle) * 0.85, 0.3));
    let sun_dir    = normalize(vec3<f32>(0.4, 0.6, -0.5));
    let sun_height = sun_dir.y;                          // –1 … +1
    let sun_dot    = dot(view_dir, sun_dir);

    // === 1. Atmospheric sky gradient (Rayleigh-style) ===
    let zenith_col  = get_sky_zenith_color(sun_height);
    let horizon_col = get_sky_horizon_color(sun_height);

    var sky_color: vec3<f32>;
    if (view_dir.y < 0.0) {
        // Below horizon: dark ground fog fading to black
        let ground_t = saturate(-view_dir.y * 4.0);
        sky_color = mix(horizon_col * 0.30, rgb(0.01, 0.01, 0.01), ground_t);
    } else {
        // Exponential altitude blend (thicker atmosphere near horizon)
        let alt_t = 1.0 - exp(-view_dir.y * 3.5);
        sky_color = mix(horizon_col, zenith_col, alt_t);
    }

    // Mie forward-scatter: orange glow only near sunset/sunrise.
    // sunset_factor = 1.0 at horizon, ramps to 0.0 when sun_height >= 0.25,
    // so the midday sky stays clean blue with no warm tint.
    // let sunset_factor = clamp(1.0 - sun_height * 4.0, 0.0, 1.0);
    // if (sunset_factor > 0.0 && sun_height > -0.15) {
    //     let mie      = pow(max(0.0, sun_dot), 6.0) * 0.40;
    //     let mie_wide = pow(max(0.0, sun_dot), 2.0) * 0.10;
    //     let mie_str  = max(0.0, sun_height + 0.15) * 0.4 * sunset_factor;
    //     sky_color   += vec3<f32>(1.0, 0.50, 0.15) * (mie + mie_wide) * mie_str;
    // }

    // === 2. Night stars ===
    sky_color += get_stars(view_dir, sun_height);

    // === 3. Volumetric clouds — sampled BEFORE the sun disc ===
    //    Correct ray-slab intersection means clouds appear at all elevations,
    //    not just directly overhead.
    let cloud_density = get_cloud_coverage_for_ray(view_dir, camera_pos, time);

    if (cloud_density > 0.005) {
        // Improved 3D lighting for side views
        // Calculate lighting based on sun angle to cloud position
        let sun_alignment = dot(view_dir, sun_dir);
        
        // Base lighting from sun direction
        let lit_frac = smoothstep(-0.2, 0.6, sun_alignment);
        
        // Self-shadow: density above the cloud midpoint toward the sun darkens interiors.
        let self_shadow = get_cloud_self_shadow(view_dir, camera_pos, sun_dir, time);
        
        // Depth-based shading: clouds closer to viewer appear brighter
        let depth_fade = smoothstep(0.7, 0.3, cloud_density);

        let lit_col    = mix(
            rgb(1.0,  0.62, 0.30),  // Warm golden at sunset
            rgb(1.0,  0.98, 0.96),  // Cool bright white at noon
            smoothstep(0.0, 0.35, sun_height)
        );
        let shadow_col = mix(
            rgb(0.28, 0.20, 0.30),  // Deep violet-grey at sunset
            rgb(0.55, 0.62, 0.76),  // Cool blue-grey at noon
            smoothstep(0.0, 0.35, sun_height)
        );
        let night_col  = rgb(0.035, 0.035, 0.055); // Almost-black night cloud
        
        // Enhanced 3D shading: mix between lit and shadow with depth
        let base_shading = mix(shadow_col, lit_col, lit_frac * self_shadow);
        let depth_enhanced = mix(base_shading * 0.7, base_shading, depth_fade);
        let cloud_base  = depth_enhanced;
        let cloud_color = mix(night_col, cloud_base, smoothstep(-0.1, 0.12, sun_height));

        // Silver-lining: thin sunlit edges (attenuated where self-shadowed)
        let edge_bright = pow(1.0 - cloud_density, 3.0) * max(0.0, sun_height) * 0.45 * self_shadow;
        let silver      = lit_col * edge_bright;

        // Sky darkens under the cloud mass (shadow on the air below)
        let sky_shadow = 1.0 - cloud_density * 0.55;
        sky_color = mix(sky_color * sky_shadow, cloud_color + silver, cloud_density * 0.92);
    }

    // === 4. Sun disc — applied last, attenuated by cloud cover ===
    let sun_col  = get_sun_disc_color(sun_height);
    let sun_occl = cloud_density; // how much cloud is blocking the sun

    // Disc
    if (sun_height > -0.08 && sun_dot > SUN_DISC_SIZE) {
        let disc_t     = smoothstep(SUN_DISC_SIZE, 1.0, sun_dot);
        let brightness = mix(SUN_BRIGHTNESS_MIN, SUN_BRIGHTNESS_MAX, smoothstep(0.0, 0.4, sun_height));
        let atten      = 1.0 - sun_occl * 0.95;
        sky_color      = mix(sky_color, sun_col * brightness, disc_t * atten);
    }

    // Corona / inner glow
    if (sun_height > -0.10 && sun_dot > SUN_GLOW_SIZE) {
        let glow       = pow((sun_dot - SUN_GLOW_SIZE) / (1.0 - SUN_GLOW_SIZE), 2.0);
        let glow_col   = sun_col * mix(2.0, 7.0, smoothstep(0.0, 0.4, sun_height));
        sky_color     += glow_col * glow * 0.35 * (1.0 - sun_occl * 0.7);
    }

    return sky_color;
}

// ===== Material Dispatch =====

fn get_material_for_fragment(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> MaterialData {
    var mat: MaterialData;
    mat.base_color       = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    mat.metallic         = 0.0;
    mat.roughness        = 0.5;
    mat.emissive_strength = 0.0;
    mat.ao               = 1.0;

    // Sky sphere detection: fragments more than 400 units from the camera
    // belong to the inverted sky sphere
    let dist = length(world_pos - camera_pos);
    if (dist > 400.0) {
        let sky_color         = calculate_sky_color(world_pos, camera_pos);
        mat.base_color        = vec4<f32>(sky_color, 1.0);
        mat.emissive_strength = 1.5;
        mat.metallic          = 0.0;
        mat.roughness         = 1.0;
    }

    return mat;
}

// Material processing functions with procedural textures

fn checkerboard_pattern(uv: vec2<f32>, scale: f32) -> f32 {
    let scaled_uv = uv * scale;
    let checker = floor(scaled_uv.x) + floor(scaled_uv.y);
    return fract(checker * 0.5) * 2.0;
}

fn get_texture_color(uv: vec2<f32>) -> vec3<f32> {
    // Create a procedural checkerboard texture
    let checker = checkerboard_pattern(uv, 8.0);

    // Alternate between two colors
    let color1 = vec3<f32>(0.9, 0.9, 0.9); // Light gray
    let color2 = vec3<f32>(0.3, 0.5, 0.7); // Blue-gray

    return mix(color2, color1, checker);
}

fn apply_material_color(base_color: vec3<f32>, tex_coords: vec2<f32>, world_pos: vec3<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    // Get material data for this fragment
    let material = get_material_for_fragment(world_pos, camera_pos);
    
    // If emissive, skip texture and return bright emissive color
    if (material.emissive_strength > 0.0) {
        return material.base_color.rgb * material.emissive_strength;
    }
    
    // Normal textured material
    let texture_color = get_texture_color(tex_coords);
    return base_color * texture_color * material.base_color.rgb;
}

// Fragment-shader GI sampling — the sole lighting path in this pipeline.
// RadianceCascades replaces BasicLighting and ProceduralShadows entirely.
// Probe radiance (pre-computed per-voxel from real lights + shadows) is sampled
// here and multiplied by the surface albedo to produce the final lit colour.

struct GpuCascadeGI {
    center_and_extent: vec4<f32>,
    resolution_and_type: vec4<f32>,
    texture_layer: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct RCUniformsGI {
    params: vec4<f32>, // x=cascade_count, y=gi_intensity, z=mode, w=blend
    cascades: array<GpuCascadeGI, 4>,
}

@group(2) @binding(0) var radiance_probes: texture_2d_array<f32>;
@group(2) @binding(1) var gi_sampler: sampler;
@group(2) @binding(2) var<uniform> gi_uniforms: RCUniformsGI;

// ACES filmic tone mapping (Narkowicz 2015)
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(x: vec3<f32>) -> vec3<f32> {
    return pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

// Convert a world position to normalised [0,1]^3 UVW for a given cascade
fn gi_world_to_uvw(world_pos: vec3<f32>, cascade: GpuCascadeGI) -> vec3<f32> {
    let center = cascade.center_and_extent.xyz;
    let extent = cascade.center_and_extent.w;
    return (world_pos - center) / (2.0 * extent) + 0.5;
}

// Trilinear sample of a 2D-array texture where the Z axis is packed as layers
fn gi_sample_2d_array(uvw: vec3<f32>) -> vec3<f32> {
    let num_layers = f32(textureNumLayers(radiance_probes));
    let layer_f    = clamp(uvw.z * num_layers, 0.0, num_layers - 1.001);
    let layer0     = i32(floor(layer_f));
    let layer1     = min(layer0 + 1, i32(num_layers) - 1);
    let t          = fract(layer_f);
    let s0 = textureSampleLevel(radiance_probes, gi_sampler, uvw.xy, layer0, 0.0).rgb;
    let s1 = textureSampleLevel(radiance_probes, gi_sampler, uvw.xy, layer1, 0.0).rgb;
    return mix(s0, s1, t);
}

// Find the finest cascade that contains world_pos, sample its probes.
// Falls back to a constant ambient if no cascade covers the position.
fn sample_probe_radiance(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let cascade_count = i32(gi_uniforms.params.x);
    if cascade_count == 0 { return vec3<f32>(0.0); }

    for (var i = 0; i < cascade_count; i++) {
        let c   = gi_uniforms.cascades[i];
        let uvw = gi_world_to_uvw(world_pos, c);
        if all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0)) {
            // Small normal offset to reduce light-bleed through thin geometry
            let offset_uvw = clamp(uvw + normalize(world_normal) * 0.008, vec3<f32>(0.001), vec3<f32>(0.999));
            return gi_sample_2d_array(offset_uvw);
        }
    }
    return vec3<f32>(0.0);
}

// Entry point called from gi_sampling.wgsl at FragmentColorCalculation.
// `base_color` is the surface albedo (already set by BasicMaterials).
// Returns the final tone-mapped, gamma-encoded pixel colour.
fn apply_radiance_cascades_gi(base_color: vec3<f32>, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // Emissive surfaces (sky, light sources) bypass probe lighting entirely
    let mat = get_material_for_fragment(world_pos, camera.position);
    if mat.emissive_strength > 0.0 {
        return base_color;
    }

    let gi_intensity = gi_uniforms.params.y;
    let probe_radiance = sample_probe_radiance(world_pos, world_normal);
    let linear_colour  = base_color * probe_radiance * gi_intensity;

    return linear_to_srgb(aces_tonemap(linear_colour));
}



// Simple efficient bloom through shader injection
// Adds glow to overlit fragments using a quick approximation

// Apply bloom glow to overlit pixels
fn apply_bloom(color: vec3<f32>) -> vec3<f32> {
    // Calculate luminance
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    // Only bloom pixels brighter than 1.0
    let bloom_threshold = 1.0;
    let excess = max(0.0, luminance - bloom_threshold);
    
    // Create bloom glow proportional to excess brightness
    let bloom_intensity = 0.3;
    let bloom = color * (excess * bloom_intensity);
    
    return color + bloom;
}



@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = vec3<f32>(0.8);

        final_color = apply_material_color(final_color, input.tex_coords, input.world_position, camera.position);


    // Radiance cascades GI injection snippet
// This single line calls the apply_radiance_cascades_gi() function
// which is defined in gi_functions.wgsl (injected at FragmentPreamble)
    final_color = apply_radiance_cascades_gi(final_color, input.world_position, input.world_normal);

    final_color = apply_bloom(final_color);


    // INJECT_FRAGMENTPOSTPROCESS

    return vec4<f32>(final_color, 1.0);
}
