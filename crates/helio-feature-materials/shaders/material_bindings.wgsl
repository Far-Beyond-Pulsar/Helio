// Material data structures and bindings

struct MaterialData {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive_strength: f32,
    ao: f32,
};

// ===== 3D Noise for Clouds =====
fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
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

fn fbm(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;
    
    for (var i = 0; i < 4; i++) {
        value += amplitude * noise3d(pos * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

// ===== Atmospheric Sky Colors =====
fn get_sky_gradient(height: f32) -> vec3<f32> {
    // More realistic blue sky gradient
    let horizon_color = vec3<f32>(0.6, 0.75, 0.95);  // Lighter blue at horizon
    let zenith_color = vec3<f32>(0.1, 0.35, 0.8);    // Deep blue at zenith
    return mix(horizon_color, zenith_color, pow(height, 0.5));
}

fn get_sun_color() -> vec3<f32> {
    return vec3<f32>(1.0, 0.98, 0.95);  // Very bright white with slight warmth
}

// ===== Cloud Rendering =====
fn get_cloud_density(world_pos: vec3<f32>, time: f32) -> f32 {
    // Cloud layer parameters
    let cloud_height_min = 200.0;
    let cloud_height_max = 400.0;
    let cloud_thickness = cloud_height_max - cloud_height_min;
    
    // Only clouds in certain height range
    if (world_pos.y < cloud_height_min || world_pos.y > cloud_height_max) {
        return 0.0;
    }
    
    // Normalize height within cloud layer
    let height_factor = (world_pos.y - cloud_height_min) / cloud_thickness;
    
    // Cloud coverage decreases at edges of layer
    let coverage = smoothstep(0.0, 0.2, height_factor) * smoothstep(1.0, 0.8, height_factor);
    
    // Animate clouds by offsetting noise
    let cloud_speed = vec3<f32>(0.5, 0.0, 0.3);  // Wind direction
    let animated_pos = world_pos * 0.003 + cloud_speed * time * 0.001;
    
    // Layered noise for cloud detail
    let base_noise = fbm(animated_pos);
    let detail_noise = fbm(animated_pos * 3.0) * 0.3;
    
    let density = (base_noise + detail_noise - 0.6) * coverage;
    return clamp(density, 0.0, 1.0);
}

// Calculate sky color based on view direction
fn calculate_sky_color(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    // Get view direction (from camera to fragment)
    let view_dir = normalize(world_pos - camera_pos);
    
    // Height in sky (0 = horizon, 1 = zenith)
    let height = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);
    
    // Base sky gradient
    var sky_color = get_sky_gradient(height);
    
    // Sun direction (elevated sun, slightly angled)
    let sun_dir = normalize(vec3<f32>(0.4, 0.6, -0.5));
    
    // Sun calculations
    let sun_dot = dot(view_dir, sun_dir);
    let sun_disc_size = 0.9998;  // Very tight cone for sun disc
    let sun_glow_size = 0.992;   // Wider cone for glow
    
    // Sun disc (VERY bright for bloom effect)
    if (sun_dot > sun_disc_size) {
        let sun_intensity = smoothstep(sun_disc_size, 1.0, sun_dot);
        sky_color = mix(sky_color, get_sun_color() * 50.0, sun_intensity);
    }
    // Sun glow/halo
    else if (sun_dot > sun_glow_size) {
        let glow_intensity = smoothstep(sun_glow_size, sun_disc_size, sun_dot);
        sky_color = mix(sky_color, get_sun_color() * 3.0, glow_intensity * 0.5);
    }
    
    // Atmospheric haze near horizon
    let horizon_factor = 1.0 - abs(view_dir.y);
    let horizon_glow = pow(horizon_factor, 4.0) * 0.3;
    sky_color += vec3<f32>(horizon_glow * 0.9, horizon_glow * 0.85, horizon_glow * 0.7);
    
    // Add clouds (simple approximation - actual raymarching would be better)
    // Use world position as proxy for cloud position
    let time = 0.0; // TODO: pass actual time uniform
    let sample_pos = camera_pos + view_dir * 300.0;  // Sample at fixed distance
    let cloud_density = get_cloud_density(sample_pos, time);
    
    if (cloud_density > 0.01) {
        let cloud_color = vec3<f32>(1.0, 1.0, 1.0);  // White clouds
        let cloud_shadow = 1.0 - cloud_density * 0.4;
        sky_color = mix(sky_color * cloud_shadow, cloud_color, cloud_density * 0.8);
    }
    
    return sky_color;
}

// Global material ID that can be set per-object
// For now, hardcoded materials by world position
fn get_material_for_fragment(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> MaterialData {
    var mat: MaterialData;
    mat.base_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    mat.metallic = 0.0;
    mat.roughness = 0.5;
    mat.emissive_strength = 0.0;
    mat.ao = 1.0;
    
    // Detect sky sphere by distance from camera (>400 units = sky sphere)
    let dist_from_camera = length(world_pos - camera_pos);
    if (dist_from_camera > 400.0) {
        // Sky sphere - calculate atmospheric sky color with clouds
        let sky_color = calculate_sky_color(world_pos, camera_pos);
        mat.base_color = vec4<f32>(sky_color, 1.0);
        mat.emissive_strength = 1.5; // Emissive so it's not affected by lighting
        mat.metallic = 0.0;
        mat.roughness = 1.0;
    }
    
    return mat;
}
