fn hash33(p: vec3<f32>) -> vec3<f32> {
    let d = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(abs(d)) * vec3<f32>(43758.5453, 22531.1719, 31241.1239));
}

fn opal_cell_noise(p: vec3<f32>) -> vec3<f32> {
    let cell = floor(p);
    let frac = p - cell;
    let f = frac * frac * (3.0 - 2.0 * frac);
    let c0 = hash33(cell + vec3<f32>(0.0, 0.0, 0.0));
    let c1 = hash33(cell + vec3<f32>(1.0, 0.0, 0.0));
    let c2 = hash33(cell + vec3<f32>(0.0, 1.0, 0.0));
    let c3 = hash33(cell + vec3<f32>(1.0, 1.0, 0.0));
    let c4 = hash33(cell + vec3<f32>(0.0, 0.0, 1.0));
    let c5 = hash33(cell + vec3<f32>(1.0, 0.0, 1.0));
    let c6 = hash33(cell + vec3<f32>(0.0, 1.0, 1.0));
    let c7 = hash33(cell + vec3<f32>(1.0, 1.0, 1.0));
    let xy0 = mix(mix(c0, c1, f.x), mix(c2, c3, f.x), f.y);
    let xy1 = mix(mix(c4, c5, f.x), mix(c6, c7, f.x), f.y);
    return mix(xy0, xy1, f.z);
}

fn radiant_eval_surface(material: GpuMaterial,
                        material_tex: MaterialTextureData,
                        input: VertexOutput) -> SurfaceData {
    var s = default_pbr_surface(material, material_tex, input);

    let V = normalize(camera.position_near.xyz - input.world_position);
    let R = reflect(-V, s.normal);
    let play = opal_cell_noise(input.world_position * 3.0 + R * 0.4);

    s.albedo.rgb = mix(vec3<f32>(0.9, 0.85, 0.8), play, 0.9);
    s.emissive = play * 3.0;
    s.roughness = 0.05;
    s.metallic = 0.0;
    s.specular_f0 = vec3<f32>(0.034);

    // RADIANT_OVERRIDE_SURFACE
    // RADIANT_OVERRIDE_END

    return s;
}
