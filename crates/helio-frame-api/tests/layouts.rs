//! The frame's WGSL contracts have the sizes their Rust constants promise:
//! a consumer sizes and binds the buffers by the constants, a shader reads
//! them by the structs.

use helio_frame_api::lighting::sky::{SKY_FRAME_BYTES, SKY_SNIPPET};

/// Byte size of struct `name` in `source` (naga's WGSL layout).
fn struct_size(source: &str, name: &str) -> u64 {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .expect("the module lays out");
    let (handle, _) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no struct `{name}`"));
    u64::from(layouter[handle].size)
}

#[test]
fn the_sky_frame_has_its_contract_size() {
    let source = helio_core::shader::resolve_with(
        "//!use frame_sky\n@group(0) @binding(0) var<storage, read> sky_frame: SkyFrame;\n",
        &[SKY_SNIPPET],
    );
    assert_eq!(struct_size(&source, "SkyFrame"), SKY_FRAME_BYTES);
}

/// Parses and validates `source` (types, packing builtins).
fn validate(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
}

/// The ray stream's rows are the sizes their Rust mirrors are
/// (`helio_frame_api::rays::stream`), and a surface goes to a row and back
/// through the frame's `SurfaceData`.
#[test]
fn the_ray_stream_rows_have_their_contract_sizes() {
    use helio_frame_api::gbuffer::SURFACE_SNIPPET;
    use helio_frame_api::rays::stream::{
        SceneHit, SceneRay, SceneSurface, StreamHeader, SNIPPET, SURFACE_SNIPPET as ROWS,
    };
    let source = helio_core::shader::resolve_with(
        "//!use frame_ray_stream
         //!use frame_ray_surface
         //!use frame_surface
         @group(0) @binding(0) var<storage, read> header: RayStreamHeader;
         @group(0) @binding(1) var<storage, read> rays: array<SceneRay>;
         @group(0) @binding(2) var<storage, read_write> hits: array<SceneHit>;
         @group(0) @binding(3) var<storage, read_write> surfaces: array<SceneSurface>;
         @compute @workgroup_size(RAY_STREAM_GROUP)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
             if id.x >= header.count { return; }
             var ray = rays[id.x];
             ray.cone = scene_ray_cone(scene_ray_width(ray), scene_ray_spread(ray), RAY_MASK_ALL);
             if !scene_ray_sees_space(ray, 3u) { ray.cone = scene_cone_in_space(ray.cone, scene_space_of_slot(0u)); }
             if scene_ray_space(ray) == SCENE_SPACE_ANY { return; }
             let reach = scene_hit_reach(ray, hits[id.x]);
             let n = scene_unpack_unit(scene_pack_unit(ray.dir));
             hits[id.x] = scene_nearest(1u, reach, false, scene_ray_mask(ray), 0u, 0u);
             hits[id.x] = scene_occlusion(scene_occlusion_through(hits[id.x]) * scene_ray_footprint(ray, reach));
             let s = surface_with_transmission(surface_data(vec3<f32>(0.5), n, 1.0, 0.0, scene_unpack_radiance(scene_pack_radiance(n))), vec3<f32>(0.25));
             surfaces[id.x] = scene_surface_pack(s, n);
             let back = scene_surface_unpack(surfaces[id.x]);
             surfaces[id.x].face = scene_pack_unit(scene_surface_face(surfaces[id.x]) + surface_transmission(back));
         }
",
        &[SNIPPET, ROWS, SURFACE_SNIPPET],
    );
    assert_eq!(struct_size(&source, "RayStreamHeader"), StreamHeader::BYTES);
    assert_eq!(struct_size(&source, "SceneRay"), SceneRay::BYTES);
    assert_eq!(struct_size(&source, "SceneHit"), SceneHit::BYTES);
    assert_eq!(struct_size(&source, "SceneSurface"), SceneSurface::BYTES);
    validate(&source);
}

/// The G-buffer's encoding of a surface: a drawer's output and a lighting
/// model's decode validate against the frame's targets.
#[test]
fn the_gbuffer_encodes_and_decodes_a_surface() {
    use helio_frame_api::gbuffer::{OUTPUT_SNIPPET, SURFACE_SNIPPET};
    let source = helio_core::shader::resolve_with(
        "//!use gbuffer_output
         //!use frame_surface
         @group(0) @binding(0) var albedo: texture_2d<f32>;
         @fragment
         fn main(@builtin(position) p: vec4<f32>) -> GBufferOutput {
             let t = textureLoad(albedo, vec2<i32>(p.xy), 0);
             let s = gbuffer_surface_at(GBufferSurfaceTexels(t, t, t, t, t, t));
             return gbuffer_output(s, vec2<f32>(-1.0), vec4<f32>(0.0));
         }
         @fragment
         fn with_depth(@builtin(position) p: vec4<f32>) -> GBufferDepthOutput {
             let s = surface_data(vec3<f32>(0.5), vec3<f32>(0.0, 1.0, 0.0), 0.5, 0.0, vec3<f32>(0.0));
             return gbuffer_depth_output(s, vec2<f32>(-1.0), vec4<f32>(0.0), 0.5);
         }
",
        &[OUTPUT_SNIPPET, SURFACE_SNIPPET],
    );
    validate(&source);
}

/// The light samples' rows are the sizes their Rust mirrors are, and a
/// sampler and a consumer written against the snippet validate: a test
/// sampler of one point light and one directional light, merging its offer
/// into the row as every sampler does.
#[test]
fn the_light_sample_rows_have_their_contract_sizes() {
    use helio_frame_api::rays::light_samples::{LightQuery, LightSample, SNIPPET};
    use helio_frame_api::rays::stream::SNIPPET as STREAM;
    let source = helio_core::shader::resolve_with(
        "//!use frame_ray_stream
         //!use frame_light_samples
         @group(0) @binding(0) var<storage, read> header: RayStreamHeader;
         @group(0) @binding(1) var<storage, read> queries: array<LightQuery>;
         @group(0) @binding(2) var<storage, read_write> samples: array<LightSample>;
         @compute @workgroup_size(RAY_STREAM_GROUP)
         fn sample(@builtin(global_invocation_id) id: vec3<u32>) {
             if id.x >= header.count { return; }
             let q = queries[id.x];
             // Uniformly one of two lights: a point at (0, 4, 0) and the sun.
             let pick = light_query_uniform(q, 0u) < 0.5;
             var offer: LightSample;
             if pick {
                 let to = vec3<f32>(0.0, 4.0, 0.0) - q.position;
                 let d = length(to);
                 if !light_query_faces(q, to / d) || !light_query_sees_space(q, 0u) { return; }
                 offer = light_sample_of(vec3<f32>(100.0) / (d * d) * 2.0, to / d, d, 0.0, 1u, 0u);
             } else {
                 let sun = normalize(vec3<f32>(0.3, 1.0, 0.2));
                 if !light_query_faces(q, sun) { return; }
                 offer = light_sample_of(vec3<f32>(3.0) * 2.0, sun, LIGHT_SAMPLE_FAR, 0.0, 1u, 1u);
             }
             samples[id.x] = light_sample_offer(samples[id.x], offer, q, 1u);
         }
         @compute @workgroup_size(RAY_STREAM_GROUP)
         fn consume(@builtin(global_invocation_id) id: vec3<u32>) {
             let s = samples[id.x];
             if !light_sample_found(s) { return; }
             let q = light_query_point(vec3<f32>(0.0), 0.01, light_sample_dir(s), false, light_query_position_uniforms(queries[id.x]), 7u, SCENE_SPACE_ANY);
             var out = s;
             out.radiance = light_sample_radiance(s) * light_sample_pdf(s) * light_sample_distance(s) * f32(q.space);
             samples[id.x] = out;
         }
",
        &[STREAM, SNIPPET],
    );
    assert_eq!(struct_size(&source, "LightQuery"), LightQuery::BYTES);
    assert_eq!(struct_size(&source, "LightSample"), LightSample::BYTES);
    validate(&source);
}

/// The radiance cache's rows are the sizes their Rust mirrors are, and a
/// reader's read and mark validate (a probe or a pixel with a hit).
#[test]
fn the_radiance_cache_rows_have_their_contract_sizes() {
    use helio_frame_api::rays::radiance_cache::{
        RadianceCacheEntry, RadianceCacheFrame, RadianceCacheKey, RadianceCacheSlot, SNIPPET,
    };
    use helio_frame_api::rays::stream::SNIPPET as STREAM;
    let source = helio_core::shader::resolve_with(
        "//!use frame_ray_stream
         //!use frame_radiance_cache
         @group(0) @binding(0) var<uniform> radiance_cache_frame: RadianceCacheFrame;
         @group(0) @binding(1) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
         @group(0) @binding(2) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
         @group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
         @compute @workgroup_size(64)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
             let p = vec3<f32>(f32(id.x), 1.0, 2.0);
             let n = vec3<f32>(0.0, 1.0, 0.0);
             let e = radiance_cache_read(p, n, 0.25, SCENE_SPACE_ANY, RADIANCE_CACHE_LEVELS);
             radiance_cache_mark(p, n, 0.25, SCENE_SPACE_ANY);
             let at = radiance_cache_entry_point(radiance_cache_entries[id.x]);
             out[id.x] = vec4<f32>(e.direct + e.indirect + at, select(0.0, 1.0, e.found) + f32(e.level) * radiance_cache_size(e.level));
         }
",
        &[STREAM, SNIPPET],
    );
    assert_eq!(
        struct_size(&source, "RadianceCacheFrame"),
        RadianceCacheFrame::BYTES
    );
    assert_eq!(
        struct_size(&source, "RadianceCacheSlot"),
        RadianceCacheSlot::BYTES
    );
    assert_eq!(
        struct_size(&source, "RadianceCacheKey"),
        std::mem::size_of::<RadianceCacheKey>() as u64
    );
    assert_eq!(
        struct_size(&source, "RadianceCacheEntry"),
        RadianceCacheEntry::BYTES
    );
    validate(&source);
}
