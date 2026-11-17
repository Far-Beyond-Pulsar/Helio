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

/// The ray stream's rows are the sizes their Rust mirrors are
/// (`helio_frame_api::rays::stream`).
#[test]
fn the_ray_stream_rows_have_their_contract_sizes() {
    use helio_frame_api::rays::stream::{SceneHit, SceneRay, SceneSurface, StreamHeader, SNIPPET};
    let source = helio_core::shader::resolve_with(
        "//!use frame_ray_stream
         @group(0) @binding(0) var<storage, read> header: RayStreamHeader;
         @group(0) @binding(1) var<storage, read> rays: array<SceneRay>;
         @group(0) @binding(2) var<storage, read_write> hits: array<SceneHit>;
         @group(0) @binding(3) var<storage, read_write> surfaces: array<SceneSurface>;
         @compute @workgroup_size(RAY_STREAM_GROUP)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
             if id.x >= header.count { return; }
             let ray = rays[id.x];
             let reach = scene_hit_reach(ray, hits[id.x]);
             let n = scene_unpack_unit(scene_pack_unit(ray.dir));
             hits[id.x] = scene_hit(1u, reach, false, scene_ray_mask(ray), 0u, 0u);
             hits[id.x] = scene_occlusion(scene_occlusion_through(hits[id.x]) * scene_ray_spread(ray));
             surfaces[id.x] = scene_surface(n, n, vec3<f32>(0.5), 0.0, 1.0, 0.0, scene_unpack_radiance(scene_pack_radiance(n)));
         }
",
        &[SNIPPET],
    );
    assert_eq!(struct_size(&source, "RayStreamHeader"), StreamHeader::BYTES);
    assert_eq!(struct_size(&source, "SceneRay"), SceneRay::BYTES);
    assert_eq!(struct_size(&source, "SceneHit"), SceneHit::BYTES);
    assert_eq!(struct_size(&source, "SceneSurface"), SceneSurface::BYTES);
    // The snippet's functions validate (types, packing builtins).
    let module = naga::front::wgsl::parse_str(&source).expect("parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("the ray stream snippet validates");
}
