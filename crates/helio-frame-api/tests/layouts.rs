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
