//! The Rust structs the pass uploads and the WGSL structs the shader reads are
//! the same layout, field by field, and the shader is valid WGSL.

use std::mem::{offset_of, size_of};

use helio_pass_status_screen::abi::{Line, Params, Rect};

const SOURCE: &str = include_str!("../shaders/status_screen.wgsl");

fn module() -> naga::Module {
    let module = naga::front::wgsl::parse_str(SOURCE)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(SOURCE)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("the status screen shader validates");
    module
}

/// The (name, offset) of every member of the shader's struct `name`, and its size.
fn shader_struct(module: &naga::Module, name: &str) -> (Vec<(String, u32)>, u32) {
    for (_, ty) in module.types.iter() {
        if ty.name.as_deref() != Some(name) {
            continue;
        }
        if let naga::TypeInner::Struct { members, span } = &ty.inner {
            let members = members
                .iter()
                .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                .collect();
            return (members, *span);
        }
    }
    panic!("the shader declares no struct `{name}`");
}

fn assert_same(
    module: &naga::Module,
    name: &str,
    rust_size: usize,
    rust_fields: &[(&str, usize)],
) {
    let (fields, span) = shader_struct(module, name);
    assert_eq!(
        fields.len(),
        rust_fields.len(),
        "{name}: the shader and Rust disagree on the number of fields ({fields:?})"
    );
    for ((shader_name, shader_offset), (rust_name, rust_offset)) in fields.iter().zip(rust_fields) {
        assert_eq!(shader_name, rust_name, "{name}: field order");
        assert_eq!(
            *shader_offset as usize, *rust_offset,
            "{name}.{shader_name}: offset"
        );
    }
    assert_eq!(span as usize, rust_size, "{name}: size");
}

#[test]
fn params_match_the_shader() {
    assert_same(
        &module(),
        "Params",
        size_of::<Params>(),
        &[
            ("screen", offset_of!(Params, screen)),
            ("unit", offset_of!(Params, unit)),
            ("flags", offset_of!(Params, flags)),
            ("dialog", offset_of!(Params, dialog)),
            ("shape", offset_of!(Params, shape)),
            ("glyphs", offset_of!(Params, glyphs)),
            ("counts", offset_of!(Params, counts)),
            ("text", offset_of!(Params, text)),
            ("icon", offset_of!(Params, icon)),
            ("void_color", offset_of!(Params, void_color)),
            ("shadow", offset_of!(Params, shadow)),
            ("panel_a", offset_of!(Params, panel_a)),
            ("panel_b", offset_of!(Params, panel_b)),
            ("edge", offset_of!(Params, edge)),
            ("rim", offset_of!(Params, rim)),
            ("chip_border", offset_of!(Params, chip_border)),
            ("icon_ring", offset_of!(Params, icon_ring)),
            ("icon_ink", offset_of!(Params, icon_ink)),
        ],
    );
    // A uniform's size is a multiple of 16.
    assert_eq!(size_of::<Params>() % 16, 0);
}

#[test]
fn line_matches_the_shader() {
    assert_same(
        &module(),
        "Line",
        size_of::<Line>(),
        &[
            ("place", offset_of!(Line, place)),
            ("color", offset_of!(Line, color)),
            ("text", offset_of!(Line, text)),
        ],
    );
}

#[test]
fn rect_matches_the_shader() {
    assert_same(
        &module(),
        "Rect",
        size_of::<Rect>(),
        &[
            ("bounds", offset_of!(Rect, bounds)),
            ("color", offset_of!(Rect, color)),
            ("shape", offset_of!(Rect, shape)),
        ],
    );
}

#[test]
fn the_shader_declares_the_bindings_the_pass_binds() {
    let module = module();
    let mut bindings: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
        .collect();
    bindings.sort_unstable();
    assert_eq!(bindings, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
}
