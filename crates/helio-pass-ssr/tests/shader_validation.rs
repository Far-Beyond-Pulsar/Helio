//! Parses and validates the SSR WGSL shaders.
//!
//! wgpu only compiles shaders at `create_shader_module`, i.e. at runtime with a
//! live device — so `cargo check` will happily pass on WGSL that cannot compile.
//! ssr_denoise.wgsl sat broken in-tree for exactly that reason: it negated a
//! `u32` loop bound (`-(KERNEL_HALF)`), which naga rejects.

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name} failed to parse:\n{}", e.emit_to_string(source)));

    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} failed validation:\n{e:?}"));
}

#[test]
fn trace_shader_is_valid() {
    validate("ssr_trace.wgsl", include_str!("../shaders/ssr_trace.wgsl"));
}

#[test]
fn denoise_shader_is_valid() {
    validate("ssr_denoise.wgsl", include_str!("../shaders/ssr_denoise.wgsl"));
}

#[test]
fn temporal_shader_is_valid() {
    validate("ssr_temporal.wgsl", include_str!("../shaders/ssr_temporal.wgsl"));
}
