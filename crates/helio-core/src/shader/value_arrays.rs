//! Finds arrays a shader indexes at run time through a value rather than
//! through their binding.
//!
//! An array reached through a pointer (a binding, a `var`) is indexed in
//! place. One reached through a value (a by-value function argument, a
//! struct or array loaded whole) cannot be: naga's SPIR-V writer stores the
//! whole value into a fresh function variable and indexes that, and the HLSL
//! and MSL writers index a local copy. For a per-frame table that is
//! kilobytes written to per-invocation memory on every call, which a
//! full-screen pass pays at every pixel: an `AtmosphereFrame` passed by value
//! with its 96 directional-light rows indexed in a loop made the deferred
//! lighting pass write two 1.5 KiB copies a pixel. WGSL has no uniform or
//! storage pointer parameters, so a shared WGSL function that indexes such a
//! table must read it through the binding by name.
//!
//! [`value_array_copies`] lists every such index in a composed shader whose
//! array is a table (`TABLE_BYTES` or more); tests assert that list is
//! empty.

use naga::valid::{Capabilities, ValidationFlags, Validator};

/// The smallest array counted as a table. Smaller local arrays (a few
/// harmonics' coefficients) are kept in registers once their constant-trip
/// loops unroll; a per-frame table is not.
const TABLE_BYTES: u32 = 256;

/// The run-time indexes of `source` (a composed shader, as
/// [`resolve_with`](super::resolve_with) builds it) into value arrays of at
/// least `TABLE_BYTES`, one line each: the function and the array's size.
/// `Err` when the shader does not parse or validate.
pub fn value_array_copies(source: &str) -> Result<Vec<String>, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("{e:?}"))?;
    let mut found = Vec::new();
    let mut scan = |name: &str, function: &naga::Function, info: &naga::valid::FunctionInfo| {
        for (handle, expression) in function.expressions.iter() {
            let naga::Expression::Access { base, .. } = *expression else {
                continue;
            };
            let naga::TypeInner::Array {
                size: naga::ArraySize::Constant(count),
                stride,
                ..
            } = *info[base].ty.inner_with(&module.types)
            else {
                continue;
            };
            let bytes = count.get().saturating_mul(stride);
            if bytes >= TABLE_BYTES {
                found.push(format!(
                    "`{name}` indexes a {bytes}-byte array by value ({handle:?})"
                ));
            }
        }
    };
    for (handle, function) in module.functions.iter() {
        let name = function.name.as_deref().unwrap_or("?");
        scan(name, function, &info[handle]);
    }
    for (index, entry) in module.entry_points.iter().enumerate() {
        scan(&entry.name, &entry.function, info.get_entry_point(index));
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::value_array_copies;

    const TABLE: &str = "
        struct Frame { rows: array<vec4<f32>, 64>, count: u32 }
        @group(0) @binding(0) var<uniform> frame: Frame;
        @group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
    ";

    #[test]
    fn a_table_indexed_through_a_by_value_argument_is_found() {
        let source = format!(
            "{TABLE}
            fn sum(f: Frame) -> vec4<f32> {{
                var s = vec4<f32>(0.0);
                for (var i = 0u; i < f.count; i++) {{ s += f.rows[i]; }}
                return s;
            }}
            @compute @workgroup_size(1) fn main() {{ out[0] = sum(frame); }}"
        );
        let found = value_array_copies(&source).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("`sum`") && found[0].contains("1024-byte"), "{found:?}");
    }

    #[test]
    fn a_table_indexed_through_its_binding_is_not() {
        let source = format!(
            "{TABLE}
            fn sum() -> vec4<f32> {{
                var s = vec4<f32>(0.0);
                for (var i = 0u; i < frame.count; i++) {{ s += frame.rows[i]; }}
                return s;
            }}
            @compute @workgroup_size(1) fn main() {{ out[0] = sum(); }}"
        );
        assert_eq!(value_array_copies(&source).unwrap(), Vec::<String>::new());
    }
}
