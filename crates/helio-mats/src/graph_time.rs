//! The clock behind shader-graph `time` nodes.
//!
//! Material graphs read time through `radiant_graph_time()`, which Radiant
//! defines from the host's `Globals.time` (see
//! [`crate::RadiantTemplate::build_shader_source`]). Every pass that uploads
//! `Globals` writes this value, so all passes — and the editor's mesh
//! viewer — agree on it.

use std::sync::OnceLock;
use std::time::Instant;

/// Seconds since the first call in this process. Monotonic and shared by
/// every pass, so a material animates identically in the G-buffer, forward
/// and transparent paths. `f32` keeps sub-frame precision for many hours.
pub fn graph_time_seconds() -> f32 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

#[cfg(test)]
mod tests {
    #[test]
    fn time_is_monotonic() {
        let a = super::graph_time_seconds();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(super::graph_time_seconds() > a);
    }
}
