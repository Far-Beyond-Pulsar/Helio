//! Canonical names for cross-pass frame resources.
//!
//! The render graph intentionally remains open-ended: individual passes can
//! publish private resources without changing `helio-core`.  These names are
//! the small set of contracts shared by the renderer facade and multiple pass
//! families.  The string constants are useful when inspecting graph state;
//! the generic helpers preserve the type marker required by [`ResourceKey`]
//! without making core depend on any pass crate.

use crate::{ResourceKey, ResourceRegistry};

/// Material descriptor bindings published for material-consuming passes.
pub const MATERIAL_TEXTURES: &str = "material_textures";
/// Current coordinate-space table published for culling and geometry passes.
pub const COORDINATE_SPACES: &str = "coordinate_spaces";
/// Previous-frame coordinate-space table, when exposed as an independent resource.
pub const COORDINATE_SPACES_PREV: &str = "coordinate_spaces_prev";
/// Per-frame environment values shared by lighting passes.
pub const RENDER_ENVIRONMENT: &str = "render_environment";
/// Shadow matrix data published by the shadow-matrix producer.
pub const SHADOW_MATRICES: &str = "shadow_matrices";
/// Optional baked lightmap view.
pub const BAKED_LIGHTMAP: &str = "baked_lightmap";
/// SceneDB buffer name containing projected portal views.
pub const PORTAL_VIEWS: &str = "portal_views";
/// What this frame draws into depth; see [`depth_draw_signature`].
pub const DEPTH_DRAW_SIGNATURE: &str = "depth_draw_signature";

/// Returns the canonical material-texture contract key for `T`.
#[inline]
pub const fn material_textures<T>() -> ResourceKey<T> {
    ResourceKey::new(MATERIAL_TEXTURES)
}

/// Returns the canonical coordinate-space contract key for `T`.
#[inline]
pub const fn coordinate_spaces<T>() -> ResourceKey<T> {
    ResourceKey::new(COORDINATE_SPACES)
}

/// Returns the canonical previous-coordinate-space contract key for `T`.
#[inline]
pub const fn coordinate_spaces_prev<T>() -> ResourceKey<T> {
    ResourceKey::new(COORDINATE_SPACES_PREV)
}

/// Returns the canonical render-environment contract key for `T`.
#[inline]
pub const fn render_environment<T>() -> ResourceKey<T> {
    ResourceKey::new(RENDER_ENVIRONMENT)
}

/// Returns the canonical shadow-matrices contract key for `T`.
#[inline]
pub const fn shadow_matrices<T>() -> ResourceKey<T> {
    ResourceKey::new(SHADOW_MATRICES)
}

/// Returns the canonical baked-lightmap contract key for `T`.
#[inline]
pub const fn baked_lightmap<T>() -> ResourceKey<T> {
    ResourceKey::new(BAKED_LIGHTMAP)
}

/// Returns the frame's depth-draw signature key.
///
/// A value that changes whenever what this frame draws into depth changes for
/// a reason `camera_generation` and the SceneDB content signature do not
/// capture -- chiefly GPU->CPU readback latency: objects uploaded before the
/// first frame only start drawing once their draw counts reach the CPU, frames
/// later, with camera and scene unchanged throughout.
///
/// Passes that cache work derived from a frame's depth (Hi-Z) treat that depth
/// as current only while camera, scene and this signature all hold. Depth
/// producers contribute with [`fold_depth_draw_signature`]; readers declare
/// [`DEPTH_DRAW_SIGNATURE`] in `reads()` so they run after every producer.
#[inline]
pub const fn depth_draw_signature() -> ResourceKey<u64> {
    ResourceKey::new(DEPTH_DRAW_SIGNATURE)
}

/// Folds `value` into this frame's [`depth_draw_signature`]. Order-independent,
/// so any number of depth producers can contribute.
pub fn fold_depth_draw_signature(registry: &mut ResourceRegistry<'_>, value: u64, writer: &'static str) {
    let mut h = value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = (h ^ (h >> 31)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let sum = registry.get(depth_draw_signature()).unwrap_or(0);
    registry.write(depth_draw_signature(), sum.wrapping_add(h ^ (h >> 29)), writer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_names_are_stable() {
        assert_eq!(material_textures::<u32>().name(), "material_textures");
        assert_eq!(coordinate_spaces::<u32>().name(), "coordinate_spaces");
        assert_eq!(coordinate_spaces_prev::<u32>().name(), "coordinate_spaces_prev");
        assert_eq!(render_environment::<u32>().name(), "render_environment");
        assert_eq!(shadow_matrices::<u32>().name(), "shadow_matrices");
        assert_eq!(baked_lightmap::<u32>().name(), "baked_lightmap");
        assert_eq!(PORTAL_VIEWS, "portal_views");
        assert_eq!(depth_draw_signature().name(), DEPTH_DRAW_SIGNATURE);
    }

    #[test]
    fn depth_draw_signature_folds_order_independently() {
        let mut a = ResourceRegistry::empty();
        fold_depth_draw_signature(&mut a, 1, "a");
        fold_depth_draw_signature(&mut a, 2, "b");
        let mut b = ResourceRegistry::empty();
        fold_depth_draw_signature(&mut b, 2, "b");
        fold_depth_draw_signature(&mut b, 1, "a");
        assert_eq!(a.get(depth_draw_signature()), b.get(depth_draw_signature()));
        assert_ne!(a.get(depth_draw_signature()), Some(0));
    }
}
