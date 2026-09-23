//! Canonical names for cross-pass frame resources.
//!
//! The render graph intentionally remains open-ended: individual passes can
//! publish private resources without changing `helio-core`.  These names are
//! the small set of contracts shared by the renderer facade and multiple pass
//! families.  The string constants are useful when inspecting graph state;
//! the generic helpers preserve the type marker required by [`ResourceKey`]
//! without making core depend on any pass crate.

use crate::ResourceKey;

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
    }
}
