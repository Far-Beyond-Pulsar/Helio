//! The shipped Radiant template sources, for what compiles a surface from
//! them outside this crate (the meshes' drawers compose [`DEFAULT_PBR`] and
//! [`TRANSPARENT_BASE`]; hosts register the others as custom templates).

/// The default surface template (class 0, `default_pbr`) and the base every
/// partial template's `radiant_eval_surface` is composed into: a complete
/// G-buffer surface program over the meshes' vertex and instance layout.
/// (Templates are whole programs today; splitting the surface function from
/// the drawer that hosts it belongs to the materials' own step.)
pub const DEFAULT_PBR: &str = include_str!("../templates/default_pbr.wgsl");
pub const TRANSPARENT_BASE: &str = include_str!("../templates/transparent_base.wgsl");
pub const OPAL: &str = include_str!("../templates/opal.wgsl");
pub const GLASS: &str = include_str!("../templates/glass.wgsl");
pub const GLASS_TRANSPARENT: &str = include_str!("../templates/glass_transparent.wgsl");
pub const WATER: &str = include_str!("../templates/water.wgsl");
pub const WATER_TRANSPARENT: &str = include_str!("../templates/water_transparent.wgsl");
