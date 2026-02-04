//! Helio - A feature-complete real-time rendering engine
//!
//! Helio provides all the advanced rendering capabilities found in modern game engines
//! like Unreal Engine, with a modular architecture where features can be optionally enabled.

pub use helio_animation as animation;
pub use helio_atmosphere as atmosphere;
pub use helio_core as core;
pub use helio_culling as culling;
pub use helio_decals as decals;
pub use helio_lighting as lighting;
pub use helio_material as material;
pub use helio_particles as particles;
pub use helio_postprocess as postprocess;
pub use helio_raytracing as raytracing;
pub use helio_render as render;
pub use helio_terrain as terrain;
pub use helio_ui as ui;
pub use helio_virtualtexture as virtualtexture;
pub use helio_water as water;

pub mod prelude {
    pub use crate::core::{
        Camera, Scene, Transform, Viewport, RenderContext, FrameGraph,
    };
    pub use crate::render::{
        Renderer, RenderPath, RendererConfig,
    };
    pub use crate::material::{
        Material, MaterialInstance,
    };
    pub use crate::lighting::{
        DirectionalLight, PointLight, SpotLight, AreaLight,
        GlobalIllumination, LightingMode,
    };
    pub use blade_graphics as gpu;
    pub use glam;
}
