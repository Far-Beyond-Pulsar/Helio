//! # nebula-light
//!
//! GPU path-traced lightmap baking.
//!
//! ## Algorithm
//!
//! 1. **Texel mapping** — each lightmap texel is mapped to a world-space position
//!    and normal via the mesh's lightmap UV set.
//! 2. **Direct illumination** — per-texel analytic direct light evaluation with
//!    GPU ray-cast shadow testing.
//! 3. **Multi-bounce GI** — iterative hemisphere path tracing dispatched as
//!    compute workgroups; radiance accumulates into ping-pong RGBA32F textures.
//! 4. **Denoising** — optional simple spatial filter pass to reduce noise.
//! 5. **Pack & export** — the final float result is read back and stored via
//!    the `nebula-serialize` chunk format.

pub mod baker;
pub mod config;
pub mod output;

pub use baker::LightmapBaker;
pub use config::LightmapConfig;
pub use output::{LightmapOutput, AtlasRegion, CHUNK_TAG};
