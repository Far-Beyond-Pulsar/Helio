//! The sky: the authored atmosphere (Hillaire 2020) and the passes that
//! render it.
//!
//! [`AtmospherePass`] resolves the first enabled [`AtmosphereComponent`] row
//! and the first directional light into the atmosphere LUTs and frame that
//! lighting reads; [`AtmosphereCompositePass`] draws the sky where nothing
//! was drawn and aerial perspective over geometry. Without an atmosphere row
//! neither draws anything, and the sky stays the graph's black background.
//!
//! The procedural `SkyPass` and its `SkyComponent` (`sky_components`) are
//! gone (Pulsar-Native #1057): a scene has one sky, its atmosphere. Clouds
//! will come back as part of the atmosphere.

pub mod atmosphere;
pub mod gpu_types;
pub use atmosphere::{AtmosphereComponent, AtmosphereCompositePass, AtmospherePass};
pub use gpu_types::*;
