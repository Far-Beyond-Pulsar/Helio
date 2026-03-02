//! Feature system V2
//!
//! Features are modular rendering components that can be enabled/disabled at runtime.
//! Unlike the old system, features don't inject shader strings - instead they:
//!
//! 1. Register passes with the render graph
//! 2. Provide specialization constants for pipeline variants
//! 3. Contribute to shared bind groups

mod traits;
mod context;
mod registry;
pub mod lighting;
pub mod bloom;
pub mod shadows;
pub mod billboards;

pub use traits::{Feature, ShaderDefine, ShaderModulePath};
pub use context::{FeatureContext, PrepareContext};
pub use registry::{FeatureRegistry, FeatureFlags};
pub use lighting::{LightingFeature, LightConfig, LightType};
pub use bloom::BloomFeature;
pub use shadows::ShadowsFeature;
pub use billboards::{BillboardsFeature, BillboardInstance};

