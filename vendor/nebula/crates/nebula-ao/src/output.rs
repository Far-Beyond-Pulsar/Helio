use serde::{Deserialize, Serialize};
use nebula_core::traits::BakeOutput;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AoOutput {
    pub width:   u32,
    pub height:  u32,
    /// R32F texels, row-major.  Each f32 ∈ [0.0, 1.0].
    pub texels:  Vec<u8>,
    pub config_json: String,
}

impl BakeOutput for AoOutput {
    fn kind_name() -> &'static str { "ao" }
}
