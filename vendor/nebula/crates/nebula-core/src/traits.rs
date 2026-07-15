use crate::{context::BakeContext, error::NebulaError, progress::ProgressReporter, scene::SceneGeometry};
use async_trait::async_trait;

/// Marker trait for any type that can be used as bake-pass input configuration.
pub trait BakeInput: Send + Sync + 'static {}

/// Marker trait for the output produced by a [`BakePass`].
pub trait BakeOutput: Send + Sync + 'static {
    /// A short human-readable name used in log messages.
    fn kind_name() -> &'static str where Self: Sized;
}

/// A single GPU-accelerated bake pass.
///
/// This is the serde `Serializer` analogue.  Each baker crate (nebula-light,
/// nebula-ao, …) implements this trait once.
///
/// ```rust,ignore
/// let output = LightmapBaker.execute(&scene, &LightmapConfig::default(), &ctx, &NullReporter).await?;
/// ```
#[async_trait(?Send)]
pub trait BakePass: Send + Sync {
    type Input:  BakeInput;
    type Output: BakeOutput;

    /// Human-readable name of this pass (e.g. `"lightmap"`, `"ao"`, …)
    fn name(&self) -> &'static str;

    /// Execute the bake on the GPU.
    async fn execute(
        &self,
        scene:    &SceneGeometry,
        input:    &Self::Input,
        ctx:      &BakeContext,
        reporter: &dyn ProgressReporter,
    ) -> Result<Self::Output, NebulaError>;
}

/// Persistence trait — each serialization backend implements this.
///
/// Corresponds to serde's format crates (json, bincode, …).
pub trait BakeSerializer<O: BakeOutput> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn serialize<W: std::io::Write>(&self, output: &O, writer: &mut W) -> Result<(), Self::Error>;
    fn deserialize<R: std::io::Read>(&self, reader: &mut R) -> Result<O, Self::Error>;
}
