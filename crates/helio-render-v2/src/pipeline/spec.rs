//! Pipeline specialization constants and material permutation types.
//!
//! Equivalent to Unreal's `FMaterialRenderProxy` / `FVertexFactory` permutation
//! system.  Each unique combination of domain + shading model + blend mode +
//! vertex factory produces a distinct compiled PSO.

// ── Material Domain ───────────────────────────────────────────────────────────

/// Which rendering domain a material belongs to.
///
/// | Domain       | Unreal equivalent          |
/// |--------------|----------------------------|
/// | `Surface`    | `MD_Surface`               |
/// | `Unlit`      | `MD_Surface` + unlit model |
/// | `PostProcess`| `MD_PostProcess`           |
/// | `Decal`      | `MD_DeferredDecal`         |
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
pub enum MaterialDomain {
    #[default]
    Surface,
    Unlit,
    PostProcess,
    Decal,
}

// ── Shading Model ─────────────────────────────────────────────────────────────

/// PBR lighting model applied to a surface material.
///
/// Each variant compiles a different fragment shader permutation.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
pub enum ShadingModel {
    #[default]
    DefaultLit,
    Unlit,
    Subsurface,
    ClearCoat,
    TwoSided,
}

// ── Blend Mode ────────────────────────────────────────────────────────────────

/// How the material blends with the background.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
pub enum BlendMode {
    /// Opaque — depth prepass + G-buffer write path.
    #[default]
    Opaque,
    /// Masked — alpha cutout; still used in the deferred GBuffer path.
    Masked,
    /// Translucent — forward-blended pass after deferred shading.
    Translucent,
    /// Additive — forward-blended, depth read-only.
    Additive,
}

impl BlendMode {
    /// Returns true if this mode requires the transparent forward pass.
    pub fn is_transparent(self) -> bool {
        matches!(self, BlendMode::Translucent | BlendMode::Additive)
    }
}

// ── Vertex Factory ────────────────────────────────────────────────────────────

/// Which vertex input layout the pipeline expects.
///
/// Determines which vertex buffer format and instance stream are used.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
pub enum VertexFactory {
    /// Static geometry — 32-byte `PackedVertex` + 4-byte `u32` primitive_id.
    #[default]
    Static,
    /// Billboard quads — generated in the vertex shader.
    Billboard,
    /// Skinned meshes (future).
    Skinned,
    /// Particle quads (future).
    Particle,
}

// ── PipelineVariant ───────────────────────────────────────────────────────────

/// Coarse pipeline variant used to select the render pipeline topology and
/// output attachments.  Fine-grained permutations (shading model, blend mode
/// etc.) are composed with this in [`PipelineKey`].
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum PipelineVariant {
    /// Forward rendering (legacy, kept for compatibility/debug)
    Forward,
    /// Forward transparent rendering (alpha blending, depth read-only)
    TransparentForward,
    /// Depth-only (shadow pass and depth prepass)
    DepthOnly,
    /// Write geometry data to 4 G-buffer textures — no lighting.
    GBufferWrite,
    /// Deferred fullscreen lighting pass — reads G-buffer, runs PBR.
    DeferredLighting,
    /// Deferred rendering (unused alias)
    Deferred,
}

// ── Composed material key ─────────────────────────────────────────────────────

/// Full material permutation key.  One compiled PSO exists per unique value of
/// this struct.  Analagous to Unreal's `FMeshPassProcessorRenderState` +
/// `FMaterialShaderTypes` combination.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
pub struct MaterialKey {
    pub domain:         MaterialDomain,
    pub shading_model:  ShadingModel,
    pub blend_mode:     BlendMode,
    pub vertex_factory: VertexFactory,
}

impl MaterialKey {
    pub fn new() -> Self { Self::default() }

    pub fn surface_lit() -> Self { Self::default() }

    pub fn surface_unlit() -> Self {
        Self { shading_model: ShadingModel::Unlit, ..Default::default() }
    }

    pub fn transparent() -> Self {
        Self { blend_mode: BlendMode::Translucent, ..Default::default() }
    }
}
