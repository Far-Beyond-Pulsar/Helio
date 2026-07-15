use serde::{Deserialize, Serialize};

// ── Math aliases ──────────────────────────────────────────────────────────────

pub type Vec2 = glam::Vec2;
pub type Vec3 = glam::Vec3;
pub type Vec4 = glam::Vec4;
pub type Mat4 = glam::Mat4;

// ── Transform ─────────────────────────────────────────────────────────────────

/// World-space transform as a 4×4 column-major matrix.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Transform(pub Mat4);

impl Transform {
    pub const IDENTITY: Self = Self(Mat4::IDENTITY);
}

impl Default for Transform {
    fn default() -> Self { Self::IDENTITY }
}

// ── Material ──────────────────────────────────────────────────────────────────

/// Per-triangle material parameters used by bakers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialDesc {
    /// Diffuse albedo (linear sRGB, pre-multiplied alpha).
    pub albedo:           [f32; 4],
    /// PBR roughness (0 = mirror, 1 = fully diffuse).
    pub roughness:        f32,
    /// PBR metallic factor.
    pub metallic:         f32,
    /// Emissive radiance (W·m⁻²·sr⁻¹).
    pub emissive:         [f32; 3],
    /// Whether this surface fully blocks light (walls, floors).
    pub casts_shadows:    bool,
    /// Acoustic absorption coefficient ∈ [0, 1] — 1 = fully absorptive.
    pub audio_absorption: f32,
    /// Acoustic scattering coefficient ∈ [0, 1].
    pub audio_scattering: f32,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            albedo:           [0.8, 0.8, 0.8, 1.0],
            roughness:        0.5,
            metallic:         0.0,
            emissive:         [0.0; 3],
            casts_shadows:    true,
            audio_absorption: 0.1,
            audio_scattering: 0.2,
        }
    }
}

// ── Mesh ──────────────────────────────────────────────────────────────────────

/// A triangle mesh ready for baking.
///
/// Positions, normals, and UVs are stored in parallel aligned arrays.
/// The lightmap UV set (channel 1) is optional — if absent the baker will
/// generate a simple planar projection.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BakeMesh {
    /// Unique stable identifier (matches the runtime mesh ID).
    pub id:           uuid::Uuid,
    pub positions:    Vec<[f32; 3]>,
    pub normals:      Vec<[f32; 3]>,
    /// Primary UVs (texture coordinates).
    pub uvs:          Vec<[f32; 2]>,
    /// Lightmap UVs — `None` = auto-generate.
    pub lightmap_uvs: Option<Vec<[f32; 2]>>,
    pub indices:      Vec<u32>,
    pub material_ids: Vec<u32>,
    pub world_transform: Transform,
}

/// A point on a surface — used as sample/probe position.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SurfacePoint {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
    /// Index of the owning mesh.
    pub mesh_idx: u32,
}

// ── Light sources ─────────────────────────────────────────────────────────────

/// Analytical light kind matching the Helio `LightType` enum so lights can be
/// passed through without conversion.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LightSourceKind {
    Directional {
        direction: [f32; 3],
    },
    Point {
        position: [f32; 3],
        range:    f32,
    },
    Spot {
        position:    [f32; 3],
        direction:   [f32; 3],
        range:       f32,
        inner_angle: f32,
        outer_angle: f32,
    },
    /// Rectangular area light — approximated with multiple point samples.
    Area {
        center:     [f32; 3],
        right:      [f32; 3],
        up:         [f32; 3],
        half_w:     f32,
        half_h:     f32,
    },
    /// Emissive mesh (index into [`SceneGeometry::meshes`]).
    Emissive { mesh_idx: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightSource {
    pub kind:          LightSourceKind,
    /// Linear HDR color (no gamma).
    pub color:         [f32; 3],
    /// Luminous intensity in candelas (point/spot) or illuminance in lux (dir).
    pub intensity:     f32,
    /// Exclude from baking when `false` (e.g. dynamic-only lights).
    pub bake_enabled:  bool,
    /// Cast shadows in baked result.
    pub casts_shadows: bool,
}

// ── Audio emitters ────────────────────────────────────────────────────────────

/// An acoustic source whose impulse response will be baked.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioEmitter {
    pub id:          uuid::Uuid,
    pub position:    [f32; 3],
    /// Source directivity — `[0,0,-1]` for omni, otherwise forward vector.
    pub direction:   [f32; 3],
    /// Frequency bands considered (e.g. octave-band centre frequencies in Hz).
    pub freq_bands:  Vec<f32>,
}

// ── Scene geometry ────────────────────────────────────────────────────────────

/// Everything a baker needs to know about the scene.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SceneGeometry {
    pub meshes:        Vec<BakeMesh>,
    pub materials:     Vec<MaterialDesc>,
    pub lights:        Vec<LightSource>,
    pub audio_emitters: Vec<AudioEmitter>,
    /// Sky/background irradiance captured as an RGBE HDR panorama (optional).
    pub sky_hdr:       Option<Vec<u8>>,
    pub sky_hdr_width: u32,
    pub sky_hdr_height: u32,
}

impl SceneGeometry {
    pub fn new() -> Self { Self::default() }

    pub fn add_mesh(&mut self, mesh: BakeMesh) -> usize {
        let idx = self.meshes.len();
        self.meshes.push(mesh);
        idx
    }

    pub fn add_light(&mut self, light: LightSource) -> usize {
        let idx = self.lights.len();
        self.lights.push(light);
        idx
    }

    pub fn total_triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len() / 3).sum()
    }
}
