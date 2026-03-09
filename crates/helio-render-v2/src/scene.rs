//! Scene database – the authoritative source for all rendered content

use crate::features::{BillboardInstance, LightType};
use crate::mesh::GpuMesh;
use crate::material::GpuMaterial;

// ── Object identity ───────────────────────────────────────────────────────────

/// Stable handle to a registered scene object.
///
/// `ObjectId` is returned by [`Renderer::add_object`] and is the only way to
/// update or remove that object.  Internally it is a monotonically increasing
/// `u64`; zero is never issued as a valid id.
///
/// This mirrors Unreal Engine's `FPrimitiveComponentId`: once issued the id is
/// stable for the entire lifetime of the object — frustum culling, LOD switches,
/// and camera movement never invalidate it.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ObjectId(pub(crate) u64);

impl ObjectId {
    /// The null / invalid sentinel (never returned by [`Renderer::add_object`]).
    pub const INVALID: Self = Self(0);
}

/// Stable handle to a registered scene light.
///
/// `LightId` is returned by [`Renderer::add_light`].  The id is stable for
/// the lifetime of the light — use it to update position/color/intensity or
/// remove the light without rebuilding the full scene environment every frame.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LightId(pub(crate) u32);

impl LightId {
    pub const INVALID: Self = Self(u32::MAX);
}

/// Stable handle to a registered billboard instance.
///
/// `BillboardId` is returned by [`Renderer::add_billboard`].  Use it to
/// update the billboard's world position/color or remove it without
/// rebuilding the full billboard list every frame.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BillboardId(pub(crate) u32);

impl BillboardId {
    pub const INVALID: Self = Self(u32::MAX);
}

// ── Sky scene objects ─────────────────────────────────────────────────────────

/// Volumetric cloud layer parameters.
#[derive(Clone, Debug)]
pub struct VolumetricClouds {
    /// Cloud coverage in [0, 1]. 0 = clear sky, 1 = overcast.
    pub coverage: f32,
    /// Density multiplier (controls how opaque clouds appear).
    pub density: f32,
    /// Cloud layer base height in world units.
    pub base_height: f32,
    /// Cloud layer top height in world units.
    pub top_height: f32,
    /// Wind XZ direction (not normalized – magnitude = speed).
    pub wind_speed: f32,
    pub wind_direction: [f32; 2],
}

impl Default for VolumetricClouds {
    fn default() -> Self {
        Self {
            coverage: 0.30,
            density: 0.7,
            base_height: 800.0,
            top_height: 1800.0,
            wind_speed: 0.08,
            wind_direction: [1.0, 0.0],
        }
    }
}

impl VolumetricClouds {
    pub fn new() -> Self { Self::default() }
    pub fn with_coverage(mut self, v: f32) -> Self { self.coverage = v; self }
    pub fn with_density(mut self, v: f32) -> Self { self.density = v; self }
    pub fn with_layer(mut self, base: f32, top: f32) -> Self {
        self.base_height = base; self.top_height = top; self
    }
    pub fn with_wind(mut self, direction: [f32; 2], speed: f32) -> Self {
        self.wind_direction = direction; self.wind_speed = speed; self
    }
}

/// Physically-based atmospheric scattering sky.
///
/// Integrates with the first directional light in the scene to determine
/// the sun's position and colour. All defaults reproduce a clear Earth sky.
#[derive(Clone, Debug)]
pub struct SkyAtmosphere {
    // ── Rayleigh (air molecules → blue sky) ────────────────────────────────
    /// Per-wavelength Rayleigh scattering coefficients (R/G/B).
    /// Default = Earth values (5.8e-3, 13.5e-3, 33.1e-3) km⁻¹.
    pub rayleigh_scatter: [f32; 3],
    /// Rayleigh scale height normalised to atmosphere thickness. Default 0.08.
    pub rayleigh_h_scale: f32,

    // ── Mie (aerosols / haze → sun glow) ────────────────────────────────────
    pub mie_scatter: f32,
    pub mie_h_scale: f32,
    /// Henyey-Greenstein asymmetry factor, -1..1. Default 0.76 (forward).
    pub mie_g: f32,

    // ── Sun disc ─────────────────────────────────────────────────────────────
    pub sun_intensity: f32,
    /// Angular size of the sun disc in radians. Default ≈ real Sun (0.0045 rad).
    pub sun_disk_angle: f32,

    // ── Atmospheric geometry ──────────────────────────────────────────────────
    /// Earth radius (km). Controls horizon shape.
    pub earth_radius: f32,
    /// Atmosphere outer radius (km).
    pub atm_radius: f32,

    // ── Post-process ──────────────────────────────────────────────────────────
    /// Exposure for sky tone mapping. Increase for brighter sky.
    pub exposure: f32,

    // ── Clouds ───────────────────────────────────────────────────────────────
    pub clouds: Option<VolumetricClouds>,
}

impl Default for SkyAtmosphere {
    fn default() -> Self {
        Self {
            rayleigh_scatter: [5.8e-3, 13.5e-3, 33.1e-3],
            rayleigh_h_scale: 0.08,
            mie_scatter: 2.1e-3,
            mie_h_scale: 0.012,
            mie_g: 0.76,
            sun_intensity: 22.0,
            sun_disk_angle: 0.0045,
            earth_radius: 6360.0,
            atm_radius: 6420.0,
            exposure: 4.0,
            clouds: None,
        }
    }
}

impl SkyAtmosphere {
    pub fn new() -> Self { Self::default() }
    pub fn with_sun_intensity(mut self, v: f32) -> Self { self.sun_intensity = v; self }
    pub fn with_exposure(mut self, v: f32) -> Self { self.exposure = v; self }
    pub fn with_mie_g(mut self, v: f32) -> Self { self.mie_g = v; self }
    pub fn with_clouds(mut self, c: VolumetricClouds) -> Self { self.clouds = Some(c); self }
}

/// Sky light – derives its colour and intensity from the atmospheric sky.
/// Inject one into the scene to enable sky-based ambient and irradiance.
#[derive(Clone, Debug)]
pub struct Skylight {
    /// Multiplier applied to the computed sky ambient colour.
    pub intensity: f32,
    /// Optional tint applied on top of the computed sky colour.
    pub color_tint: [f32; 3],
}

impl Default for Skylight {
    fn default() -> Self {
        Self { intensity: 1.0, color_tint: [1.0, 1.0, 1.0] }
    }
}

impl Skylight {
    pub fn new() -> Self { Self::default() }
    pub fn with_intensity(mut self, v: f32) -> Self { self.intensity = v; self }
    pub fn with_tint(mut self, c: [f32; 3]) -> Self { self.color_tint = c; self }
}

// ── Existing scene types ──────────────────────────────────────────────────────

/// A single renderable object in the scene
///
/// `mesh` contains geometry in *model space*; its world transform is stored
/// separately so that identical meshes can be instanced.  Older code that
/// baked the transform into the vertex data will continue to work because the
/// default transform is identity.
#[derive(Clone)]
pub struct SceneObject {
    pub mesh: GpuMesh,
    /// Per-object material.  `None` → renderer uses its built-in default white material.
    pub material: Option<GpuMaterial>,
    /// Model‑to‑world transform for this object.
    pub transform: glam::Mat4,
}

impl SceneObject {
    pub fn new(mesh: GpuMesh) -> Self {
        Self { mesh, material: None, transform: glam::Mat4::IDENTITY }
    }

    pub fn with_material(mesh: GpuMesh, material: GpuMaterial) -> Self {
        Self { mesh, material: Some(material), transform: glam::Mat4::IDENTITY }
    }

    /// Specify a world transform.
    pub fn with_transform(mut self, t: glam::Mat4) -> Self {
        self.transform = t;
        self
    }
}

/// A light source in the scene
#[derive(Clone, Debug)]
pub struct SceneLight {
    pub light_type: LightType,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

impl SceneLight {
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            light_type: LightType::Directional,
            position: [0.0; 3],
            direction,
            color,
            intensity,
            range: 1000.0,
        }
    }

    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            light_type: LightType::Point,
            position,
            direction: [0.0, -1.0, 0.0],
            color,
            intensity,
            range,
        }
    }

    /// Spot light — a cone emitter.
    ///
    /// `direction` is the axis the cone points along (should be normalised).
    /// `inner_angle` / `outer_angle` are the half-angles of the soft-edge cone
    /// in radians, e.g. `inner = 0.8`, `outer = 1.0` for a ~60°/115° spread.
    /// Only one shadow-map face is rendered for spots (vs 6 for point lights).
    pub fn spot(
        position:    [f32; 3],
        direction:   [f32; 3],
        color:       [f32; 3],
        intensity:   f32,
        range:       f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            light_type: LightType::Spot { inner_angle, outer_angle },
            position,
            direction,
            color,
            intensity,
            range,
        }
    }
}

/// The scene database – defines all rendered content
///
/// This is the single authoritative source for everything the renderer draws.
/// No content is hardcoded in the renderer itself.
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub lights: Vec<SceneLight>,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
    pub billboards: Vec<BillboardInstance>,
    /// Background/sky clear color. Ignored when `sky_atmosphere` is set.
    pub sky_color: [f32; 3],
    /// Physical atmosphere. When set the sky is rendered by `SkyPass`.
    pub sky_atmosphere: Option<SkyAtmosphere>,
    /// Sky-driven ambient lighting. Requires `sky_atmosphere`.
    pub skylight: Option<Skylight>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
            ambient_color: [0.0, 0.0, 0.0],
            ambient_intensity: 0.0,
            billboards: Vec::new(),
            sky_color: [0.0, 0.0, 0.0],
            sky_atmosphere: None,
            skylight: None,
        }
    }

    pub fn with_sky(mut self, color: [f32; 3]) -> Self {
        self.sky_color = color;
        self
    }

    /// Enable physical atmospheric scattering for this scene.
    pub fn with_sky_atmosphere(mut self, atm: SkyAtmosphere) -> Self {
        self.sky_atmosphere = Some(atm);
        self
    }

    /// Add a sky-driven ambient light. Requires `with_sky_atmosphere`.
    pub fn with_skylight(mut self, sl: Skylight) -> Self {
        self.skylight = Some(sl);
        self
    }

    pub fn add_object(mut self, mesh: GpuMesh) -> Self {
        self.objects.push(SceneObject::new(mesh));
        self
    }

    /// Add object with custom material and an explicit transform.
    pub fn add_object_transform(mut self, mesh: GpuMesh, transform: glam::Mat4) -> Self {
        let mut obj = SceneObject::new(mesh);
        obj.transform = transform;
        self.objects.push(obj);
        self
    }

    /// Add an object with a custom PBR material (identity transform).
    pub fn add_object_with_material(mut self, mesh: GpuMesh, material: GpuMaterial) -> Self {
        self.objects.push(SceneObject::with_material(mesh, material));
        self
    }

    /// Add an object with material and world transform.
    pub fn add_object_with_material_transform(
        mut self,
        mesh: GpuMesh,
        material: GpuMaterial,
        transform: glam::Mat4,
    ) -> Self {
        let mut obj = SceneObject::with_material(mesh, material);
        obj.transform = transform;
        self.objects.push(obj);
        self
    }

    pub fn add_light(mut self, light: SceneLight) -> Self {
        self.lights.push(light);
        self
    }

    pub fn add_billboard(mut self, billboard: BillboardInstance) -> Self {
        self.billboards.push(billboard);
        self
    }

    pub fn with_ambient(mut self, color: [f32; 3], intensity: f32) -> Self {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
        self
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
