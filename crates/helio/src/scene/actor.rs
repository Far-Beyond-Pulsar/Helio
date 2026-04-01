use crate::handles::{LightId, MeshId, ObjectId, VirtualObjectId, WaterVolumeId};
use crate::mesh::MeshUpload;
use crate::scene::types::ObjectDescriptor;
use crate::vg::{VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor};
use helio_v3::{GpuLight, SkyContext};
use libhelio::{GpuWaterVolume, SkyActor};

/// Result of inserting a typed scene actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneActorId {
    None,
    Mesh(MeshId),
    Light(LightId),
    VirtualMesh(VirtualMeshId),
    Object(ObjectId),
    WaterVolume(WaterVolumeId),
}

impl SceneActorId {
    pub fn as_mesh(self) -> Option<MeshId> {
        if let SceneActorId::Mesh(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_light(self) -> Option<LightId> {
        if let SceneActorId::Light(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_virtual_mesh(self) -> Option<VirtualMeshId> {
        if let SceneActorId::VirtualMesh(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_object(self) -> Option<ObjectId> {
        if let SceneActorId::Object(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_water_volume(self) -> Option<WaterVolumeId> {
        if let SceneActorId::WaterVolume(id) = self {
            Some(id)
        } else {
            None
        }
    }
}

/// Common behavior for scene actors (custom and built-in).
pub trait SceneActorTrait {
    /// Whether the actor should be ticked each frame.
    fn is_active(&self) -> bool {
        true
    }

    /// Called once when the actor is inserted into the scene.
    fn on_attach(&mut self, _scene: &mut crate::scene::Scene) {}

    /// Called once per frame when the actor is active.
    fn on_tick(&mut self, _scene: &mut crate::scene::Scene) {}

    /// Optional scene sky context contributed by this actor.
    fn sky_context(&self) -> Option<SkyContext> {
        None
    }

    /// Actor id generated during insertion (if applicable).
    fn inserted_id(&self) -> SceneActorId {
        SceneActorId::None
    }
}

/// A mesh actor (upload + optional resource handle).
#[derive(Debug, Clone)]
pub struct MeshActor {
    pub upload: MeshUpload,
    pub mesh_id: Option<MeshId>,
}

impl MeshActor {
    pub fn new(upload: MeshUpload) -> Self {
        Self {
            upload,
            mesh_id: None,
        }
    }

    pub fn id(&self) -> Option<MeshId> {
        self.mesh_id
    }
}

impl SceneActorTrait for MeshActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.mesh_id.is_none() {
            self.mesh_id = Some(scene.insert_mesh(self.upload.clone()));
        }
    }

    fn inserted_id(&self) -> SceneActorId {
        self.mesh_id
            .map(SceneActorId::Mesh)
            .unwrap_or(SceneActorId::None)
    }
}

/// A light actor (GPU light descriptor + optional light handle).
#[derive(Debug, Clone, Copy)]
pub struct LightActor {
    pub light: GpuLight,
    pub light_id: Option<LightId>,
}

impl LightActor {
    pub fn new(light: GpuLight) -> Self {
        Self {
            light,
            light_id: None,
        }
    }

    pub fn id(&self) -> Option<LightId> {
        self.light_id
    }
}

impl SceneActorTrait for LightActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.light_id.is_none() {
            self.light_id = Some(scene.insert_light(self.light));
        }
    }

    fn inserted_id(&self) -> SceneActorId {
        self.light_id
            .map(SceneActorId::Light)
            .unwrap_or(SceneActorId::None)
    }
}

/// A virtual mesh actor (meshletized upload + optional handle).
#[derive(Debug, Clone)]
pub struct VirtualMeshActor {
    pub upload: VirtualMeshUpload,
    pub virtual_mesh_id: Option<VirtualMeshId>,
}

impl VirtualMeshActor {
    pub fn new(upload: VirtualMeshUpload) -> Self {
        Self {
            upload,
            virtual_mesh_id: None,
        }
    }

    pub fn id(&self) -> Option<VirtualMeshId> {
        self.virtual_mesh_id
    }
}

impl SceneActorTrait for VirtualMeshActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.virtual_mesh_id.is_none() {
            self.virtual_mesh_id = Some(scene.insert_virtual_mesh(self.upload.clone()));
        }
    }

    fn inserted_id(&self) -> SceneActorId {
        self.virtual_mesh_id
            .map(SceneActorId::VirtualMesh)
            .unwrap_or(SceneActorId::None)
    }
}

/// A virtual object actor (instance of a virtual mesh).
#[derive(Debug, Clone, Copy)]
pub struct VirtualObjectActor {
    pub descriptor: VirtualObjectDescriptor,
    pub object_id: Option<VirtualObjectId>,
}

impl VirtualObjectActor {
    pub fn new(descriptor: VirtualObjectDescriptor) -> Self {
        Self {
            descriptor,
            object_id: None,
        }
    }

    pub fn id(&self) -> Option<VirtualObjectId> {
        self.object_id
    }
}

impl SceneActorTrait for VirtualObjectActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.object_id.is_none() {
            if let Ok(id) = scene.insert_virtual_object(self.descriptor) {
                self.object_id = Some(id);
            }
        }
    }
}

/// A standard object actor (mesh+material instance).
#[derive(Debug, Clone, Copy)]
pub struct ObjectActor {
    pub descriptor: ObjectDescriptor,
    pub object_id: Option<ObjectId>,
}

impl ObjectActor {
    pub fn new(descriptor: ObjectDescriptor) -> Self {
        Self {
            descriptor,
            object_id: None,
        }
    }

    pub fn id(&self) -> Option<ObjectId> {
        self.object_id
    }
}

impl SceneActorTrait for ObjectActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.object_id.is_none() {
            if let Ok(id) = scene.insert_object(self.descriptor) {
                self.object_id = Some(id);
            }
        }
    }

    fn inserted_id(&self) -> SceneActorId {
        self.object_id
            .map(SceneActorId::Object)
            .unwrap_or(SceneActorId::None)
    }
}

/// Water volume configuration descriptor.
///
/// Defines all parameters for heightfield-simulation water rendering including
/// waves, visual properties, reflections, caustics, and underwater effects.
/// This maps directly onto the new webgpu-water-style sim + render pipeline.
#[derive(Debug, Clone, Copy)]
pub struct WaterVolumeDescriptor {
    /// AABB minimum corner in world space
    pub bounds_min: [f32; 3],
    /// AABB maximum corner in world space
    pub bounds_max: [f32; 3],
    /// Water surface height (Y coordinate, local to bounds)
    pub surface_height: f32,

    // Wave parameters (legacy Gerstner — kept for compatibility; heightfield uses sim)
    /// Wave amplitude (height in meters)
    pub wave_amplitude: f32,
    /// Wave frequency (spacing between waves)
    pub wave_frequency: f32,
    /// Wave animation speed
    pub wave_speed: f32,
    /// Primary wave direction (XZ plane)
    pub wave_direction: [f32; 2],
    /// Wave steepness (0.0 = sine wave, 1.0 = sharp peaks)
    pub wave_steepness: f32,

    // Visual properties
    /// Base water color (deep water)
    pub water_color: [f32; 3],
    /// RGB absorption per meter depth (Beer-Lambert)
    pub extinction: [f32; 3],
    /// Wave steepness threshold to spawn foam
    pub foam_threshold: f32,
    /// Foam intensity multiplier
    pub foam_amount: f32,

    // Reflection/refraction
    /// Screen-space reflection intensity (0-1)
    pub reflection_strength: f32,
    /// Refraction distortion amount
    pub refraction_strength: f32,
    /// Fresnel exponent (higher = sharper falloff)
    pub fresnel_power: f32,

    // Caustics
    /// Enable caustics rendering
    pub caustics_enabled: bool,
    /// Caustics brightness multiplier (caustics_intensity fed to sim_params)
    pub caustics_intensity: f32,
    /// Caustics pattern scale
    pub caustics_scale: f32,
    /// Caustics animation speed
    pub caustics_speed: f32,

    // Underwater effects
    /// Volumetric fog density
    pub fog_density: f32,
    /// God rays (volumetric light shafts) intensity
    pub god_rays_intensity: f32,

    // Heightfield simulation surface parameters
    /// Index of refraction (default 1.333 for water)
    pub ior: f32,
    /// Fresnel minimum reflectance at normal incidence (default 0.1)
    pub fresnel_min: f32,
    /// Effective water density for fog (default 0.03)
    pub density: f32,

    // Shadow / lighting parameters
    /// Rim light intensity for pool walls (default 1.0)
    pub shadow_rim: f32,
    /// Hitbox shadow (0.0 = no hitbox shadow, 1.0 = full shadow under hitbox)
    pub shadow_hitbox: f32,
    /// Ambient occlusion strength (default 1.0)
    pub shadow_ao: f32,

    /// Sun / dominant directional light direction (world space, need not be normalized —
    /// will be normalised in `to_gpu()`). Default: [0.5, 1.0, 0.5] (upper-right).
    pub sun_direction: [f32; 3],

    // Heightfield simulation physics
    /// Wave spring constant: restoring force toward the mean height.
    /// Range [0.5, 2.0]. Lower (~1.0) feels fluid; higher (~2.0) feels jelly-like.
    /// Pass to `WaterSimPass::set_sim_dynamics()` after updating the volume.
    pub wave_spring: f32,
    /// Per-step energy damping multiplier (0.0..1.0).
    /// Closer to 1.0 = waves linger; closer to 0.9 = waves die quickly.
    /// Pass to `WaterSimPass::set_sim_dynamics()` after updating the volume.
    pub wave_damping: f32,

    // Wind
    /// Wind direction in world XZ space. Does not need to be normalised.
    /// Set [0, 0] for calm water. Pass to `WaterSimPass::set_wind()`.
    pub wind_direction: [f32; 2],
    /// Wind strength. 0 = calm, ~1 = gentle ripples, ~5 = choppy.
    /// Pass to `WaterSimPass::set_wind()` after updating the volume.
    pub wind_strength: f32,
    /// Wave spatial scale factor. 1.0 = default size; 0.25 = fine ripples; 2.0 = large swells.
    /// Controls the footprint of gust impulses on the heightfield surface.
    pub wave_scale: f32,
}

impl WaterVolumeDescriptor {
    /// Converts descriptor to GPU-side representation.
    pub fn to_gpu(&self) -> GpuWaterVolume {
        let sun = {
            let [x, y, z] = self.sun_direction;
            let len = (x * x + y * y + z * z).sqrt().max(1e-6);
            [x / len, y / len, z / len, 0.0]
        };
        GpuWaterVolume {
            bounds_min: [self.bounds_min[0], self.bounds_min[1], self.bounds_min[2], 0.0],
            bounds_max: [self.bounds_max[0], self.bounds_max[1], self.bounds_max[2], self.surface_height],
            wave_params: [self.wave_amplitude, self.wave_frequency, self.wave_speed, self.wave_steepness],
            wave_direction: [self.wave_direction[0], self.wave_direction[1], 0.0, 0.0],
            water_color: [self.water_color[0], self.water_color[1], self.water_color[2], self.foam_threshold],
            extinction: [self.extinction[0], self.extinction[1], self.extinction[2], self.foam_amount],
            reflection_refraction: [self.reflection_strength, self.refraction_strength, self.fresnel_power, 0.0],
            caustics_params: [if self.caustics_enabled { 1.0 } else { 0.0 }, self.caustics_intensity, self.caustics_scale, self.caustics_speed],
            fog_params: [self.fog_density, self.god_rays_intensity, 0.0, 0.0],
            sim_params: [self.ior, self.caustics_intensity, self.fresnel_min, self.density],
            shadow_params: [self.shadow_rim, self.shadow_hitbox, self.shadow_ao, 0.0],
            sun_direction: sun,
            ssr_params: [1.0, 32.0, 0.05, 0.02],  // Default SSR: enabled, 32 steps
            sim_dynamics: [self.wave_spring, self.wave_damping, self.wave_scale, 0.0],
            wind_params: [self.wind_direction[0], self.wind_direction[1], self.wind_strength, 0.0],
            _pad6: [0.0; 4],
        }
    }

    /// Creates a default ocean water volume.
    pub fn ocean() -> Self {
        Self {
            bounds_min: [-100.0, -10.0, -100.0],
            bounds_max: [100.0, 50.0, 100.0],
            surface_height: 0.0,
            wave_amplitude: 0.5,
            wave_frequency: 0.3,
            wave_speed: 1.5,
            wave_direction: [1.0, 0.0],
            wave_steepness: 0.5,
            water_color: [0.0, 0.2, 0.4],
            extinction: [0.1, 0.05, 0.02],
            foam_threshold: 0.8,
            foam_amount: 0.6,
            reflection_strength: 0.8,
            refraction_strength: 0.2,
            fresnel_power: 5.0,
            caustics_enabled: true,
            caustics_intensity: 1.5,
            caustics_scale: 5.0,
            caustics_speed: 0.5,
            fog_density: 0.03,
            god_rays_intensity: 1.0,
            ior: 1.333,
            fresnel_min: 0.1,
            density: 0.03,
            shadow_rim: 1.0,
            shadow_hitbox: 0.0,
            shadow_ao: 1.0,
            sun_direction: [0.5, 1.0, 0.5],
            wave_spring: 0.04,   // c ≈ 22 m/s at wave_speed=1.0 — realistic ocean swell
            wave_damping: 0.988,
            wind_direction: [0.0, 0.0],
            wind_strength: 0.0,
            wave_scale: 1.0,
        }
    }

    /// Creates a default lake / pool water volume.
    pub fn lake() -> Self {
        Self {
            bounds_min: [-50.0, -5.0, -50.0],
            bounds_max: [50.0, 20.0, 50.0],
            surface_height: 0.0,
            wave_amplitude: 0.2,
            wave_frequency: 0.5,
            wave_speed: 0.8,
            wave_direction: [1.0, 0.0],
            wave_steepness: 0.3,
            water_color: [0.1, 0.3, 0.2],
            extinction: [0.2, 0.1, 0.08],
            foam_threshold: 0.7,
            foam_amount: 0.5,
            reflection_strength: 0.6,
            refraction_strength: 0.3,
            fresnel_power: 4.0,
            caustics_enabled: true,
            caustics_intensity: 1.2,
            caustics_scale: 4.0,
            caustics_speed: 0.4,
            fog_density: 0.05,
            god_rays_intensity: 0.5,
            ior: 1.333,
            fresnel_min: 0.1,
            density: 0.05,
            shadow_rim: 1.0,
            shadow_hitbox: 0.0,
            shadow_ao: 1.0,
            sun_direction: [0.5, 1.0, 0.5],
            wave_spring: 0.06,   // c ≈ 27 m/s, slightly choppier than flat ocean
            wave_damping: 0.984,
            wind_direction: [0.0, 0.0],
            wind_strength: 0.0,
            wave_scale: 1.0,
        }
    }
}

impl Default for WaterVolumeDescriptor {
    fn default() -> Self {
        Self::ocean()
    }
}

/// A water volume actor (descriptor + optional volume handle).
#[derive(Debug, Clone, Copy)]
pub struct WaterVolumeActor {
    pub descriptor: WaterVolumeDescriptor,
    pub volume_id: Option<WaterVolumeId>,
}

impl WaterVolumeActor {
    pub fn new(descriptor: WaterVolumeDescriptor) -> Self {
        Self {
            descriptor,
            volume_id: None,
        }
    }

    pub fn id(&self) -> Option<WaterVolumeId> {
        self.volume_id
    }
}

impl SceneActorTrait for WaterVolumeActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.volume_id.is_none() {
            if let Ok(id) = scene.insert_water_volume(self.descriptor) {
                self.volume_id = Some(id);
            }
        }
    }

    fn inserted_id(&self) -> SceneActorId {
        self.volume_id
            .map(SceneActorId::WaterVolume)
            .unwrap_or(SceneActorId::None)
    }
}

// ── Water Hitbox ─────────────────────────────────────────────────────────────

/// Descriptor for a water hitbox — an AABB that displaces the heightfield simulation.
///
/// A hitbox records where an object *was* (old bounds) and where it *is* (new bounds).
/// The simulation computes the volume that was vacated minus the new volume to produce
/// a realistic rise-and-fall displacement pattern on the water surface.
///
/// # Usage
/// ```ignore
/// let hitbox_id = scene.insert_water_hitbox(WaterHitboxDescriptor {
///     old_min: [-0.5, 0.0, -0.5],
///     old_max: [0.5, 1.0, 0.5],
///     new_min: [-0.5, -0.3, -0.5],  // moved downward into the water
///     new_max: [0.5, 0.7, 0.5],
///     edge_softness: 0.5,
///     strength: 1.0,
/// })?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WaterHitboxDescriptor {
    /// Previous frame AABB minimum (world space XYZ)
    pub old_min: [f32; 3],
    /// Previous frame AABB maximum (world space XYZ)
    pub old_max: [f32; 3],
    /// Current frame AABB minimum (world space XYZ)
    pub new_min: [f32; 3],
    /// Current frame AABB maximum (world space XYZ)
    pub new_max: [f32; 3],
    /// Gaussian falloff width at the AABB edges (lower = sharper, typical range 0.3–2.0)
    pub edge_softness: f32,
    /// Displacement strength multiplier (default 1.0)
    pub strength: f32,
}

impl WaterHitboxDescriptor {
    /// Converts to GPU representation.
    pub fn to_gpu(&self) -> libhelio::GpuWaterHitbox {
        libhelio::GpuWaterHitbox {
            old_min: [self.old_min[0], self.old_min[1], self.old_min[2], 0.0],
            old_max: [self.old_max[0], self.old_max[1], self.old_max[2], 0.0],
            new_min: [self.new_min[0], self.new_min[1], self.new_min[2], 0.0],
            new_max: [self.new_max[0], self.new_max[1], self.new_max[2], 0.0],
            params: [self.edge_softness, self.strength, 0.0, 0.0],
        }
    }
}

/// Water hitbox actor — wraps a [`WaterHitboxDescriptor`] for the scene actor system.
#[derive(Debug, Clone, Copy)]
pub struct WaterHitboxActor {
    pub descriptor: WaterHitboxDescriptor,
    pub hitbox_id: Option<crate::handles::WaterHitboxId>,
}

impl WaterHitboxActor {
    pub fn new(descriptor: WaterHitboxDescriptor) -> Self {
        Self { descriptor, hitbox_id: None }
    }

    pub fn id(&self) -> Option<crate::handles::WaterHitboxId> {
        self.hitbox_id
    }
}

impl SceneActorTrait for WaterHitboxActor {
    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        if self.hitbox_id.is_none() {
            if let Ok(id) = scene.insert_water_hitbox(self.descriptor) {
                self.hitbox_id = Some(id);
            }
        }
    }

    fn inserted_id(&self) -> SceneActorId {
        // Hitboxes don't map to a top-level SceneActorId variant; return None
        SceneActorId::None
    }
}

/// Unified scene actor type. Includes shading, geometry, and user custom logic.
#[derive(Debug, Clone)]
pub enum SceneActor {
    Sky(SkyActor),
    Mesh(MeshActor),
    Light(LightActor),
    VirtualMesh(VirtualMeshActor),
    VirtualObject(VirtualObjectActor),
    Object(ObjectActor),
    WaterVolume(WaterVolumeActor),
    WaterHitbox(WaterHitboxActor),
}

impl SceneActor {
    pub fn sky(sky: SkyActor) -> Self {
        SceneActor::Sky(sky)
    }

    pub fn mesh(upload: MeshUpload) -> Self {
        SceneActor::Mesh(MeshActor::new(upload))
    }

    pub fn light(light: GpuLight) -> Self {
        SceneActor::Light(LightActor::new(light))
    }

    pub fn virtual_mesh(upload: VirtualMeshUpload) -> Self {
        SceneActor::VirtualMesh(VirtualMeshActor::new(upload))
    }

    pub fn virtual_object(desc: VirtualObjectDescriptor) -> Self {
        SceneActor::VirtualObject(VirtualObjectActor::new(desc))
    }

    pub fn object(desc: ObjectDescriptor) -> Self {
        SceneActor::Object(ObjectActor::new(desc))
    }

    pub fn water_volume(descriptor: WaterVolumeDescriptor) -> Self {
        SceneActor::WaterVolume(WaterVolumeActor::new(descriptor))
    }
}

impl SceneActorTrait for SceneActor {
    fn is_active(&self) -> bool {
        true
    }

    fn inserted_id(&self) -> SceneActorId {
        match self {
            SceneActor::Sky(_) => SceneActorId::None,
            SceneActor::Mesh(actor) => actor.inserted_id(),
            SceneActor::Light(actor) => actor.inserted_id(),
            SceneActor::VirtualMesh(actor) => actor.inserted_id(),
            SceneActor::VirtualObject(actor) => actor.inserted_id(),
            SceneActor::Object(actor) => actor.inserted_id(),
            SceneActor::WaterVolume(actor) => actor.inserted_id(),
            SceneActor::WaterHitbox(actor) => actor.inserted_id(),
        }
    }

    fn on_attach(&mut self, scene: &mut crate::scene::Scene) {
        match self {
            SceneActor::Sky(_) => {
                // No additional per-frame state. Scene will query context from actors.
            }
            SceneActor::Mesh(actor) => actor.on_attach(scene),
            SceneActor::Light(actor) => actor.on_attach(scene),
            SceneActor::VirtualMesh(actor) => actor.on_attach(scene),
            SceneActor::VirtualObject(actor) => actor.on_attach(scene),
            SceneActor::Object(actor) => actor.on_attach(scene),
            SceneActor::WaterVolume(actor) => actor.on_attach(scene),
            SceneActor::WaterHitbox(actor) => actor.on_attach(scene),
        }
    }

    fn on_tick(&mut self, scene: &mut crate::scene::Scene) {
        match self {
            SceneActor::Mesh(actor) => actor.on_tick(scene),
            SceneActor::Light(actor) => actor.on_tick(scene),
            SceneActor::VirtualMesh(actor) => actor.on_tick(scene),
            SceneActor::VirtualObject(actor) => actor.on_tick(scene),
            SceneActor::Object(actor) => actor.on_tick(scene),
            SceneActor::WaterVolume(actor) => actor.on_tick(scene),
            SceneActor::WaterHitbox(_) => {}
            SceneActor::Sky(_) => {}
        }
    }

    fn sky_context(&self) -> Option<SkyContext> {
        match self {
            SceneActor::Sky(sky) => Some(sky.context()),
            _ => None,
        }
    }
}
