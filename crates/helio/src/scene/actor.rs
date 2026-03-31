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
/// Defines all parameters for realistic water rendering including waves,
/// visual properties, reflections, caustics, and underwater effects.
#[derive(Debug, Clone, Copy)]
pub struct WaterVolumeDescriptor {
    /// AABB minimum corner in world space
    pub bounds_min: [f32; 3],
    /// AABB maximum corner in world space
    pub bounds_max: [f32; 3],
    /// Water surface height (Y coordinate)
    pub surface_height: f32,

    // Wave parameters (Gerstner waves)
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
    /// Caustics brightness multiplier
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
}

impl WaterVolumeDescriptor {
    /// Converts descriptor to GPU-side representation.
    pub fn to_gpu(&self) -> GpuWaterVolume {
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
            _pad0: [0.0; 4],
            _pad1: [0.0; 4],
            _pad2: [0.0; 4],
            _pad3: [0.0; 4],
            _pad4: [0.0; 4],
            _pad5: [0.0; 4],
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
        }
    }

    /// Creates a default lake water volume.
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
            caustics_enabled: false,
            caustics_intensity: 0.0,
            caustics_scale: 0.0,
            caustics_speed: 0.0,
            fog_density: 0.05,
            god_rays_intensity: 0.5,
        }
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
