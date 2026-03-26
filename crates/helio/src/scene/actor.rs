use crate::handles::{LightId, MeshId, ObjectId, VirtualObjectId};
use crate::mesh::MeshUpload;
use crate::scene::types::ObjectDescriptor;
use crate::vg::{VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor};
use helio_v3::{GpuLight, SkyContext};
use libhelio::SkyActor;

/// Result of inserting a typed scene actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneActorId {
    None,
    Mesh(MeshId),
    Light(LightId),
    VirtualMesh(VirtualMeshId),
    Object(ObjectId),
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

/// Unified scene actor type. Includes shading, geometry, and user custom logic.
#[derive(Debug, Clone)]
pub enum SceneActor {
    Sky(SkyActor),
    Mesh(MeshActor),
    Light(LightActor),
    VirtualMesh(VirtualMeshActor),
    VirtualObject(VirtualObjectActor),
    Object(ObjectActor),
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
        }
    }

    fn on_tick(&mut self, scene: &mut crate::scene::Scene) {
        match self {
            SceneActor::Mesh(actor) => actor.on_tick(scene),
            SceneActor::Light(actor) => actor.on_tick(scene),
            SceneActor::VirtualMesh(actor) => actor.on_tick(scene),
            SceneActor::VirtualObject(actor) => actor.on_tick(scene),
            SceneActor::Object(actor) => actor.on_tick(scene),
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
