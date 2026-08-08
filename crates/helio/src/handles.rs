macro_rules! define_handle {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name {
            slot: u32,
            generation: u32,
        }

        impl $name {
            pub const fn from_raw(slot: u32, generation: u32) -> Self {
                Self { slot, generation }
            }

            pub const fn slot(self) -> u32 {
                self.slot
            }

            pub const fn generation(self) -> u32 {
                self.generation
            }
        }

        impl super::handles::Handle for $name {
            fn from_parts(slot: u32, generation: u32) -> Self {
                Self::from_raw(slot, generation)
            }

            fn slot(self) -> u32 {
                self.slot
            }

            fn generation(self) -> u32 {
                self.generation
            }
        }
    };
}

pub trait Handle: Copy {
    fn from_parts(slot: u32, generation: u32) -> Self;
    fn slot(self) -> u32;
    fn generation(self) -> u32;
}

define_handle!(MeshId);
define_handle!(MultiMeshId);
define_handle!(SectionedInstanceId);
define_handle!(MaterialId);
define_handle!(TextureId);
/// Handle to a scene object, backed by `pulsar_scenedb::Entity`.
///
/// Does **not** implement [`Handle`]: `Entity`'s constructor is private outside
/// `pulsar_scenedb` by design, so `Handle::from_parts` cannot be implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ObjectId(pub(crate) pulsar_scenedb::Entity);

impl ObjectId {
    /// Sentinel — never a live object handle.
    pub const INVALID: ObjectId = ObjectId(pulsar_scenedb::Entity::DANGLING);

    pub(crate) fn from_entity(entity: pulsar_scenedb::Entity) -> Self { Self(entity) }
    pub(crate) fn entity(self) -> pulsar_scenedb::Entity { self.0 }
    pub fn slot(self) -> u32 { self.0.index() }
    pub fn generation(self) -> u32 { self.0.generation() }
}

define_handle!(LightId);
define_handle!(VirtualObjectId);
define_handle!(WaterVolumeId);
define_handle!(WaterHitboxId);
define_handle!(PostProcessVolumeId);
define_handle!(ReflectionCaptureId);
define_handle!(VoxelVolumeId);
define_handle!(DecalId);
define_handle!(FoliageTypeId);
define_handle!(FoliageLayerId);
define_handle!(FoliageInteractorId);
define_handle!(SublevelId);
define_handle!(PortalId);

