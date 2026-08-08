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

/// Handle to a scene object, backed by `pulsar_scenedb::Entity` instead of
/// `define_handle!`'s bespoke `{slot, generation}` pair — see
/// `docs/scenedb_object_storage_migration.md`. Still its own distinct
/// newtype (not a bare `Entity` re-export), matching every other handle in
/// this file.
///
/// Does **not** implement [`Handle`]: that trait exists for types stored in
/// this crate's own `DenseArena`/`SparsePool`, and `ObjectId`-backed data no
/// longer lives there. It also could not be implemented honestly —
/// `Entity`'s constructor is private outside `pulsar_scenedb` by design (no
/// forging a handle from raw parts), so `Handle::from_parts` would have
/// nothing valid to return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ObjectId(pub(crate) pulsar_scenedb::Entity);

impl ObjectId {
    /// Sentinel value — never a live object handle. `World::spawn` never
    /// hands out `Entity::DANGLING` (its generation, `u32::MAX`, is not a
    /// value the generation counter reaches in practice), so this is safe
    /// to use as a "not yet assigned" placeholder the way `from_raw(0, 0)`
    /// served before this handle was backed by `Entity`.
    pub const INVALID: ObjectId = ObjectId(pulsar_scenedb::Entity::DANGLING);

    pub(crate) fn from_entity(entity: pulsar_scenedb::Entity) -> Self {
        Self(entity)
    }

    pub(crate) fn entity(self) -> pulsar_scenedb::Entity {
        self.0
    }

    /// Slot index, for debug/logging parity with the other handle types in
    /// this file. Not a dense/reusable raw array index the way the
    /// `SparsePool`-backed handles' `slot()` is — construct GPU-facing slot
    /// numbers from `ObjectRecord::gpu_slot` instead.
    pub fn slot(self) -> u32 {
        self.0.index()
    }

    pub fn generation(self) -> u32 {
        self.0.generation()
    }
}

define_handle!(MeshId);
define_handle!(MultiMeshId);
define_handle!(SectionedInstanceId);
define_handle!(MaterialId);
define_handle!(TextureId);
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

