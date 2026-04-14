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
define_handle!(LightId);
define_handle!(ObjectId);
define_handle!(VirtualObjectId);
define_handle!(WaterVolumeId);
define_handle!(WaterHitboxId);

