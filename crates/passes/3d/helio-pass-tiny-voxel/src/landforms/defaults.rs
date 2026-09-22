use super::{BoundedLandforms, LandformSnapshot, SnapshotId, VoxelField};
use std::sync::{Arc, OnceLock};

pub const DEFAULT_LANDFORM_ID: &str =
    "edb3e9b2907cabca78aadcc622d41fb3cd319a4b40b40b5a35798ca94b89d127";

/// Provisional fixed integration dataset, shared by CPU queries and pass uploads.
/// It is not the final geographic/appearance recipe.
pub fn default_field() -> &'static VoxelField {
    static FIELD: OnceLock<VoxelField> = OnceLock::new();
    FIELD.get_or_init(|| {
        let id = SnapshotId(std::array::from_fn(|i| {
            u8::from_str_radix(&DEFAULT_LANDFORM_ID[i * 2..i * 2 + 2], 16).unwrap()
        }));
        let snapshot =
            LandformSnapshot::decode(include_bytes!("assets/default-landforms.hlfm"), id)
                .expect("embedded landform identity");
        VoxelField::new(Arc::new(BoundedLandforms::new(snapshot)), 419)
    })
}

impl VoxelField {
    pub fn outer_radius(&self) -> f64 {
        let maximum = *self
            .landforms()
            .snapshot()
            .height_atlas()
            .iter()
            .max()
            .unwrap();
        (127420000.0 + f64::from(maximum) + 8280.0) * 0.05 + 2.0
    }
}

pub fn default_clearance() -> super::FieldClearance {
    static BOUND: OnceLock<super::FieldClearance> = OnceLock::new();
    *BOUND.get_or_init(|| {
        let snapshot = default_field().landforms().snapshot();
        super::FieldClearance::from_atlas(snapshot.resolution(), snapshot.height_atlas())
    })
}
