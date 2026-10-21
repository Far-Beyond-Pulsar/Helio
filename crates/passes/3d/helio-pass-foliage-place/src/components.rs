//! SceneDB-owned foliage authoring records.

use crate::{GpuFoliageLayer, GpuFoliageType};
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_types")]
pub struct FoliageTypeComponent {
    #[gpu]
    pub density: f32,
    #[gpu]
    pub height_range: [f32; 2],
    #[gpu]
    pub width_range: [f32; 2],
    #[gpu]
    pub slope_range: [f32; 2],
    #[gpu]
    pub altitude_range: [f32; 2],
    #[gpu]
    pub lod_distances: [f32; 4],
    #[gpu]
    pub wind_response: [f32; 3],
    #[gpu]
    pub interaction_stiffness: f32,
    #[gpu]
    pub material_id: u32,
    #[gpu]
    pub density_layer: u32,
    #[gpu]
    pub kind_and_flags: u32,
    #[gpu]
    pub mesh_or_impostor_id: u32,
    #[gpu]
    pub base_color: u32,
    #[gpu]
    pub roughness_metallic: u32,
    #[gpu]
    pub ground_mask: u32,
    #[gpu]
    pub blades_per_tuft: u32,
    #[gpu]
    pub lean_range: [f32; 2],
    #[gpu]
    pub curve: f32,
    #[gpu]
    pub fall_chance: f32,
    #[gpu]
    pub tip_color: u32,
    #[gpu]
    pub patch_scale: f32,
    #[gpu]
    pub patch_strength: f32,
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_layers")]
pub struct FoliageLayerComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
}

/// Something that pushes foliage aside: a character's feet, a vehicle, a thrown rock.
/// Blades within `radius` of it lie over away from it, and stay pressed down along its
/// wake (where it was over the last `wake` seconds, from its velocity) and rise again
/// behind it. World positions in f64 halves (`position_hi + position_lo`), so it
/// holds at planet scale: build rows with [`FoliageInteractorComponent::new`].
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_interactors")]
pub struct FoliageInteractorComponent {
    /// xyz the world position's f32 part, w the radius (metres; 0: inactive).
    #[gpu]
    pub position_hi_radius: [f32; 4],
    /// xyz the rest of the position (world − hi), w how hard it pushes (1: flat at its
    /// centre for a type of stiffness 1).
    #[gpu]
    pub position_lo_strength: [f32; 4],
    /// xyz its velocity (m/s), w its wake (seconds of track that stay pressed).
    #[gpu]
    pub velocity_wake: [f32; 4],
}

impl FoliageInteractorComponent {
    /// An interactor at world `position` (the point that presses the ground: feet, a
    /// wheel's contact) moving at `velocity`.
    pub fn new(position: glam::DVec3, radius: f32, strength: f32, velocity: glam::Vec3, wake: f32) -> Self {
        let mut row = Self { position_hi_radius: [0.0, 0.0, 0.0, radius], position_lo_strength: [0.0, 0.0, 0.0, strength], velocity_wake: [0.0; 4] };
        row.set_motion(position, velocity);
        row.velocity_wake[3] = wake;
        row
    }

    /// Move it to `position`, moving at `velocity`.
    pub fn set_motion(&mut self, position: glam::DVec3, velocity: glam::Vec3) {
        let hi = position.as_vec3();
        let lo = (position - hi.as_dvec3()).as_vec3();
        self.position_hi_radius = [hi.x, hi.y, hi.z, self.position_hi_radius[3]];
        self.position_lo_strength = [lo.x, lo.y, lo.z, self.position_lo_strength[3]];
        self.velocity_wake = [velocity.x, velocity.y, velocity.z, self.velocity_wake[3]];
    }

    /// Its world position.
    pub fn position(&self) -> glam::DVec3 {
        let [hx, hy, hz, _] = self.position_hi_radius;
        let [lx, ly, lz, _] = self.position_lo_strength;
        glam::DVec3::new(f64::from(hx) + f64::from(lx), f64::from(hy) + f64::from(ly), f64::from(hz) + f64::from(lz))
    }
}

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_wind")]
pub struct FoliageWindComponent {
    #[gpu]
    pub direction_speed: [f32; 4],
    #[gpu]
    pub gust: [f32; 4],
    #[gpu]
    pub time_prev_time: [f32; 2],
    #[gpu]
    pub _pad: [f32; 2],
}

impl From<GpuFoliageType> for FoliageTypeComponent {
    fn from(v: GpuFoliageType) -> Self {
        bytemuck::cast(v)
    }
}
impl From<FoliageTypeComponent> for GpuFoliageType {
    fn from(v: FoliageTypeComponent) -> Self {
        bytemuck::cast(v)
    }
}
impl From<GpuFoliageLayer> for FoliageLayerComponent {
    fn from(v: GpuFoliageLayer) -> Self {
        bytemuck::cast(v)
    }
}
impl From<FoliageLayerComponent> for GpuFoliageLayer {
    fn from(v: FoliageLayerComponent) -> Self {
        bytemuck::cast(v)
    }
}
impl From<crate::GpuWind> for FoliageWindComponent {
    fn from(v: crate::GpuWind) -> Self {
        bytemuck::cast(v)
    }
}
impl From<FoliageWindComponent> for crate::GpuWind {
    fn from(v: FoliageWindComponent) -> Self {
        bytemuck::cast(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    #[test]
    fn scene_records_preserve_gpu_abi() {
        assert_eq!(std::mem::size_of::<FoliageTypeComponent>(), 128);
        assert_eq!(std::mem::size_of::<FoliageLayerComponent>(), 32);
        assert_eq!(std::mem::size_of::<FoliageInteractorComponent>(), 48);
        assert_eq!(std::mem::size_of::<FoliageWindComponent>(), 48);
    }

    #[test]
    fn scene_rows_support_insert_mutate_remove() {
        let mut world = pulsar_scenedb::World::new();
        let entity = world.spawn();
        let ty = FoliageTypeComponent::zeroed();
        let layer = FoliageLayerComponent::zeroed();
        let interactor = FoliageInteractorComponent::zeroed();
        let wind = FoliageWindComponent::zeroed();
        world.insert(entity, ty);
        world.insert(entity, layer);
        world.insert(entity, interactor);
        world.insert(entity, wind);

        world
            .get_mut::<FoliageInteractorComponent>(entity)
            .unwrap()
            .position_hi_radius[3] = 2.0;
        assert_eq!(
            world
                .get::<FoliageInteractorComponent>(entity)
                .unwrap()
                .position_hi_radius[3],
            2.0
        );
        assert_eq!(world.remove::<FoliageTypeComponent>(entity), Some(ty));
        assert_eq!(world.remove::<FoliageLayerComponent>(entity), Some(layer));
        assert_eq!(
            world.remove::<FoliageInteractorComponent>(entity),
            Some(FoliageInteractorComponent {
                position_hi_radius: [0.0, 0.0, 0.0, 2.0],
                position_lo_strength: [0.0; 4],
                velocity_wake: [0.0; 4],
            })
        );
        assert_eq!(world.remove::<FoliageWindComponent>(entity), Some(wind));
    }

    #[test]
    fn interactors_hold_planet_scale_positions() {
        let at = glam::DVec3::new(6_371_000.123_456, -1_234_567.891, 42.5);
        let row = FoliageInteractorComponent::new(at, 1.0, 1.0, glam::Vec3::X, 1.5);
        assert!((row.position() - at).length() < 1.0e-6);
        assert_eq!(row.velocity_wake, [1.0, 0.0, 0.0, 1.5]);
    }
}
