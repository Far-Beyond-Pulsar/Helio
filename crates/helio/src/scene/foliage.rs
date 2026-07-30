//! Foliage authoring API for [`Scene`].
//!
//! This is the CPU-facing half of the foliage stack: registering foliage *types* (what a
//! grass blade or a tree is), grouping them into *layers* (where they grow), driving the
//! global wind clock, and publishing *interactors* — the moving bodies that push grass
//! aside.
//!
//! # Two things here are load-bearing and easy to break
//!
//! **The `generation` counter must not advance when the wind changes.** `generation` gates
//! re-upload of the foliage type table. Wind changes every single frame; if it bumped
//! `generation`, the type table would re-upload every frame and the residency cache's
//! whole reason for existing — that steady-state foliage costs the CPU nothing — would be
//! gone. Wind travels in the same [`FoliageFrameData`] struct but is deliberately outside
//! the versioned part. See the doc comment on `FoliageFrameData::generation` in `libhelio`.
//!
//! **The `foliage` frame slot is left unwritten when no foliage types are registered.** The
//! foliage passes early-out on the unwritten slot, and that is the entire mechanism behind
//! the zero-overhead-when-absent guarantee. Publishing an empty-but-present struct would
//! silently turn "no foliage in this scene" into "foliage machinery runs and finds nothing".

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use helio_foliage_core::{
    pack_kind_and_flags, FoliageKind, GpuFoliageType, FOLIAGE_FLAG_CASTS_SHADOW,
    FOLIAGE_FLAG_RECEIVES_INTERACTION, FOLIAGE_FLAG_TWO_SIDED,
};
use libhelio::{FoliageFrameData, Wind};

use super::core::Scene;
use super::errors::{invalid, Result};
use crate::handles::{FoliageInteractorId, FoliageLayerId, FoliageTypeId};

// ─── GPU interactor record ──────────────────────────────────────────────────

/// One moving body that displaces foliage. 32 bytes.
///
/// Lives here rather than in `helio-foliage-core` only because nothing on the GPU consumes
/// it yet — `FoliageInteractionPass` is the next phase. Move it there when that pass lands
/// so the producer and consumer share one definition, which is the rule the rest of the
/// foliage POD types follow.
///
/// `velocity` is carried separately from position rather than differenced on the GPU
/// because the interaction field takes the *max* of existing and incoming displacement:
/// a body crossing several texels in one frame must leave a full-strength track along its
/// whole path, not a single splat wherever it happened to land on the frame boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuFoliageInteractor {
    /// World-space position (xyz) + radius (w).
    pub position_radius: [f32; 4],
    /// World-space velocity (xyz) + unused (w).
    pub velocity: [f32; 4],
}

const _: () = {
    assert!(
        std::mem::size_of::<GpuFoliageInteractor>() == 32,
        "GpuFoliageInteractor must be exactly 32 bytes"
    );
};

// ─── Public descriptors ─────────────────────────────────────────────────────

/// What a single kind of foliage is.
///
/// All fields are public because the documented call style uses functional-update syntax
/// (`FoliageTypeDescriptor { density: 40.0, ..Default::default() }`), which requires every
/// field to be visible at the call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoliageTypeDescriptor {
    /// Blade, card, or a full mesh driven through virtual geometry.
    pub kind: FoliageKind,
    /// Instances per square metre at full density weight.
    pub density: f32,
    /// Minimum and maximum blade/plant height in metres.
    pub height_range: [f32; 2],
    /// Minimum and maximum width in metres.
    pub width_range: [f32; 2],
    /// Acceptance band on terrain slope, in radians from vertical.
    pub slope_range: [f32; 2],
    /// Acceptance band on world altitude in metres.
    pub altitude_range: [f32; 2],
    /// Distances at which the four LOD transitions happen, in metres.
    pub lod_distances: [f32; 4],
    /// Per-band wind gain: `[trunk sway, branch flutter, leaf jitter]`.
    pub wind_response: [f32; 3],
    /// How fast a bent plant recovers. Larger is stiffer.
    pub interaction_stiffness: f32,
    /// Material table index.
    pub material_id: u32,
    /// Slice in the density texture array driving where this type grows.
    pub density_layer: u32,
    /// Virtual mesh or impostor page, for [`FoliageKind::Mesh`].
    pub mesh_or_impostor_id: u32,
    /// Render both faces. Almost always wanted for vegetation.
    pub two_sided: bool,
    /// Contribute to the shadow atlas.
    pub casts_shadow: bool,
    /// Respond to [`FoliageInteractor`]s.
    pub receives_interaction: bool,
}

impl Default for FoliageTypeDescriptor {
    fn default() -> Self {
        Self {
            kind: FoliageKind::Blade,
            density: 40.0,
            height_range: [0.15, 0.45],
            width_range: [0.01, 0.03],
            slope_range: [0.0, std::f32::consts::FRAC_PI_4],
            altitude_range: [f32::MIN, f32::MAX],
            lod_distances: helio_foliage_core::DEFAULT_LOD_DISTANCES,
            wind_response: [0.0, 0.3, 1.0],
            interaction_stiffness: 6.0,
            material_id: 0,
            density_layer: 0,
            mesh_or_impostor_id: u32::MAX,
            two_sided: true,
            casts_shadow: false,
            receives_interaction: true,
        }
    }
}

impl FoliageTypeDescriptor {
    /// Convert to the GPU record.
    ///
    /// Non-finite inputs are clamped to inert values rather than passed through. A NaN
    /// reaching a foliage shader does not produce an error anywhere — it makes every
    /// affected primitive fail its comparisons and vanish, which presents to the user as
    /// "all the grass disappeared" with a clean log. Degrading to zero density is at least
    /// diagnosable.
    pub fn to_gpu(self) -> GpuFoliageType {
        fn finite(value: f32, fallback: f32) -> f32 {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        }
        fn finite_pair(pair: [f32; 2], fallback: [f32; 2]) -> [f32; 2] {
            let low = finite(pair[0], fallback[0]);
            let high = finite(pair[1], fallback[1]);
            if low <= high {
                [low, high]
            } else {
                [high, low]
            }
        }

        let flags = u32::from(self.two_sided) * FOLIAGE_FLAG_TWO_SIDED
            | u32::from(self.casts_shadow) * FOLIAGE_FLAG_CASTS_SHADOW
            | u32::from(self.receives_interaction) * FOLIAGE_FLAG_RECEIVES_INTERACTION;

        let mut gpu = GpuFoliageType {
            density: finite(self.density, 0.0).max(0.0),
            height_range: finite_pair(self.height_range, [0.15, 0.45]),
            width_range: finite_pair(self.width_range, [0.01, 0.03]),
            slope_range: finite_pair(self.slope_range, [0.0, std::f32::consts::FRAC_PI_4]),
            altitude_range: finite_pair(self.altitude_range, [f32::MIN, f32::MAX]),
            lod_distances: helio_foliage_core::DEFAULT_LOD_DISTANCES,
            wind_response: [
                finite(self.wind_response[0], 0.0),
                finite(self.wind_response[1], 0.0),
                finite(self.wind_response[2], 0.0),
            ],
            interaction_stiffness: finite(self.interaction_stiffness, 6.0).max(0.0),
            material_id: self.material_id,
            density_layer: self.density_layer,
            kind_and_flags: pack_kind_and_flags(self.kind, flags),
            mesh_or_impostor_id: self.mesh_or_impostor_id,
            _pad: [0; 3],
        };

        // LOD distances must be non-decreasing or `select_blade_lod` picks a level whose
        // band it is not actually inside, and the cross-fade blends two representations
        // that are never both valid.
        let mut previous = 0.0_f32;
        for (slot, raw) in gpu.lod_distances.iter_mut().zip(self.lod_distances) {
            let value = finite(raw, previous).max(previous);
            *slot = value;
            previous = value;
        }
        gpu
    }
}

/// Where a set of foliage types grows.
///
/// The GPU-side layer table is not built yet — for now a scene behaves as if it had a
/// single implicit layer covering every registered type, which is enough for uniform
/// ground cover. Density and exclusion maps are stored and will be bound once
/// `FoliageTerrainPass` exists to sample them.
#[derive(Debug, Clone)]
pub struct FoliageLayer {
    /// Types this layer places.
    pub types: Vec<FoliageTypeId>,
    /// World-space AABB the layer covers, as `[min_xyz, max_xyz]`.
    pub bounds: [Vec3; 2],
    /// Placement seed. Two layers with the same seed and bounds place identically.
    pub seed: u32,
}

/// A moving body that pushes foliage aside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoliageInteractor {
    /// World-space position.
    pub position: Vec3,
    /// Influence radius in metres.
    pub radius: f32,
    /// World-space velocity, used to smear the track across a frame.
    pub velocity: Vec3,
}

impl FoliageInteractor {
    fn to_gpu(self) -> GpuFoliageInteractor {
        // Same reasoning as `FoliageTypeDescriptor::to_gpu`: an inert interactor is far
        // easier to diagnose than a NaN that silently deletes vegetation.
        let position = if self.position.is_finite() {
            self.position
        } else {
            Vec3::ZERO
        };
        let velocity = if self.velocity.is_finite() {
            self.velocity
        } else {
            Vec3::ZERO
        };
        let radius = if self.radius.is_finite() {
            self.radius.max(0.0)
        } else {
            0.0
        };
        GpuFoliageInteractor {
            position_radius: [position.x, position.y, position.z, radius],
            velocity: [velocity.x, velocity.y, velocity.z, 0.0],
        }
    }
}

// ─── Internal records ───────────────────────────────────────────────────────

pub(in crate::scene) struct FoliageTypeRecord {
    pub descriptor: FoliageTypeDescriptor,
}

pub(in crate::scene) struct FoliageLayerRecord {
    pub layer: FoliageLayer,
}

pub(in crate::scene) struct FoliageInteractorRecord {
    pub interactor: FoliageInteractor,
}

// ─── Scene API ──────────────────────────────────────────────────────────────

impl Scene {
    // ── Types ───────────────────────────────────────────────────────────────

    /// Register a foliage type. Advances the publication generation.
    pub fn add_foliage_type(&mut self, descriptor: FoliageTypeDescriptor) -> FoliageTypeId {
        let (id, _) = self
            .foliage_types
            .insert(FoliageTypeRecord { descriptor });
        self.foliage_types_dirty = true;
        id
    }

    /// Replace a registered type's descriptor. Advances the publication generation.
    ///
    /// This re-rolls placement for every tile using the type, because density and the
    /// acceptance bands are placement inputs. Prefer setting a type up once.
    pub fn update_foliage_type(
        &mut self,
        id: FoliageTypeId,
        descriptor: FoliageTypeDescriptor,
    ) -> Result<()> {
        let record = self
            .foliage_types
            .get_mut(id)
            .ok_or_else(|| invalid("foliage type"))?;
        record.descriptor = descriptor;
        self.foliage_types_dirty = true;
        Ok(())
    }

    /// Remove a foliage type. Advances the publication generation.
    pub fn remove_foliage_type(&mut self, id: FoliageTypeId) -> Result<()> {
        self.foliage_types
            .remove(id)
            .ok_or_else(|| invalid("foliage type"))?;
        self.foliage_types_dirty = true;
        Ok(())
    }

    /// Number of registered foliage types.
    pub fn foliage_type_count(&self) -> u32 {
        self.foliage_types.len() as u32
    }

    // ── Layers ──────────────────────────────────────────────────────────────

    /// Register a foliage layer.
    pub fn add_foliage_layer(&mut self, layer: FoliageLayer) -> FoliageLayerId {
        let (id, _) = self.foliage_layers.insert(FoliageLayerRecord { layer });
        self.foliage_types_dirty = true;
        id
    }

    /// Remove a foliage layer.
    pub fn remove_foliage_layer(&mut self, id: FoliageLayerId) -> Result<()> {
        self.foliage_layers
            .remove(id)
            .ok_or_else(|| invalid("foliage layer"))?;
        self.foliage_types_dirty = true;
        Ok(())
    }

    // ── Wind ────────────────────────────────────────────────────────────────

    /// Replace the global wind state.
    ///
    /// Deliberately does **not** advance the publication generation — see the module
    /// header. Wind is per-frame data riding in a versioned struct, not versioned data.
    pub fn set_wind(&mut self, wind: Wind) {
        let (time, prev_time) = (self.wind.time, self.wind.prev_time);
        self.wind = wind;
        // Keep the clock continuous across a parameter change. Taking the caller's
        // `time`/`prev_time` (usually a fresh `Wind::default()`, so both zero) would make
        // the motion-vector pair describe a jump back to t=0, and every foliage pixel
        // would emit a wild velocity for one frame — a full-screen TAA smear on any wind
        // tweak.
        self.wind.time = time;
        self.wind.prev_time = prev_time;
    }

    /// The current wind state.
    pub fn wind(&self) -> Wind {
        self.wind
    }

    /// Advance the wind clock by `dt` seconds. Call once per frame.
    ///
    /// This is what rolls `time` into `prev_time`, which is what lets foliage vertex
    /// shaders emit a correct previous-frame position and therefore correct motion
    /// vectors. Skip it and the wind freezes; call it twice and vegetation reports double
    /// its true screen-space velocity to TAA.
    pub fn advance_wind(&mut self, dt: f32) {
        self.wind.advance(dt);
    }

    // ── Interactors ─────────────────────────────────────────────────────────

    /// Add a body that displaces foliage.
    pub fn add_foliage_interactor(&mut self, interactor: FoliageInteractor) -> FoliageInteractorId {
        let (id, index) = self
            .foliage_interactors
            .insert(FoliageInteractorRecord { interactor });
        self.foliage_interactors_dirty = true;
        self.mark_foliage_interactor_dirty(index, index + 1);
        id
    }

    /// Move an interactor. O(1); call every physics tick.
    pub fn update_foliage_interactor(
        &mut self,
        id: FoliageInteractorId,
        position: Vec3,
        velocity: Vec3,
    ) -> Result<()> {
        let (index, record) = self
            .foliage_interactors
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("foliage interactor"))?;
        record.interactor.position = position;
        record.interactor.velocity = velocity;
        self.foliage_interactors_dirty = true;
        self.mark_foliage_interactor_dirty(index, index + 1);
        Ok(())
    }

    /// Remove an interactor.
    pub fn remove_foliage_interactor(&mut self, id: FoliageInteractorId) -> Result<()> {
        let removed = self
            .foliage_interactors
            .remove(id)
            .ok_or_else(|| invalid("foliage interactor"))?;
        self.foliage_interactors_dirty = true;
        // A dense arena backfills the hole with the last element, so the moved element's
        // slot is dirty too.
        let index = removed.dense_index;
        self.mark_foliage_interactor_dirty(index, index + 1);
        Ok(())
    }

    /// Number of live interactors.
    pub fn foliage_interactor_count(&self) -> u32 {
        self.foliage_interactors.len() as u32
    }

    fn mark_foliage_interactor_dirty(&mut self, start: usize, end: usize) {
        self.foliage_interactors_dirty_range = Some(match self.foliage_interactors_dirty_range {
            Some((existing_start, existing_end)) => {
                (existing_start.min(start), existing_end.max(end))
            }
            None => (start, end),
        });
    }

    // ── Publication ─────────────────────────────────────────────────────────

    /// Rebuild the CPU mirrors the foliage passes consume. Called from `flush`.
    pub(in crate::scene) fn rebuild_foliage_buffers(&mut self) {
        if self.foliage_types_dirty {
            self.foliage_cpu_types.clear();
            self.foliage_cpu_types.extend(
                self.foliage_types
                    .iter()
                    .map(|(_, record)| record.descriptor.to_gpu()),
            );
            // Only topology changes reach here — see the module header on why wind must
            // never take this path.
            self.foliage_generation = self.foliage_generation.wrapping_add(1);
            self.foliage_types_dirty = false;
        }

        if self.foliage_interactors_dirty {
            self.foliage_cpu_interactors.clear();
            self.foliage_cpu_interactors.extend(
                self.foliage_interactors
                    .iter()
                    .map(|(_, record)| record.interactor.to_gpu()),
            );
        }
    }

    /// Per-frame foliage publication, or `None` when the scene has no foliage.
    ///
    /// Returning `None` is what keeps the `foliage` frame slot unwritten, which is what
    /// makes the foliage passes cost nothing. Do not make this return an empty struct.
    pub fn foliage_frame_data(&self) -> Option<FoliageFrameData<'_>> {
        if self.foliage_cpu_types.is_empty() {
            return None;
        }
        Some(FoliageFrameData {
            types: bytemuck::cast_slice(&self.foliage_cpu_types),
            layers: &[],
            type_count: self.foliage_cpu_types.len() as u32,
            layer_count: 0,
            wind: self.wind.to_gpu(),
            generation: self.foliage_generation,
        })
    }

    /// CPU mirror of the interactor buffer, for the renderer's upload path.
    pub fn foliage_interactors_gpu_slice(&self) -> &[GpuFoliageInteractor] {
        &self.foliage_cpu_interactors
    }

    /// Whether the interactor mirror needs re-uploading.
    pub fn foliage_interactors_dirty(&self) -> bool {
        self.foliage_interactors_dirty
    }

    /// The dirty span of the interactor mirror.
    pub fn foliage_interactors_dirty_range(&self) -> Option<(usize, usize)> {
        self.foliage_interactors_dirty_range
    }

    /// Clear the interactor dirty state after upload.
    pub fn clear_foliage_interactors_dirty(&mut self) {
        self.foliage_interactors_dirty = false;
        self.foliage_interactors_dirty_range = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_packs_kind_and_flags() {
        let gpu = FoliageTypeDescriptor {
            kind: FoliageKind::Blade,
            two_sided: true,
            casts_shadow: false,
            receives_interaction: true,
            ..Default::default()
        }
        .to_gpu();

        assert_eq!(gpu.kind(), Some(FoliageKind::Blade));
        assert!(gpu.has_flag(FOLIAGE_FLAG_TWO_SIDED));
        assert!(!gpu.has_flag(FOLIAGE_FLAG_CASTS_SHADOW));
        assert!(gpu.has_flag(FOLIAGE_FLAG_RECEIVES_INTERACTION));
    }

    #[test]
    fn non_finite_descriptor_inputs_degrade_to_inert_values() {
        let gpu = FoliageTypeDescriptor {
            density: f32::NAN,
            height_range: [f32::NAN, f32::INFINITY],
            interaction_stiffness: f32::NEG_INFINITY,
            wind_response: [f32::NAN; 3],
            ..Default::default()
        }
        .to_gpu();

        assert_eq!(gpu.density, 0.0);
        assert!(gpu.height_range.iter().all(|v| v.is_finite()));
        assert!(gpu.interaction_stiffness.is_finite());
        assert!(gpu.wind_response.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn inverted_ranges_are_ordered_not_rejected() {
        let gpu = FoliageTypeDescriptor {
            height_range: [0.9, 0.1],
            ..Default::default()
        }
        .to_gpu();
        assert!(gpu.height_range[0] <= gpu.height_range[1]);
    }

    #[test]
    fn lod_distances_are_forced_non_decreasing() {
        let gpu = FoliageTypeDescriptor {
            lod_distances: [40.0, 10.0, 80.0, 5.0],
            ..Default::default()
        }
        .to_gpu();
        let d = gpu.lod_distances;
        assert!(d[0] <= d[1] && d[1] <= d[2] && d[2] <= d[3], "{d:?}");
    }

    #[test]
    fn interactor_rejects_non_finite_transform() {
        let gpu = FoliageInteractor {
            position: Vec3::new(f32::NAN, 0.0, 0.0),
            radius: -3.0,
            velocity: Vec3::splat(f32::INFINITY),
        }
        .to_gpu();

        assert!(gpu.position_radius.iter().all(|v| v.is_finite()));
        assert!(gpu.velocity.iter().all(|v| v.is_finite()));
        assert!(gpu.position_radius[3] >= 0.0);
    }

    #[test]
    fn interactor_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuFoliageInteractor>(), 32);
    }
}
