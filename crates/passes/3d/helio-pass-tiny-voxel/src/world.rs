use glam::DVec3;
use serde::{Deserialize, Serialize};

pub const RADIUS: f64 = 6_371_000.0;
pub const VOXEL: f64 = 0.1;
pub const MAX_EDITS: usize = 65_536;
pub const GENERATOR_REVISION: u32 = 5;

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edit {
    pub cell: [i32; 3],
    pub radius: f32,
    pub material: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct World {
    #[serde(default)]
    pub generator_revision: u32,
    #[serde(default)]
    pub landform_id: String,
    pub edits: Vec<Edit>,
    #[serde(skip)]
    pub(crate) edit_index: crate::edits::EditIndex,
    #[serde(skip)]
    pub chunks: crate::chunks::ChunkStore,
}
impl Default for World {
    fn default() -> Self {
        Self {
            generator_revision: GENERATOR_REVISION,
            landform_id: crate::landforms::DEFAULT_LANDFORM_ID.into(),
            edits: Vec::new(),
            edit_index: Default::default(),
            chunks: Default::default(),
        }
    }
}

impl Edit {
    pub fn radius_units(self) -> u32 {
        (self.radius * 20.0).round() as u32
    }
    pub fn canonical_radius(self) -> f64 {
        f64::from(self.radius_units()) * 0.05
    }
    pub fn contains(self, c: [i32; 3]) -> bool {
        let d = std::array::from_fn::<_, 3, _>(|a| i64::from(c[a]) - i64::from(self.cell[a]));
        let radius2 = i64::from(self.radius_units());
        d.iter().all(|d| d.abs() <= (radius2 + 1) / 2)
            && d.iter().map(|d| d * d).sum::<i64>() * 4 <= radius2 * radius2
    }
}

#[cfg(test)]
fn hash(p: [i32; 3], seed: u32) -> u32 {
    let mut v = (p[0] as u32).wrapping_mul(0x8da6b343)
        ^ (p[1] as u32).wrapping_mul(0xd8163841)
        ^ (p[2] as u32).wrapping_mul(0xcb1ab31f)
        ^ seed;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846ca68b);
    v ^ (v >> 16)
}
pub fn terrain_units(cell: [i32; 3]) -> i32 {
    crate::landforms::default_field()
        .height_units(cell)
        .unwrap_or(0)
}
pub fn base_material(c: [i32; 3]) -> u32 {
    crate::landforms::default_field()
        .sample_cell(c)
        .map_or(0, |sample| sample.material)
}
pub fn procedural_outer_radius() -> f64 {
    static R: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::landforms::default_field().outer_radius())
}
fn procedural_air_radius(c: [i32; 3], air_depth: f64) -> f64 {
    let outer_clearance = center(c).length() - procedural_outer_radius();
    if outer_clearance > 1.0 {
        return outer_clearance;
    }
    crate::landforms::default_clearance().empty_radius(air_depth)
}
pub fn cell_of(p: DVec3) -> [i32; 3] {
    (p / VOXEL).floor().as_ivec3().to_array()
}
/// Keep the integer render anchor representable even when the camera is far
/// outside voxel space. Nearby cameras retain their exact cell and fraction.
pub fn render_origin(p: DVec3) -> [i32; 3] {
    (p / VOXEL)
        .floor()
        .clamp(
            DVec3::splat(-1_000_000_000.0),
            DVec3::splat(1_000_000_000.0),
        )
        .as_ivec3()
        .to_array()
}
pub fn center(c: [i32; 3]) -> DVec3 {
    (glam::IVec3::from_array(c).as_dvec3() + DVec3::splat(0.5)) * VOXEL
}
impl World {
    pub fn ground_spawn(&self, x: f64, z: f64, clearance: f64) -> DVec3 {
        let mut outer = procedural_outer_radius();
        if let Some((minimum, maximum)) = self.edit_index.bounds {
            let extent = DVec3::from_array(std::array::from_fn(|a| {
                f64::from(minimum[a]).abs().max(f64::from(maximum[a]).abs())
            })) * VOXEL;
            outer = outer.max(extent.length() + 2.0);
        }
        let top = DVec3::new(x, outer, z);
        if let Some((_, _, distance)) = self.cast_hit_impl(top, -DVec3::Y, outer * 2.0, true) {
            top - DVec3::Y * (distance - clearance)
        } else {
            DVec3::new(x, RADIUS + 3.0, z)
        }
    }
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut records =
            serde_json::Deserializer::from_slice(&bytes).into_iter::<serde_json::Value>();
        let first = records
            .next()
            .ok_or("Empty voxel journal")?
            .map_err(|e| e.to_string())?;
        let mut world: Self = serde_json::from_value(first).map_err(|e| e.to_string())?;
        for record in records {
            world.edits.push(
                serde_json::from_value(record.map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?,
            );
        }
        world.validate_recipe()
    }
    /// Decode the opaque recipe held by a generic voxel terrain source.
    /// An empty recipe selects the versioned built-in landform.
    pub fn from_recipe_json(json: &str) -> Result<Self, String> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str::<Self>(json)
            .map_err(|error| error.to_string())?
            .validate_recipe()
    }
    fn validate_recipe(mut self) -> Result<Self, String> {
        if (self.generator_revision != GENERATOR_REVISION
            || self.landform_id != crate::landforms::DEFAULT_LANDFORM_ID)
            && !self.edits.is_empty()
        {
            return Err(format!("Edit journal uses terrain revision {} or a different landform identity; this renderer uses {}. Use a new journal so edits are not silently applied to different terrain.",self.generator_revision,GENERATOR_REVISION));
        }
        self.generator_revision = GENERATOR_REVISION;
        self.landform_id = crate::landforms::DEFAULT_LANDFORM_ID.into();
        if self.edits.len() > MAX_EDITS
            || self.edits.iter().any(|e| {
                !e.radius.is_finite()
                    || e.radius <= 0.0
                    || e.radius > (RADIUS * 2.0) as f32
                    || e.radius_units() == 0
                    || e.cell.iter().any(|v| v.unsigned_abs() > 100_000_000)
                    || e.material > 3
            })
        {
            return Err("Invalid or oversized voxel edit journal".into());
        }
        self.rebuild_edits();
        Ok(self)
    }
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_vec(&self).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        std::fs::rename(temp, path).map_err(|e| e.to_string())
    }
    pub fn density(&self, c: [i32; 3]) -> f64 {
        RADIUS - center(c).length() + f64::from(terrain_units(c)) * 0.05
    }
    /// Conservative empty radius around a point, using certified field regions
    /// and ordered edits. Useful for camera near-plane placement;
    /// a one-metre allowance covers cell extents and numeric approximation.
    pub fn air_clearance(&self, position: DVec3) -> f64 {
        let c = cell_of(position);
        if self.material(c) != 0 {
            return 0.0;
        }
        let latest = self.latest_edit(c);
        let mut safe = if let Some(i) = latest {
            self.edits[i].canonical_radius() * 0.9999996
                - position.distance(center(self.edits[i].cell))
                - 1.0
        } else {
            procedural_air_radius(c, -self.density(c)) - 1.0
        };
        for e in self
            .edits
            .iter()
            .skip(latest.map_or(0, |i| i + 1))
            .filter(|e| e.material != 0)
        {
            safe = safe
                .min(position.distance(center(e.cell)) - e.canonical_radius() * 1.0000004 - 1.0);
        }
        safe.max(0.0)
    }
    pub fn apply_edit(&mut self, edit: Edit) -> Result<(), String> {
        if self.edits.len() >= MAX_EDITS
            || !edit.radius.is_finite()
            || edit.radius <= 0.0
            || edit.radius > (RADIUS * 2.0) as f32
            || edit.radius_units() == 0
            || edit.material > 3
            || edit.cell.iter().any(|v| v.unsigned_abs() > 100_000_000)
        {
            return Err("Invalid edit or edit capacity reached".into());
        }
        if self.edit_index.len() != self.edits.len() {
            self.rebuild_edits();
        }
        self.edits.push(edit);
        self.edit_index.append(edit);
        self.chunks.invalidate_edit(edit);
        Ok(())
    }
    pub fn rebuild_edits(&mut self) {
        self.chunks.reconcile(&self.edits);
        self.edit_index.reconcile(&self.edits);
    }
    pub fn latest_edit(&self, c: [i32; 3]) -> Option<usize> {
        if self.edit_index.len() != self.edits.len() {
            return self.edits.iter().rposition(|e| e.contains(c));
        }
        self.edit_index.latest(c)
    }
    pub fn material(&self, c: [i32; 3]) -> u32 {
        if self.edit_index.len() == self.edits.len() {
            return self.chunks.material(self, c);
        }
        self.latest_edit(c)
            .map_or_else(|| base_material(c), |i| self.edits[i].material)
    }
    pub(crate) fn region_edits(&self, low: [i32; 3], high: [i32; 3]) -> Vec<usize> {
        if self.edit_index.len() != self.edits.len() {
            return (0..self.edits.len()).collect();
        }
        self.edit_index.region(low, high)
    }
    pub fn cast(&self, origin: DVec3, dir: DVec3, max: f64) -> Option<([i32; 3], [i32; 3])> {
        self.cast_hit(origin, dir, max)
            .map(|(cell, last, _)| (cell, last))
    }
    pub fn cast_hit(
        &self,
        origin: DVec3,
        dir: DVec3,
        max: f64,
    ) -> Option<([i32; 3], [i32; 3], f64)> {
        self.cast_hit_impl(origin, dir, max, false)
    }
    pub(crate) fn cast_hit_impl(
        &self,
        origin: DVec3,
        dir: DVec3,
        max: f64,
        accelerated: bool,
    ) -> Option<([i32; 3], [i32; 3], f64)> {
        let mut t = 0.0;
        let mut last = cell_of(origin);
        while t < max {
            let p = origin + dir * t;
            let c = cell_of(p);
            if self.material(c) != 0 {
                return Some((c, last, t));
            }
            last = c;
            if accelerated {
                let edit = self.latest_edit(c);
                let mut safe = if let Some(i) = edit {
                    self.edits[i].canonical_radius()
                        - center(c).distance(center(self.edits[i].cell))
                        - self.edits[i].canonical_radius() * 0.0000004
                        - 1.0
                } else {
                    procedural_air_radius(c, -self.density(c)) - 1.0
                };
                for e in self
                    .edits
                    .iter()
                    .skip(edit.map_or(0, |i| i + 1))
                    .filter(|e| e.material != 0)
                {
                    safe = safe.min(
                        center(c).distance(center(e.cell)) - e.canonical_radius() * 1.0000004 - 1.0,
                    );
                }
                if safe > 0.1 {
                    t += safe;
                    continue;
                }
            }
            let local = p / VOXEL - glam::IVec3::from_array(c).as_dvec3();
            let mut step = f64::INFINITY;
            for a in 0..3 {
                if dir[a].abs() > 1e-12 {
                    let edge = if dir[a] > 0.0 {
                        1.0 - local[a]
                    } else {
                        local[a]
                    };
                    step = step.min(edge * VOXEL / dir[a].abs());
                }
            }
            t += step.max(0.000001) + 0.000001;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_air_clearance_does_not_cross_voxels_or_ordered_edits() {
        let mut world = World::default();
        let anchor = world.ground_spawn(0.03, -256.03, 3.0);
        for altitude in [0.0, 10.0, 1000.0, 100_000.0, 8_000_000.0] {
            let eye = anchor + anchor.normalize() * altitude;
            let safe = world.air_clearance(eye);
            assert!(safe.is_finite() && safe >= 0.0);
            for ray in [
                DVec3::NEG_Y,
                DVec3::new(0.3, -1.0, 0.2).normalize(),
                DVec3::new(-0.4, -1.0, -0.3).normalize(),
            ] {
                if let Some((_, _, distance)) = world.raycast(eye, ray, 30_000_000.0) {
                    assert!(
                        distance + 0.000001 >= safe,
                        "near plane air bound crossed original terrain"
                    );
                }
            }
        }
        let eye = anchor + DVec3::Y * 100.0;
        let target = cell_of(eye - DVec3::Y * 4.0);
        world.edits.push(Edit {
            cell: target,
            radius: 1.0,
            material: 3,
        });
        world.rebuild_edits();
        let safe = world.air_clearance(eye);
        assert!(safe < 3.0, "near plane must respect floating construction");
        world.edits.push(Edit {
            cell: cell_of(eye),
            radius: 8.0,
            material: 0,
        });
        world.rebuild_edits();
        assert!(
            world.air_clearance(eye) > 6.0,
            "later excavation removes construction"
        );
        world.edits.push(Edit {
            cell: target,
            radius: 1.0,
            material: 3,
        });
        world.rebuild_edits();
        assert!(
            world.air_clearance(eye) < 3.0,
            "new construction inside an old cut remains visible"
        );
    }
    #[test]
    fn ground_spawn_has_support_and_headroom() {
        let w = World::default();
        let p = w.ground_spawn(0.0, -256.0, 1.8);
        assert_eq!(w.material(cell_of(p)), 0);
        let floor = w
            .cast_hit(p, -DVec3::Y, 2.0)
            .expect("spawn must have nearby support");
        assert!((floor.2 - 1.8).abs() < 0.002);
        assert!(w.cast_hit(p, DVec3::Y, 50.0).is_none());
    }
    #[test]
    fn ground_spawn_clears_planetary_construction() {
        let mut w = World::default();
        w.edits.push(Edit {
            cell: [0, 63_710_000, 0],
            radius: 2_000_000.0,
            material: 3,
        });
        w.rebuild_edits();
        let p = w.ground_spawn(0.0, 0.0, 1.8);
        assert!(p.y > RADIUS + 1_999_999.0);
        assert_eq!(w.material(cell_of(p)), 0);
        assert!(w.cast_hit(p, -DVec3::Y, 2.0).is_some());
    }
    #[test]
    fn spatial_index_matches_ordered_replay_for_overlapping_edits() {
        let mut w = World::default();
        for i in 0..4096 {
            let h = hash([i, 37, -91], 117);
            w.edits.push(Edit {
                cell: [
                    (h % 300) as i32 - 150,
                    63_710_000 + ((h >> 10) % 100) as i32 - 50,
                    ((h >> 20) % 300) as i32 - 150,
                ],
                radius: ((h % 80) + 1) as f32 * 0.05,
                material: h % 4,
            });
        }
        w.rebuild_edits();
        assert_eq!(w.edit_index.len(), 4096);
        for i in 0..3000 {
            let h = hash([i, 41, 77], 231);
            let c = [
                (h % 350) as i32 - 175,
                63_710_000 + ((h >> 10) % 150) as i32 - 75,
                ((h >> 20) % 350) as i32 - 175,
            ];
            assert_eq!(
                w.latest_edit(c),
                w.edits.iter().rposition(|e| e.contains(c)),
                "cell {c:?}"
            );
        }
    }
    #[test]
    fn earth_coordinates_retain_ten_centimeter_neighbors() {
        let c = [0, 63_710_000, 0];
        assert!((center([0, c[1] + 1, 0]).distance(center(c)) - 0.1).abs() < 1e-8);
        assert_eq!(cell_of(center(c)), c);
    }
    #[test]
    fn destruction_and_construction_replay_in_order_at_planet_scale() {
        let mut w = World::default();
        let c = [0, 63_709_990, 0];
        assert_ne!(w.material(c), 0);
        w.edits.push(Edit {
            cell: c,
            radius: 0.15,
            material: 0,
        });
        assert_eq!(w.material(c), 0);
        w.edits.push(Edit {
            cell: c,
            radius: 0.05,
            material: 3,
        });
        assert_eq!(w.material(c), 3);
        let restored: World = serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
        assert_eq!(restored.material(c), 3);
    }
    #[test]
    fn journals_roundtrip_and_reject_edits_from_different_terrain() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "helio-voxel-revision-{}-{unique}.json",
            std::process::id()
        ));
        let mut w = World::default();
        let c = [0, 63710000, 0];
        w.edits.push(Edit {
            cell: c,
            radius: 0.05,
            material: 3,
        });
        w.rebuild_edits();
        w.save(&path).unwrap();
        let loaded = World::load(&path).unwrap();
        assert_eq!(loaded.generator_revision, GENERATOR_REVISION);
        assert_eq!(loaded.material(c), 3);
        let old = br#"{"edits":[{"cell":[0,63710000,0],"radius":0.05,"material":0}]}"#;
        std::fs::write(&path, old).unwrap();
        assert!(World::load(&path)
            .err()
            .unwrap()
            .contains("terrain revision"));
        assert_eq!(std::fs::read(&path).unwrap(), old);
        std::fs::write(&path, br#"{"edits":[]}"#).unwrap();
        assert_eq!(
            World::load(&path).unwrap().generator_revision,
            GENERATOR_REVISION
        );
        std::fs::remove_file(&path).unwrap();
    }
}
