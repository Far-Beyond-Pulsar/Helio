use helio_planet_voxel_core::{AddressError, PageKey, TransitionFace, MAX_ADDRESSABLE_LOD};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

const TANGENT_AXES: [usize; 2] = [0, 2];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainLodTopologyStats {
    pub pages: usize,
    pub minimum_lod: u8,
    pub maximum_lod: u8,
    pub transition_faces: u32,
}

/// Renderer-side oracle for one immutable mixed-LOD visible page set.
///
/// Pulsar remains authoritative for view demand and residency. This type
/// validates the resulting spatial contract and derives the coarse-owned face
/// masks consumed by Helio's Transvoxel transition extractor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainLodTopology {
    pages: BTreeSet<PageKey>,
    transition_masks: BTreeMap<PageKey, u8>,
    stats: TerrainLodTopologyStats,
}

impl TerrainLodTopology {
    pub fn new(pages: impl IntoIterator<Item = PageKey>) -> Result<Self, TerrainLodTopologyError> {
        let mut unique = BTreeSet::new();
        for page in pages {
            page.validate()?;
            if !unique.insert(page) {
                return Err(TerrainLodTopologyError::DuplicatePage(page));
            }
        }
        if unique.is_empty() {
            return Err(TerrainLodTopologyError::Empty);
        }

        let pages = unique.iter().copied().collect::<Vec<_>>();
        let bounds = pages
            .iter()
            .copied()
            .map(PageBounds::new)
            .collect::<Result<Vec<_>, _>>()?;
        let mut transition_masks = pages
            .iter()
            .copied()
            .map(|page| (page, 0_u8))
            .collect::<BTreeMap<_, _>>();

        for left in 0..pages.len() {
            for right in left + 1..pages.len() {
                if bounds[left].overlaps_volume(bounds[right]) {
                    return Err(TerrainLodTopologyError::OverlappingPages {
                        first: pages[left],
                        second: pages[right],
                    });
                }
                let Some((axis, left_positive)) = bounds[left].shared_face(bounds[right]) else {
                    continue;
                };
                let lod_difference = pages[left].lod.abs_diff(pages[right].lod);
                if lod_difference > 1 {
                    return Err(TerrainLodTopologyError::UnbalancedFace {
                        first: pages[left],
                        second: pages[right],
                    });
                }
                if lod_difference == 1 {
                    let (coarse, coarse_positive) = if pages[left].lod > pages[right].lod {
                        (pages[left], left_positive)
                    } else {
                        (pages[right], !left_positive)
                    };
                    *transition_masks
                        .get_mut(&coarse)
                        .expect("every validated page has a transition mask") |=
                        transition_face(axis, coarse_positive).bit();
                }
            }
        }

        let minimum_lod = pages.iter().map(|page| page.lod).min().unwrap_or(0);
        let maximum_lod = pages.iter().map(|page| page.lod).max().unwrap_or(0);
        let transition_faces = transition_masks
            .values()
            .map(|mask| mask.count_ones())
            .sum();
        Ok(Self {
            pages: unique,
            transition_masks,
            stats: TerrainLodTopologyStats {
                pages: pages.len(),
                minimum_lod,
                maximum_lod,
                transition_faces,
            },
        })
    }

    pub fn pages(&self) -> impl ExactSizeIterator<Item = PageKey> + '_ {
        self.pages.iter().copied()
    }

    pub fn transition_mask(&self, page: PageKey) -> Option<u8> {
        self.transition_masks.get(&page).copied()
    }

    pub fn transition_masks(&self) -> &BTreeMap<PageKey, u8> {
        &self.transition_masks
    }

    pub const fn stats(&self) -> TerrainLodTopologyStats {
        self.stats
    }

    /// Proves exact, non-overlapping coverage of one tangent-plane root.
    ///
    /// The standalone horizon fixture represents the planet surface in X/Z
    /// while canonical Y remains the surface-bearing page layer. The generic
    /// three-dimensional overlap and face checks above still guard the actual
    /// renderer page addresses.
    pub fn validate_tangent_root(&self, root: PageKey) -> Result<(), TerrainLodTopologyError> {
        root.validate()?;
        let root_bounds = PageBounds::new(root)?;
        let root_area = root_bounds.tangent_area()?;
        let mut covered_area = 0_u128;
        for page in &self.pages {
            let bounds = PageBounds::new(*page)?;
            if !root_bounds.contains_tangent(bounds) {
                return Err(TerrainLodTopologyError::OutsideTangentRoot { root, page: *page });
            }
            covered_area = covered_area
                .checked_add(bounds.tangent_area()?)
                .ok_or(TerrainLodTopologyError::CoordinateOverflow)?;
        }
        if covered_area != root_area {
            return Err(TerrainLodTopologyError::TangentCoverage {
                expected: root_area,
                actual: covered_area,
            });
        }
        Ok(())
    }
}

/// Deterministic bounded mixed-LOD fixture for `planet_voxel_demo`.
///
/// This is a renderer validation source, not Helio-owned production demand.
/// It refines the camera's canonical ground cell to LOD0, balances adjacent
/// tangent leaves to 2:1, and retains coarse coverage out to the root extent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HorizonLodFixturePlan {
    root: PageKey,
    focus_lod0_cell: [i64; 3],
    topology: TerrainLodTopology,
}

impl HorizonLodFixturePlan {
    pub fn build(
        focus_lod0_cell: [i64; 3],
        root_lod: u8,
        max_pages: usize,
    ) -> Result<Self, TerrainLodTopologyError> {
        Self::build_with_minimum_lod(focus_lod0_cell, root_lod, 0, max_pages)
    }

    pub fn build_with_minimum_lod(
        focus_lod0_cell: [i64; 3],
        root_lod: u8,
        minimum_lod: u8,
        max_pages: usize,
    ) -> Result<Self, TerrainLodTopologyError> {
        if root_lod == 0 || root_lod > MAX_ADDRESSABLE_LOD {
            return Err(TerrainLodTopologyError::UnsupportedRootLod(root_lod));
        }
        if minimum_lod >= root_lod {
            return Err(TerrainLodTopologyError::UnsupportedMinimumLod {
                minimum: minimum_lod,
                root: root_lod,
            });
        }
        if max_pages < 4 {
            return Err(TerrainLodTopologyError::PageBudget {
                maximum: max_pages,
                required: 4,
            });
        }
        let (mut root, _) = PageKey::address_lod0_cell(root_lod, focus_lod0_cell)?;
        root.page_xyz[1] = -1;
        let mut leaves = BTreeSet::from([root]);

        for target_lod in (minimum_lod..root_lod).rev() {
            let target = PageKey::address_lod0_cell(target_lod + 1, focus_lod0_cell)?.0;
            let target = PageKey::new(target.lod, [target.page_xyz[0], -1, target.page_xyz[2]]);
            if !leaves.remove(&target) {
                return Err(TerrainLodTopologyError::MissingRefinementParent(target));
            }
            leaves.extend(tangent_children(target)?);
            balance_tangent_leaves(&mut leaves, max_pages)?;
            if leaves.len() > max_pages {
                return Err(TerrainLodTopologyError::PageBudget {
                    maximum: max_pages,
                    required: leaves.len(),
                });
            }
        }

        let topology = TerrainLodTopology::new(leaves)?;
        topology.validate_tangent_root(root)?;
        Ok(Self {
            root,
            focus_lod0_cell,
            topology,
        })
    }

    pub const fn root(&self) -> PageKey {
        self.root
    }

    pub const fn focus_lod0_cell(&self) -> [i64; 3] {
        self.focus_lod0_cell
    }

    pub const fn topology(&self) -> &TerrainLodTopology {
        &self.topology
    }
}

#[derive(Clone, Copy, Debug)]
struct PageBounds {
    min: [i128; 3],
    max: [i128; 3],
}

impl PageBounds {
    fn new(page: PageKey) -> Result<Self, TerrainLodTopologyError> {
        let minimum = page.lod0_cell_min()?;
        let span = i128::from(page.lod0_cell_span()?);
        let min = minimum.map(i128::from);
        let max = min.map(|coordinate| coordinate + span);
        Ok(Self { min, max })
    }

    fn overlaps_volume(self, other: Self) -> bool {
        (0..3).all(|axis| self.min[axis] < other.max[axis] && other.min[axis] < self.max[axis])
    }

    fn shared_face(self, other: Self) -> Option<(usize, bool)> {
        for axis in 0..3 {
            let self_positive = self.max[axis] == other.min[axis];
            let self_negative = other.max[axis] == self.min[axis];
            if !self_positive && !self_negative {
                continue;
            }
            let tangential_overlap =
                (0..3)
                    .filter(|other_axis| *other_axis != axis)
                    .all(|other_axis| {
                        self.min[other_axis] < other.max[other_axis]
                            && other.min[other_axis] < self.max[other_axis]
                    });
            if tangential_overlap {
                return Some((axis, self_positive));
            }
        }
        None
    }

    fn contains_tangent(self, other: Self) -> bool {
        TANGENT_AXES
            .into_iter()
            .all(|axis| self.min[axis] <= other.min[axis] && other.max[axis] <= self.max[axis])
    }

    fn tangent_area(self) -> Result<u128, TerrainLodTopologyError> {
        TANGENT_AXES.into_iter().try_fold(1_u128, |area, axis| {
            let span = u128::try_from(self.max[axis] - self.min[axis])
                .map_err(|_| TerrainLodTopologyError::CoordinateOverflow)?;
            area.checked_mul(span)
                .ok_or(TerrainLodTopologyError::CoordinateOverflow)
        })
    }
}

fn tangent_children(parent: PageKey) -> Result<[PageKey; 4], TerrainLodTopologyError> {
    let lod = parent
        .lod
        .checked_sub(1)
        .ok_or(TerrainLodTopologyError::CannotRefineLod0)?;
    let base_x = parent.page_xyz[0]
        .checked_mul(2)
        .ok_or(TerrainLodTopologyError::CoordinateOverflow)?;
    let base_z = parent.page_xyz[2]
        .checked_mul(2)
        .ok_or(TerrainLodTopologyError::CoordinateOverflow)?;
    Ok([
        PageKey::new(lod, [base_x, -1, base_z]),
        PageKey::new(lod, [base_x + 1, -1, base_z]),
        PageKey::new(lod, [base_x, -1, base_z + 1]),
        PageKey::new(lod, [base_x + 1, -1, base_z + 1]),
    ])
}

fn balance_tangent_leaves(
    leaves: &mut BTreeSet<PageKey>,
    max_pages: usize,
) -> Result<(), TerrainLodTopologyError> {
    loop {
        let pages = leaves.iter().copied().collect::<Vec<_>>();
        let bounds = pages
            .iter()
            .copied()
            .map(PageBounds::new)
            .collect::<Result<Vec<_>, _>>()?;
        let mut coarse = None;
        'pairs: for left in 0..pages.len() {
            for right in left + 1..pages.len() {
                if tangent_shared_edge(bounds[left], bounds[right])
                    && pages[left].lod.abs_diff(pages[right].lod) > 1
                {
                    coarse = Some(if pages[left].lod > pages[right].lod {
                        pages[left]
                    } else {
                        pages[right]
                    });
                    break 'pairs;
                }
            }
        }
        let Some(coarse) = coarse else {
            return Ok(());
        };
        let required = leaves.len().saturating_add(3);
        if required > max_pages {
            return Err(TerrainLodTopologyError::PageBudget {
                maximum: max_pages,
                required,
            });
        }
        leaves.remove(&coarse);
        leaves.extend(tangent_children(coarse)?);
    }
}

fn tangent_shared_edge(left: PageBounds, right: PageBounds) -> bool {
    TANGENT_AXES.into_iter().any(|axis| {
        let other_axis = if axis == TANGENT_AXES[0] {
            TANGENT_AXES[1]
        } else {
            TANGENT_AXES[0]
        };
        (left.max[axis] == right.min[axis] || right.max[axis] == left.min[axis])
            && left.min[other_axis] < right.max[other_axis]
            && right.min[other_axis] < left.max[other_axis]
    })
}

const fn transition_face(axis: usize, positive: bool) -> TransitionFace {
    match (axis, positive) {
        (0, false) => TransitionFace::NegativeX,
        (0, true) => TransitionFace::PositiveX,
        (1, false) => TransitionFace::NegativeY,
        (1, true) => TransitionFace::PositiveY,
        (2, false) => TransitionFace::NegativeZ,
        (2, true) => TransitionFace::PositiveZ,
        _ => unreachable!(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TerrainLodTopologyError {
    #[error("mixed-LOD topology must contain at least one page")]
    Empty,
    #[error(transparent)]
    Address(#[from] AddressError),
    #[error("mixed-LOD topology contains duplicate page {0:?}")]
    DuplicatePage(PageKey),
    #[error("mixed-LOD pages {first:?} and {second:?} overlap")]
    OverlappingPages { first: PageKey, second: PageKey },
    #[error("mixed-LOD pages {first:?} and {second:?} violate the 2:1 face constraint")]
    UnbalancedFace { first: PageKey, second: PageKey },
    #[error("LOD{0} cannot be used as the horizon fixture root")]
    UnsupportedRootLod(u8),
    #[error("minimum LOD{minimum} must be finer than root LOD{root}")]
    UnsupportedMinimumLod { minimum: u8, root: u8 },
    #[error("LOD0 pages cannot be refined")]
    CannotRefineLod0,
    #[error("horizon fixture lost refinement parent {0:?}")]
    MissingRefinementParent(PageKey),
    #[error("page budget {maximum} cannot hold the required {required} pages")]
    PageBudget { maximum: usize, required: usize },
    #[error("page coordinate arithmetic overflowed")]
    CoordinateOverflow,
    #[error("page {page:?} lies outside tangent root {root:?}")]
    OutsideTangentRoot { root: PageKey, page: PageKey },
    #[error("tangent coverage is {actual} LOD0 cells squared; expected {expected}")]
    TangentCoverage { expected: u128, actual: u128 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_page_owns_transition_bits_on_every_face_and_quadrant() {
        for face in TransitionFace::ALL {
            let axis = face.axis();
            let coarse = PageKey::new(1, [-3, -3, -3]);
            let tangential = match axis {
                0 => [1, 2],
                1 => [0, 2],
                2 => [0, 1],
                _ => unreachable!(),
            };
            for quadrant in 0..4 {
                let mut fine_xyz = coarse.page_xyz.map(|coordinate| coordinate * 2);
                fine_xyz[axis] += if face.is_positive() { 2 } else { -1 };
                fine_xyz[tangential[0]] += (quadrant & 1) as i64;
                fine_xyz[tangential[1]] += ((quadrant >> 1) & 1) as i64;
                let fine = PageKey::new(0, fine_xyz);
                let topology = TerrainLodTopology::new([coarse, fine]).unwrap();
                assert_eq!(topology.transition_mask(coarse), Some(face.bit()));
                assert_eq!(topology.transition_mask(fine), Some(0));
            }
        }
    }

    #[test]
    fn edge_and_corner_contacts_do_not_create_false_transition_faces() {
        let coarse = PageKey::new(1, [-1, -1, -1]);
        let edge = PageKey::new(0, [0, 0, -2]);
        let corner = PageKey::new(0, [0, 0, 0]);
        let topology = TerrainLodTopology::new([coarse, edge, corner]).unwrap();
        assert_eq!(topology.transition_mask(coarse), Some(0));
        assert_eq!(topology.stats().transition_faces, 0);
    }

    #[test]
    fn overlap_and_unbalanced_faces_fail_explicitly() {
        let coarse = PageKey::new(2, [-1, -1, -1]);
        let descendant = PageKey::new(0, [-4, -4, -4]);
        assert!(matches!(
            TerrainLodTopology::new([coarse, descendant]),
            Err(TerrainLodTopologyError::OverlappingPages { .. })
        ));

        let coarse = PageKey::new(2, [-1, -1, -1]);
        let fine = PageKey::new(0, [0, -4, -4]);
        assert!(matches!(
            TerrainLodTopology::new([coarse, fine]),
            Err(TerrainLodTopologyError::UnbalancedFace { .. })
        ));
    }

    #[test]
    fn horizon_plan_is_deterministic_bounded_balanced_and_exact() {
        let focus = [63_710_000, -1, -17];
        let first = HorizonLodFixturePlan::build(focus, 11, 96).unwrap();
        let second = HorizonLodFixturePlan::build(focus, 11, 96).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.topology().stats().minimum_lod, 0);
        assert!(first.topology().stats().maximum_lod >= 7);
        assert!(first.topology().stats().pages <= 96);
        assert!(first.topology().stats().transition_faces > 0);
        first
            .topology()
            .validate_tangent_root(first.root())
            .unwrap();
    }

    #[test]
    fn horizon_plan_crosses_signed_boundaries_and_teleports_without_growth() {
        let traces = [
            [63_710_000, -1, -1],
            [63_710_032, -1, 0],
            [-63_710_001, -1, -33],
            [-63_710_033, -1, 32],
        ];
        let plans = traces
            .into_iter()
            .map(|focus| HorizonLodFixturePlan::build(focus, 11, 96).unwrap())
            .collect::<Vec<_>>();
        assert!(plans.iter().all(|plan| plan.topology().stats().pages <= 96));
        assert_ne!(plans[0].topology().pages, plans[2].topology().pages);
        for plan in plans {
            plan.topology().validate_tangent_root(plan.root()).unwrap();
        }
    }

    #[test]
    fn altitude_fixture_coarsens_the_near_field_without_losing_exact_coverage() {
        let focus = [63_710_000, -1, 17];
        let ground = HorizonLodFixturePlan::build_with_minimum_lod(focus, 11, 0, 96).unwrap();
        let orbit = HorizonLodFixturePlan::build_with_minimum_lod(focus, 11, 5, 96).unwrap();
        assert_eq!(ground.topology().stats().minimum_lod, 0);
        assert_eq!(orbit.topology().stats().minimum_lod, 5);
        assert!(orbit.topology().stats().pages < ground.topology().stats().pages);
        orbit
            .topology()
            .validate_tangent_root(orbit.root())
            .unwrap();
    }

    #[test]
    fn randomized_signed_horizon_neighborhoods_are_exact_balanced_and_single_owned() {
        let mut random = 0x4D59_5DF4_D0F3_3173_u64;
        for case_index in 0..512 {
            let focus = [
                random_signed_coordinate(&mut random),
                -1,
                random_signed_coordinate(&mut random),
            ];
            let minimum_lod = (next_random(&mut random) % 6) as u8;
            let plan = HorizonLodFixturePlan::build_with_minimum_lod(focus, 11, minimum_lod, 192)
                .unwrap_or_else(|error| panic!("case {case_index}, focus {focus:?}: {error}"));
            let replay =
                HorizonLodFixturePlan::build_with_minimum_lod(focus, 11, minimum_lod, 192).unwrap();
            assert_eq!(plan, replay, "case {case_index}");
            assert!(plan.topology().stats().pages <= 192, "case {case_index}");
            assert_eq!(
                plan.topology().stats().minimum_lod,
                minimum_lod,
                "case {case_index}"
            );
            plan.topology()
                .validate_tangent_root(plan.root())
                .unwrap_or_else(|error| panic!("case {case_index}: {error}"));
            assert_transition_ownership_is_exact(plan.topology(), case_index);
        }
    }

    #[test]
    fn declared_page_budget_fails_instead_of_degrading_topology() {
        assert!(matches!(
            HorizonLodFixturePlan::build([63_710_000, -1, 0], 10, 32),
            Err(TerrainLodTopologyError::PageBudget { maximum: 32, .. })
        ));
    }

    #[test]
    fn page_edge_constant_matches_canonical_addressing() {
        assert_eq!(helio_planet_voxel_core::PAGE_EDGE_CELLS, 32);
    }

    fn assert_transition_ownership_is_exact(topology: &TerrainLodTopology, case_index: usize) {
        let pages = topology.pages().collect::<Vec<_>>();
        let bounds = pages
            .iter()
            .copied()
            .map(PageBounds::new)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut expected = BTreeSet::new();
        for left in 0..pages.len() {
            for right in left + 1..pages.len() {
                assert!(
                    !bounds[left].overlaps_volume(bounds[right]),
                    "case {case_index}: overlapping {:?} and {:?}",
                    pages[left],
                    pages[right]
                );
                let Some((axis, left_positive)) = bounds[left].shared_face(bounds[right]) else {
                    continue;
                };
                assert!(
                    pages[left].lod.abs_diff(pages[right].lod) <= 1,
                    "case {case_index}: unbalanced {:?} and {:?}",
                    pages[left],
                    pages[right]
                );
                if pages[left].lod == pages[right].lod {
                    continue;
                }
                let (coarse, coarse_positive) = if pages[left].lod > pages[right].lod {
                    (pages[left], left_positive)
                } else {
                    (pages[right], !left_positive)
                };
                expected.insert((coarse, transition_face(axis, coarse_positive)));
            }
        }

        let actual = topology
            .transition_masks()
            .iter()
            .flat_map(|(page, mask)| {
                TransitionFace::ALL
                    .into_iter()
                    .filter(move |face| mask & face.bit() != 0)
                    .map(move |face| (*page, face))
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected, "case {case_index}");
        assert_eq!(
            actual.len() as u32,
            topology.stats().transition_faces,
            "case {case_index}"
        );
    }

    fn random_signed_coordinate(random: &mut u64) -> i64 {
        let magnitude = (next_random(random) % 127_420_000) as i64;
        if next_random(random) & 1 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    fn next_random(random: &mut u64) -> u64 {
        *random ^= *random << 13;
        *random ^= *random >> 7;
        *random ^= *random << 17;
        *random
    }
}
