//! Decorating the terrain: trees, cacti and structures.
//!
//! The planet is divided into square regions of `REGION_BLOCKS` columns per
//! face. A region is decorated once, from a random stream that is a pure
//! function of the world seed and the region, reading only the terrain (and
//! what neighbours already built, so features never share a block). The result
//! is a list of brushes the edit worker applies.

use super::edits::{box_brush, BlockEdit};
use crate::blocks::Block;
use crate::config::{REGION_BLOCKS, REGION_RADIUS, WORLD_SEED};
use crate::structures::Kind;
use crate::terrain::climate::{forest_density, forest_kind, hash, unit};
use crate::terrain::{self, Biome};
use crate::trees::{Rng, Species, Tree};
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{Brush, Cell, Planet};
use std::collections::HashSet;

/// A region of a face: its column block indices.
pub type Region = (u8, i32, i32);

/// Chance a region holds a structure.
const STRUCTURE_CHANCE: f64 = 0.09;
/// Regions within this many of the player's are decorated when the worker idles.
const RADIUS: i32 = REGION_RADIUS;

pub struct Decorator {
    seed: u64,
    done: HashSet<Region>,
    /// Where the outward scan for undecorated regions stands: the focus
    /// region it started from and the square ring it has reached (rings
    /// inside it are all done), so a scan never starts over from the player.
    scan: Option<(Region, i32)>,
}

impl Decorator {
    pub fn new() -> Self {
        Self { seed: WORLD_SEED, done: HashSet::new(), scan: None }
    }

    pub fn regions_done(&self) -> usize {
        self.done.len()
    }

    /// The undecorated regions around `focus` within reach, nearest rings
    /// first, at most `limit` of them. The scan resumes from the ring it
    /// reached (it only starts over when the player moves to another region),
    /// so a world-sized radius costs nothing until the rings are walked.
    pub fn next_batch(&mut self, planet: &Planet, focus: Cell, limit: usize) -> Vec<Region> {
        let centre = (focus.face, focus.i.div_euclid(REGION_BLOCKS), focus.j.div_euclid(REGION_BLOCKS));
        // Past the face's own width there is nothing to decorate.
        let reach = RADIUS.min(planet.grid().cells() / REGION_BLOCKS + 1);
        let mut ring = match self.scan {
            Some((from, ring)) if from == centre => ring,
            _ => 0,
        };
        let mut out = Vec::new();
        while ring <= reach && out.len() < limit {
            let mut visit = |dx: i32, dy: i32, out: &mut Vec<Region>| {
                if dx * dx + dy * dy > RADIUS * RADIUS || out.len() >= limit {
                    return;
                }
                let region = (centre.0, centre.1 + dx, centre.2 + dy);
                if !self.done.contains(&region) && terrain::buildable(planet, region.1 * REGION_BLOCKS + REGION_BLOCKS / 2, region.2 * REGION_BLOCKS + REGION_BLOCKS / 2) {
                    out.push(region);
                }
            };
            // The cells of the square ring at Chebyshev distance `ring`.
            if ring == 0 {
                visit(0, 0, &mut out);
            } else {
                for k in -ring..=ring {
                    visit(k, -ring, &mut out);
                    visit(k, ring, &mut out);
                }
                for k in (-ring + 1)..ring {
                    visit(-ring, k, &mut out);
                    visit(ring, k, &mut out);
                }
            }
            // A ring that filled the batch may have more left for the next call.
            if out.len() >= limit {
                break;
            }
            ring += 1;
        }
        self.scan = Some((centre, ring));
        out
    }

    /// Whether any region around `focus` is still waiting.
    pub fn has_work(&mut self, planet: &Planet, focus: Cell) -> bool {
        !self.next_batch(planet, focus, 1).is_empty()
    }

    /// Decorates `regions` in parallel against the same planet state, marking
    /// them done. The brushes come back per region, in the order given.
    pub fn decorate_all(&mut self, planet: &Planet, regions: &[Region]) -> Vec<Vec<Brush>> {
        self.done.extend(regions.iter().copied());
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(regions.len()).max(1);
        let this = &*self;
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut slots: Vec<(usize, Vec<Brush>)> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut mine = Vec::new();
                        loop {
                            let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&region) = regions.get(n) else { break };
                            mine.push((n, this.decorate(planet, region)));
                        }
                        mine
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().expect("decorate thread")).collect()
        });
        slots.sort_by_key(|(n, _)| *n);
        slots.into_iter().map(|(_, b)| b).collect()
    }

    /// Decorates `region`: the brushes to apply, in order.
    fn decorate(&self, planet: &Planet, region: Region) -> Vec<Brush> {
        let (face, rx, ry) = region;
        let (i0, j0) = (rx * REGION_BLOCKS, ry * REGION_BLOCKS);
        let mut rng = Rng::new(hash(self.seed, i64::from(face), i64::from(rx), i64::from(ry)));
        let mut brushes = Vec::new();
        // Footprints taken: (i, j, radius).
        let mut taken: Vec<(i32, i32, i32)> = Vec::new();
        let grid = *planet.grid();

        // A structure, near the region's middle.
        if unit(hash(self.seed ^ 0xabc, i64::from(face), i64::from(rx), i64::from(ry))) < STRUCTURE_CHANCE {
            let (ci, cj) = (i0 + REGION_BLOCKS / 2 + rng.int(-5, 5), j0 + REGION_BLOCKS / 2 + rng.int(-5, 5));
            if let Some(g) = terrain::ground(planet, face, ci, cj) {
                if let Some(kind) = Kind::for_biome(Biome::of(g.material), rng.f64()) {
                    let blueprint = kind.build(rng.u64()).oriented(rng.int(0, 3) as u32, rng.f64() < 0.5);
                    let [hx, hz] = blueprint.half;
                    let reach = hx.max(hz) + 1;
                    // The ground under the footprint's corners and middle.
                    let tops: Vec<i32> = [(0, 0), (hx, hz), (-hx, hz), (hx, -hz), (-hx, -hz)].into_iter().map(|(di, dj)| planet.column_top(face, ci + di, cj + dj, 0)).collect();
                    let (low, high) = (*tops.iter().min().unwrap(), *tops.iter().max().unwrap());
                    if high - low <= 7 {
                        let floor = (tops.iter().sum::<i32>() as f64 / tops.len() as f64).round() as i32;
                        let columns = 2 * reach + 3;
                        // Clear what stands in the way, set a foundation, build.
                        brushes.push(box_brush(&grid, face, ci, cj, floor, columns, blueprint.height + 6, None));
                        brushes.push(box_brush(&grid, face, ci, cj, floor - 5, columns - 2, 5, Some(Block::Cobblestone)));
                        let mut ops: Vec<_> = blueprint.blocks.iter().collect();
                        // Clears after sets would undo nothing: clears first.
                        ops.sort_by_key(|(p, b)| (b.is_some(), p[1], p[0], p[2]));
                        for (p, block) in ops {
                            let cell = Cell::new(face, ci + p[0], cj + p[2], floor + p[1]);
                            brushes.push(match block {
                                Some(block) => BlockEdit::set(cell, *block).brush(&grid),
                                None => BlockEdit::clear(cell).brush(&grid),
                            });
                        }
                        taken.push((ci, cj, reach + 3));
                    }
                }
            }
        }

        // Trees and cacti.
        let (mid_i, mid_j) = (i0 + REGION_BLOCKS / 2, j0 + REGION_BLOCKS / 2);
        let density = forest_density(self.seed, face, mid_i, mid_j);
        let kind = forest_kind(self.seed, face, mid_i, mid_j);
        let attempts = 3 + (density * 22.0) as i32;
        let mut trunks: Vec<(i32, i32)> = Vec::new();
        for _ in 0..attempts {
            let (i, j) = (i0 + rng.int(3, REGION_BLOCKS - 4), j0 + rng.int(3, REGION_BLOCKS - 4));
            if taken.iter().any(|&(ti, tj, r)| (ti - i).abs() <= r && (tj - j).abs() <= r) {
                continue;
            }
            // Trunks keep apart: close ones would merge their crowns.
            if trunks.iter().any(|&(ti, tj)| (ti - i).abs() < 4 && (tj - j).abs() < 4) {
                continue;
            }
            let Some(g) = terrain::ground(planet, face, i, j) else { continue };
            if planet.column_top(face, i, j, 0) != g.cell.k + 1 || terrain::relief(planet, face, i, j, 2) > 3 {
                continue;
            }
            let base = g.above();
            match Biome::of(g.material) {
                Biome::Desert => {
                    // Cacti on sand, spread thin.
                    if g.material == material::SAND && rng.f64() < 0.08 + density * 0.1 {
                        for y in 0..rng.int(2, 4) {
                            brushes.push(BlockEdit::set(Cell::new(face, i, j, base.k + y), Block::Cactus).brush(&grid));
                        }
                        trunks.push((i, j));
                    }
                }
                Biome::Rock => {}
                biome => {
                    let height_m = f64::from(g.cell.k) * planet.grid().voxel_size();
                    let species = pick_species(biome, height_m, kind, density, &mut rng);
                    let Some(species) = species else { continue };
                    let tree = Tree::grow(species, hash(self.seed, i64::from(i), i64::from(j), 77));
                    // A trunk of a big tree takes room: keep clear of the rest.
                    let reach = if species == Species::Jungle { 8 } else { 3 };
                    if trunks.iter().any(|&(ti, tj)| (ti - i).abs() < reach && (tj - j).abs() < reach) {
                        continue;
                    }
                    let mut cells: Vec<(Cell, Block)> = Vec::with_capacity(tree.blocks.len());
                    for (off, block) in &tree.blocks {
                        let cell = Cell::new(face, i + off[0], j + off[2], base.k + off[1]);
                        // Crowns stop at hillsides.
                        if planet.solid(cell) && !matches!(block, Block::OakLog | Block::BirchLog | Block::SpruceLog | Block::JungleLog) {
                            continue;
                        }
                        cells.push((cell, *block));
                    }
                    push_columns(&grid, &mut brushes, cells);
                    // Dirt under the trunk, as the roots would make it.
                    if matches!(g.material, material::GRASS) {
                        brushes.push(BlockEdit::set(g.cell, Block::Dirt).brush(&grid));
                    }
                    trunks.push((i, j));
                }
            }
        }
        brushes
    }
}

/// The tree a column grows, if any.
fn pick_species(biome: Biome, height_m: f64, kind: f64, density: f64, rng: &mut Rng) -> Option<Species> {
    // Sparse where the forest field is thin; none on open meadows.
    if rng.f64() > 0.35 + density {
        return None;
    }
    if biome == Biome::Snow || height_m > 55.0 {
        return Some(if rng.f64() < 0.65 { Species::Spruce } else { Species::Pine });
    }
    Some(match kind {
        k if k < 0.30 => {
            if rng.f64() < 0.18 {
                Species::DarkOak
            } else {
                Species::Oak
            }
        }
        k if k < 0.55 => {
            if rng.f64() < 0.7 {
                Species::Birch
            } else {
                Species::Oak
            }
        }
        k if k < 0.78 => {
            if rng.f64() < 0.4 {
                Species::Jungle
            } else {
                Species::Oak
            }
        }
        _ => {
            if rng.f64() < 0.55 {
                Species::Spruce
            } else {
                Species::Birch
            }
        }
    })
}

/// Brushes for `cells`, with vertical runs of one block made into a single
/// box brush: a tree is a few hundred brushes, not a few thousand (every brush
/// is history the planet keeps and the renderer follows).
fn push_columns(grid: &helio_pass_voxel_planet::Grid, brushes: &mut Vec<Brush>, mut cells: Vec<(Cell, Block)>) {
    cells.sort_by_key(|(c, b)| (c.face, c.i, c.j, b.material(), c.k));
    let mut run: Option<(Cell, Block, i32)> = None;
    let mut flush = |run: Option<(Cell, Block, i32)>, brushes: &mut Vec<Brush>| {
        if let Some((start, block, length)) = run {
            brushes.push(box_brush(grid, start.face, start.i, start.j, start.k, 1, length, Some(block)));
        }
    };
    for (cell, block) in cells {
        run = match run {
            Some((start, b, length)) if b == block && start.face == cell.face && start.i == cell.i && start.j == cell.j && start.k + length == cell.k => Some((start, b, length + 1)),
            Some((start, b, length)) if b == block && start.face == cell.face && start.i == cell.i && start.j == cell.j && start.k + length > cell.k => Some((start, b, length)),
            other => {
                flush(other, brushes);
                Some((cell, block, 1))
            }
        };
    }
    flush(run, brushes);
}
