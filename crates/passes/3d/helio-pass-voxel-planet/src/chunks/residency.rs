//! Chunk residency (`docs/chunk-octree.md`): which nodes of the chunk
//! octree are drawn, generated next and evicted, for a moving eye.
//!
//! A Mixed node is refined while its cells would project larger than
//! `lod_pixels` pixels from the eye or from where the eye will be
//! `prefetch_seconds` ahead (it stays refined until they fall under
//! `coarsen` of that); out of view, `offscreen_pixels` times that; where
//! no ray has met it for `seen_frames` (occluded, or not yet sampled),
//! `unseen_pixels` times that; where the eye's motion sweeps the image, up
//! to `motion_max` times that. Empty, Solid and Buried nodes are never
//! refined. A refined node keeps being drawn until every Mixed child is
//! generated, so nothing is ever missing: detail arrives as whole nodes
//! replacing their parent. The cut covers the whole world: roots are
//! always visited, and nodes the eye cannot see are drawn as they are.
//! Generated nodes no plan has visited for `release_frames` are evicted,
//! and when chunk slots run short, those no plan has visited for longest.
//!
//! Pure CPU logic over node keys, ranges and edits, run on a [`Worker`]
//! thread; the GPU work lives in `chunks`.

use super::classify::{Class, EditView, Footprint, Kind, NodeKey, Ranges};
use super::{tile_distance, Eye, CHUNK};
use crate::grid::Grid;
use glam::DVec3;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct PlanConfig {
    /// Radians a pixel spans.
    pub pixel_angle: f64,
    /// Refine while a cell would cover more pixels than this.
    pub lod_pixels: f64,
    /// Out of view (past the view cone widened by `view_guard` radians),
    /// cells may cover this many times `lod_pixels`: a level coarser at 2,
    /// an eighth of the nodes there.
    pub offscreen_pixels: f64,
    pub view_guard: f64,
    /// In view but met by no ray for `seen_frames` frames (occluded), cells
    /// may cover this many times `lod_pixels` (1: rays decide nothing).
    pub unseen_pixels: f64,
    pub seen_frames: u64,
    /// Coarsen again only under this share of `lod_pixels`.
    pub coarsen: f64,
    /// Planes: nodes farther than this are not drawn (spheres draw up to
    /// the horizon).
    pub max_distance: f64,
    /// Also refine for where the eye will be this far ahead.
    pub prefetch_seconds: f64,
    /// Bookkeeping of nodes not generated (wanted once, or found empty) is
    /// dropped after this many frames unvisited.
    pub keep_frames: u64,
    /// Generated nodes no plan has visited for this many frames are
    /// released (`MAX_RELEASE` a plan, least recently visited first), so
    /// residency follows the cut rather than the eye's history; a cut
    /// that comes back within them finds its chunks (0: released only
    /// when chunk slots run short).
    pub release_frames: u64,
    /// Where the eye's motion sweeps a node's image by `flow` pixels a
    /// frame, its cells may cover `1 + flow / motion_flow` times
    /// `lod_pixels`, at most `motion_max` (0: no motion allowance).
    pub motion_flow: f64,
    pub motion_max: f64,
    /// Buried nodes within this many of their cells of the eye are
    /// generated: those holding an eye under the ground (the rest of its
    /// cave comes as rays reach it).
    pub near_cells: f64,
}

/// The frame time motion allowances assume (60 frames a second).
const MOTION_FRAME_SECONDS: f64 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug, Default)]
struct Node {
    generated: bool,
    /// Generated and found to hold nothing at its level: drawn as air,
    /// still refined (its descendants may hold ground its coarse cells
    /// rounded away).
    empty: bool,
    /// Its children were drawn in its place last time it was visited.
    refined: bool,
    used: u64,
    /// The plan that last drew it on screen (frame + 1; 0: never).
    drawn: u64,
    /// Buried, and met by a ray: generated like a Mixed node.
    reached: bool,
    /// The layers (level cells, [lo, hi)) its chunk occupies, once its
    /// result came back (generated or restored).
    occupied: Option<(i32, i32)>,
    /// What it was drawn as when it was last a leaf of the cut (Mixed: its
    /// chunk; Empty: air), and that plan (frame + 1).
    leaf: Option<(Kind, u64)>,
    /// Edits made it Mixed and its chunk is not generated yet: drawn as it
    /// was (Empty: air; Solid or Buried: rock) meanwhile, without holding
    /// its parent back, so an edit never shows a coarse ancestor in its
    /// place.
    stand_in: Option<Kind>,
    /// The frame an edit last marked it.
    edited: u64,
}

/// Frames rock an edit added stays air while its footprint's new surface
/// is generated, at most (`Planner::surface_pending`).
const ADDED_ROCK_WAITS: u64 = 120;

struct Fp {
    ranges: Ranges,
    /// Measured on the GPU (else estimated from the parent's quarter over
    /// it, or the parent's whole ranges while it is estimated too).
    exact: bool,
    /// Measured: each quarter's ranges (32x32 columns, x then y: a child
    /// footprint's; none past a face's edge).
    quarters: [Option<Ranges>; 4],
    /// Levels under the nearest measured ancestor (0: measured, or a root):
    /// an estimate's margin doubles at each.
    guess: u8,
    used: u64,
    /// Its classification, and the edits' version it was made for.
    class: Option<(u64, Class)>,
}

/// A band's refinement this plan ([`Planner::decide`]).
#[derive(Clone, Copy, Debug)]
struct Decision {
    error: f64,
    seen: bool,
    allowance: f64,
    ready: bool,
    exact: bool,
    guess: u8,
    /// Its cells' pixels from the eye now, measured to the cells its
    /// chunks occupy, once known.
    cells: Option<f64>,
}

/// Where the eye looks: nodes in view are generated first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub forward: DVec3,
    /// Cosine of the half-angle of the cone around `forward` in view.
    pub cos_half: f64,
}

/// One frame's decisions.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Nodes to draw (they cover the world once each, Empty nodes left
    /// out) and what they hold.
    pub drawn: Vec<(NodeKey, Kind)>,
    /// Nodes to generate, most wanted first, those of a footprint
    /// together (at most `MAX_REQUESTS`).
    pub requests: Vec<NodeKey>,
    /// Nodes waiting to be generated in all.
    pub waiting: usize,
    /// Drawn Mixed nodes in view whose cells project larger than wanted,
    /// as a share of them (the refinement still to come).
    pub coarse_share: f64,
    /// Pixels the largest cell of a drawn node in view spans from the eye
    /// (above the finest level; measured to the cells its chunks occupy,
    /// once their results are back), and the largest of those over its
    /// allowances (`PlanConfig::motion_flow`, `unseen_pixels`; 1: as
    /// wanted).
    pub worst_px: f64,
    pub worst_wanted: f64,
    pub worst: Option<Worst>,
    /// Nodes in view this plan switches between levels, and the pixels the
    /// largest cell of those spans from the eye (measured to their cells,
    /// as `worst_px`).
    pub switches: usize,
    pub switch_px: f64,
}

/// The coarsest node in view (`Plan::worst`).
#[derive(Clone, Copy, Debug)]
pub struct Worst {
    pub key: NodeKey,
    pub exact: bool,
    pub guess: u8,
    pub children: usize,
    pub generated: usize,
}

pub struct Planner {
    nodes: FxHashMap<NodeKey, Node>,
    footprints: FxHashMap<Footprint, Fp>,
    /// The frame rays last met each node or a node under it.
    seen: FxHashMap<NodeKey, u64>,
    /// Whether any ray report came yet (before one, nothing is unseen).
    rays: bool,
    roots: Vec<NodeKey>,
    root_level: u32,
    /// The planet's height range in base layers.
    range: (i32, i32),
    config: PlanConfig,
    edits: Arc<EditView>,
    /// Spheres: the radius of the lowest ground, whose horizon hides what
    /// lies beyond it (none when that ground reaches the centre).
    occluder: Option<f64>,
    /// The last plan's frame + 1 (0: none yet).
    last_plan: u64,
}

impl Planner {
    /// Roots: every node of the coarsest level from the planet's centre (a
    /// plane's lowest ground) to its highest, and as deep and high as the
    /// edits reach, with the planet's whole height range (`range`, base
    /// layers) as their footprints' estimate.
    pub fn new(grid: &Grid, range: (i32, i32), config: PlanConfig, edits: Arc<EditView>) -> Self {
        let level = grid.levels() - 1;
        let occluder = (!grid.is_plane())
            .then(|| grid.radius() + f64::from(range.0) * f64::from(grid.layer_mm()) * 1e-3)
            .filter(|r| *r > 0.0);
        let mut planner = Self {
            nodes: FxHashMap::default(),
            footprints: FxHashMap::default(),
            seen: FxHashMap::default(),
            rays: false,
            roots: Vec::new(),
            root_level: level,
            range,
            config,
            edits,
            occluder,
            last_plan: 0,
        };
        planner.place_roots(grid);
        planner
    }

    /// The roots for the planet and its edits.
    fn place_roots(&mut self, grid: &Grid) {
        let level = self.root_level;
        let n = tiles_at(grid, level);
        let layers = ((self.range.0 >> level) - 1, (self.range.1 >> level) + 1);
        let ranges = Ranges {
            volume: layers,
            surface: layers,
        };
        // Base layers the roots span: the centre (or the lowest ground)
        // to the highest ground, widened by the edits'.
        let bottom = if grid.is_plane() {
            self.range.0
        } else {
            -(grid.radius() / grid.voxel_size()).ceil() as i32
        };
        let (mut lo, mut hi) = (bottom.min(self.range.0), self.range.1);
        if let Some((a, b)) = self.edits.reach {
            lo = lo.min(a);
            hi = hi.max(b);
        }
        if let Some((a, b)) = self.edits.bricks.layers {
            lo = lo.min(a);
            hi = hi.max(b);
        }
        let cz = |k: i32| (k >> level).div_euclid(CHUNK);
        self.roots.clear();
        for &face in grid.faces() {
            for cx in 0..n {
                for cy in 0..n {
                    let fp = Footprint {
                        face,
                        level,
                        cx,
                        cy,
                    };
                    self.footprints.entry(fp).or_insert(Fp {
                        ranges,
                        exact: false,
                        quarters: [None; 4],
                        guess: 0,
                        used: 0,
                        class: None,
                    });
                    self.roots.extend((cz(lo)..=cz(hi)).map(|z| fp.node(z)));
                }
            }
        }
    }

    pub fn config(&self) -> &PlanConfig {
        &self.config
    }

    /// What `key` holds, from its footprint (which must exist).
    fn kind(&mut self, key: NodeKey) -> Kind {
        let edits = &self.edits;
        let fp = self
            .footprints
            .get_mut(&key.footprint())
            .expect("a visited node's footprint");
        if fp.class.as_ref().is_none_or(|(v, _)| *v != edits.version) {
            fp.class = Some((edits.version, edits.classify(key.footprint(), fp.ranges)));
        }
        let class = &fp.class.as_ref().unwrap().1;
        edits.kind(key.footprint(), class, key.cz)
    }

    /// The footprints under `key`'s, estimated unless measured: from the
    /// measured quarter of its columns over each (a coarse footprint spans
    /// all its children's relief: one on the Moon 200 m across holds crater
    /// walls tens of metres over the plain beside them), else from its
    /// ranges.
    fn estimate_children(&mut self, key: NodeKey, frame: u64) {
        let Some(parent) = self.footprints.get(&key.footprint()) else {
            return;
        };
        let (whole, quarters, guess) = (
            parent.ranges,
            parent.quarters,
            if parent.exact {
                1
            } else {
                parent.guess.saturating_add(1)
            },
        );
        for (q, (a, b)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let ranges = quarters[q].unwrap_or(whole).child();
            let fp = Footprint {
                face: key.face,
                level: key.level - 1,
                cx: key.cx * 2 + a,
                cy: key.cy * 2 + b,
            };
            let entry = self.footprints.entry(fp).or_insert(Fp {
                ranges,
                exact: false,
                quarters: [None; 4],
                guess,
                used: frame,
                class: None,
            });
            entry.used = frame;
            if !entry.exact && entry.ranges != ranges {
                entry.ranges = ranges;
                entry.guess = guess;
                entry.class = None;
            }
        }
    }

    fn guess_of(&self, key: NodeKey) -> u8 {
        self.footprints
            .get(&key.footprint())
            .map_or(0, |f| if f.exact { 0 } else { f.guess })
    }

    /// Whether layers `layers` of `fp` cannot be seen from `eye`: below
    /// its horizon over the lowest ground (spheres; an eye under that ground
    /// sees past it), or past `max_distance` (planes). Conservative: their
    /// nearest point against the horizon distances of the eye and of their
    /// top.
    fn beyond(&self, grid: &Grid, fp: Footprint, layers: (i32, i32), eye: &Eye) -> bool {
        let distance = tile_distance(grid, fp, layers, eye);
        let Some(r) = self.occluder else {
            return distance > self.config.max_distance;
        };
        let over = |radius: f64| (radius * radius - r * r).max(0.0).sqrt();
        let e = eye.position.length();
        if e <= r {
            return false;
        }
        let top = grid.radius() + f64::from(layers.1) * grid.level_size(fp.level);
        distance > over(e) + over(top)
    }

    /// Cells of `key` project this many times `lod_pixels` from the
    /// nearest of `eyes`.
    fn error(&self, grid: &Grid, key: NodeKey, eyes: &[Eye]) -> f64 {
        self.error_box(grid, key.footprint(), key.layers(), eyes)
    }

    /// Cells of `fp`'s layers `layers` project this many times
    /// `lod_pixels` from the nearest of `eyes`.
    fn error_box(&self, grid: &Grid, fp: Footprint, layers: (i32, i32), eyes: &[Eye]) -> f64 {
        let distance = eyes
            .iter()
            .map(|e| tile_distance(grid, fp, layers, e))
            .fold(f64::INFINITY, f64::min);
        grid.level_size(fp.level)
            / distance.max(1e-3)
            / (self.config.pixel_angle * self.config.lod_pixels)
    }

    /// How many times `lod_pixels` the cells of `fp`'s layers `layers` may
    /// cover where the eye's motion (`velocity`) sweeps their image.
    fn motion_allowance(
        &self,
        grid: &Grid,
        fp: Footprint,
        layers: (i32, i32),
        eye: &Eye,
        velocity: DVec3,
    ) -> f64 {
        if self.config.motion_flow <= 0.0 {
            return 1.0;
        }
        let offset = eye.toward(grid, fp, layers);
        let distance = offset.length().max(1e-3);
        let across = velocity - offset * (velocity.dot(offset) / (distance * distance));
        let flow = across.length() / distance * MOTION_FRAME_SECONDS / self.config.pixel_angle;
        (1.0 + flow / self.config.motion_flow).min(self.config.motion_max.max(1.0))
    }

    /// The band `key` switches levels with: its footprint's contiguous Mixed
    /// layers holding it (itself when it is not Mixed by class: a Buried
    /// node generated).
    fn band_of(&mut self, key: NodeKey) -> (i32, i32) {
        let _ = self.kind(key);
        let class = &self.footprints[&key.footprint()]
            .class
            .as_ref()
            .expect("classified above")
            .1;
        class
            .mixed
            .iter()
            .find(|(a, b)| (*a..=*b).contains(&key.cz))
            .copied()
            .unwrap_or((key.cz, key.cz))
    }

    /// The layers (level cells) the chunks of nodes `members` (a band)
    /// occupy, once every one's result came back (the same whether
    /// generated or restored).
    fn occupied_of(&self, members: &[NodeKey]) -> Option<(i32, i32)> {
        let mut occupied: Option<(i32, i32)> = None;
        for m in members {
            match self.nodes.get(m) {
                Some(n) if n.generated && n.empty => {}
                Some(Node {
                    generated: true,
                    occupied: Some(o),
                    ..
                }) => occupied = Some(occupied.map_or(*o, |a| (a.0.min(o.0), a.1.max(o.1)))),
                _ => return None,
            }
        }
        occupied
    }

    /// Whether a Mixed node of `fp` (classified) is not generated yet.
    fn surface_pending(&self, fp: Footprint) -> bool {
        let Some((_, class)) = self.footprints.get(&fp).and_then(|f| f.class.as_ref()) else {
            return false;
        };
        class.mixed.iter().any(|&(a, b)| {
            (a..=b).any(|cz| !self.nodes.get(&fp.node(cz)).is_some_and(|n| n.generated))
        })
    }

    /// Whether `key`'s band (`band`, its footprint's layers) is refined,

    /// from its box: its error, in view, allowances; ready once every Mixed
    /// child of every node in it is generated (the children wanted, and the
    /// nodes under the eye when it is far too coarse, are queued in
    /// `wanted`).
    #[allow(clippy::too_many_arguments)]
    fn decide(
        &mut self,
        grid: &Grid,
        key: NodeKey,
        band: (i32, i32),
        eyes: &[Eye; 2],
        velocity: DVec3,
        view: Option<View>,
        guarded: f64,
        frame: u64,
        tiles: &[i32],
        wanted: &mut Vec<(f64, NodeKey)>,
    ) -> Decision {
        let fp = key.footprint();
        // Refinement measures to the band's chunks, air over its ground
        // included (under a hovering eye, more than its cells' own distance
        // asks); the plan's worst cell, to the cells drawn (`cells`).
        let layers = (band.0 * CHUNK, (band.1 + 1) * CHUNK);
        let members: Vec<NodeKey> = (band.0..=band.1).map(|cz| fp.node(cz)).collect();
        let error = self.error_box(grid, fp, layers, eyes);
        let now = self.error_box(grid, fp, layers, &eyes[..1]);
        let cells = self
            .occupied_of(&members)
            .map(|o| self.error_box(grid, fp, o, &eyes[..1]));
        // In view: the band's nearest point within the cone (generous for
        // its extent: anything beside it is near it).
        let cone = |cos_half: f64| -> bool {
            let Some(v) = view else { return true };
            let offset = eyes[0].toward(grid, fp, layers);
            let length = offset.length();
            let size = grid.level_size(fp.level) * f64::from(CHUNK) * 1.5;
            length < size || offset.dot(v.forward) > cos_half * length - size
        };
        let seen = cone(view.map_or(-1.0, |v| v.cos_half));
        let motion = self.motion_allowance(grid, fp, layers, &eyes[0], velocity);
        // Met by rays lately (or no ray report yet): as fine as wanted.
        let met = !self.rays
            || self.config.unseen_pixels <= 1.0
            || members.iter().any(|m| {
                self.seen
                    .get(m)
                    .is_some_and(|&f| f + self.config.seen_frames >= frame)
            });
        let refined = members
            .iter()
            .any(|m| self.nodes.get(m).is_some_and(|n| n.refined));
        // Rays not meeting a band only keep it from refining: a refined
        // band they stop meeting (its children hidden) stays refined, or its
        // coarser cells, poking out, would be met and refine it again.
        let unmet = !met && !refined;
        let allowance = motion
            * if !seen && !cone(guarded) {
                self.config.offscreen_pixels
            } else if unmet {
                self.config.unseen_pixels
            } else {
                1.0
            };
        // Bands in view first, by what they need now.
        let priority = if seen {
            now / allowance * 1e6
        } else {
            now / allowance
        };
        let wanted_error = error / allowance;
        let finer = key.level > 0
            && if refined {
                wanted_error > self.config.coarsen
            } else {
                wanted_error > 1.0
            };
        // Children wait for their footprint's measured ranges unless the
        // band is far too coarse (an eye that dropped from orbit) and a
        // measured ancestor is near: then they come estimated.
        let deep = wanted_error >= DEEP_REFINE;
        let guess = self.guess_of(key);
        let exact = self.footprints.get(&fp).is_some_and(|f| f.exact);
        let refine = finer && (exact || deep && guess < GUESS_LEVELS);
        let mut ready = refine;
        if refine {
            self.estimate_children(key, frame);
            for &m in &members {
                // A member drawn as it was before an edit hands that to its
                // children (rock's children are rock, air's air).
                let inherited = self.nodes.get(&m).and_then(|n| n.stand_in);
                let below: Vec<NodeKey> = m.children(tiles[m.level as usize - 1]).collect();
                for child in below {
                    let kind = self.kind(child);
                    let needs = kind != Kind::Empty && self.needs_data(grid, child, kind, &eyes[0]);
                    let entry = self.nodes.entry(child).or_default();
                    entry.used = frame;
                    if !entry.generated && needs {
                        if entry.stand_in.is_none() {
                            entry.stand_in = inherited;
                            entry.edited = frame;
                        }
                        // A Buried child is rock meanwhile, an edited one
                        // what it was; another Mixed one has no stand-in
                        // but its parent.
                        if kind == Kind::Mixed && entry.stand_in.is_none() {
                            ready = false;
                        }
                        wanted.push((priority, child));
                    }
                }
            }
        }
        // Far too coarse: the nodes under the eye down to the level it
        // wants are wanted now (each level's eight, along the child
        // nearest it holding ground), so the cut refines through them as
        // they come.
        if refine && deep {
            let mut at = key;
            loop {
                let near = |k: &NodeKey| tile_distance(grid, k.footprint(), k.layers(), &eyes[0]);
                let below: Vec<NodeKey> = at.children(tiles[at.level as usize - 1]).collect();
                let mut next = None;
                for c in below {
                    if self.kind(c) == Kind::Mixed
                        && next.is_none_or(|n: NodeKey| near(&c) < near(&n))
                    {
                        next = Some(c);
                    }
                }
                let Some(next) = next else { break };
                if next.level == 0
                    || self.guess_of(next) >= GUESS_LEVELS
                    || self.error(grid, next, eyes) <= 1.0
                {
                    break;
                }
                self.estimate_children(next, frame);
                let below: Vec<NodeKey> = next.children(tiles[next.level as usize - 1]).collect();
                for c in below {
                    let kind = self.kind(c);
                    let needs = kind != Kind::Empty && self.needs_data(grid, c, kind, &eyes[0]);
                    let entry = self.nodes.entry(c).or_default();
                    entry.used = frame;
                    if !entry.generated && needs {
                        wanted.push((priority, c));
                    }
                }
                at = next;
            }
        }
        Decision {
            error,
            seen,
            allowance,
            ready,
            exact,
            guess,
            cells,
        }
    }

    /// Whether `kind` at `key` must be generated before it is drawn: Mixed,
    /// or Buried and reached by a ray or near the eye.
    fn needs_data(&self, grid: &Grid, key: NodeKey, kind: Kind, eye: &Eye) -> bool {
        match kind {
            Kind::Mixed => true,
            Kind::Buried => {
                self.nodes.get(&key).is_some_and(|n| n.reached)
                    || tile_distance(grid, key.footprint(), key.layers(), eye)
                        < self.config.near_cells * grid.level_size(key.level)
            }
            Kind::Empty | Kind::Solid => false,
        }
    }

    /// Visit the octree from the roots for an eye at `eye` moving at
    /// `velocity`.
    pub fn plan(
        &mut self,
        grid: &Grid,
        eye: DVec3,
        velocity: DVec3,
        view: Option<View>,
        frame: u64,
    ) -> Plan {
        let eyes = [
            Eye::new(grid, eye),
            Eye::new(grid, eye + velocity * self.config.prefetch_seconds),
        ];
        let guarded = view.map_or(-1.0, |v| {
            (v.cos_half.clamp(-1.0, 1.0).acos() + self.config.view_guard)
                .min(std::f64::consts::PI)
                .cos()
        });
        let tiles: Vec<i32> = (0..=self.root_level).map(|l| tiles_at(grid, l)).collect();
        let mut drawn_in_view = 0usize;
        let mut plan = Plan::default();
        let mut wanted: Vec<(f64, NodeKey)> = Vec::new();
        let mut coarse = 0usize;
        let mut stack: Vec<NodeKey> = self.roots.clone();
        let mut groups: FxHashMap<(Footprint, i32), Decision> = FxHashMap::default();
        while let Some(key) = stack.pop() {
            if let Some(fp) = self.footprints.get_mut(&key.footprint()) {
                fp.used = frame;
            }
            let kind = self.kind(key);
            let node = *self.nodes.entry(key).or_default();
            let this_plan = frame + 1;
            let n = self.nodes.get_mut(&key).unwrap();
            n.used = frame;
            n.leaf = None;
            if kind == Kind::Empty {
                n.stand_in = None;
                n.leaf = Some((Kind::Empty, this_plan));
                continue;
            }
            if !self.needs_data(grid, key, kind, &eyes[0]) {
                // Rock an edit added where there was air (a mound's inside)
                // stays air until its footprint's new surface is generated:
                // drawn at once, its boxes covered the view while the
                // surface came.
                if kind != Kind::Empty
                    && node.stand_in == Some(Kind::Empty)
                    && frame < node.edited + ADDED_ROCK_WAITS
                    && self.surface_pending(key.footprint())
                {
                    self.nodes.get_mut(&key).unwrap().leaf = Some((Kind::Empty, this_plan));
                    continue;
                }
                let n = self.nodes.get_mut(&key).unwrap();
                n.stand_in = None;
                n.leaf = Some((kind, this_plan));
                plan.drawn.push((key, kind));
                continue;
            }
            if !node.generated {
                // An edited node too coarse for the eye in view hands what
                // it stands in as to its children, which are visited in its
                // place: the edit is generated at the level the eye wants,
                // not first at the coarse level it was drawn at (a 60 m
                // stroke drew rock boxes 200 m across, then refined).
                if let Some(shown) = node.stand_in.filter(|_| key.level > 0) {
                    let fp = key.footprint();
                    let in_view = view.is_none_or(|v| {
                        let offset = eyes[0].toward(grid, fp, key.layers());
                        let length = offset.length();
                        let size = grid.level_size(fp.level) * f64::from(CHUNK) * 1.5;
                        length < size || offset.dot(v.forward) > guarded * length - size
                    });
                    let exact = self.footprints.get(&fp).is_some_and(|f| f.exact);
                    if in_view
                        && self.error(grid, key, &eyes) > 1.0
                        && (exact || self.guess_of(key) < GUESS_LEVELS)
                    {
                        self.estimate_children(key, frame);
                        self.nodes.get_mut(&key).unwrap().refined = true;
                        for child in key.children(tiles[key.level as usize - 1]) {
                            let edited = node.edited;
                            let c = self.nodes.entry(child).or_default();
                            if !c.generated && c.stand_in.is_none() {
                                c.stand_in = Some(shown);
                                c.edited = edited;
                            }
                            stack.push(child);
                        }
                        continue;
                    }
                }
                // Else it is drawn as it was until its chunk comes; a
                // Buried node a ray met, or holding the eye, is rock (it
                // never holds its parent back). Else only a root can be
                // ungenerated here: children are visited once every Mixed
                // one is generated or stands in.
                let shown = node
                    .stand_in
                    .or((kind == Kind::Buried).then_some(Kind::Buried));

                if let Some(shown) = shown {
                    self.nodes.get_mut(&key).unwrap().leaf = Some((shown, this_plan));
                    if shown != Kind::Empty {
                        plan.drawn.push((key, shown));
                    }
                }
                wanted.push((f64::INFINITY, key));
                continue;
            }
            // Generated: drawn from its chunk (a Buried one too).
            let kind = Kind::Mixed;
            // The node's band (its footprint's contiguous Mixed layers)
            // switches levels as one: a coarse cell's ground, rounded, may
            // lie one level finer under a node above or below it, so nodes
            // of a band refined apart would leave the surface between them
            // (holes through a fade's dither). Bands apart (caves, digs)
            // switch apart. Every test below is the band's.
            let band = self.band_of(key);
            let band_layers = (band.0 * CHUNK, (band.1 + 1) * CHUNK);
            // Bands that cannot be seen are drawn as they are, not refined;
            // those the prefetch eye sees are ready when they come over the
            // horizon.
            let hidden = self.beyond(grid, key.footprint(), band_layers, &eyes[0]);
            if hidden && self.beyond(grid, key.footprint(), band_layers, &eyes[1]) {
                let n = self.nodes.get_mut(&key).unwrap();
                n.refined = false;
                n.leaf = Some((if node.empty { Kind::Empty } else { kind }, this_plan));
                if !node.empty {
                    plan.drawn.push((key, kind));
                }
                continue;
            }
            let d = match groups.get(&(key.footprint(), band.0)) {
                Some(d) => *d,
                None => {
                    let d = self.decide(
                        grid,
                        key,
                        band,
                        &eyes,
                        velocity,
                        view,
                        guarded,
                        frame,
                        &tiles,
                        &mut wanted,
                    );
                    groups.insert((key.footprint(), band.0), d);
                    d
                }
            };
            let (error, seen, allowance, ready) = (d.error, d.seen, d.allowance, d.ready);
            let shows = seen && !hidden;
            let (exact, guess) = (d.exact, d.guess);
            let children: Vec<NodeKey> = if ready {
                key.children(tiles[key.level as usize - 1]).collect()
            } else {
                Vec::new()
            };
            // A switch shows where the last plan drew the side it leaves.
            let last = self.last_plan;
            let shown = |nodes: &FxHashMap<NodeKey, Node>, k: &NodeKey| {
                last != 0 && nodes.get(k).is_some_and(|n| n.drawn == last)
            };
            if shows
                && node.refined != ready
                && if ready {
                    shown(&self.nodes, &key)
                } else {
                    key.children(tiles[key.level as usize - 1])
                        .any(|c| shown(&self.nodes, &c))
                }
            {
                plan.switches += 1;
                plan.switch_px = plan.switch_px.max(d.cells.unwrap_or(0.0));
            }
            // Measured to the cells drawn (an eye in a chunk's air is not
            // at its cells); a band whose chunks' results are not back yet
            // counts once they are.
            if let Some(cells) = d.cells.filter(|_| shows && !ready && key.level > 0) {
                plan.worst_wanted = plan.worst_wanted.max(cells / allowance);
                if cells > plan.worst_px {
                    plan.worst_px = cells;
                    let below: Vec<NodeKey> = key.children(tiles[key.level as usize - 1]).collect();
                    let generated = below
                        .iter()
                        .filter(|c| self.nodes.get(c).is_some_and(|n| n.generated))
                        .count();
                    plan.worst = Some(Worst {
                        key,
                        exact,
                        guess,
                        children: below.len(),
                        generated,
                    });
                }
            }
            let n = self.nodes.get_mut(&key).unwrap();
            n.refined = ready;
            if ready {
                stack.extend(children);
            } else {
                n.leaf = Some((if node.empty { Kind::Empty } else { kind }, this_plan));
                if shows {
                    n.drawn = frame + 1;
                    drawn_in_view += 1;
                    if error > 1.5 && key.level > 0 {
                        coarse += 1;
                    }
                }
                // An empty chunk is air: nothing to draw.
                if !node.empty {
                    plan.drawn.push((key, kind));
                }
            }
        }
        plan.waiting = wanted.len();
        if wanted.len() > MAX_REQUESTS {
            wanted.select_nth_unstable_by(MAX_REQUESTS, |a, b| b.0.total_cmp(&a.0));
            wanted.truncate(MAX_REQUESTS);
        }
        wanted.sort_by(|a, b| b.0.total_cmp(&a.0));
        // A footprint's nodes together, where its most wanted one is.
        let mut order: FxHashMap<Footprint, usize> = FxHashMap::default();
        let mut groups: Vec<Vec<NodeKey>> = Vec::new();
        for (_, key) in wanted {
            let n = *order.entry(key.footprint()).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            if !groups[n].contains(&key) {
                groups[n].push(key);
            }
        }
        plan.requests = groups.into_iter().flatten().collect();
        plan.coarse_share = coarse as f64 / drawn_in_view.max(1) as f64;
        self.last_plan = frame + 1;
        plan
    }

    /// The edits changed: classifications follow, roots cover them.
    pub fn set_edits(&mut self, grid: &Grid, edits: Arc<EditView>) {
        self.edits = edits;
        self.place_roots(grid);
    }

    pub fn apply(&mut self, event: Event, frame: u64) {
        match event {
            Event::Generated(k) => {
                // (Its layers stay until its result comes back: a chunk
                // generated again holds about what it held.)
                let n = self.nodes.entry(k).or_default();
                n.generated = true;
                n.empty = false;
                n.stand_in = None;
            }
            Event::Emptied(k) => {
                let n = self.nodes.entry(k).or_default();
                n.generated = true;
                n.empty = true;
                n.occupied = None;
                n.stand_in = None;
            }
            Event::Occupied(k, layers) => {
                if let Some(n) = self.nodes.get_mut(&k).filter(|n| n.generated) {
                    n.occupied = Some(layers);
                }
            }
            Event::Stale(k) => {
                if let Some(n) = self.nodes.get_mut(&k) {
                    n.generated = false;
                    n.empty = false;
                    n.occupied = None;
                }
            }
            Event::Edited(face, boxes) => {
                // The boxes' nodes, looked up where they are fewer than the
                // nodes known, else found by a scan.
                let volume: u64 = boxes.iter().map(NodeBox::volume).sum();
                let keys: Vec<NodeKey> = if volume < self.nodes.len() as u64 {
                    boxes
                        .iter()
                        .flat_map(|b| b.keys(face))
                        .filter(|k| self.nodes.contains_key(k))
                        .collect()
                } else {
                    self.nodes
                        .keys()
                        .filter(|k| k.face == face && boxes.iter().any(|b| b.contains(k)))
                        .copied()
                        .collect()
                };
                let last = self.last_plan;
                for k in keys {
                    let n = self.nodes.get_mut(&k).unwrap();
                    n.edited = frame;
                    // Drawn as it was until it is generated again: air (one
                    // generated empty: its chunk dropped, known only here)
                    // or rock.
                    let was = n.leaf.filter(|&(_, plan)| plan == last).map(|(kind, _)| kind);
                    if n.generated && n.empty {
                        n.generated = false;
                        n.empty = false;
                        n.stand_in = Some(Kind::Empty);
                    } else if !n.generated {
                        if let Some(kind) = was.filter(|k| *k != Kind::Mixed) {
                            n.stand_in = Some(kind);
                        }
                    }
                }
            }
            Event::Measured(fp, quarters) => {
                if let (Some(f), Some(ranges)) =
                    (self.footprints.get_mut(&fp), Ranges::hull(&quarters))
                {
                    if f.ranges != ranges || !f.exact {
                        f.class = None;
                    }
                    f.ranges = ranges;
                    f.quarters = quarters;
                    f.exact = true;
                    f.guess = 0;
                }
            }
            Event::Seen(k) => {
                self.rays = true;
                // It and its ancestors (the drawn node over it among them).
                let mut at = Some(k);
                while let Some(key) = at {
                    if self.seen.insert(key, frame) == Some(frame) {
                        break;
                    }
                    at = (key.level < self.root_level).then(|| key.parent());
                }
            }
            Event::Reached(k) => {
                self.nodes.entry(k).or_default().reached = true;
            }
        }
    }

    /// Bookkeeping not visited for `keep_frames` dropped, and generated
    /// nodes evicted, least recently visited first: those unvisited for
    /// `release_frames` (`MAX_RELEASE` at most), and when chunk slots run
    /// short, `need` at least (never one visited in the last `IDLE_FRAMES`:
    /// it may still be fading out).
    pub fn evict(&mut self, frame: u64, need: usize) -> Vec<NodeKey> {
        let keep = self.config.keep_frames;
        self.nodes
            .retain(|_, n| (n.generated && !n.empty) || n.used + keep >= frame);
        let mut evicted = Vec::new();
        let idle = IDLE_FRAMES.max(self.config.release_frames);
        let released = if self.config.release_frames > 0 {
            self.nodes
                .values()
                .filter(|n| n.generated && !n.empty && n.used + idle < frame)
                .count()
                .min(MAX_RELEASE)
        } else {
            0
        };
        let need = need.max(released);
        if need > 0 {
            let mut old: Vec<(u64, NodeKey)> = self
                .nodes
                .iter()
                .filter(|(_, n)| n.generated && !n.empty && n.used + IDLE_FRAMES < frame)
                .map(|(k, n)| (n.used, *k))
                .collect();
            if old.len() > need {
                old.select_nth_unstable_by_key(need, |o| o.0);
                old.truncate(need);
            }
            for (_, key) in old {
                self.nodes.remove(&key);
                evicted.push(key);
            }
        }
        let held: FxHashSet<Footprint> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.generated)
            .map(|(k, _)| k.footprint())
            .collect();
        let root = self.root_level;
        self.footprints
            .retain(|fp, f| fp.level == root || f.used + keep >= frame || held.contains(fp));
        let seen_keep = self.config.seen_frames;
        if self.seen.len() > 4 * self.nodes.len().max(4096) {
            self.seen.retain(|_, f| *f + seen_keep >= frame);
        }
        evicted
    }
}

/// What happened to nodes since the worker's last plan.
#[derive(Clone, Debug)]
pub enum Event {
    /// Queued for generation: drawn from the next plan.
    Generated(NodeKey),
    /// Generated and found to hold nothing.
    Emptied(NodeKey),
    /// Its chunk came back (generated or restored) occupying these layers
    /// (level cells, [lo, hi)).
    Occupied(NodeKey, (i32, i32)),
    /// Its chunk dropped (out of the cut, or evicted): wanted again before
    /// it is drawn.
    Stale(NodeKey),
    /// A footprint's ranges by quarter (`Fp::quarters`).
    Measured(Footprint, [Option<Ranges>; 4]),
    /// A ray met it or a node under it.
    Seen(NodeKey),
    /// A ray met this Buried node: generate it.
    Reached(NodeKey),
    /// Edits changed what the nodes of `face` in these boxes hold: those
    /// generated empty (their chunks dropped, known only here) are
    /// generated again when wanted.
    Edited(u8, Arc<[NodeBox]>),
}

/// Nodes of one level: footprints `x` by `y` and layers `z` (inclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeBox {
    pub level: u32,
    pub x: [i32; 2],
    pub y: [i32; 2],
    pub z: [i32; 2],
}

impl NodeBox {
    fn volume(&self) -> u64 {
        let side = |r: [i32; 2]| (i64::from(r[1]) - i64::from(r[0]) + 1).max(0) as u64;
        side(self.x)
            .saturating_mul(side(self.y))
            .saturating_mul(side(self.z))
    }
    /// Its nodes on `face`.
    fn keys(&self, face: u8) -> impl Iterator<Item = NodeKey> + '_ {
        (self.x[0]..=self.x[1]).flat_map(move |cx| {
            (self.y[0]..=self.y[1]).flat_map(move |cy| {
                (self.z[0]..=self.z[1]).map(move |cz| NodeKey {
                    face,
                    level: self.level,
                    cx,
                    cy,
                    cz,
                })
            })
        })
    }
    pub fn contains(&self, k: &NodeKey) -> bool {

        k.level == self.level
            && (self.x[0]..=self.x[1]).contains(&k.cx)
            && (self.y[0]..=self.y[1]).contains(&k.cy)
            && (self.z[0]..=self.z[1]).contains(&k.cz)
    }
}

impl Event {
    /// Whether a plan must see it before it can be settled: all but ray
    /// reports of nodes met, which come every frame (a plan that wants
    /// nothing and switches nothing is settled whatever rays report).
    pub fn structural(&self) -> bool {
        !matches!(self, Event::Seen(_))
    }
}

/// A plan the worker made, and the nodes it evicted.
pub struct Planned {
    pub plan: Plan,
    /// Nodes drawn now and not by the last plan sent (or drawn as
    /// something else), and the reverse.
    pub added: Vec<(NodeKey, Kind)>,
    pub removed: Vec<NodeKey>,
    pub evicted: Vec<NodeKey>,
    /// What the plan was made for.
    pub made_for: PlannedFor,
    /// The frame whose eye it was made for, and the worker's milliseconds
    /// on it (plan and eviction).
    pub frame: u64,
    pub ms: f64,
}

/// What a plan was made for: structural events up to this number had
/// reached the planner, and this many frames of ray reports; the eye and
/// the view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedFor {
    pub events_seen: u64,
    pub reports_seen: u64,
    pub eye: DVec3,
    pub view: Option<View>,
    /// The version of the edits it classified with (`EditView::version`).
    pub edits: u64,
}

struct Job {
    events_seen: u64,
    reports_seen: u64,
    eye: DVec3,
    velocity: DVec3,
    view: Option<View>,
    frame: u64,
    events: Vec<Event>,
    /// Generated nodes to evict for chunk slots.
    need: usize,
    edits: Option<Arc<EditView>>,
}

/// The planner on its own thread: a job at a time, the latest plan taken
/// without waiting.
pub struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    results: std::sync::Mutex<std::sync::mpsc::Receiver<Planned>>,
    busy: bool,
    /// Structural events and frames of ray reports submitted so far.
    events_sent: u64,
    reports_sent: u64,
}

impl Worker {
    pub fn spawn(grid: Grid, range: (i32, i32), config: PlanConfig, edits: Arc<EditView>) -> Self {
        let (jobs, inbox) = std::sync::mpsc::channel::<Job>();
        let (outbox, results) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("chunk residency".into())
            .spawn(move || {
                let mut planner = Planner::new(&grid, range, config, edits);
                let mut drawn: FxHashMap<NodeKey, Kind> = FxHashMap::default();
                while let Ok(job) = inbox.recv() {
                    if let Some(edits) = job.edits {
                        planner.set_edits(&grid, edits);
                    }
                    for event in job.events {
                        planner.apply(event, job.frame);
                    }
                    let started = std::time::Instant::now();
                    let plan = planner.plan(&grid, job.eye, job.velocity, job.view, job.frame);
                    let evicted = planner.evict(job.frame, job.need);
                    let now: FxHashMap<NodeKey, Kind> = plan.drawn.iter().copied().collect();
                    let added = plan
                        .drawn
                        .iter()
                        .filter(|(k, kind)| drawn.get(k) != Some(kind))
                        .copied()
                        .collect();
                    let removed = drawn
                        .iter()
                        .filter(|(k, kind)| now.get(k) != Some(kind))
                        .map(|(k, _)| *k)
                        .collect();
                    drawn = now;
                    if outbox
                        .send(Planned {
                            plan,
                            added,
                            removed,
                            evicted,
                            made_for: PlannedFor {
                                events_seen: job.events_seen,
                                reports_seen: job.reports_seen,
                                eye: job.eye,
                                view: job.view,
                                edits: planner.edits.version,
                            },
                            frame: job.frame,
                            ms: started.elapsed().as_secs_f64() * 1e3,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("chunk residency thread");
        Self {
            jobs,
            results: std::sync::Mutex::new(results),
            busy: false,
            events_sent: 0,
            reports_sent: 0,
        }
    }

    /// Plan for this eye with these events (`reports` frames of ray
    /// reports among them; and edits, when they changed) unless a plan is
    /// under way (then they stay with the caller).
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        eye: DVec3,
        velocity: DVec3,
        view: Option<View>,
        frame: u64,
        events: &mut Vec<Event>,
        reports: &mut u64,
        need: usize,
        edits: &mut Option<Arc<EditView>>,
    ) {
        if !self.busy {
            self.events_sent += events.iter().filter(|e| e.structural()).count() as u64;
            self.reports_sent += std::mem::take(reports);
            self.busy = self
                .jobs
                .send(Job {
                    events_seen: self.events_sent,
                    reports_seen: self.reports_sent,
                    eye,
                    velocity,
                    view,
                    frame,
                    events: std::mem::take(events),
                    need,
                    edits: edits.take(),
                })
                .is_ok();
        }
    }

    /// The finished plan, if any.
    pub fn poll(&mut self) -> Option<Planned> {
        let planned = self.results.get_mut().ok()?.try_recv().ok()?;
        self.busy = false;
        Some(planned)
    }

    /// Structural events submitted so far ([`Event::structural`]).
    pub fn events_sent(&self) -> u64 {
        self.events_sent
    }

    /// Frames of ray reports submitted so far.
    pub fn reports_sent(&self) -> u64 {
        self.reports_sent
    }
}

/// Frames a node must go unvisited before eviction: longer than a fade
/// between levels (`drawn::FADE_FRAMES`) and a plan's lag.
const IDLE_FRAMES: u64 = 32;

/// Nodes released for age a plan at most (`PlanConfig::release_frames`):
/// a flight's trail goes over some frames, not in one.
const MAX_RELEASE: usize = 4096;

/// Requests a plan lists at most (more than a frame generates).
pub const MAX_REQUESTS: usize = 4096;

/// Cells this many times `lod_pixels` across are far too coarse: their
/// children and the nodes under the eye are wanted before measurements.
const DEEP_REFINE: f64 = 4.0;

/// Levels under a measured footprint whose ranges may be estimated: a
/// margin of two of its cells is eight cells three levels down.
const GUESS_LEVELS: u8 = 3;

/// Footprints along a face edge at `level`.
pub fn tiles_at(grid: &Grid, level: u32) -> i32 {
    ((grid.cells() >> level) + CHUNK - 1) / CHUNK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PlanConfig {
        PlanConfig {
            pixel_angle: 7.67e-4,
            lod_pixels: 1.0,
            offscreen_pixels: 2.0,
            view_guard: 0.35,
            unseen_pixels: 1.0,
            seen_frames: 90,
            coarsen: 0.8,
            max_distance: 400_000.0,
            prefetch_seconds: 0.5,
            keep_frames: 60,
            release_frames: 0,
            motion_flow: 0.0,
            motion_max: 1.0,
            near_cells: 2.0,
        }
    }

    fn no_edits(grid: &Grid) -> Arc<EditView> {
        Arc::new(EditView {
            grid: *grid,
            faces: Default::default(),
            index: Default::default(),
            recent_index: Default::default(),
            reach: None,
            large_reach: None,
            large_faces: 0,
            large_brushes: 0,
            large_prefix: 0,
            recent_counts: Vec::new(),
            baked: Default::default(),
            bricks: Default::default(),
            version: 0,
            volume_depth: 0,
        })
    }

    /// An Earth whose ground lies between -200 and 200 m.
    fn setup() -> (Grid, DVec3, Planner) {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let eye = grid.position(
            2,
            [
                f64::from(grid.cells()) * 0.47,
                f64::from(grid.cells()) * 0.53,
                20.0,
            ],
        );
        let planner = Planner::new(&grid, (-2000, 2000), config(), no_edits(&grid));
        (grid, eye, planner)
    }

    /// Generate every request as if the budget were unlimited, measuring
    /// each footprint at the ground band (-20..20 m).
    fn settle(
        grid: &Grid,
        eye: DVec3,
        view: Option<View>,
        planner: &mut Planner,
        frame: &mut u64,
    ) -> Plan {
        loop {
            *frame += 1;
            let plan = planner.plan(grid, eye, DVec3::ZERO, view, *frame);
            if plan.requests.is_empty() {
                return plan;
            }
            for k in plan.requests {
                planner.apply(Event::Generated(k), *frame);
                let band = ((-200 >> k.level) - 1, (200 >> k.level) + 1);
                planner.apply(
                    Event::Measured(
                        k.footprint(),
                        [Some(Ranges {
                            volume: band,
                            surface: band,
                        }); 4],
                    ),
                    *frame,
                );
            }
        }
    }

    /// The drawn nodes over a base column: one shows at every height.
    fn assert_covers_once(grid: &Grid, plan: &Plan) {
        let drawn: FxHashSet<NodeKey> = plan.drawn.iter().map(|d| d.0).collect();
        for &(k, _) in &plan.drawn {
            let mut p = k;
            while p.level + 1 < grid.levels() {
                p = p.parent();
                assert!(!drawn.contains(&p), "{k:?} under drawn {p:?}");
            }
        }
    }

    #[test]
    fn a_settled_cut_reaches_level_zero_once_over_each_point() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        let plan = settle(&grid, eye, None, &mut planner, &mut frame);
        assert!(plan
            .drawn
            .iter()
            .any(|(k, kind)| k.level == 0 && *kind == Kind::Mixed));
        assert_eq!(plan.coarse_share, 0.0);
        assert_covers_once(&grid, &plan);
        // Uniform nodes are never refined: the planet's core is drawn coarse.
        assert!(plan
            .drawn
            .iter()
            .any(|(k, kind)| *kind == Kind::Solid && k.level > 10));
    }

    /// A footprint's band switches levels as one: every node of a drawn
    /// Mixed node's band is drawn at its level too (or holds nothing), so
    /// ground a coarse cell rounded into the node above or below is never
    /// left between levels.
    #[test]
    fn a_band_switches_levels_as_one() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        for eye in [
            eye,
            eye + eye.normalize() * 3_000.0,
            eye + eye.normalize() * 60_000.0,
        ] {
            let plan = settle(&grid, eye, None, &mut planner, &mut frame);
            let drawn: FxHashSet<NodeKey> = plan.drawn.iter().map(|d| d.0).collect();
            for &(k, kind) in &plan.drawn {
                if kind != Kind::Mixed {
                    continue;
                }
                let band = planner.band_of(k);
                for cz in band.0..=band.1 {
                    let m = k.footprint().node(cz);
                    assert!(
                        drawn.contains(&m) || planner.nodes.get(&m).is_some_and(|n| n.empty),
                        "{k:?} drawn, {m:?} of its band {band:?} not"
                    );
                }
            }
        }
    }

    /// Height is a dimension of the cut: an eye high over the ground
    /// refines the ground under it less than an eye on it.
    #[test]
    fn nodes_refine_by_their_own_distance() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        let low = settle(&grid, eye, None, &mut planner, &mut frame);
        let finest = |p: &Plan| {
            p.drawn
                .iter()
                .filter(|(_, kind)| *kind == Kind::Mixed)
                .map(|(k, _)| k.level)
                .min()
                .unwrap()
        };
        let (_, _, mut high_planner) = setup();
        let high = settle(
            &grid,
            eye + eye.normalize() * 2_000.0,
            None,
            &mut high_planner,
            &mut frame,
        );
        assert!(
            finest(&low) == 0 && finest(&high) >= 3,
            "{} {}",
            finest(&low),
            finest(&high)
        );
    }

    #[test]
    fn a_parent_stays_drawn_until_all_its_mixed_children_are_generated() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, None, &mut planner, &mut frame);
        // A drawn level-1 Mixed node's children dropped (stale): it is
        // drawn again until they come back.
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        let child = plan
            .drawn
            .iter()
            .find(|(k, kind)| k.level == 0 && *kind == Kind::Mixed)
            .unwrap()
            .0;
        let parent = child.parent();
        planner.apply(Event::Stale(child), frame);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(plan.drawn.iter().any(|d| d.0 == parent));
        assert!(plan.requests.contains(&child));
        planner.apply(Event::Generated(child), frame);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(!plan.drawn.iter().any(|d| d.0 == parent));
    }

    /// An edit over a node drawn as air (generated empty) wants it again
    /// without its parent drawn in its place: it stays air until its chunk
    /// comes, the cut as it was.
    #[test]
    fn an_edited_node_stands_in_as_it_was() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, None, &mut planner, &mut frame);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        let child = plan
            .drawn
            .iter()
            .find(|(k, kind)| k.level == 0 && *kind == Kind::Mixed)
            .unwrap()
            .0;
        let parent = child.parent();
        planner.apply(Event::Emptied(child), frame);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(!plan.drawn.iter().any(|d| d.0 == child || d.0 == parent));
        let around = |v: i32| [v, v];
        planner.apply(
            Event::Edited(
                child.face,
                vec![NodeBox {
                    level: 0,
                    x: around(child.cx),
                    y: around(child.cy),
                    z: around(child.cz),
                }]
                .into(),
            ),
            frame,
        );
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(plan.requests.contains(&child), "the edited node is wanted");
        assert!(
            !plan.drawn.iter().any(|d| d.0 == child || d.0 == parent),
            "neither it nor its parent drawn meanwhile"
        );
        planner.apply(Event::Generated(child), frame);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(plan.drawn.iter().any(|d| d.0 == child));
        assert!(!plan.drawn.iter().any(|d| d.0 == parent));
    }

    /// Generated nodes stay until slots run short; then the least
    /// recently visited go, never one the plan drew.
    #[test]
    fn eviction_is_by_pressure_least_recently_visited_first() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, None, &mut planner, &mut frame);
        let far = eye + eye.normalize() * 200_000.0;
        let here = settle(&grid, far, None, &mut planner, &mut frame);
        frame += 500;
        planner.plan(&grid, far, DVec3::ZERO, None, frame);
        frame += IDLE_FRAMES + 1;
        planner.plan(&grid, far, DVec3::ZERO, None, frame);
        assert!(
            planner.evict(frame, 0).is_empty(),
            "nothing evicted without pressure"
        );
        let fine = planner
            .nodes
            .iter()
            .filter(|(k, n)| k.level == 0 && n.generated)
            .count();
        assert!(fine > 0, "the first eye's fine nodes are still resident");
        let evicted = planner.evict(frame, fine);
        assert_eq!(evicted.len(), fine);
        let drawn: FxHashSet<NodeKey> = here.drawn.iter().map(|d| d.0).collect();
        assert!(evicted.iter().all(|k| !drawn.contains(k)));
    }

    /// With `release_frames`, nodes the cut left go once that old without
    /// pressure, a plan's `MAX_RELEASE` at a time, never one the plan drew;
    /// what is left is about what a fresh planner settles with there.
    #[test]
    fn nodes_the_cut_left_are_released_by_age() {
        let (grid, eye, _) = setup();
        let release = 200;
        let mut planner = Planner::new(
            &grid,
            (-2000, 2000),
            PlanConfig {
                release_frames: release,
                ..config()
            },
            no_edits(&grid),
        );
        let resident = |p: &Planner| p.nodes.values().filter(|n| n.generated && !n.empty).count();
        let mut frame = 0;
        settle(&grid, eye, None, &mut planner, &mut frame);
        let far = eye + eye.normalize() * 200_000.0;
        let here = settle(&grid, far, None, &mut planner, &mut frame);
        let before = resident(&planner);
        frame += 1;
        planner.plan(&grid, far, DVec3::ZERO, None, frame);
        assert!(
            planner.evict(frame, 0).is_empty(),
            "nothing released before {release} frames"
        );
        frame += release;
        planner.plan(&grid, far, DVec3::ZERO, None, frame);
        let mut released = Vec::new();
        loop {
            let batch = planner.evict(frame, 0);
            assert!(batch.len() <= MAX_RELEASE);
            if batch.is_empty() {
                break;
            }
            released.extend(batch);
        }
        let drawn: FxHashSet<NodeKey> = here.drawn.iter().map(|d| d.0).collect();
        assert!(released.iter().all(|k| !drawn.contains(k)));
        assert!(
            released.iter().any(|k| k.level == 0),
            "the first eye's fine nodes released"
        );
        let (_, _, mut fresh) = setup();
        let mut fresh_frame = 0;
        settle(&grid, far, None, &mut fresh, &mut fresh_frame);
        // What is left the plan visits (a band refined in passing may stay
        // refined), within a fifth of a fresh cut's.
        assert!(planner
            .nodes
            .values()
            .all(|n| !n.generated || n.empty || n.used + release >= frame));
        let (left, wanted) = (resident(&planner), resident(&fresh));
        assert!(
            left * 5 <= wanted * 6,
            "{before} resident, {left} after release, {wanted} fresh"
        );
    }

    /// In view but met by no ray: a level coarser; met: as fine as wanted.
    #[test]
    fn rays_decide_where_detail_goes() {
        let (grid, eye, _) = setup();
        let up = eye.normalize();
        let forward = (up.any_orthonormal_vector() - up * 0.3).normalize();
        let view = Some(View {
            forward,
            cos_half: 0.6f64.cos(),
        });
        let cut = |seen: bool| {
            let mut planner = Planner::new(
                &grid,
                (-2000, 2000),
                PlanConfig {
                    unseen_pixels: 2.0,
                    ..config()
                },
                no_edits(&grid),
            );
            let mut frame = 0;
            // One ray report somewhere else: rays are on, nothing here met.
            planner.apply(
                Event::Seen(NodeKey {
                    face: 0,
                    level: 0,
                    cx: 0,
                    cy: 0,
                    cz: 0,
                }),
                0,
            );
            loop {
                frame += 1;
                if seen {
                    // Every drawn node met (a clear view).
                    let drawn: Vec<NodeKey> = planner
                        .plan(&grid, eye, DVec3::ZERO, view, frame)
                        .drawn
                        .iter()
                        .map(|d| d.0)
                        .collect();
                    for k in drawn {
                        planner.apply(Event::Seen(k), frame);
                    }
                }
                let plan = planner.plan(&grid, eye, DVec3::ZERO, view, frame);
                if plan.requests.is_empty() {
                    return plan;
                }
                for k in plan.requests {
                    planner.apply(Event::Generated(k), frame);
                    let band = ((-200 >> k.level) - 1, (200 >> k.level) + 1);
                    planner.apply(
                        Event::Measured(
                            k.footprint(),
                            [Some(Ranges {
                                volume: band,
                                surface: band,
                            }); 4],
                        ),
                        frame,
                    );
                }
            }
        };
        let (met, unmet) = (cut(true), cut(false));
        let mixed = |p: &Plan| p.drawn.iter().filter(|d| d.1 == Kind::Mixed).count();
        // In view only (out of view is a level coarser either way).
        assert!(
            mixed(&unmet) * 4 < mixed(&met) * 3,
            "{} {}",
            mixed(&unmet),
            mixed(&met)
        );
        assert_eq!(met.coarse_share, 0.0);
    }

    /// An eye 20 m over flat ground is inside the level-2 chunk over it
    /// (25.6 m tall), not at its cells: with that node drawn for want of
    /// its children, the plan's worst cell is measured to the cells its
    /// chunks occupy (about 43 pixels), not to a chunk at no distance; a
    /// band whose chunks' results are not back does not count yet.
    #[test]
    fn the_worst_cell_is_measured_to_the_cells_drawn() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let at = [
            f64::from(grid.cells()) * 0.47,
            f64::from(grid.cells()) * 0.53,
        ];
        let eye = grid.position(2, [at[0], at[1], 200.0]);
        let mut planner = Planner::new(&grid, (-2000, 2000), config(), no_edits(&grid));
        let mut frame = 0;
        // Ground within a metre of layer 0 everywhere.
        loop {
            frame += 1;
            let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
            if plan.requests.is_empty() {
                break;
            }
            for k in plan.requests {
                planner.apply(Event::Generated(k), frame);
                let band = ((-10 >> k.level) - 1, (10 >> k.level) + 1);
                planner.apply(
                    Event::Measured(
                        k.footprint(),
                        [Some(Ranges {
                            volume: band,
                            surface: band,
                        }); 4],
                    ),
                    frame,
                );
            }
        }
        // The level-2 node holding the eye, its children dropped.
        let tiles: Vec<i32> = (0..grid.levels()).map(|l| tiles_at(&grid, l)).collect();
        let held = NodeKey {
            face: 2,
            level: 2,
            cx: (at[0] as i32 >> 2) / CHUNK,
            cy: (at[1] as i32 >> 2) / CHUNK,
            cz: 0,
        };
        assert!(held.layers().1 << 2 > 200, "the eye is inside {held:?}");
        for child in held.children(tiles[1]) {
            planner.apply(Event::Stale(child), frame);
        }
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        assert!(plan.drawn.iter().any(|d| d.0 == held), "{held:?} drawn");
        assert!(
            plan.requests.iter().any(|k| k.level == 1),
            "its children wanted"
        );
        assert_eq!(plan.worst_px, 0.0, "no cells known");
        // Its band's chunks back, the ground's top 8 m over the plain.
        planner.apply(Event::Occupied(held, (0, 20)), frame);
        planner.apply(Event::Occupied(NodeKey { cz: -1, ..held }, (-3, 0)), frame);
        frame += 1;
        let plan = planner.plan(&grid, eye, DVec3::ZERO, None, frame);
        let cells_px = grid.level_size(2) / 12.0 / config().pixel_angle;
        assert!(
            plan.worst_px > cells_px * 0.9 && plan.worst_px < cells_px * 1.1,
            "worst {} px, its cells' {cells_px:.1}",
            plan.worst_px
        );
    }

    /// A measured footprint's children are estimated from the quarter of
    /// its columns over each: a crater wall in one quarter raises that
    /// child's band only.
    #[test]
    fn children_are_estimated_from_their_quarter() {
        let (grid, _, mut planner) = setup();
        let key = NodeKey {
            face: 2,
            level: 6,
            cx: 10,
            cy: 12,
            cz: 0,
        };
        planner.footprints.insert(
            key.footprint(),
            Fp {
                ranges: Ranges {
                    volume: (0, 0),
                    surface: (0, 0),
                },
                exact: false,
                quarters: [None; 4],
                guess: 0,
                used: 0,
                class: None,
            },
        );
        let plain = Ranges {
            volume: (-3, 2),
            surface: (-2, 2),
        };
        let wall = Ranges {
            volume: (-3, 40),
            surface: (-2, 40),
        };
        planner.apply(
            Event::Measured(
                key.footprint(),
                [Some(plain), Some(plain), Some(wall), None],
            ),
            1,
        );
        assert_eq!(
            planner.footprints[&key.footprint()].ranges,
            Ranges {
                volume: (-3, 40),
                surface: (-2, 40)
            }
        );
        planner.estimate_children(key, 1);
        let child = |a: i32, b: i32| {
            planner.footprints[&Footprint {
                face: 2,
                level: 5,
                cx: 20 + a,
                cy: 24 + b,
            }]
                .ranges
        };
        assert_eq!(child(0, 0), plain.child());
        assert_eq!(child(1, 0), plain.child());
        assert_eq!(child(0, 1), wall.child());
        // A quarter that measured nothing (past a face's edge): the whole.
        assert_eq!(child(1, 1), wall.child());
        let _ = grid;
    }

    #[test]
    fn prefetch_refines_ahead_of_a_moving_eye() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, None, &mut planner, &mut frame);
        let ahead = (grid.position(
            2,
            [
                f64::from(grid.cells()) * 0.47 + 2000.0,
                f64::from(grid.cells()) * 0.53,
                20.0,
            ],
        ) - eye)
            .normalize();
        frame += 1;
        let plan = planner.plan(&grid, eye, ahead * 400.0, None, frame);
        assert!(
            plan.requests.iter().any(|k| k.level == 0),
            "nothing requested ahead"
        );
    }

    /// A climb faster than the plan: from the ground to orbit in one plan,
    /// the cut still covers everything once.
    #[test]
    fn a_sudden_climb_leaves_no_overlap() {
        let (grid, eye, mut planner) = setup();
        let mut frame = 0;
        settle(&grid, eye, None, &mut planner, &mut frame);
        frame += 1;
        let plan = planner.plan(
            &grid,
            eye.normalize() * (grid.radius() + 400_000.0),
            DVec3::ZERO,
            None,
            frame,
        );
        assert_covers_once(&grid, &plan);
        assert!(!plan.drawn.is_empty());
    }
}
