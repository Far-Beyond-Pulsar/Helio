//! Procedural backrooms map with VHS camcorder effect.
//!
//! Generates a random but reliably navigable map each run using a
//! room-scatter + graph-connection approach: varied room sizes, varied
//! ceiling heights (low/normal/tall/atrium), looping corridors of varied
//! width, dead-end alcoves, and pillar halls in larger rooms — all
//! immersed in a degraded VHS look (CA, grain, vignette, desaturated).
//!
//! A subset of the fluorescent lights are "flickering" lights: they run
//! their own little state machine (steady / rapid-burst / dark-or-dim)
//! independently of everything else, so the space feels alive rather than
//! uniformly lit.
//!
//! Rendered through the OpenXR path when a headset is connected (see `examples/vr` and
//! `examples/foliage_demo`), and a plain desktop mirror with WASD + mouse look when
//! OpenXR initialisation fails. The camcorder bob/sway is desktop-only — a headset
//! already supplies real head motion, and stacking a synthetic bob on top of it is a
//! reliable way to make someone sick.
//!
//! Controls:
//!   WASD        — move (desktop mirror mode)
//!   Space/Shift — up/down (desktop mirror mode)
//!   R           — regenerate map
//!   Mouse drag  — look around
//!   Left stick  — move (XR)
//!   Right stick — turn (XR)

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, GroupMask, HelioAction, HelioCommandBridge, LightId, MaterialId, MeshId,
    Movability, ObjectDescriptor, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph_with_user_effects;
use helio_pass_postprocess::PostProcessPass;
use libhelio::{PostProcessSettings, PostProcessVolumeDescriptor};
use v3_demo_common::{box_mesh, make_material, point_light};

// User shader snippet injected into the post-process pipeline.
// Uses noise_tex, noise_samp, and pp_custom from the core bindings.
//
// v2 changes vs original, aimed at "found VHS footage" realism rather than
// a generic retro filter:
//   1. Lens identity: barrel distortion + vignette + edge chromatic aberration
//      (cheap camcorder wide-angle glass, not a flat digital frame)
//   2. CCD highlight bloom/smear: bright sources streak vertically, a classic
//      tell of consumer CCD sensors that's otherwise completely absent
//   3. Grain now samples noise_tex with a slow scroll instead of pure hash
//      noise -> spatially/temporally correlated grain instead of TV static
//   4. Head-switching noise bar near the bottom of frame (real decks show
//      this on every single frame - previously missing entirely)
//   5. Rare, severe tracking tears layered on top of the existing smooth
//      jitter, so damage reads as bursty/eventful instead of constant
// Everything from the original (YIQ separation, chroma phase drift, rolling
// scanline error, per-line jitter, crushed blacks/highlights, dot crawl,
// flicker) is kept - it was solid and just needed a physical frame around it.
const VHS_SHADER_SNIPPET: &str = include_str!("vhs_effects.wgsl");
use std::io::{self, BufRead};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;

// ── Map Generator ─────────────────────────────────────────────────────────────
//
// Room-scatter + graph-connection generator:
//   1. Scatter non-overlapping rectangular rooms of varied size across the grid.
//   2. Assign each room a ceiling height (low / normal / tall / atrium).
//   3. Connect all rooms with corridors (chain + extra random links -> loops),
//      corridors have randomized width and L-bend direction.
//   4. Branch a handful of dead-end alcoves off corridors for texture.
//   5. Flood-fill from a room to strip any unreachable pockets.
//   6. Scatter pillars through big/atrium rooms (purely decorative).
//   7. Place lights sparsely: a few per room, one every few corridor cells.
//      A fraction of those lights are flagged to flicker at runtime.

const CELL: f32 = 4.0; // metres per grid cell
const H_CELL: f32 = CELL / 2.0;
const GRID_W: usize = 56;
const GRID_H: usize = 42;

const WALL_H: f32 = 3.2; // baseline / corridor ceiling height
const LOW_H: f32 = 2.4; // cramped storage-nook feel
const TALL_H: f32 = 4.6; // office hall
const ATRIUM_H: f32 = 6.8; // big open pillar hall

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cell {
    Wall,
    Corridor,
    Room,
}

/// Small deterministic-per-seed PRNG (PCG-style LCG), reseeded from the
/// system clock each generation so every regenerate (R key) is different.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    /// Inclusive-lo, exclusive-hi integer range.
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo).max(1) as u32;
        lo + (self.next_u32() % span) as i32
    }

    fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}

#[derive(Clone, Copy)]
struct RoomRect {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl RoomRect {
    fn center(&self) -> (i32, i32) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }

    fn overlaps(&self, other: &RoomRect, pad: i32) -> bool {
        self.x - pad < other.x + other.w
            && self.x + self.w + pad > other.x
            && self.y - pad < other.y + other.h
            && self.y + self.h + pad > other.y
    }
}

struct BackroomsMap {
    grid: Vec<Vec<Cell>>,
    ceiling_h: Vec<Vec<f32>>,
    lights: Vec<(f32, f32, f32)>, // world x, world z, ceiling height there
    pillars: Vec<(f32, f32, f32)>, // world x, world z, pillar height
}

fn carve_pt(grid: &mut [Vec<Cell>], ceiling_h: &mut [Vec<f32>], x: i32, y: i32, width_extra: i32) {
    for ox in 0..=width_extra {
        for oy in 0..=width_extra {
            let gx = x + ox;
            let gy = y + oy;
            if gx > 0 && gx < GRID_W as i32 - 1 && gy > 0 && gy < GRID_H as i32 - 1 {
                let (ux, uy) = (gx as usize, gy as usize);
                if grid[ux][uy] == Cell::Wall {
                    grid[ux][uy] = Cell::Corridor;
                    ceiling_h[ux][uy] = WALL_H;
                }
            }
        }
    }
}

fn carve_corridor(
    grid: &mut Vec<Vec<Cell>>,
    ceiling_h: &mut Vec<Vec<f32>>,
    rng: &mut Rng,
    from: (i32, i32),
    to: (i32, i32),
) {
    // 30% chance of a 2-wide corridor instead of the usual 1-wide.
    let width_extra = if rng.chance(0.3) { 1 } else { 0 };
    let horizontal_first = rng.chance(0.5);
    let (x1, y1) = from;
    let (x2, y2) = to;

    if horizontal_first {
        let (lo, hi) = (x1.min(x2), x1.max(x2));
        for x in lo..=hi {
            carve_pt(grid, ceiling_h, x, y1, width_extra);
        }
        let (lo, hi) = (y1.min(y2), y1.max(y2));
        for y in lo..=hi {
            carve_pt(grid, ceiling_h, x2, y, width_extra);
        }
    } else {
        let (lo, hi) = (y1.min(y2), y1.max(y2));
        for y in lo..=hi {
            carve_pt(grid, ceiling_h, x1, y, width_extra);
        }
        let (lo, hi) = (x1.min(x2), x1.max(x2));
        for x in lo..=hi {
            carve_pt(grid, ceiling_h, x, y2, width_extra);
        }
    }
}

fn generate_map() -> BackroomsMap {
    let mut grid = vec![vec![Cell::Wall; GRID_H]; GRID_W];
    let mut ceiling_h = vec![vec![WALL_H; GRID_H]; GRID_W];

    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let mut rng = Rng::new(0x5EED_F00D_1234_5678 ^ time_seed);

    // ---- 1. Scatter non-overlapping rooms of varied size ----
    let mut rooms: Vec<RoomRect> = Vec::new();
    let target_rooms = 26;
    let mut attempts = 0;
    while rooms.len() < target_rooms && attempts < 800 {
        attempts += 1;
        let w = rng.range_i32(3, 10);
        let h = rng.range_i32(3, 8);
        let x = rng.range_i32(2, GRID_W as i32 - w - 2);
        let y = rng.range_i32(2, GRID_H as i32 - h - 2);
        let candidate = RoomRect { x, y, w, h };
        if rooms.iter().any(|r| candidate.overlaps(r, 2)) {
            continue;
        }
        rooms.push(candidate);
    }

    // ---- 2. Stamp rooms, rolling a ceiling-height variant per room ----
    let mut room_heights: Vec<(f32, bool)> = Vec::with_capacity(rooms.len());
    for room in &rooms {
        let roll = rng.next_f32();
        let (h, is_atrium) = if roll < 0.10 && room.w >= 6 && room.h >= 5 {
            (ATRIUM_H, true)
        } else if roll < 0.28 {
            (TALL_H, false)
        } else if roll < 0.50 {
            (LOW_H, false)
        } else {
            (WALL_H, false)
        };
        room_heights.push((h, is_atrium));
        for dx in 0..room.w {
            for dy in 0..room.h {
                let (ux, uy) = ((room.x + dx) as usize, (room.y + dy) as usize);
                grid[ux][uy] = Cell::Room;
                ceiling_h[ux][uy] = h;
            }
        }
    }

    // ---- 3. Connect every room: a chain first (guarantees reachability),
    //         then extra random links layered on top so the map has loops
    //         instead of reading as a strict tree/maze. ----
    let mut order: Vec<usize> = (0..rooms.len()).collect();
    for i in (1..order.len()).rev() {
        let j = rng.range_i32(0, i as i32 + 1) as usize;
        order.swap(i, j);
    }
    for pair in order.windows(2) {
        let a = rooms[pair[0]].center();
        let b = rooms[pair[1]].center();
        carve_corridor(&mut grid, &mut ceiling_h, &mut rng, a, b);
    }
    let extra_links = ((rooms.len() as f32) * 0.45) as usize;
    for _ in 0..extra_links {
        if rooms.len() < 2 {
            break;
        }
        let i = rng.range_i32(0, rooms.len() as i32) as usize;
        let j = rng.range_i32(0, rooms.len() as i32) as usize;
        if i == j {
            continue;
        }
        carve_corridor(
            &mut grid,
            &mut ceiling_h,
            &mut rng,
            rooms[i].center(),
            rooms[j].center(),
        );
    }

    // ---- 4. Dead-end alcoves branching off corridors, for texture ----
    let mut corridor_cells: Vec<(i32, i32)> = Vec::new();
    for x in 0..GRID_W {
        for y in 0..GRID_H {
            if grid[x][y] == Cell::Corridor {
                corridor_cells.push((x as i32, y as i32));
            }
        }
    }
    let alcove_tries = corridor_cells.len() / 10;
    for _ in 0..alcove_tries {
        if corridor_cells.is_empty() {
            break;
        }
        let idx = rng.range_i32(0, corridor_cells.len() as i32) as usize;
        let (cx, cy) = corridor_cells[idx];
        let dirs = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)];
        let d = dirs[(rng.next_u32() as usize) & 3];
        let aw = rng.range_i32(2, 4);
        let ah = rng.range_i32(2, 4);
        let ax = cx + d.0 * 2;
        let ay = cy + d.1 * 2;
        let cand = RoomRect {
            x: ax,
            y: ay,
            w: aw,
            h: ah,
        };
        if cand.x < 1
            || cand.y < 1
            || cand.x + cand.w >= GRID_W as i32 - 1
            || cand.y + cand.h >= GRID_H as i32 - 1
        {
            continue;
        }
        let mut clear = true;
        for dx in 0..cand.w {
            for dy in 0..cand.h {
                if grid[(cand.x + dx) as usize][(cand.y + dy) as usize] != Cell::Wall {
                    clear = false;
                }
            }
        }
        if !clear {
            continue;
        }
        let h = if rng.chance(0.3) { LOW_H } else { WALL_H };
        for dx in 0..cand.w {
            for dy in 0..cand.h {
                let (ux, uy) = ((cand.x + dx) as usize, (cand.y + dy) as usize);
                grid[ux][uy] = Cell::Room;
                ceiling_h[ux][uy] = h;
            }
        }
        carve_corridor(&mut grid, &mut ceiling_h, &mut rng, (cx, cy), cand.center());
    }

    // ---- 5. Flood-fill connectivity: strip anything unreachable ----
    let mut visited = vec![vec![false; GRID_H]; GRID_W];
    let (start_x, start_y) = if let Some(r) = rooms.first() {
        let c = r.center();
        (c.0 as usize, c.1 as usize)
    } else {
        (GRID_W / 2, GRID_H / 2)
    };
    let mut stack = vec![(start_x, start_y)];
    visited[start_x][start_y] = true;
    while let Some((x, y)) = stack.pop() {
        for &(dx, dy) in &[(0, -1), (0, 1), (-1, 0), (1, 0)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && nx < GRID_W as i32 && ny >= 0 && ny < GRID_H as i32 {
                let (nx, ny) = (nx as usize, ny as usize);
                if !visited[nx][ny] && grid[nx][ny] != Cell::Wall {
                    visited[nx][ny] = true;
                    stack.push((nx, ny));
                }
            }
        }
    }
    for x in 0..GRID_W {
        for y in 0..GRID_H {
            if grid[x][y] != Cell::Wall && !visited[x][y] {
                grid[x][y] = Cell::Wall;
            }
        }
    }

    // ---- 6. Pillars scattered through big/atrium rooms (decorative only —
    //         this demo has no collision, camera moves freely regardless) ----
    let mut pillars = Vec::new();
    for (room, &(h, is_atrium)) in rooms.iter().zip(room_heights.iter()) {
        let big_enough = room.w >= 6 && room.h >= 5;
        if !(is_atrium || (big_enough && rng.chance(0.35))) {
            continue;
        }
        let mut px = room.x + 1;
        while px < room.x + room.w - 1 {
            let mut py = room.y + 1;
            while py < room.y + room.h - 1 {
                if visited[px as usize][py as usize] {
                    let wx = px as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                    let wz = py as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;
                    pillars.push((wx, wz, h));
                }
                py += 3;
            }
            px += 3;
        }
    }

    // ---- 7. Sparse light placement: a few per room, plus a spaced-out
    //         string along corridors — not one light per walkable tile.
    //         Which of these end up flickering is decided later, when the
    //         actual light entities are created. ----
    let mut lights = Vec::new();
    for (room, &(h, _)) in rooms.iter().zip(room_heights.iter()) {
        let count = (1 + (room.w * room.h) / 16).max(1);
        for i in 0..count {
            let lx = (room.x + 1 + (i % room.w.max(1))).clamp(room.x, room.x + room.w - 1) as usize;
            let ly = (room.y + 1 + ((i * 2) % room.h.max(1))).clamp(room.y, room.y + room.h - 1)
                as usize;
            if visited[lx][ly] {
                let wx = lx as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                let wz = ly as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;
                lights.push((wx, wz, h));
            }
        }
    }
    let mut since_last_light = 0;
    for x in 0..GRID_W {
        for y in 0..GRID_H {
            if grid[x][y] == Cell::Corridor && visited[x][y] {
                since_last_light += 1;
                if since_last_light >= 3 {
                    since_last_light = 0;
                    let wx = x as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                    let wz = y as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;
                    lights.push((wx, wz, ceiling_h[x][y]));
                }
            }
        }
    }

    BackroomsMap {
        grid,
        ceiling_h,
        lights,
        pillars,
    }
}

// ── End Map Generator ─────────────────────────────────────────────────────────

// ── Flickering fluorescent lights ──────────────────────────────────────────────
//
// A subset of tube lights run a little state machine instead of shining at a
// constant intensity:
//   - Fine  : steady baseline brightness, with rare single-frame blips
//   - Burst : rapid, uneven on/off strobing for a short stretch
//   - Dark  : an extended period fully off or knocked down to a dim glow,
//             with an occasional failed-reignition flicker partway through
//
// Since this renderer doesn't expose a "set light intensity" call, a change
// in brightness is applied by removing the old light and inserting a new one
// with the updated intensity — using only APIs already used elsewhere in
// this file (remove_light / insert_actor).

#[derive(Clone, Copy, PartialEq)]
enum FlickerPhase {
    Fine,
    Burst,
    Dark,
}

struct FlickerLight {
    main_id: LightId,
    fill_id: LightId,

    pos: [f32; 3],
    fill_pos: [f32; 3],
    color: [f32; 3],
    fill_color: [f32; 3],
    base_intensity: f32,
    fill_base_intensity: f32,
    radius: f32,
    fill_radius: f32,

    phase: FlickerPhase,
    phase_timer: f32,
    toggle_timer: f32,
    lit: bool,
    dark_is_full_off: bool,
    last_ratio: f32,
    rng: Rng,
}

impl FlickerLight {
    #[allow(clippy::too_many_arguments)]
    fn new(
        main_id: LightId,
        fill_id: LightId,
        pos: [f32; 3],
        fill_pos: [f32; 3],
        color: [f32; 3],
        fill_color: [f32; 3],
        base_intensity: f32,
        fill_base_intensity: f32,
        radius: f32,
        fill_radius: f32,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        // Stagger initial "fine" durations so lights don't sync up.
        let phase_timer = 3.0 + rng.next_f32() * 8.0;
        FlickerLight {
            main_id,
            fill_id,
            pos,
            fill_pos,
            color,
            fill_color,
            base_intensity,
            fill_base_intensity,
            radius,
            fill_radius,
            phase: FlickerPhase::Fine,
            phase_timer,
            toggle_timer: 0.0,
            lit: true,
            dark_is_full_off: false,
            last_ratio: 1.0,
            rng,
        }
    }

    /// Advance the flicker state machine by `dt` seconds, swapping the
    /// underlying lights in `scene` whenever brightness changes enough to
    /// matter (a full phase change, a toggle, or a blip).
    fn update(&mut self, scene: &mut Scene, dt: f32) {
        self.phase_timer -= dt;
        let mut ratio = self.last_ratio;
        let mut force_apply = false;

        match self.phase {
            FlickerPhase::Fine => {
                ratio = 1.0;
                if self.toggle_timer > 0.0 {
                    self.toggle_timer -= dt;
                    if self.toggle_timer <= 0.0 {
                        self.lit = true;
                        ratio = 1.0;
                        force_apply = true;
                    } else {
                        ratio = if self.lit { 1.0 } else { 0.05 };
                    }
                } else if self.rng.chance(dt * 0.15) {
                    // A rare single-frame blip during an otherwise steady stretch.
                    self.lit = false;
                    self.toggle_timer = 0.03 + self.rng.next_f32() * 0.06;
                    ratio = 0.05;
                    force_apply = true;
                }

                if self.phase_timer <= 0.0 {
                    self.toggle_timer = 0.0;
                    self.lit = true;
                    if self.rng.chance(0.6) {
                        self.phase = FlickerPhase::Burst;
                        self.phase_timer = 0.4 + self.rng.next_f32() * 1.2;
                    } else {
                        self.phase = FlickerPhase::Dark;
                        self.phase_timer = 1.5 + self.rng.next_f32() * 4.5;
                        self.dark_is_full_off = self.rng.chance(0.5);
                    }
                    ratio = 1.0;
                    force_apply = true;
                }
            }
            FlickerPhase::Burst => {
                self.toggle_timer -= dt;
                if self.toggle_timer <= 0.0 {
                    self.lit = !self.lit;
                    self.toggle_timer = 0.02 + self.rng.next_f32() * 0.07;
                    force_apply = true;
                }
                ratio = if self.lit { 1.0 } else { 0.04 };

                if self.phase_timer <= 0.0 {
                    self.lit = true;
                    if self.rng.chance(0.55) {
                        self.phase = FlickerPhase::Dark;
                        self.phase_timer = 1.5 + self.rng.next_f32() * 4.5;
                        self.dark_is_full_off = self.rng.chance(0.5);
                    } else {
                        self.phase = FlickerPhase::Fine;
                        self.phase_timer = 4.0 + self.rng.next_f32() * 10.0;
                    }
                    ratio = 1.0;
                    force_apply = true;
                }
            }
            FlickerPhase::Dark => {
                let base_ratio = if self.dark_is_full_off { 0.0 } else { 0.18 };
                if self.toggle_timer > 0.0 {
                    self.toggle_timer -= dt;
                    if self.toggle_timer <= 0.0 {
                        ratio = base_ratio;
                        force_apply = true;
                    } else {
                        ratio = 0.9;
                    }
                } else if self.rng.chance(dt * 0.1) {
                    // A dying tube occasionally tries to reignite mid-dark.
                    self.toggle_timer = 0.03 + self.rng.next_f32() * 0.05;
                    ratio = 0.9;
                    force_apply = true;
                } else {
                    ratio = base_ratio;
                }

                if self.phase_timer <= 0.0 {
                    self.toggle_timer = 0.0;
                    self.lit = true;
                    if self.rng.chance(0.7) {
                        self.phase = FlickerPhase::Fine;
                        self.phase_timer = 4.0 + self.rng.next_f32() * 10.0;
                    } else {
                        self.phase = FlickerPhase::Burst;
                        self.phase_timer = 0.4 + self.rng.next_f32() * 1.2;
                    }
                    ratio = 1.0;
                    force_apply = true;
                }
            }
        }

        if force_apply || (ratio - self.last_ratio).abs() > 0.02 {
            self.last_ratio = ratio;
            let main_intensity = (self.base_intensity * ratio).max(0.0);
            let fill_intensity = (self.fill_base_intensity * (0.3 + 0.7 * ratio)).max(0.0);

            let _ = scene.remove_light(self.main_id);
            self.main_id = scene
                .insert_actor(helio::SceneActor::light_with_movability(
                    point_light(self.pos, self.color, main_intensity, self.radius),
                    Some(Movability::Movable),
                ))
                .as_light()
                .unwrap();

            let _ = scene.remove_light(self.fill_id);
            self.fill_id = scene
                .insert_actor(helio::SceneActor::light_with_movability(
                    point_light(
                        self.fill_pos,
                        self.fill_color,
                        fill_intensity,
                        self.fill_radius,
                    ),
                    Some(Movability::Movable),
                ))
                .as_light()
                .unwrap();
        }
    }
}

// ── End Flickering Lights ──────────────────────────────────────────────────────

// ── OpenXR state (native only) ───────────────────────────────────────────────
//
// Mirrors `examples/vr/main.rs` / `examples/foliage_demo.rs`: a headset session
// requires the Vulkan instance/device to be created *through* OpenXR, so this has
// to happen before any wgpu device exists, and it either fully succeeds or the
// demo falls back to the desktop mirror path.

#[cfg(not(target_arch = "wasm32"))]
struct XrBundle {
    instance: helio_xr::XrInstance,
    session: helio_xr::XrSession,
    swapchain: helio_xr::XrSwapchain,
    input: helio_xr::XrInput,
    wgpu_instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

/// The features the XR device is asked for. `create_wgpu_device` masks this
/// down to what the HMD's Vulkan adapter actually supports.
#[cfg(not(target_arch = "wasm32"))]
fn xr_features() -> wgpu::Features {
    let required = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::INDIRECT_FIRST_INSTANCE
        | wgpu::Features::MULTIVIEW
        | wgpu::Features::MULTISAMPLE_ARRAY;
    let optional = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::VERTEX_WRITABLE_STORAGE;
    required | optional
}

/// Bring up the full OpenXR stack: OpenXR instance → wgpu Vulkan instance →
/// wgpu device → session → swapchain. Any failure degrades to `None`, which
/// switches the demo to desktop mirror mode.
#[cfg(not(target_arch = "wasm32"))]
fn try_init_xr() -> Option<XrBundle> {
    let result = (|| -> helio_xr::Result<XrBundle> {
        let instance = helio_xr::XrInstance::create("helio_vhs_backrooms_demo")?;
        let wgpu_instance = helio_xr::create_wgpu_instance(&instance.instance, instance.system)?;
        let (adapter, device, queue) = helio_xr::create_wgpu_device(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            xr_features(),
        )?;
        // Actions must be declared and their bindings suggested BEFORE the session is
        // created; the runtime resolves them at session creation and will not accept new
        // suggestions afterwards.
        let input = helio_xr::XrInput::new(&instance.instance)?;
        let session = helio_xr::XrSession::create(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            &device,
            &queue,
        )?;
        // Attach exactly once, after creation and before the first sync.
        input.attach(&session.session)?;
        let swapchain = helio_xr::XrSwapchain::create(
            &device,
            &session.session,
            session.width,
            session.height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )?;
        log::info!(
            "[XR] OpenXR ready — {}x{} eye buffer, {} array layer(s), format {:?}",
            session.width,
            session.height,
            swapchain.array_size,
            swapchain.format,
        );
        Ok(XrBundle {
            instance,
            session,
            swapchain,
            input,
            wgpu_instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })();

    match result {
        Ok(bundle) => Some(bundle),
        Err(e) => {
            log::warn!("[XR] OpenXR init failed ({e}); running in desktop mirror mode instead");
            None
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

// Ids we need to keep for regeneration
struct MapResources {
    walls: Vec<MeshId>,
    floors: Vec<MeshId>,
    ceilings: Vec<MeshId>,
    pillars: Vec<MeshId>,
    light_ids: Vec<LightId>,
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    /// Present only in desktop mode; idle/absent while the headset is driven.
    surface: Option<wgpu::Surface<'static>>,
    alpha_mode: wgpu::CompositeAlphaMode,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Arc<Mutex<Renderer>>,
    action_rx: Receiver<HelioAction>,
    last_frame: std::time::Instant,

    map_resources: Option<MapResources>,
    flicker_lights: Vec<FlickerLight>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    start_time: std::time::Instant,

    /// True when a live OpenXR session is driving `renderer.render_xr()`.
    xr_active: bool,
    #[cfg(not(target_arch = "wasm32"))]
    xr_input: Option<(helio_xr::XrInput, helio_xr::XrSessionHandle)>,
    /// Player locomotion, applied as the XR stage transform. Position is the stage
    /// origin in world space; yaw is snap/smooth turn about it.
    player_position: glam::Vec3,
    player_yaw: f32,
}

impl AppState {
    fn configure_surface(&self, width: u32, height: u32) {
        let Some(surface) = &self.surface else { return };
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: width.max(1),
                height: height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
    }

    /// Read the controllers and walk the player, then publish the result as the XR
    /// stage transform. Movement is head-relative on the yaw axis only, matching
    /// `examples/vr`.
    #[cfg(not(target_arch = "wasm32"))]
    fn update_locomotion(&mut self, dt: f32) {
        const MOVE_SPEED: f32 = 2.0;
        const TURN_SPEED: f32 = 1.8;
        const DEADZONE: f32 = 0.15;

        let Some((input, session)) = &self.xr_input else {
            return;
        };
        let Ok(controls) = input.sync(session) else {
            return;
        };

        let deadzone = |v: glam::Vec2| -> glam::Vec2 {
            if v.length() < DEADZONE {
                glam::Vec2::ZERO
            } else {
                v
            }
        };
        let move_stick = deadzone(controls.left_stick);
        let turn_stick = deadzone(controls.right_stick);

        self.player_yaw -= turn_stick.x * TURN_SPEED * dt;

        let (sin, cos) = self.player_yaw.sin_cos();
        let forward = glam::Vec3::new(-sin, 0.0, -cos);
        let right = glam::Vec3::new(cos, 0.0, -sin);
        self.player_position += (forward * move_stick.y + right * move_stick.x) * MOVE_SPEED * dt;

        if let Ok(mut renderer) = self.renderer.lock() {
            renderer.set_xr_stage_transform(
                glam::Mat4::from_translation(self.player_position)
                    * glam::Mat4::from_rotation_y(self.player_yaw),
            );
        }
    }
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }

    fn place(
        scene: &mut Scene,
        mesh: MeshId,
        material: MaterialId,
        transform: glam::Mat4,
        radius: f32,
    ) {
        let _ = scene.insert_actor(helio::SceneActor::object(ObjectDescriptor {
            mesh,
            material,
            transform,
            bounds: [
                transform.w_axis.x,
                transform.w_axis.y,
                transform.w_axis.z,
                radius,
            ],
            flags: 0,
            groups: GroupMask::NONE,
            movability: None,
            user_tag: 0,
        }));
    }

    fn regenerate_map(state: &mut AppState) {
        let map = generate_map();
        let mut renderer = state.renderer.lock().unwrap();
        let scene = renderer.scene_mut();

        // Remove previous map resources
        if let Some(res) = &state.map_resources {
            for &id in &res.walls {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.floors {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.ceilings {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.pillars {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.light_ids {
                let _ = scene.remove_light(id);
            }
        }
        // Flickering lights carry their own, runtime-evolving light ids that
        // aren't tracked in MapResources, so they need their own cleanup pass.
        for fl in state.flicker_lights.drain(..) {
            let _ = scene.remove_light(fl.main_id);
            let _ = scene.remove_light(fl.fill_id);
        }

        let wall_mat = scene.insert_material(make_material(
            [0.82, 0.72, 0.52, 1.0],
            0.6,
            0.05,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let floor_mat = scene.insert_material(make_material(
            [0.45, 0.38, 0.28, 1.0],
            0.3,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let ceiling_mat = scene.insert_material(make_material(
            [0.88, 0.88, 0.85, 1.0],
            0.7,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let trim_mat = scene.insert_material(make_material(
            [0.35, 0.32, 0.28, 1.0],
            0.4,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let pillar_mat = scene.insert_material(make_material(
            [0.5, 0.46, 0.36, 1.0],
            0.5,
            0.02,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let mut walls = Vec::new();
        let mut floors = Vec::new();
        let mut ceilings = Vec::new();
        let mut pillars = Vec::new();
        let mut light_ids = Vec::new();

        for x in 0..GRID_W {
            for y in 0..GRID_H {
                if map.grid[x][y] == Cell::Wall {
                    continue;
                }

                let wx = x as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                let wz = y as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;
                let h = map.ceiling_h[x][y];

                // Floor tile
                let f = scene
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [H_CELL, 0.05, H_CELL],
                    )))
                    .as_mesh()
                    .unwrap();
                Self::place(
                    scene,
                    f,
                    floor_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(wx, 0.0, wz)),
                    H_CELL,
                );
                floors.push(f);

                // Ceiling tile — height varies per room (low/normal/tall/atrium)
                let c = scene
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [H_CELL, 0.03, H_CELL],
                    )))
                    .as_mesh()
                    .unwrap();
                Self::place(
                    scene,
                    c,
                    ceiling_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(wx, h, wz)),
                    H_CELL,
                );
                ceilings.push(c);

                // Walls on edges adjacent to Wall cells. Rotation is chosen so
                // the wall mesh's long axis (local z) lies along the shared
                // boundary: south/north boundaries run along world x (need a
                // 90-degree turn), west/east boundaries run along world z
                // (need no turn).
                let neighbors = [
                    (0, -1, 0.0, -H_CELL, std::f32::consts::FRAC_PI_2 * 3.0), // south
                    (0, 1, 0.0, H_CELL, std::f32::consts::FRAC_PI_2),         // north
                    (-1, 0, -H_CELL, 0.0, 0.0),                               // west
                    (1, 0, H_CELL, 0.0, std::f32::consts::PI),                // east
                ];
                for &(dx, dy, ox, oz, rot_y) in &neighbors {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let is_wall = nx < 0
                        || nx >= GRID_W as i32
                        || ny < 0
                        || ny >= GRID_H as i32
                        || map.grid[nx as usize][ny as usize] == Cell::Wall;
                    if !is_wall {
                        continue;
                    }

                    // Wall — height matches this cell's own ceiling height,
                    // so height transitions between neighbouring rooms show
                    // up as an open ceiling ledge rather than a mismatched
                    // wall (there's no wall between two walkable cells).
                    let w = scene
                        .insert_actor(helio::SceneActor::mesh(box_mesh(
                            [0.0, 0.0, 0.0],
                            [0.1, h / 2.0, CELL / 2.0],
                        )))
                        .as_mesh()
                        .unwrap();
                    let t =
                        glam::Mat4::from_translation(glam::Vec3::new(wx + ox, h / 2.0, wz + oz))
                            * glam::Mat4::from_rotation_y(rot_y);
                    Self::place(scene, w, wall_mat, t, H_CELL);
                    walls.push(w);

                    // Baseboard trim
                    let t2 = scene
                        .insert_actor(helio::SceneActor::mesh(box_mesh(
                            [0.0, 0.0, 0.0],
                            [0.12, 0.05, CELL / 2.0],
                        )))
                        .as_mesh()
                        .unwrap();
                    let tt = glam::Mat4::from_translation(glam::Vec3::new(wx + ox, 0.05, wz + oz))
                        * glam::Mat4::from_rotation_y(rot_y);
                    Self::place(scene, t2, trim_mat, tt, H_CELL);
                    walls.push(t2);
                }
            }
        }

        // Pillars — decorative columns scattered through big/atrium rooms
        for &(px, pz, ph) in &map.pillars {
            let p = scene
                .insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0],
                    [0.35, ph / 2.0, 0.35],
                )))
                .as_mesh()
                .unwrap();
            Self::place(
                scene,
                p,
                pillar_mat,
                glam::Mat4::from_translation(glam::Vec3::new(px, ph / 2.0, pz)),
                0.5,
            );
            pillars.push(p);
        }

        // Fluorescent lights — sparse: a few per room plus spaced corridor
        // lights. ~16% of them are picked to run the flicker state machine
        // instead of just shining steadily.
        let light_colors = [
            [0.95, 0.92, 0.85],
            [0.92, 0.93, 0.88],
            [0.96, 0.91, 0.82],
            [0.90, 0.94, 0.86],
        ];
        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15 ^ (map.lights.len() as u64));
        let mut new_flicker_lights = Vec::new();
        for &(lx, lz, h) in &map.lights {
            let ci = (rng.next_u32() as usize) % light_colors.len();
            let color = light_colors[ci];
            let fill_color = [0.95, 0.92, 0.85];
            let main_pos = [lx, h - 0.2, lz];
            let fill_pos = [lx, (h * 0.5).max(1.0), lz];
            let main_intensity = 3.5;
            let fill_intensity = 1.2;
            let radius = 8.0;
            let fill_radius = 4.0;

            let is_flicker = rng.chance(0.16);

            let main_id = scene
                .insert_actor(helio::SceneActor::light_with_movability(
                    point_light(main_pos, color, main_intensity, radius),
                    Some(Movability::Movable),
                ))
                .as_light()
                .unwrap();
            let fill_id = scene
                .insert_actor(helio::SceneActor::light_with_movability(
                    point_light(fill_pos, fill_color, fill_intensity, fill_radius),
                    Some(Movability::Movable),
                ))
                .as_light()
                .unwrap();

            if is_flicker {
                let seed = 0xD1B5_4A32_9C3F_2717_u64
                    ^ (lx.to_bits() as u64)
                    ^ ((lz.to_bits() as u64) << 32)
                    ^ (rng.next_u32() as u64);
                new_flicker_lights.push(FlickerLight::new(
                    main_id,
                    fill_id,
                    main_pos,
                    fill_pos,
                    color,
                    fill_color,
                    main_intensity,
                    fill_intensity,
                    radius,
                    fill_radius,
                    seed,
                ));
            } else {
                light_ids.push(main_id);
                light_ids.push(fill_id);
            }
        }
        state.flicker_lights = new_flicker_lights;

        state.map_resources = Some(MapResources {
            walls,
            floors,
            ceilings,
            pillars,
            light_ids,
        });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – VHS Backrooms")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        // Try OpenXR before creating wgpu: a headset session requires the
        // Vulkan instance/device to be created through OpenXR.
        #[cfg(not(target_arch = "wasm32"))]
        let xr_owned = try_init_xr();
        #[cfg(target_arch = "wasm32")]
        let xr_owned: Option<XrBundle> = None;

        let (device, queue, surface, surface_format, alpha_mode, config, xr_owned) = match xr_owned
        {
            Some(bundle) => {
                let config = RendererConfig::new(
                    bundle.session.width,
                    bundle.session.height,
                    bundle.swapchain.format,
                )
                .with_shadow_quality(helio::ShadowQuality::Ultra)
                // The graph's internal resolution must match the XR eye buffer exactly
                // (it becomes the swapchain target); no scaling. The graph is built in
                // single-layer (non-multiview) mode — render_xr renders each eye
                // separately (dual-pass stereo) so every existing pass and shader,
                // including the VHS user-effects snippet, works unchanged.
                .with_render_scale(1.0);
                // PC mirror surface: the OpenXR-created device presents both eye
                // buffers side-by-side into this window.
                let mirror_surface = bundle
                    .wgpu_instance
                    .create_surface(window.clone())
                    .expect("Failed to create mirror surface");
                let caps = mirror_surface.get_capabilities(&bundle.adapter);
                let mirror_format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);
                let mirror_alpha = caps.alpha_modes[0];
                let mirror_size = window.inner_size();
                mirror_surface.configure(
                    &bundle.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: mirror_format,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                        width: mirror_size.width.max(1),
                        height: mirror_size.height.max(1),
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: mirror_alpha,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                (
                    bundle.device.clone(),
                    bundle.queue.clone(),
                    Some(mirror_surface),
                    mirror_format,
                    mirror_alpha,
                    config,
                    Some(bundle),
                )
            }
            None => {
                log::warn!("[XR] no headset — running desktop mirror");
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    flags: wgpu::InstanceFlags::empty(),
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                let surface = instance.create_surface(window.clone()).expect("surface");
                let adapter =
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    }))
                    .expect("adapter");
                let (device, queue) =
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                        label: Some("Device"),
                        required_features: required_wgpu_features(adapter.features()),
                        required_limits: required_wgpu_limits(adapter.limits()),
                        experimental_features: required_experimental_features(adapter.features()),
                        ..Default::default()
                    }))
                    .expect("device");
                let device = Arc::new(device);
                let queue = Arc::new(queue);

                let caps = surface.get_capabilities(&adapter);
                let format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);
                let alpha_mode = caps.alpha_modes[0];
                let size = window.inner_size();
                let config = RendererConfig::new(size.width, size.height, format)
                    .with_shadow_quality(helio::ShadowQuality::Ultra)
                    .with_render_scale(1.0);
                (
                    device,
                    queue,
                    Some(surface),
                    format,
                    alpha_mode,
                    config,
                    None,
                )
            }
        };

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));

        let scene = Scene::new(device.clone(), queue.clone());
        let debug_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let graph = build_default_graph_with_user_effects(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None,
            VHS_SHADER_SNIPPET,
        );
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            config.surface_format,
            config.width,
            config.height,
            config.render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buf,
            cull_stats_buf,
        );

        #[cfg(not(target_arch = "wasm32"))]
        let mut xr_input = None;
        #[cfg(not(target_arch = "wasm32"))]
        let xr_active = if let Some(bundle) = xr_owned {
            let template = Camera::perspective_look_at(
                glam::Vec3::new(0.0, 1.6, 0.0),
                glam::Vec3::new(0.0, 1.6, -1.0),
                glam::Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                1.0,
                0.1,
                200.0,
            );
            renderer.set_xr_camera(template);
            renderer.set_xr_mirror_format(surface_format);
            let session_handle = bundle.session.session.clone();
            xr_input = Some((bundle.input, session_handle));
            renderer.set_xr_session(
                Some(bundle.instance),
                Some(bundle.session),
                Some(bundle.swapchain),
            );
            // No temporal AA/TSR jitter in VR (render_xr disables it anyway).
            renderer.set_jitter_enabled(false);
            true
        } else {
            false
        };
        #[cfg(target_arch = "wasm32")]
        let xr_active = false;

        // ── VHS camcorder post-process volume ─────────────────────────────────
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::post_process_volume(
                PostProcessVolumeDescriptor {
                    bounds_min: [-1000.0, -1000.0, -1000.0],
                    bounds_max: [1000.0, 1000.0, 1000.0],
                    blend_radius: 0.0,
                    unbound: true,
                    priority: 100.0,
                    blend_weight: 1.0,
                    settings: PostProcessSettings {
                        // All effects are handled by the user_effects WGSL snippet.
                        // Keep the volume at defaults so the built-in chain is a
                        // no-op — the VHS shader owns the entire post-process look.
                        ..PostProcessSettings::default()
                    },
                },
            ));

        renderer.set_ambient([0.75, 0.7, 0.6], 0.04);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        let renderer = Arc::new(Mutex::new(renderer));
        let (bridge, action_rx) = HelioCommandBridge::new();
        let command_bridge = Arc::new(bridge);

        {
            let bridge = command_bridge.clone();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    match line {
                        Ok(cmd) if !cmd.trim().is_empty() => match bridge.run(&cmd) {
                            Ok(()) => println!("OK: {}", cmd),
                            Err(e) => println!("ERR: {} -> {}", cmd, e),
                        },
                        _ => {}
                    }
                }
            });
        }

        let state = AppState {
            window,
            surface,
            alpha_mode,
            device,
            queue,
            surface_format,
            renderer,
            action_rx,
            last_frame: std::time::Instant::now(),
            map_resources: None,
            flicker_lights: Vec::new(),
            cam_pos: glam::Vec3::new(0.0, 1.6, 0.0),
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            start_time: std::time::Instant::now(),
            xr_active,
            #[cfg(not(target_arch = "wasm32"))]
            xr_input,
            player_position: glam::Vec3::ZERO,
            player_yaw: 0.0,
        };
        state.configure_surface(
            state.window.inner_size().width,
            state.window.inner_size().height,
        );
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyR),
                        ..
                    },
                ..
            } => {
                App::regenerate_map(state);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(code),
                        ..
                    },
                ..
            } => {
                state.keys.insert(code);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Released,
                        physical_key: PhysicalKey::Code(code),
                        ..
                    },
                ..
            } => {
                state.keys.remove(&code);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    state.cursor_grabbed = true;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::Locked);
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.configure_surface(s.width, s.height);
                // In XR mode the graph resolution is fixed by the headset's eye
                // buffer; resizing the mirror window must not rebuild the graph
                // at the window's resolution (that destroys resources cached in
                // pass bind groups and breaks the XR composite).
                if !state.xr_active {
                    if let Ok(mut renderer) = state.renderer.lock() {
                        renderer.set_render_size(s.width, s.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                #[cfg(not(target_arch = "wasm32"))]
                if state.xr_active {
                    if state.map_resources.is_none() {
                        App::regenerate_map(state);
                    }
                    state.update_locomotion(dt);
                    if let Ok(mut renderer) = state.renderer.lock() {
                        AppState::update_common_frame(
                            &mut state.flicker_lights,
                            &state.action_rx,
                            state.start_time,
                            &mut renderer,
                            dt,
                        );
                        // Headset path: render_xr() polls session events, locates the
                        // per-eye poses, uploads the stereo camera and renders both
                        // eyes. The mirror surface (if any) receives both eye buffers
                        // side by side via the renderer's mirror blit.
                        let mirror = match &state.surface {
                            Some(surface) => match surface.get_current_texture() {
                                wgpu::CurrentSurfaceTexture::Success(texture)
                                | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Some(texture),
                                _ => None,
                            },
                            None => None,
                        };
                        let mirror_view = mirror.as_ref().map(|t| {
                            t.texture
                                .create_view(&wgpu::TextureViewDescriptor::default())
                        });
                        if let Err(e) = renderer.render_xr(mirror_view.as_ref()) {
                            log::error!("[XR] render_xr error: {e:?}");
                        }
                        if let Some(output) = mirror {
                            state.queue.present(output);
                        }
                    }
                    state.mouse_delta = (0.0, 0.0);
                    state.window.request_redraw();
                    return;
                }

                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl AppState {
    /// Advances the flicker lights, drains the command bridge, and refreshes the
    /// VHS post-process params. Shared between the desktop and XR redraw paths.
    ///
    /// Takes individual fields rather than `&mut self` so it can be called while a
    /// `MutexGuard` borrowed from `self.renderer` is already held — `self.method()`
    /// would need to reborrow all of `self`, which conflicts with that guard.
    fn update_common_frame(
        flicker_lights: &mut [FlickerLight],
        action_rx: &Receiver<HelioAction>,
        start_time: std::time::Instant,
        renderer: &mut Renderer,
        dt: f32,
    ) {
        while let Ok(action) = action_rx.try_recv() {
            match action {
                HelioAction::SetDebugMode(mode) => renderer.set_debug_mode(mode),
                HelioAction::SetEditorMode(enabled) => renderer.set_editor_mode(enabled),
                HelioAction::DebugClear => renderer.debug_clear(),
            }
        }

        // Advance every flickering light's state machine this frame.
        {
            let scene = renderer.scene_mut();
            for fl in flicker_lights.iter_mut() {
                fl.update(scene, dt);
            }
        }

        // Write VHS parameters to post-process custom params buffer
        let time = start_time.elapsed().as_secs_f32();
        let vhs_params: [[f32; 4]; 2] = [
            [
                0.0,  // unused
                0.12, // tape jitter — ~0.6 px max horizontal displacement
                8.0,  // jitter frequency
                0.2,
            ], // flicker intensity
            [
                0.4,  // noise amount
                time, // animation time
                0.0, 0.0,
            ],
        ];
        if let Some(pass) = renderer.find_pass_mut::<helio_pass_postprocess::PostProcessPass>() {
            pass.set_custom_params(&vhs_params);
        }
    }

    /// Desktop mirror path: WASD + mouse free camera with camcorder bob, then a
    /// normal render. The bob is desktop-only — see the module doc comment.
    fn render(&mut self, dt: f32) {
        // Generate map on first render (after renderer is fully set up)
        if self.map_resources.is_none() {
            App::regenerate_map(self);
        }

        const SPEED: f32 = 3.0;
        const SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.2, 1.2);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos += glam::Vec3::Y * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= glam::Vec3::Y * SPEED * dt;
        }

        let time = self.start_time.elapsed().as_secs_f32();
        let bob_amt = 0.015;
        let bob_speed = 3.5;
        let bob = (time * bob_speed).sin() * bob_amt;
        let bob_sway = (time * bob_speed * 0.5).cos() * bob_amt * 0.5;

        let cam_pos = self.cam_pos + glam::Vec3::new(bob_sway, bob, 0.0);

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(
            cam_pos,
            cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
        );

        let mut renderer = self.renderer.lock().unwrap();
        Self::update_common_frame(
            &mut self.flicker_lights,
            &self.action_rx,
            self.start_time,
            &mut renderer,
            dt,
        );

        let Some(surface) = &self.surface else {
            return;
        };
        let output = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        self.queue.present(output);
    }
}
