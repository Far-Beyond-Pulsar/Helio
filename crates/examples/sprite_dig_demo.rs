//! 2D side-scrolling sandbox/mining platformer demo — every sprite in `assets/sprites/` (one
//! file per sprite, extracted from the original non-uniform composite sheet,
//! each file's own tight pixel bounds, no shared grid) gets packed at
//! startup into a single non-uniform shelf-packed atlas texture and used
//! somewhere in the world, grouped into themed zones (a forest, a village, a
//! mining camp, a monster den guarding treasure, a market stall) rather than
//! scattered everywhere.
//!
//! Controls:
//! - A/D or Left/Right to move, Space to jump.
//! - Left-click and hold on anything to mine it — it cracks in three stages
//!   before breaking, then joins the hotbar along the top of the screen.
//! - Mouse wheel scrolls the hotbar selection.
//! - Right-click places the selected hotbar item at the cursor (snapped to
//!   the terrain grid). Stacks are tracked internally (no on-screen counts);
//!   a stack's icon disappears once it hits zero.
//!
//! Every placed object — terrain tile, tree, monster, chest, torch, the lot —
//! is breakable; breaking a terrain tile actually opens a hole (collision
//! reads the same broken-tile set), so digging is real, not just cosmetic.
//!
//! Rendering is the same GPU cull + radix-sort → batch pipeline as the other
//! sprite demos (`SpriteCullPass` → `SpriteBatchPass`) — the terrain alone is
//! ~5,500 instanced quads, inserted once at startup; only the player, a
//! handful of animated critters/items, the in-progress crack overlay, and
//! the hotbar icons re-upload their instance bytes after that.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use helio_core::{GpuScene, RenderGraph};
use helio_pass_radiance_cascades_2d::{RadianceCascades2DPass, RadianceCascadesCompositePass, RadianceCascadesConfig};
use helio_pass_sprite_batch::{SpriteBatchPass, SpriteHandle, SpriteInstance};
use helio_pass_sprite_cull::SpriteCullPass;
use image::RgbaImage;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// ── World layout ─────────────────────────────────────────────────────────

const TILE: f32 = 48.0;
const WORLD_COLS: i32 = 240;
const DIRT_ROWS: i32 = 8;
const STONE_ROWS: i32 = 14;
const POOL_CAPACITY: usize = 7500;

const GRAVITY: f32 = -2400.0;
const MOVE_SPEED: f32 = 260.0;
const JUMP_VEL: f32 = 780.0;

const BREAK_STAGE_DURATION: f32 = 0.22;
const BREAK_TOTAL_STAGES: u32 = 3;
const HOTBAR_SLOT_SPACING: f32 = 56.0;
const HOTBAR_ICON_SIZE: f32 = 40.0;
const HOTBAR_MARGIN_TOP: f32 = 46.0;

// ── Lighting (2D radiance cascades) ─────────────────────────────────────
//
// The occupancy grid the lighting pass reads is a flat world-space grid
// (unlike terrain's own per-column-relative storage), sized generously to
// cover every tile the undulating heightfield can ever place: surface_row's
// three summed sines bound it to roughly ±11 tiles, and terrain goes
// `DIRT_ROWS + STONE_ROWS` (22) tiles deep from there.
const OCC_COLS: u32 = WORLD_COLS as u32;
const OCC_ROWS: u32 = 52;
const OCC_ORIGIN: [f32; 2] = [-TILE * 0.5, -38.0 * TILE];

fn occ_cell(pos: [f32; 2]) -> Option<(u32, u32)> {
    let cf = (pos[0] - OCC_ORIGIN[0]) / TILE;
    let rf = (pos[1] - OCC_ORIGIN[1]) / TILE;
    if cf < 0.0 || rf < 0.0 {
        return None;
    }
    let (c, r) = (cf.floor() as u32, rf.floor() as u32);
    if c < OCC_COLS && r < OCC_ROWS { Some((c, r)) } else { None }
}

fn occ_index(c: u32, r: u32) -> u32 {
    r * OCC_COLS + c
}

/// Mirrors `helio-pass-radiance-cascades-2d`'s `Emitter` WGSL struct layout
/// exactly (32 bytes) — no Cargo-level type dependency, matching how every
/// other pass pair in this codebase shares a byte layout as a protocol
/// rather than a shared Rust type.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmitter {
    pos: [f32; 2],
    radius: f32,
    r: f32,
    g: f32,
    b: f32,
    _pad: f32,
}

const LIGHT_EMITTER_NAMES: &[&str] = &["torch", "torch_post", "torch_small", "torch_staff", "lantern_hanging", "lantern_wall", "campfire", "furnace_lit"];

fn emitter_style(name: &str) -> ([f32; 3], f32) {
    match name {
        "campfire" | "furnace_lit" => ([1.0, 0.42, 0.10], 260.0),
        "lantern_hanging" | "lantern_wall" => ([1.0, 0.75, 0.42], 200.0),
        _ => ([1.0, 0.58, 0.22], 240.0), // torches
    }
}

// Every one of the 77 sprites in `assets/sprites/` is placed exactly once
// below, grouped into themed zones along the world (a forest, a village, a
// mining camp, a monster den guarding treasure, a second forest, a market
// stall) rather than scattered uniformly everywhere — `grass_b`/`stone_block`
// (terrain) and the landmarks (`sign`, `cabin`, the three chests, `hut`,
// `gold_ore`, `torch`, `player*`) account for the rest.
const TREES: &[&str] = &["pine_tree", "oak_tree"];
const FOREST_CLUTTER: &[&str] = &["flower_blue", "flower_orange", "mushroom", "leaf_sprig", "bush", "grass_a"];
const FOREST_CRITTERS: &[&str] = &["bunny", "lizard_creature"];
const MINING_ROCKS: &[&str] = &["rock", "rockpile", "rock_small", "rock_cluster", "stone_chunk_pile", "stone_pile_2"];
const MINING_TOOLS: &[&str] = &["pickaxe_1", "pickaxe_2", "pickaxe_hammer", "axe"];
const MINING_PROPS: &[&str] = &["anvil", "grindstone", "crate_pile", "log_pile", "workbench_small"];
const DEN_MONSTERS: &[&str] = &["slime_blue", "slime_green", "zombie_green", "dragon_red", "eyeball_red"];
const DEN_WEAPONS: &[&str] = &["sword_1", "sword_2", "dagger"];
// Village furniture isn't scattered loose in the open anymore — the cabin
// and a couple of `hut`s (plus two villager-posed NPCs standing by the
// cabin door) carry the village's presence instead.
const VILLAGE_LIGHTS: &[&str] = &["torch_post", "lantern_hanging", "campfire"];
const MARKET_STALL: &[&str] = &["wood_table_2", "wood_wall_bracket", "stone_wall_segment"];
const MARKET_WARES: &[&str] =
    &["heart_red", "star_blue", "potion_red", "potion_blue", "coin_gold", "coin_silver", "coin_copper", "bomb"];

enum Animated {
    None,
    Critter,
    Item,
}

fn surface_row(col: i32) -> i32 {
    let t = col as f32;
    let h = (t * 0.09).sin() * 3.5 + (t * 0.03).sin() * 6.0 + (t * 0.23).sin() * 1.2;
    h.round() as i32
}

fn surface_top_world_y(col: i32) -> f32 {
    surface_row(col) as f32 * TILE
}

/// Ground height at `col` accounting for mined-out terrain: walks down from
/// the surface through consecutive broken tiles, so digging out the row(s)
/// under your feet actually opens a hole you fall into, rather than just
/// changing what's drawn.
fn ground_y_at(col: i32, broken: &HashSet<(i32, i32)>) -> f32 {
    let mut r = 0i32;
    while r <= DIRT_ROWS + STONE_ROWS && broken.contains(&(col, r)) {
        r += 1;
    }
    surface_top_world_y(col) - r as f32 * TILE
}

fn hash01(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(2654435761);
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    x as f32 / u32::MAX as f32
}

fn flip_u(uv: [f32; 4]) -> [f32; 4] {
    [uv[2], uv[1], uv[0], uv[3]]
}

fn world_from_screen(mouse: (f64, f64), window_size: (u32, u32), camera_center: [f32; 2]) -> [f32; 2] {
    let sx = mouse.0 as f32 - window_size.0 as f32 * 0.5;
    let sy = mouse.1 as f32 - window_size.1 as f32 * 0.5;
    [camera_center[0] + sx, camera_center[1] - sy]
}

/// Point-in-AABB hit test over every breakable object, picking the
/// topmost (highest-depth) match under the cursor.
fn hit_test(objects: &HashMap<SpriteHandle, Breakable>, p: [f32; 2]) -> Option<SpriteHandle> {
    let mut best: Option<(SpriteHandle, f32)> = None;
    for (&handle, b) in objects.iter() {
        let hw = b.size[0] * 0.5;
        let hh = b.size[1] * 0.5;
        if p[0] >= b.pos[0] - hw && p[0] <= b.pos[0] + hw && p[1] >= b.pos[1] - hh && p[1] <= b.pos[1] + hh
            && best.map(|(_, d)| b.depth > d).unwrap_or(true)
        {
            best = Some((handle, b.depth));
        }
    }
    best.map(|(h, _)| h)
}

fn hotbar_slot_world_pos(camera_center: [f32; 2], window_size: (u32, u32), index: usize, total: usize) -> [f32; 2] {
    let n = total.max(1) as f32;
    let x_offset = (index as f32 - (n - 1.0) * 0.5) * HOTBAR_SLOT_SPACING;
    let y_offset = window_size.1 as f32 * 0.5 - HOTBAR_MARGIN_TOP;
    [camera_center[0] + x_offset, camera_center[1] + y_offset]
}

/// Small deterministic PRNG (PCG-style LCG) — avoids pulling in `rand`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_f32() * (hi - lo) as f32) as i32
    }
    fn range_usize(&mut self, n: usize) -> usize {
        ((self.next_f32() * n as f32) as usize).min(n - 1)
    }
    fn bool(&mut self) -> bool {
        self.next_f32() < 0.5
    }
}

// ── Sprite atlas: load every PNG in assets/sprites/, shelf-pack them (no
// padding on the individual sprites, native pixel dimensions, non-uniform
// packing — not a grid) into one runtime texture, plus a few procedurally
// generated extras (a solid-white swatch, three crack-overlay stages). ──────

#[derive(Clone, Copy)]
struct PackedSprite {
    uv: [f32; 4],
    w: f32,
    h: f32,
}

// Generated by `build.rs` from every PNG in `assets/sprites/` at compile
// time (`include_bytes!` per file) — the sprite set ships inside the binary,
// not read from disk at runtime.
include!(concat!(env!("OUT_DIR"), "/embedded_sprites.rs"));

fn load_all_sprites() -> Vec<(String, RgbaImage)> {
    EMBEDDED_SPRITES
        .iter()
        .map(|(name, bytes)| {
            let img = image::load_from_memory(bytes).unwrap_or_else(|e| panic!("decode embedded sprite {name}: {e}")).to_rgba8();
            (name.to_string(), img)
        })
        .collect()
}

fn point_seg_dist(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { ((px - x0) * dx + (py - y0) * dy) / len2 } else { 0.0 };
    let t = t.clamp(0.0, 1.0);
    let (cx, cy) = (x0 + t * dx, y0 + t * dy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

/// A procedural crack overlay for mining stage `stage` (1..=3) — more, wider
/// cracks radiating from the center at higher stages. Straight alpha, on a
/// transparent background, stretched over whatever it's overlaid on.
fn make_crack_image(stage: u32) -> RgbaImage {
    const SIZE: u32 = 64;
    let mut img = RgbaImage::new(SIZE, SIZE);
    let mut rng = Rng::new(0x0C7A_CC00 + stage as u64);
    let num_lines = 2 + stage * 2;
    let thickness = 1.6 + stage as f32 * 0.5;
    let (cx, cy) = (SIZE as f32 * 0.5, SIZE as f32 * 0.5);
    for _ in 0..num_lines {
        let ang = rng.next_f32() * std::f32::consts::TAU;
        let len = SIZE as f32 * (0.25 + rng.next_f32() * 0.35);
        let (x1, y1) = (cx + ang.cos() * len, cy + ang.sin() * len);
        for py in 0..SIZE {
            for px in 0..SIZE {
                let d = point_seg_dist(px as f32 + 0.5, py as f32 + 0.5, cx, cy, x1, y1);
                if d < thickness {
                    let a = ((1.0 - d / thickness) * 210.0) as u8;
                    let p = img.get_pixel_mut(px, py);
                    if a > p[3] {
                        *p = image::Rgba([15, 12, 10, a]);
                    }
                }
            }
        }
    }
    img
}

/// Shelf-packs `images` into one atlas texture at their native sizes — no
/// padding on the sprites themselves, just a small gap between packed items
/// to avoid sampling bleed at their shared edges.
fn pack_atlas(images: Vec<(String, RgbaImage)>) -> (RgbaImage, HashMap<String, PackedSprite>) {
    const ATLAS_W: u32 = 1536;
    const GAP: u32 = 2;

    let mut order: Vec<usize> = (0..images.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(images[i].1.height()));

    let mut placements: Vec<(usize, u32, u32)> = Vec::new();
    let mut cursor_y = GAP;
    let mut shelf_x = GAP;
    let mut shelf_h = 0u32;
    for &i in &order {
        let (iw, ih) = images[i].1.dimensions();
        if shelf_x + iw + GAP > ATLAS_W {
            cursor_y += shelf_h + GAP;
            shelf_x = GAP;
            shelf_h = 0;
        }
        placements.push((i, shelf_x, cursor_y));
        shelf_x += iw + GAP;
        shelf_h = shelf_h.max(ih);
    }
    let atlas_h = cursor_y + shelf_h + GAP;

    let mut atlas = RgbaImage::new(ATLAS_W, atlas_h);
    let mut uvs = HashMap::new();
    for (i, x, y) in placements {
        let (name, img) = &images[i];
        image::imageops::overlay(&mut atlas, img, x as i64, y as i64);
        let (iw, ih) = img.dimensions();
        uvs.insert(
            name.clone(),
            PackedSprite {
                uv: [
                    x as f32 / ATLAS_W as f32,
                    y as f32 / atlas_h as f32,
                    (x + iw) as f32 / ATLAS_W as f32,
                    (y + ih) as f32 / atlas_h as f32,
                ],
                w: iw as f32,
                h: ih as f32,
            },
        );
    }
    (atlas, uvs)
}

// ── Breakable world objects ──────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Breakable {
    pos: [f32; 2],
    size: [f32; 2],
    depth: f32,
    name: &'static str,
    /// `Some((col, row))` for terrain tiles only — lets breaking one open an
    /// actual hole in the collision heightfield.
    terrain_cell: Option<(i32, i32)>,
}

struct Breaking {
    handle: SpriteHandle,
    target: Breakable,
    start: Instant,
    crack_handle: Option<SpriteHandle>,
    stage: u32,
}

struct HotbarSlot {
    name: &'static str,
    count: u32,
    handle: SpriteHandle,
    uv: [f32; 4],
    w: f32,
    h: f32,
}

fn place_prop(
    sprite_pass: &mut SpriteBatchPass,
    atlas: &HashMap<String, PackedSprite>,
    atlas_layer: u32,
    objects: &mut HashMap<SpriteHandle, Breakable>,
    name: &'static str,
    x: f32,
    depth: f32,
    flip: bool,
) -> PackedSprite {
    let s = atlas[name];
    let col = (x / TILE).round() as i32;
    let top = surface_top_world_y(col);
    let uv = if flip { flip_u(s.uv) } else { s.uv };
    let pos = [x, top + s.h * 0.5];
    let handle = sprite_pass.insert_sprite(
        SpriteInstance::new(pos, [s.w, s.h]).with_uv_rect(uv).with_depth(depth).with_atlas_layer(atlas_layer),
    );
    objects.insert(handle, Breakable { pos, size: [s.w, s.h], depth, name, terrain_cell: None });
    s
}

/// Lays `names` out left-to-right starting at `start_x`, each spaced by its
/// own real width plus `gap` — used for the mining/market rows so every
/// listed sprite appears exactly once, never overlapping its neighbor.
/// Returns the cursor position after the last item, for chaining rows.
fn lay_row(
    sprite_pass: &mut SpriteBatchPass,
    atlas: &HashMap<String, PackedSprite>,
    atlas_layer: u32,
    objects: &mut HashMap<SpriteHandle, Breakable>,
    names: &[&'static str],
    start_x: f32,
    depth: f32,
    gap: f32,
) -> f32 {
    let mut cursor = start_x;
    for &name in names {
        let s = atlas[name];
        cursor += s.w * 0.5;
        place_prop(sprite_pass, atlas, atlas_layer, objects, name, cursor, depth, false);
        cursor += s.w * 0.5 + gap;
    }
    cursor
}

/// Scatters instances of `names` across `[col_start, col_end)`, spaced by a
/// random step in `step`. Used *within* one themed zone (a forest band, a
/// mining band, a monster den) rather than across the whole world, so each
/// category stays visually grouped where it thematically belongs.
#[allow(clippy::too_many_arguments)]
fn scatter_band(
    sprite_pass: &mut SpriteBatchPass,
    atlas: &HashMap<String, PackedSprite>,
    atlas_layer: u32,
    objects: &mut HashMap<SpriteHandle, Breakable>,
    rng: &mut Rng,
    critters: &mut Vec<Critter>,
    items: &mut Vec<Item>,
    col_start: i32,
    col_end: i32,
    step: (i32, i32),
    names: &[&'static str],
    animated: Animated,
    depth: f32,
) {
    let mut col = col_start;
    loop {
        col += rng.range_i32(step.0, step.1);
        if col >= col_end {
            break;
        }
        let x = col as f32 * TILE + rng.range_i32(-6, 6) as f32;
        let name = names[rng.range_usize(names.len())];
        let s = atlas[name];
        let top = surface_top_world_y((x / TILE).round() as i32);
        let flip = rng.bool();
        let uv = if flip { flip_u(s.uv) } else { s.uv };
        match animated {
            Animated::None => {
                let pos = [x, top + s.h * 0.5];
                let handle = sprite_pass.insert_sprite(
                    SpriteInstance::new(pos, [s.w, s.h]).with_uv_rect(uv).with_depth(depth).with_atlas_layer(atlas_layer),
                );
                objects.insert(handle, Breakable { pos, size: [s.w, s.h], depth, name, terrain_cell: None });
            }
            Animated::Critter => {
                let base_pos = [x, top + s.h * 0.5];
                let handle = sprite_pass.insert_sprite(
                    SpriteInstance::new(base_pos, [s.w, s.h]).with_uv_rect(uv).with_depth(depth).with_atlas_layer(atlas_layer),
                );
                objects.insert(handle, Breakable { pos: base_pos, size: [s.w, s.h], depth, name, terrain_cell: None });
                critters.push(Critter { handle, base_pos, phase: rng.next_f32() * std::f32::consts::TAU, spr: PackedSprite { uv, ..s } });
            }
            Animated::Item => {
                let base_pos = [x, top + s.h * 0.5 + 10.0];
                let handle = sprite_pass.insert_sprite(
                    SpriteInstance::new(base_pos, [s.w, s.h]).with_uv_rect(s.uv).with_depth(depth).with_atlas_layer(atlas_layer),
                );
                objects.insert(handle, Breakable { pos: base_pos, size: [s.w, s.h], depth, name, terrain_cell: None });
                items.push(Item {
                    handle,
                    base_pos,
                    phase: rng.next_f32() * std::f32::consts::TAU,
                    spin: rng.range_i32(-100, 100) as f32 / 100.0,
                    spr: s,
                });
            }
        }
    }
}

// ── App scaffolding ──────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    state: Option<AppState>,
}
impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

struct Critter {
    handle: SpriteHandle,
    base_pos: [f32; 2],
    phase: f32,
    spr: PackedSprite,
}

struct Item {
    handle: SpriteHandle,
    base_pos: [f32; 2],
    phase: f32,
    spin: f32,
    spr: PackedSprite,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    graph: RenderGraph,
    scene: GpuScene,
    dummy_depth_view: wgpu::TextureView,

    atlas: HashMap<String, PackedSprite>,
    atlas_layer: u32,
    crack_uvs: [[f32; 4]; 3],

    player_handle: SpriteHandle,
    player_pos: [f32; 2],
    player_vel: [f32; 2],
    player_on_ground: bool,
    player_facing_right: bool,
    player_spr: PackedSprite,

    camera_center: [f32; 2],
    keys: HashSet<KeyCode>,
    mouse_pos: (f64, f64),

    critters: Vec<Critter>,
    items: Vec<Item>,

    objects: HashMap<SpriteHandle, Breakable>,
    broken_terrain: HashSet<(i32, i32)>,
    breaking: Option<Breaking>,
    hotbar: Vec<HotbarSlot>,
    hotbar_selected: usize,

    occupancy_buf: Arc<wgpu::Buffer>,
    occupancy_words: Vec<u32>,

    start_time: Instant,
    last_frame: Instant,
    fps_frames: u32,
    fps_last_print: Instant,
    window_size: (u32, u32),
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
                        .with_title("Helio – Sprite Sandbox/Mining Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let mut loaded = load_all_sprites();
        log::info!("[sprite_dig_demo] loaded {} embedded sprite files", loaded.len());
        loaded.push(("__white".to_string(), RgbaImage::from_pixel(4, 4, image::Rgba([255, 255, 255, 255]))));
        for stage in 1..=3u32 {
            loaded.push((format!("__crack_{stage}"), make_crack_image(stage)));
        }
        let (atlas_img, atlas) = pack_atlas(loaded);
        log::info!("[sprite_dig_demo] packed atlas: {}x{}", atlas_img.width(), atlas_img.height());
        let crack_uvs = [atlas["__crack_1"].uv, atlas["__crack_2"].uv, atlas["__crack_3"].uv];

        let mut graph = RenderGraph::new(&device, &queue);
        let mut sprite_pass = SpriteBatchPass::new(&device, &queue, format);
        sprite_pass.set_clear_color(Some(wgpu::Color { r: 0.53, g: 0.75, b: 0.95, a: 1.0 }));
        let atlas_layer =
            sprite_pass.add_atlas_layer(&device, &queue, atlas_img.width(), atlas_img.height(), atlas_img.as_raw());

        sprite_pass.reserve(&device, POOL_CAPACITY);
        let mut sprite_cull = SpriteCullPass::new(
            &device,
            &queue,
            sprite_pass.instances_buffer(),
            sprite_pass.alive_buffer(),
            POOL_CAPACITY as u32,
            POOL_CAPACITY as u32,
        );
        sprite_cull.set_view_rect([0.0, 0.0], [size.width as f32 * 0.5, size.height as f32 * 0.5]);
        sprite_pass.use_gpu_culling(sprite_cull.draw_order_buf.clone(), sprite_cull.indirect_buf.clone());

        let grass_uv = atlas["grass_b"].uv;
        let stone_uv = atlas["stone_block"].uv;
        let mut objects: HashMap<SpriteHandle, Breakable> = HashMap::new();
        let mut occupancy_words = vec![0u32; ((OCC_COLS * OCC_ROWS + 31) / 32) as usize];
        let mark_occupied = |pos: [f32; 2], words: &mut [u32]| {
            if let Some((c, r)) = occ_cell(pos) {
                let idx = occ_index(c, r);
                words[(idx / 32) as usize] |= 1 << (idx % 32);
            }
        };

        // ── Terrain: a heightfield of tiled sprite art (grass_b on the
        // surface row, stone_block for every row underneath), inserted once.
        // Every tile is breakable and tagged with its (col, row) cell so
        // mining it out actually opens a hole in the collision heightfield
        // *and* the radiance-cascades occupancy grid (real digging, real
        // light bouncing into the hole).
        for col in 0..WORLD_COLS {
            let top = surface_top_world_y(col);
            let jitter = 0.92 + hash01(col as u32 * 7 + 1) * 0.16;
            let pos = [col as f32 * TILE, top - TILE * 0.5];
            let handle = sprite_pass.insert_sprite(
                SpriteInstance::new(pos, [TILE + 1.0, TILE + 1.0])
                    .with_uv_rect(grass_uv)
                    .with_color([jitter, jitter, jitter, 1.0])
                    .with_atlas_layer(atlas_layer),
            );
            objects.insert(handle, Breakable { pos, size: [TILE, TILE], depth: 0.0, name: "grass_b", terrain_cell: Some((col, 0)) });
            mark_occupied(pos, &mut occupancy_words);
            for r in 1..=(DIRT_ROWS + STONE_ROWS) {
                let jitter = 0.9 + hash01(col as u32 * 17 + r as u32 * 53) * 0.2;
                let pos = [col as f32 * TILE, top - TILE * 0.5 - r as f32 * TILE];
                let handle = sprite_pass.insert_sprite(
                    SpriteInstance::new(pos, [TILE + 1.0, TILE + 1.0])
                        .with_uv_rect(stone_uv)
                        .with_color([jitter, jitter, jitter, 1.0])
                        .with_atlas_layer(atlas_layer),
                );
                objects.insert(handle, Breakable { pos, size: [TILE, TILE], depth: 0.0, name: "stone_block", terrain_cell: Some((col, r)) });
                mark_occupied(pos, &mut occupancy_words);
            }
        }

        let mut rng = Rng::new(0xC0FF_EE12_3456_7890);
        let mut critters: Vec<Critter> = Vec::new();
        let mut items: Vec<Item> = Vec::new();

        // ── Zone boundaries (columns) — each themed band gets its own
        // sprites, so the world reads as "a forest, then a village, then a
        // mine, then a monster den guarding treasure, then a second forest,
        // then a market" instead of one uniform scatter of everything.
        const SPAWN_END: i32 = 14;
        const FOREST_A_END: i32 = 46;
        const CABIN_COL: i32 = 48;
        const MINING_START: i32 = 84;
        const MINING_END: i32 = 108;
        const DEN_END: i32 = 132;
        const HUT_COL: i32 = 148;
        const FOREST_B_END: i32 = 166;
        const MARKET_START: i32 = 168;
        const TAIL_END: i32 = WORLD_COLS - 4;

        macro_rules! scatter {
            ($start:expr, $end:expr, $step:expr, $names:expr, $anim:expr, $depth:expr) => {
                scatter_band(
                    &mut sprite_pass, &atlas, atlas_layer, &mut objects, &mut rng, &mut critters, &mut items,
                    $start, $end, $step, $names, $anim, $depth,
                )
            };
        }

        // ── Spawn: the welcome sign, lightly decorated.
        place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "sign", 8.0 * TILE, 0.2, false);
        scatter!(2, SPAWN_END, (4, 7), FOREST_CLUTTER, Animated::None, 0.2);

        // ── Forest A: trees, undergrowth, a little wildlife.
        scatter!(SPAWN_END, FOREST_A_END, (2, 4), TREES, Animated::None, 0.2);
        scatter!(SPAWN_END, FOREST_A_END, (3, 6), FOREST_CLUTTER, Animated::None, 0.2);
        scatter!(SPAWN_END, FOREST_A_END, (10, 16), FOREST_CRITTERS, Animated::Critter, 0.3);

        // ── Village: the cabin, a second hut nearby, and two villager-posed
        // NPCs standing by the door — no furniture scattered in the open.
        let cabin_x = CABIN_COL as f32 * TILE;
        let cabin = place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "cabin", cabin_x, 0.15, false);
        let hut = place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "hut", cabin_x + cabin.w * 0.5 + 90.0, 0.2, false);
        lay_row(
            &mut sprite_pass, &atlas, atlas_layer, &mut objects, VILLAGE_LIGHTS,
            cabin_x + cabin.w * 0.5 + 90.0 + hut.w * 0.5 + 30.0, 0.2, 24.0,
        );
        let mut npc_cursor = cabin_x - cabin.w * 0.5 - 30.0;
        for &name in &["player_pickaxe", "player_pickaxe_hood"] {
            let s = atlas[name];
            npc_cursor -= s.w * 0.5;
            place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, name, npc_cursor, 0.3, false);
            npc_cursor -= s.w * 0.5 + 20.0;
        }

        // ── Mining camp: rocks, a blacksmith's tools/props, an ore vein, a
        // stash chest.
        scatter!(MINING_START, MINING_END, (3, 5), MINING_ROCKS, Animated::None, 0.2);
        scatter!(MINING_START, MINING_END, (6, 9), MINING_TOOLS, Animated::None, 0.2);
        let mining_center = (MINING_START + MINING_END) / 2;
        lay_row(&mut sprite_pass, &atlas, atlas_layer, &mut objects, MINING_PROPS, mining_center as f32 * TILE, 0.2, 16.0);
        place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "gold_ore", (MINING_START + 4) as f32 * TILE, 0.2, false);
        place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "gold_chest_small", (MINING_END - 3) as f32 * TILE, 0.2, false);

        // ── Monster den: guards clustered around the gold chest, with
        // weapons dropped by a less fortunate adventurer.
        place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "chest_gold", ((MINING_END + DEN_END) / 2) as f32 * TILE, 0.2, false);
        scatter!(MINING_END, DEN_END, (5, 8), DEN_MONSTERS, Animated::Critter, 0.3);
        scatter!(MINING_END, DEN_END, (9, 13), DEN_WEAPONS, Animated::None, 0.2);

        // ── Forest B: a second, wilder patch of woods around a shrine.
        scatter!(DEN_END, FOREST_B_END, (2, 4), TREES, Animated::None, 0.2);
        scatter!(DEN_END, FOREST_B_END, (3, 6), FOREST_CLUTTER, Animated::None, 0.2);
        scatter!(DEN_END, FOREST_B_END, (12, 18), FOREST_CRITTERS, Animated::Critter, 0.3);
        place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "hut", HUT_COL as f32 * TILE, 0.2, false);

        // ── Market stall: a small wood-and-stone stall structure with wares
        // laid out on the table, and the shopkeeper's dark storage chest.
        let stall_cursor = lay_row(&mut sprite_pass, &atlas, atlas_layer, &mut objects, MARKET_STALL, MARKET_START as f32 * TILE, 0.2, 10.0);
        scatter!(MARKET_START, MARKET_START + 10, (1, 3), MARKET_WARES, Animated::Item, 0.3);
        place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "chest_dark", stall_cursor + 20.0, 0.2, false);

        // ── Tail end of the world: the path continues, lit at intervals.
        scatter!(MARKET_START + 20, TAIL_END, (4, 7), FOREST_CLUTTER, Animated::None, 0.2);

        // ── Regular path lighting the whole way through, distinct from the
        // village's own torch variants — every ~22 columns.
        let mut torch_col = 20;
        while torch_col < TAIL_END {
            place_prop(&mut sprite_pass, &atlas, atlas_layer, &mut objects, "torch", torch_col as f32 * TILE, 0.2, false);
            torch_col += 22;
        }

        // ── Lighting: 2D radiance cascades reading the occupancy grid built
        // above (occluders) plus every placed torch/lantern/campfire (the
        // only actually-placed light emitters — see `LIGHT_EMITTER_NAMES`).
        let mut gpu_emitters: Vec<GpuEmitter> = objects
            .values()
            .filter(|b| LIGHT_EMITTER_NAMES.contains(&b.name))
            .map(|b| {
                let (color, radius) = emitter_style(b.name);
                GpuEmitter { pos: b.pos, radius, r: color[0], g: color[1], b: color[2], _pad: 0.0 }
            })
            .collect();
        let real_emitter_count = gpu_emitters.len() as u32;
        let max_emitters = real_emitter_count.max(1);
        gpu_emitters.resize(max_emitters as usize, GpuEmitter { pos: [0.0, 0.0], radius: 0.0, r: 0.0, g: 0.0, b: 0.0, _pad: 0.0 });
        log::info!("[sprite_dig_demo] {real_emitter_count} light emitters");

        let occupancy_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Occupancy Grid"),
            size: (occupancy_words.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&occupancy_buf, 0, bytemuck::cast_slice(&occupancy_words));
        let emitters_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Emitters"),
            size: (gpu_emitters.len() * std::mem::size_of::<GpuEmitter>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&emitters_buf, 0, bytemuck::cast_slice(&gpu_emitters));

        let mut radiance_pass = RadianceCascades2DPass::new(
            &device,
            &queue,
            RadianceCascadesConfig { max_emitters, ..Default::default() },
            occupancy_buf.clone(),
            (OCC_COLS, OCC_ROWS),
            TILE,
            OCC_ORIGIN,
            emitters_buf,
        );
        radiance_pass.set_emitter_count(real_emitter_count);
        let radiance_composite = RadianceCascadesCompositePass::new(
            &device,
            &queue,
            format,
            radiance_pass.radiance_view(),
            [0.12, 0.11, 0.16],
            1.6,
        );

        // ── Player, spawned standing on the surface near the sign. ────────
        let player_spr = atlas["player"];
        let spawn_col = 4;
        let player_pos = [spawn_col as f32 * TILE, surface_top_world_y(spawn_col) + player_spr.h * 0.5];
        let player_handle = sprite_pass.insert_sprite(
            SpriteInstance::new(player_pos, [player_spr.w, player_spr.h])
                .with_uv_rect(player_spr.uv)
                .with_depth(0.5)
                .with_atlas_layer(atlas_layer),
        );

        graph.add_pass(Box::new(sprite_cull));
        graph.add_pass(Box::new(sprite_pass));
        graph.add_pass(Box::new(radiance_pass));
        graph.add_pass(Box::new(radiance_composite));
        graph.lock(size.width.max(1), size.height.max(1));

        let scene = GpuScene::new(device.clone(), queue.clone());
        let dummy_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Depth (unused by 2D passes)"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth.create_view(&wgpu::TextureViewDescriptor::default());

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            graph,
            scene,
            dummy_depth_view,
            atlas,
            atlas_layer,
            crack_uvs,
            player_handle,
            player_pos,
            player_vel: [0.0, 0.0],
            player_on_ground: false,
            player_facing_right: true,
            player_spr,
            camera_center: player_pos,
            keys: HashSet::new(),
            mouse_pos: (0.0, 0.0),
            critters,
            items,
            objects,
            broken_terrain: HashSet::new(),
            breaking: None,
            hotbar: Vec::new(),
            hotbar_selected: 0,
            occupancy_buf,
            occupancy_words,
            start_time: Instant::now(),
            last_frame: Instant::now(),
            fps_frames: 0,
            fps_last_print: Instant::now(),
            window_size: (size.width.max(1), size.height.max(1)),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                state.graph.set_render_size(s.width, s.height);
                state.window_size = (s.width, s.height);
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    match key_event.state {
                        ElementState::Pressed => {
                            state.keys.insert(code);
                        }
                        ElementState::Released => {
                            state.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => match (button, btn_state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    let world = world_from_screen(state.mouse_pos, state.window_size, state.camera_center);
                    if let Some(handle) = hit_test(&state.objects, world) {
                        let target = state.objects[&handle];
                        state.breaking = Some(Breaking { handle, target, start: Instant::now(), crack_handle: None, stage: 0 });
                    }
                }
                (MouseButton::Left, ElementState::Released) => {
                    if let Some(b) = state.breaking.take() {
                        if let Some(ch) = b.crack_handle {
                            state
                                .graph
                                .find_pass_mut::<SpriteBatchPass>()
                                .expect("sprite batch pass missing from graph")
                                .remove_sprite(ch);
                        }
                    }
                }
                (MouseButton::Right, ElementState::Pressed) => {
                    if !state.hotbar.is_empty() {
                        let world = world_from_screen(state.mouse_pos, state.window_size, state.camera_center);
                        let sel = state.hotbar_selected.min(state.hotbar.len() - 1);
                        let (name, uv, w, h) = {
                            let s = &state.hotbar[sel];
                            (s.name, s.uv, s.w, s.h)
                        };
                        let snapped = [(world[0] / TILE).round() * TILE, world[1]];
                        let atlas_layer = state.atlas_layer;
                        let sprite_pass = state
                            .graph
                            .find_pass_mut::<SpriteBatchPass>()
                            .expect("sprite batch pass missing from graph");
                        let handle = sprite_pass.insert_sprite(
                            SpriteInstance::new(snapped, [w, h]).with_uv_rect(uv).with_depth(0.2).with_atlas_layer(atlas_layer),
                        );
                        state.objects.insert(handle, Breakable { pos: snapped, size: [w, h], depth: 0.2, name, terrain_cell: None });
                        state.hotbar[sel].count -= 1;
                        if state.hotbar[sel].count == 0 {
                            let removed = state.hotbar.remove(sel);
                            sprite_pass.remove_sprite(removed.handle);
                            if state.hotbar_selected >= state.hotbar.len() {
                                state.hotbar_selected = state.hotbar.len().saturating_sub(1);
                            }
                        }
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if !state.hotbar.is_empty() {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.02,
                    };
                    if dy.abs() > 0.01 {
                        let n = state.hotbar.len() as i32;
                        let dir = if dy > 0.0 { -1 } else { 1 };
                        state.hotbar_selected = (state.hotbar_selected as i32 + dir).rem_euclid(n) as usize;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                let time = state.start_time.elapsed().as_secs_f32();

                let sprite_pass =
                    state.graph.find_pass_mut::<SpriteBatchPass>().expect("sprite batch pass missing from graph");
                let atlas_layer = state.atlas_layer;

                // ── Mining: advance the crack overlay, or finalize a break.
                let mut should_finish = false;
                if let Some(breaking) = state.breaking.as_mut() {
                    let elapsed = breaking.start.elapsed().as_secs_f32();
                    let stage = ((elapsed / BREAK_STAGE_DURATION).floor() as u32).min(BREAK_TOTAL_STAGES);
                    if stage != breaking.stage {
                        breaking.stage = stage;
                        if stage >= 1 {
                            let crack_uv = state.crack_uvs[(stage - 1) as usize];
                            let inst = SpriteInstance::new(breaking.target.pos, breaking.target.size)
                                .with_uv_rect(crack_uv)
                                .with_depth(breaking.target.depth + 0.01)
                                .with_atlas_layer(atlas_layer);
                            match breaking.crack_handle {
                                Some(ch) => sprite_pass.update_sprite(ch, inst),
                                None => breaking.crack_handle = Some(sprite_pass.insert_sprite(inst)),
                            }
                        }
                    }
                    should_finish = stage >= BREAK_TOTAL_STAGES;
                }
                if should_finish {
                    let breaking = state.breaking.take().unwrap();
                    sprite_pass.remove_sprite(breaking.handle);
                    if let Some(ch) = breaking.crack_handle {
                        sprite_pass.remove_sprite(ch);
                    }
                    state.objects.remove(&breaking.handle);
                    state.critters.retain(|c| c.handle != breaking.handle);
                    state.items.retain(|it| it.handle != breaking.handle);
                    if let Some(cell) = breaking.target.terrain_cell {
                        state.broken_terrain.insert(cell);
                        // Clear the same tile in the lighting occupancy grid
                        // (a flat world-space grid, not `terrain_cell`'s
                        // column-relative one — see `occ_cell`) so light can
                        // actually pour into the hole just dug.
                        if let Some((c, r)) = occ_cell(breaking.target.pos) {
                            let idx = occ_index(c, r);
                            let word = (idx / 32) as usize;
                            state.occupancy_words[word] &= !(1 << (idx % 32));
                            state.queue.write_buffer(&state.occupancy_buf, (word * 4) as u64, bytemuck::bytes_of(&state.occupancy_words[word]));
                        }
                    }
                    if let Some(slot) = state.hotbar.iter_mut().find(|s| s.name == breaking.target.name) {
                        slot.count += 1;
                    } else {
                        let s = state.atlas[breaking.target.name];
                        let index = state.hotbar.len();
                        let pos = hotbar_slot_world_pos(state.camera_center, state.window_size, index, index + 1);
                        let handle = sprite_pass.insert_sprite(
                            SpriteInstance::new(pos, [HOTBAR_ICON_SIZE, HOTBAR_ICON_SIZE])
                                .with_uv_rect(s.uv)
                                .with_depth(0.9)
                                .with_atlas_layer(atlas_layer),
                        );
                        state.hotbar.push(HotbarSlot { name: breaking.target.name, count: 1, handle, uv: s.uv, w: s.w, h: s.h });
                    }
                }

                // ── Player physics: simple gravity + blocky heightfield
                // collision (snap to the nearest terrain column's surface,
                // accounting for any tiles mined out from under it).
                let mut move_dir = 0.0f32;
                if state.keys.contains(&KeyCode::KeyA) || state.keys.contains(&KeyCode::ArrowLeft) {
                    move_dir -= 1.0;
                }
                if state.keys.contains(&KeyCode::KeyD) || state.keys.contains(&KeyCode::ArrowRight) {
                    move_dir += 1.0;
                }
                if move_dir != 0.0 {
                    state.player_facing_right = move_dir > 0.0;
                }
                state.player_vel[0] = move_dir * MOVE_SPEED;
                state.player_vel[1] += GRAVITY * dt;
                if state.keys.contains(&KeyCode::Space) && state.player_on_ground {
                    state.player_vel[1] = JUMP_VEL;
                    state.player_on_ground = false;
                }
                state.player_pos[0] += state.player_vel[0] * dt;
                state.player_pos[1] += state.player_vel[1] * dt;
                state.player_pos[0] = state.player_pos[0].clamp(TILE * 1.0, (WORLD_COLS as f32 - 2.0) * TILE);

                let col = (state.player_pos[0] / TILE).round() as i32;
                let ground_y = ground_y_at(col, &state.broken_terrain) + state.player_spr.h * 0.5;
                if state.player_pos[1] <= ground_y {
                    state.player_pos[1] = ground_y;
                    state.player_vel[1] = 0.0;
                    state.player_on_ground = true;
                } else {
                    state.player_on_ground = false;
                }

                let uv = if state.player_facing_right { state.player_spr.uv } else { flip_u(state.player_spr.uv) };
                let player_pos = state.player_pos;
                sprite_pass.update_sprite(
                    state.player_handle,
                    SpriteInstance::new(player_pos, [state.player_spr.w, state.player_spr.h])
                        .with_uv_rect(uv)
                        .with_depth(0.5)
                        .with_atlas_layer(atlas_layer),
                );

                for c in &state.critters {
                    let mut pos = c.base_pos;
                    pos[1] += (time * 2.0 + c.phase).sin() * 6.0;
                    sprite_pass.update_sprite(
                        c.handle,
                        SpriteInstance::new(pos, [c.spr.w, c.spr.h])
                            .with_uv_rect(c.spr.uv)
                            .with_depth(0.3)
                            .with_atlas_layer(atlas_layer),
                    );
                }
                for it in &state.items {
                    let mut pos = it.base_pos;
                    pos[1] += (time * 1.5 + it.phase).sin() * 4.0;
                    sprite_pass.update_sprite(
                        it.handle,
                        SpriteInstance::new(pos, [it.spr.w, it.spr.h])
                            .with_uv_rect(it.spr.uv)
                            .with_rotation(time * it.spin + it.phase)
                            .with_depth(0.3)
                            .with_atlas_layer(atlas_layer),
                    );
                }

                // ── Camera: smoothly follows the player, biased up a bit so
                // more sky/foreground is visible ahead of the player.
                let target = [state.player_pos[0], state.player_pos[1] + 100.0];
                let smoothing = (dt * 5.0).min(1.0);
                state.camera_center[0] += (target[0] - state.camera_center[0]) * smoothing;
                state.camera_center[1] += (target[1] - state.camera_center[1]) * smoothing;

                // ── Hotbar: screen-locked (re-anchored to the camera every
                // frame), the selected slot tinted — no on-screen counts,
                // just internal bookkeeping per the design.
                let n = state.hotbar.len();
                for (i, slot) in state.hotbar.iter().enumerate() {
                    let pos = hotbar_slot_world_pos(state.camera_center, state.window_size, i, n);
                    let tint = if i == state.hotbar_selected { [1.35, 1.25, 0.55, 1.0] } else { [1.0, 1.0, 1.0, 1.0] };
                    sprite_pass.update_sprite(
                        slot.handle,
                        SpriteInstance::new(pos, [HOTBAR_ICON_SIZE, HOTBAR_ICON_SIZE])
                            .with_uv_rect(slot.uv)
                            .with_color(tint)
                            .with_depth(0.9)
                            .with_atlas_layer(atlas_layer),
                    );
                }

                sprite_pass.set_camera(state.camera_center, None);
                let (win_w, win_h) = state.window_size;
                state
                    .graph
                    .find_pass_mut::<SpriteCullPass>()
                    .expect("sprite cull pass missing from graph")
                    .set_view_rect(state.camera_center, [win_w as f32 * 0.5, win_h as f32 * 0.5]);
                state
                    .graph
                    .find_pass_mut::<RadianceCascades2DPass>()
                    .expect("radiance cascades pass missing from graph")
                    .set_view(state.camera_center, [win_w as f32 * 0.5, win_h as f32 * 0.5]);

                state.fps_frames += 1;
                if state.fps_last_print.elapsed().as_secs_f32() >= 1.0 {
                    let elapsed = state.fps_last_print.elapsed().as_secs_f32();
                    log::info!(
                        "[sprite_dig_demo] {:.0} fps | pool={} | hotbar_slots={}",
                        state.fps_frames as f32 / elapsed,
                        POOL_CAPACITY,
                        state.hotbar.len(),
                    );
                    state.fps_frames = 0;
                    state.fps_last_print = Instant::now();
                }

                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => {
                        state.window.request_redraw();
                        return;
                    }
                };
                let view = output.texture.create_view(&Default::default());
                if let Err(e) = state.graph.execute(&state.scene, &view, &state.dummy_depth_view) {
                    log::error!("graph execute error: {e:?}");
                }
                state.queue.present(output);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}
