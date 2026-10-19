//! The game: its state and one frame of it.

use crate::blocks::Block;
use crate::config::*;
use crate::gfx::{Gfx, View};
use crate::input::Input;
use crate::player::interaction::Interaction;
use crate::player::Player;
use crate::terrain::{self, Biome};
use crate::world::World;
use glam::DVec3;
use helio_pass_voxel_planet::{Cell, Planet};
use std::time::Instant;
use winit::keyboard::KeyCode;

pub struct Game {
    pub gfx: Gfx,
    pub world: World,
    pub player: Player,
    pub interaction: Interaction,
    pub input: Input,
    last_frame: Instant,
    started: Instant,
    last_title: Instant,
    frames: u32,
    fps: f64,
    /// Seconds of game time, and its speed relative to the clock.
    time: f64,
    time_speed: f64,
    /// The region the world was last told the player is in.
    /// The sun's path: a great circle through the sky over the spawn.
    sun_axes: (DVec3, DVec3),
    show_help: bool,
    total_frames: u64,
    tour: Option<Tour>,
    pub quit: bool,
    luma: bool,
    walk: bool,
    edit_started: Option<Instant>,
    edit_two: bool,
    pending_places: Vec<Vec<crate::world::edits::BlockEdit>>,
}

/// A grassy, level place to start on.
fn find_spawn(planet: &Planet) -> DVec3 {
    let grid = planet.grid();
    let face = grid.faces()[0];
    let n = grid.cells();
    let (ci, cj) = (n / 2, n / 2);
    let mut fallback = None;
    // Rings outward from the middle of the face.
    for ring in 0..400 {
        for step in 0..(8 * ring.max(1)) {
            let t = f64::from(step) / f64::from(8 * ring.max(1)) * std::f64::consts::TAU;
            let (i, j) = (ci + (t.cos() * f64::from(ring) * 12.0) as i32, cj + (t.sin() * f64::from(ring) * 12.0) as i32);
            let Some(ground) = terrain::ground(planet, face, i, j) else { continue };
            let top = planet.column_top(face, i, j, 0);
            let height = f64::from(top) * grid.voxel_size();
            if ground.cell.k + 1 != top || terrain::relief(planet, face, i, j, 3) > 2 {
                continue;
            }
            fallback.get_or_insert((i, j, top));
            if Biome::of(ground.material) == Biome::Temperate && (4.0..40.0).contains(&height) {
                return grid.at_radial(grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5), grid.layer_radius(f64::from(top)) + 0.01);
            }
        }
    }
    let (i, j, top) = fallback.unwrap_or((ci, cj, 8));
    grid.at_radial(grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5), grid.layer_radius(f64::from(top)) + 0.01)
}

impl Game {
    pub fn new(gfx: impl FnOnce(&Planet) -> Gfx) -> Self {
        let started = Instant::now();
        let planet = Planet::new(terrain::recipe::recipe()).expect("the planet's recipe is valid");
        log::info!("planet built in {:.2} s", started.elapsed().as_secs_f64());
        let spawn = find_spawn(&planet);
        let gfx = gfx(&planet);
        let up = spawn.normalize();
        let e1 = up.any_orthonormal_vector();
        let e2 = up.cross(e1);
        let world = World::new(planet);
        let mut player = Player::new(spawn);
        player.heading = e2;
        let tour_active = std::env::var_os("VOXEL_CRAFT_SHOTS").is_some();
        Self {
            gfx,
            world,
            player,
            interaction: Interaction::default(),
            input: Input::default(),
            last_frame: Instant::now(),
            started,
            last_title: Instant::now(),
            frames: 0,
            fps: 0.0,
            // Mid-morning.
            time: DAY_SECONDS * 0.08,
            time_speed: if tour_active || std::env::var_os("VOXEL_CRAFT_FREEZE").is_some() { 0.0 } else { 1.0 },
            sun_axes: (up, e1),
            show_help: true,
            total_frames: 0,
            tour: std::env::var("VOXEL_CRAFT_SHOTS").ok().map(Tour::new),
            // A tour keeps the sun where it is: shots compare.
            // (set below)
            quit: false,
            luma: std::env::var_os("VOXEL_CRAFT_LUMA").is_some(),
            walk: std::env::var_os("VOXEL_CRAFT_WALK").is_some(),
            edit_started: None,
            edit_two: false,
            pending_places: Vec::new(),
        }
    }

    /// The direction to the sun at game time `time`: a great circle that
    /// passes over the spawn at noon.
    fn sun(&self) -> DVec3 {
        let a = self.time / DAY_SECONDS * std::f64::consts::TAU + 0.55;
        let (up, side) = self.sun_axes;
        (up * a.sin() + side * a.cos()).normalize()
    }

    /// Runs one frame.
    pub fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().min(0.1);
        self.last_frame = now;
        if dt > 0.05 {
            log::info!("HITCH {:.0} ms at frame {}", dt * 1e3, self.total_frames);
        }
        self.time += dt * self.time_speed;

        // Keys that are not movement.
        if self.input.just_pressed(KeyCode::KeyF) {
            self.player.flying = !self.player.flying;
            self.player.velocity = DVec3::ZERO;
        }
        if self.input.just_pressed(KeyCode::Space) && self.player.jump_pressed() {
            self.player.flying = !self.player.flying;
            self.player.velocity = DVec3::ZERO;
        }
        if self.input.just_pressed(KeyCode::F5) || self.input.just_pressed(KeyCode::KeyV) {
            self.player.third_person = !self.player.third_person;
        }
        if self.input.just_pressed(KeyCode::KeyR) {
            self.player.respawn();
        }
        if self.input.just_pressed(KeyCode::KeyH) {
            self.show_help = !self.show_help;
        }
        if self.input.just_pressed(KeyCode::KeyT) {
            self.time_speed = if self.time_speed > 1.5 { 1.0 } else { 40.0 };
        }
        if let Some(slot) = self.input.hotbar_key() {
            self.player.hotbar = slot;
        }
        let scroll = self.input.scroll();
        if scroll != 0.0 {
            let n = Block::HOTBAR.len() as i32;
            self.player.hotbar = (self.player.hotbar as i32 - scroll.signum() as i32).rem_euclid(n) as usize;
        }

        let mut controls = self.input.controls();
        // `VOXEL_CRAFT_WALK=1`: walk forward by itself and log the eye's height each frame.
        if self.walk || std::env::var_os("VOXEL_CRAFT_EYELOG").is_some() {
            controls.forward = if self.walk { 1.0 } else { 0.0 };
            log::info!("WALK {:.5} {:.3} ground {} rev {} pos {:.4}", self.world.planet().grid().height(self.player.eye()), dt * 1e3, self.player.on_ground, self.world.revision(), self.player.eye().length());
        }
        self.world.update();
        self.player.update(&self.world, &controls, dt);
        self.interaction.update(&mut self.player, &mut self.world, &controls, dt);

        self.total_frames += 1;
        // `VOXEL_CRAFT_REPLAY="place:face:i:j:k:Block_Name;break:face:i:j:k;..."` (as printed by
        // the game): stand near the first edit, wait for the world to be decorated
        // (`VOXEL_CRAFT_WAIT` frames), apply them one at a time like a player, and look.
        if let Ok(script) = std::env::var("VOXEL_CRAFT_REPLAY") {
            let steps: Vec<&str> = script.split(';').filter(|s| !s.is_empty()).collect();
            let parse = |s: &str| -> Option<(bool, Cell, Option<Block>)> {
                let f: Vec<&str> = s.split(':').collect();
                let place = f.first()? == &"place";
                let cell = Cell::new(f.get(1)?.parse().ok()?, f.get(2)?.parse().ok()?, f.get(3)?.parse().ok()?, f.get(4)?.parse().ok()?);
                let block = if place { let name = f.get(5)?.to_string(); Block::ALL.iter().copied().find(|b| b.name().replace(' ', "_") == name) } else { None };
                Some((place, cell, block))
            };
            let start: u64 = std::env::var("VOXEL_CRAFT_REPLAY_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
            if self.total_frames == start {
                if let Some((_, cell, _)) = steps.first().and_then(|s| parse(s)) {
                    let up = self.world.cell_centre(cell).normalize();
                    let side = up.any_orthonormal_vector();
                    self.player.flying = true;
                    self.player.position = self.world.cell_centre(cell) + side * 5.0 + up * 3.5 - up * 1.62;
                    self.player.heading = -side;
                    self.player.pitch = -0.4;
                    self.player.velocity = DVec3::ZERO;
                    for dj in -6..=2 {
                        let row: Vec<String> = (-8..=8).map(|di| {
                            let c = Cell::new(cell.face, cell.i + di, cell.j + dj, cell.k);
                            self.world.block_at(c).map(|b| b.name().chars().take(2).collect::<String>()).unwrap_or("..".into())
                        }).collect();
                        println!("CPU j{:+} {}", dj, row.join(" "));
                    }
                }
            }
            if self.total_frames > start && (self.total_frames - start) % 90 == 0 {
                let n = ((self.total_frames - start) / 90 - 1) as usize;
                if let Some((place, cell, block)) = steps.get(n).and_then(|s| parse(s)) {
                    let edit = if place { crate::world::edits::BlockEdit::set(cell, block.unwrap_or(Block::Cobblestone)) } else { crate::world::edits::BlockEdit::clear(cell) };
                    self.world.edit(&[edit]);
                    println!("REPLAY {n}: {}", steps[n]);
                }
            }
        }
        // `VOXEL_CRAFT_EDITTEST=1`: place five different blocks in a row, time how long until
        // the world has them, and look at them.
        if std::env::var_os("VOXEL_CRAFT_EDITTEST").is_some() {
            if self.total_frames == 700 {
                let here = self.world.cell_at(self.player.position);
                if std::env::var("VOXEL_CRAFT_EDITTEST").as_deref() == Ok("4") {
                    // Dig pits in the ground, then place blocks in them and against their walls, in two steps like a player.
                    let mut dig = Vec::new();
                    for a in -3..=3 {
                        let (i, j) = (here.i + a * 4, here.j + 6);
                        let top = self.world.planet().column_top(here.face, i, j, 0);
                        for d in 1..=2 {
                            dig.push(crate::world::edits::BlockEdit::clear(Cell::new(here.face, i, j, top - d)));
                        }
                    }
                    self.world.edit(&dig);
                    self.pending_places = (-3..=3).map(|a| {
                        let (i, j) = (here.i + a * 4, here.j + 6);
                        let top = self.world.planet().column_top(here.face, i, j, 0);
                        let types = [Block::Glowstone, Block::OakPlanks, Block::Bricks, Block::Sand, Block::Cobblestone, Block::Snow, Block::StoneBricks];
                        vec![crate::world::edits::BlockEdit::set(Cell::new(here.face, i, j, top - 2), types[(a + 3) as usize]), crate::world::edits::BlockEdit::set(Cell::new(here.face, i + 1, j, top - 1), types[(a + 4).rem_euclid(7) as usize])]
                    }).collect();
                    let c = Cell::new(here.face, here.i, here.j + 11, here.k + 9);
                    let along = self.world.cell_centre(Cell::new(here.face, here.i, here.j + 1, here.k)) - self.world.cell_centre(here);
                    self.player.flying = true;
                    self.player.position = self.world.cell_centre(c);
                    self.player.heading = along.normalize();
                    self.player.pitch = -0.9;
                    self.player.velocity = DVec3::ZERO;
                    self.edit_two = true;
                } else if std::env::var("VOXEL_CRAFT_EDITTEST").as_deref() == Ok("3") {
                    // A grid of different blocks on the ground wherever it is, seen from above.
                    let types = [Block::Glowstone, Block::OakPlanks, Block::Bricks, Block::Sand, Block::Cobblestone, Block::Snow];
                    let mut edits = Vec::new();
                    for a in -4..=4 {
                        for b in 1..=8 {
                            let (i, j) = (here.i + a * 3, here.j + b * 3);
                            let k = self.world.planet().column_top(here.face, i, j, 0);
                            edits.push(crate::world::edits::BlockEdit::set(Cell::new(here.face, i, j, k), types[((a + 4 + b) as usize) % types.len()]));
                        }
                    }
                    self.world.edit(&edits);
                    let c = Cell::new(here.face, here.i, here.j + 14, here.k + 14);
                    let along = self.world.cell_centre(Cell::new(here.face, here.i, here.j + 1, here.k)) - self.world.cell_centre(here);
                    self.player.flying = true;
                    self.player.position = self.world.cell_centre(c);
                    self.player.heading = along.normalize();
                    self.player.pitch = -1.0;
                    self.player.velocity = DVec3::ZERO;
                    self.edit_started = Some(Instant::now());
                    self.edit_two = true;
                } else if std::env::var("VOXEL_CRAFT_EDITTEST").as_deref() == Ok("2") {
                    let c = |di: i32, dk: i32| Cell::new(here.face, here.i + di, here.j + 3, here.k + dk);
                    let edits = vec![
                        crate::world::edits::BlockEdit::set(c(0, 0), Block::Glowstone),
                        crate::world::edits::BlockEdit::set(c(3, 0), Block::Cobblestone),
                        crate::world::edits::BlockEdit::set(c(3, 1), Block::Cobblestone),
                        crate::world::edits::BlockEdit::set(c(3, 2), Block::Cobblestone),
                        crate::world::edits::BlockEdit::set(c(-3, 0), Block::Sand),
                        crate::world::edits::BlockEdit::set(c(-4, 0), Block::Bricks),
                    ];
                    self.world.edit(&edits);
                    let along = self.world.cell_centre(Cell::new(here.face, here.i, here.j + 1, here.k)) - self.world.cell_centre(here);
                    self.player.heading = along.normalize();
                    self.player.pitch = -0.35;
                    self.edit_started = Some(Instant::now());
                    self.edit_two = true;
                }
                if !self.edit_two {
                let blocks = [Block::Glowstone, Block::OakPlanks, Block::StoneBricks, Block::Bricks, Block::Sand];
                let edits: Vec<_> = blocks.iter().enumerate().map(|(n, b)| crate::world::edits::BlockEdit::set(Cell::new(here.face, here.i + n as i32 * 2 - 4, here.j + 5, here.k), *b)).collect();
                self.world.edit(&edits);
                self.edit_started = Some(Instant::now());
                let along = self.world.cell_centre(Cell::new(here.face, here.i, here.j + 1, here.k)) - self.world.cell_centre(here);
                self.player.heading = along.normalize();
                self.player.pitch = -0.25;
                }
            }
            if self.total_frames > 700 && self.total_frames % 20 == 0 {
                if let Some(next) = self.pending_places.pop() {
                    self.world.edit(&next);
                }
            }
            if let Some(t) = self.edit_started {
                if self.world.pending_edits() == 0 {
                    log::info!("EDIT in world after {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
                    self.edit_started = None;
                }
            }
        }
        // `VOXEL_CRAFT_PALETTE=1`: every block in a row, for checking the textures.
        if std::env::var_os("VOXEL_CRAFT_PALETTE").is_some() && self.tour.is_some() {
            if self.total_frames == 300 {
                let here = self.world.cell_at(self.player.position);
                let edits: Vec<_> = Block::ALL.iter().enumerate().map(|(n, b)| crate::world::edits::BlockEdit::set(Cell::new(here.face, here.i + (n as i32 % 8) * 2 - 7, here.j + 12, here.k + 60 + (n as i32 / 8) * 3), *b)).collect();
                self.world.edit(&edits);
                let start = Cell::new(here.face, here.i, here.j + 12, here.k + 63);
                let along = self.world.cell_centre(Cell::new(start.face, start.i, start.j + 1, start.k)) - self.world.cell_centre(start);
                self.player.flying = true;
                self.player.position = self.world.cell_centre(Cell::new(start.face, start.i, start.j - 12, start.k)) - self.player.position.normalize() * 1.62;
                self.player.heading = along.normalize();
                self.player.pitch = 0.0;
                self.player.velocity = DVec3::ZERO;
            }
        }
        if let Some(t) = &self.tour {
            if t.trunk && self.total_frames == 700 {
                let here = self.world.cell_at(self.player.position);
                match nearest_wood(&self.world, here, 45) {
                    Some((cell, block)) => {
                        log::info!("TRUNK {:?} at {:?} is {:?}, drawn material {}", block, cell, self.world.block_at(cell), block.material());
                        let target = self.world.cell_centre(cell);
                        let up = target.normalize();
                        let along = (self.world.cell_centre(Cell::new(cell.face, cell.i, cell.j + 1, cell.k)) - target).normalize();
                        for back in 1..=3 {
                            let c = Cell::new(cell.face, cell.i, cell.j - back, cell.k);
                            log::info!("TRUNK in front: {:?} -> {:?}", c, self.world.block_at(c));
                        }
                        self.player.flying = true;
                        self.player.position = target - along * 4.0 - up * 1.12;
                        self.player.heading = along;
                        self.player.pitch = 0.0;
                        self.player.velocity = DVec3::ZERO;
                    }
                    None => log::info!("TRUNK none found"),
                }
            }
        }
        let mut shot = self.tour.as_mut().and_then(|t| t.step(&mut self.player, self.gfx.settled(), &mut self.quit));
        // `VOXEL_CRAFT_REPLAY_SHOTS=<dir>`: a shot 60 frames after each replay step (and before the first).
        if let (Ok(dir), Ok(_)) = (std::env::var("VOXEL_CRAFT_REPLAY_SHOTS"), std::env::var("VOXEL_CRAFT_REPLAY")) {
            let start: u64 = std::env::var("VOXEL_CRAFT_REPLAY_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
            if self.total_frames >= start + 30 && (self.total_frames - start) % 90 == 60 && (self.total_frames - start) / 90 <= 3 {
                let _ = std::fs::create_dir_all(&dir);
                shot = Some(format!("{dir}/step_{:02}.png", (self.total_frames - start) / 90));
            }
        }
        let (eye, look) = self.player.camera(&self.world);
        let outline = self.interaction.outline(&self.world);
        let view = View {
            planet: self.world.planet(),
            player: &self.player,
            eye,
            look,
            sun: self.sun(),
            outline: &outline,
            time: self.started.elapsed().as_secs_f64(),
            screenshot: shot.as_deref(),
            luma: self.luma,
        };
        self.gfx.draw(&view);
        self.input.end_frame();

        self.frames += 1;
        let since = self.last_title.elapsed().as_secs_f64();
        if since > 0.5 {
            self.fps = f64::from(self.frames) / since;
            self.frames = 0;
            self.last_title = Instant::now();
            self.update_title();
        }
    }

    fn update_title(&self) {
        let grid = self.world.planet().grid();
        let height = grid.height(self.player.position);
        let hotbar: Vec<String> = Block::HOTBAR
            .iter()
            .enumerate()
            .map(|(n, b)| if n == self.player.hotbar { format!("[{} {}]", n + 1, b.name()) } else { format!("{} {}", n + 1, b.name()) })
            .collect();
        let hour = (self.time / DAY_SECONDS * 24.0 + 9.0).rem_euclid(24.0);
        let help = if self.show_help {
            "  |  WASD move, Space jump (tap twice or F: fly), Ctrl sprint, LMB break, RMB place, MMB pick, wheel/1-9 hotbar, V camera, T time, R respawn, H help, Esc free mouse"
        } else {
            ""
        };
        let target = self.interaction.target.and_then(|t| self.world.block_at(t.cell)).map_or(String::new(), |b| format!("  |  looking at {}", b.name()));
        self.gfx.window.set_title(&format!(
            "Voxel Craft  |  {:.0} fps  |  {:02}:{:02}  |  height {:.0} m  |  {}{}{}{}",
            self.fps,
            hour as u32,
            ((hour.fract()) * 60.0) as u32,
            height,
            hotbar.join("  "),
            target,
            if self.player.flying { "  |  FLYING" } else { "" },
            help
        ));
    }
}

/// `VOXEL_CRAFT_SHOTS=<dir>`: a scripted tour that poses the camera, waits for
/// the terrain to settle, saves `<dir>/shot_N.png` for each pose and quits.
/// For comparing renders without a person at the keyboard.
struct Tour {
    dir: String,
    pose: usize,
    /// Frames since the pose was set.
    waited: u32,
    /// Look at the nearest tree trunk instead of the posed views.
    trunk: bool,
}

/// (turn about up in degrees, pitch, third person, metres above the start).
const POSES: &[(f64, f64, bool, f64)] = &[
    (0.0, -0.08, false, 0.0),
    (90.0, 0.0, false, 0.0),
    (90.0, 0.55, false, 0.0),
    (90.0, -0.75, false, 0.0),
    (0.0, -0.2, true, 0.0),
    (90.0, -0.35, true, 0.0),
    (0.0, -0.55, false, 45.0),
    (180.0, 0.05, false, 0.0),
    // 450 m up, looking out: forests to the horizon (their far canopies).
    (30.0, -0.22, false, 400.0),
];

impl Tour {
    fn new(dir: String) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self { dir, pose: 0, waited: u32::MAX, trunk: std::env::var_os("VOXEL_CRAFT_TRUNK").is_some() }
    }

    /// Advances the tour by a frame; the path to save this frame to, if it
    /// should be.
    fn step(&mut self, player: &mut Player, settled: bool, quit: &mut bool) -> Option<String> {
        if self.pose >= usize::MAX / 2 {
            return None;
        }
        if self.waited == u32::MAX {
            self.set_pose(player);
        }
        self.waited += 1;
        // A pose is shot once the terrain has settled and the temporal
        // anti-aliasing has had frames to converge (or after a long wait).
        let minimum: u32 = std::env::var("VOXEL_CRAFT_WAIT").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let ready = (settled && self.waited >= minimum) || self.waited >= minimum.max(900);
        if !ready {
            return None;
        }
        let path = format!("{}/shot_{:02}.png", self.dir, self.pose);
        log::info!("shot {} at {} frames, settled {settled}", self.pose, self.waited);
        self.pose += 1;
        let limit = std::env::var("VOXEL_CRAFT_POSES").ok().and_then(|v| v.parse().ok()).unwrap_or(POSES.len());
        if self.pose >= limit.min(POSES.len()) {
            // `VOXEL_CRAFT_HOLD=1` keeps the last pose running (for LUMA / DIFF).
            *quit = std::env::var_os("VOXEL_CRAFT_HOLD").is_none();
            self.pose = usize::MAX / 2;
        } else {
            self.set_pose(player);
        }
        Some(path)
    }

    fn set_pose(&mut self, player: &mut Player) {
        let (turn, pitch, third, lift) = POSES[self.pose];
        let _ = (turn, pitch, third, lift);
        let frame = player.frame();
        let (s, c) = turn.to_radians().sin_cos();
        player.heading = frame.forward * c + frame.right * s;
        player.pitch = pitch;
        player.third_person = third;
        player.flying = lift > 0.0;
        if lift > 0.0 && self.pose > 0 {
            player.position += frame.up * lift;
        }
        player.velocity = DVec3::ZERO;
        self.waited = 0;
    }
}

/// The nearest wood block within `reach` blocks of `from`, as (cell, block).
fn nearest_wood(world: &World, from: Cell, reach: i32) -> Option<(Cell, Block)> {
    let logs = [Block::OakLog, Block::BirchLog, Block::SpruceLog, Block::JungleLog];
    let mut best: Option<(i32, Cell, Block)> = None;
    for di in -reach..=reach {
        for dj in -reach..=reach {
            for dk in 0..16 {
                let cell = Cell::new(from.face, from.i + di, from.j + dj, from.k + dk);
                if let Some(block) = world.block_at(cell).filter(|b| logs.contains(b)) {
                    let d = di * di + dj * dj + dk * dk;
                    if best.is_none_or(|(bd, _, _)| d < bd) {
                        best = Some((d, cell, block));
                    }
                }
            }
        }
    }
    best.map(|(_, c, b)| (c, b))
}
