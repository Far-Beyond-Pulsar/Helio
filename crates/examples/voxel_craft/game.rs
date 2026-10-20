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
        let tour = std::env::var("VOXEL_CRAFT_SHOTS").ok().map(Tour::new);
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
            // A tour keeps the sun where it is: shots compare.
            time_speed: if tour.is_some() { 0.0 } else { 1.0 },
            sun_axes: (up, e1),
            show_help: true,
            total_frames: 0,
            tour,
            quit: false,
        }
    }

    /// Whether the screenshot tour is running (`VOXEL_CRAFT_SHOTS`): it
    /// poses the camera itself and takes no input.
    pub fn touring(&self) -> bool {
        self.tour.is_some()
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

        let controls = self.input.controls();
        self.world.update();
        self.player.update(&self.world, &controls, dt);
        self.interaction.update(&mut self.player, &mut self.world, &controls, dt);

        self.total_frames += 1;
        if let Some(t) = &self.tour {
            if t.trunk && self.total_frames == 700 {
                let here = self.world.cell_at(self.player.position);
                match nearest_wood(&self.world, here, (15.0 / crate::config::VOXEL_M) as i32) {
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
        let shot = self.tour.as_mut().and_then(|t| t.step(&mut self.player, self.gfx.settled(), &mut self.quit));
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
            weather: self.time,
            screenshot: shot.as_deref(),
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
    /// Consecutive frames saved a pose (`VOXEL_CRAFT_BURST`), and those
    /// saved of this one: still frames, to measure flicker.
    burst: u32,
    shots: u32,
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
        let burst = std::env::var("VOXEL_CRAFT_BURST").ok().and_then(|v| v.parse().ok()).unwrap_or(1u32).max(1);
        Self { dir, pose: 0, waited: u32::MAX, trunk: std::env::var_os("VOXEL_CRAFT_TRUNK").is_some(), burst, shots: 0 }
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
        let path = if self.burst == 1 { format!("{}/shot_{:02}.png", self.dir, self.pose) } else { format!("{}/shot_{:02}_{:03}.png", self.dir, self.pose, self.shots) };
        log::info!("shot {} at {} frames, settled {settled}", self.pose, self.waited);
        self.shots += 1;
        if self.shots < self.burst {
            return Some(path);
        }
        self.shots = 0;
        self.pose += 1;
        let limit = std::env::var("VOXEL_CRAFT_POSES").ok().and_then(|v| v.parse().ok()).unwrap_or(POSES.len());
        if self.pose >= limit.min(POSES.len()) {
            *quit = true;
            self.pose = usize::MAX / 2;
        } else {
            // The next pose from the next frame: this one is shot as posed.
            self.waited = u32::MAX;
        }
        Some(path)
    }

    fn set_pose(&mut self, player: &mut Player) {
        let (turn, pitch, third, lift) = POSES[self.pose];
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
