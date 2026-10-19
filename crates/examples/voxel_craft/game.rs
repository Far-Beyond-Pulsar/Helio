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
    focus: (i32, i32),
    /// The sun's path: a great circle through the sky over the spawn.
    sun_axes: (DVec3, DVec3),
    show_help: bool,
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
        let cell = world.cell_at(spawn);
        world.focus(cell);
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
            time_speed: 1.0,
            focus: (i32::MIN, i32::MIN),
            sun_axes: (up, e1),
            show_help: true,
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

        // The world decorates around the player's region.
        let cell: Cell = self.world.cell_at(self.player.position);
        let region = (cell.i.div_euclid(REGION_BLOCKS), cell.j.div_euclid(REGION_BLOCKS));
        if region != self.focus {
            self.focus = region;
            self.world.focus(cell);
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
