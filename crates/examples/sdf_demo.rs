//! SDF Planet Demo — helio v3
//!
//! A procedurally generated spherical planet whose terrain is the isosurface
//! of a 3-D signed distance field:
//!
//! ```text
//!   h(n̂) = R + fbm(n̂ × freq) × scale     n̂ = unit sphere direction
//!   sdf(p) = |p| − h(p/|p|)
//! ```
//!
//! The icosphere mesh is split into 7 biome bands (deep ocean · shallow water ·
//! beach · grassland · highland · mountain rock · snow cap), each with its own
//! PBR material.  The sun light rotates slowly for a day/night effect.
//! Camera auto-orbits the planet — press WASD to switch to free-fly mode.
//!
//! Controls:
//!   WASD / Space / Shift  — free-fly (auto-orbit disabled on first input)
//!   Mouse drag            — look (click window to grab cursor)
//!   Escape                — release cursor / exit

mod v3_demo_common;
use v3_demo_common::{make_material, point_light, directional_light, insert_object};
use crate::v3_demo_common::box_mesh;
use helio::{
    required_wgpu_features, required_wgpu_limits,
    Camera, LightId, MaterialId, MeshUpload, PackedVertex, Renderer, RendererConfig,
};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};

use std::collections::HashSet;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// PLANET / TERRAIN CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const PLANET_RADIUS: f32 = 100.0;
const TERRAIN_SCALE: f32 = 26.0;  // max height above sea level
const OCEAN_DEPTH:   f32 = 16.0;  // max depth below sea level
const SUBDIVISIONS:  u32 = 5;     // 20 480 triangles, 10 242 vertices

// Absolute height thresholds for each biome zone
const H_DEEP_OCEAN:  f32 = PLANET_RADIUS - 10.0; //  90
const H_SHALLOW:     f32 = PLANET_RADIUS -  3.0; //  97
const H_BEACH:       f32 = PLANET_RADIUS +  1.5; // 101.5
const H_GRASS:       f32 = PLANET_RADIUS +  7.0; // 107
const H_HIGHLAND:    f32 = PLANET_RADIUS + 13.0; // 113
const H_MOUNTAIN:    f32 = PLANET_RADIUS + 20.0; // 120
// Snow: everything above H_MOUNTAIN

const ZONE_COUNT: usize = 7;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Zone { DeepOcean = 0, ShallowWater, Beach, Grassland, Highland, Mountain, Snow }

fn height_to_zone(h: f32) -> Zone {
    if      h < H_DEEP_OCEAN { Zone::DeepOcean }
    else if h < H_SHALLOW    { Zone::ShallowWater }
    else if h < H_BEACH      { Zone::Beach }
    else if h < H_GRASS      { Zone::Grassland }
    else if h < H_HIGHLAND   { Zone::Highland }
    else if h < H_MOUNTAIN   { Zone::Mountain }
    else                     { Zone::Snow }
}

// ─────────────────────────────────────────────────────────────────────────────
// PROCEDURAL NOISE — smooth value noise + fBm
// ─────────────────────────────────────────────────────────────────────────────

fn hash3(ix: i32, iy: i32, iz: i32) -> f32 {
    let mut h = ix.wrapping_mul(73856093)
                  .wrapping_add(iy.wrapping_mul(19349663))
                  .wrapping_add(iz.wrapping_mul(83492791));
    h ^= h >> 16;
    h = h.wrapping_mul(0x45d9f3b_u32 as i32);
    h ^= h >> 16;
    (h as u32 as f32) * (1.0 / u32::MAX as f32)
}

#[inline] fn smooth(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }

fn value_noise(px: f32, py: f32, pz: f32) -> f32 {
    let ix = px.floor() as i32;
    let iy = py.floor() as i32;
    let iz = pz.floor() as i32;
    let fx = smooth(px - ix as f32);
    let fy = smooth(py - iy as f32);
    let fz = smooth(pz - iz as f32);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let v000 = hash3(ix,   iy,   iz  );
    let v100 = hash3(ix+1, iy,   iz  );
    let v010 = hash3(ix,   iy+1, iz  );
    let v110 = hash3(ix+1, iy+1, iz  );
    let v001 = hash3(ix,   iy,   iz+1);
    let v101 = hash3(ix+1, iy,   iz+1);
    let v011 = hash3(ix,   iy+1, iz+1);
    let v111 = hash3(ix+1, iy+1, iz+1);
    lerp(
        lerp(lerp(v000, v100, fx), lerp(v010, v110, fx), fy),
        lerp(lerp(v001, v101, fx), lerp(v011, v111, fx), fy),
        fz,
    )
}

fn fbm(px: f32, py: f32, pz: f32, octaves: u32) -> f32 {
    let (mut v, mut a, mut sum) = (0.0_f32, 0.5_f32, 0.0_f32);
    let (mut ox, mut oy, mut oz) = (px, py, pz);
    for _ in 0..octaves {
        v += a * value_noise(ox, oy, oz);
        sum += a;  a *= 0.5;
        ox *= 2.0; oy *= 2.0; oz *= 2.0;
    }
    v / sum
}

// ─────────────────────────────────────────────────────────────────────────────
// TERRAIN HEIGHT  (the SDF: sdf(p) = |p| − h(p/|p|))
// ─────────────────────────────────────────────────────────────────────────────

fn terrain_height(n: [f32; 3]) -> f32 {
    // Broad continental masses — low-freq, controls land/ocean split
    let continent = fbm(n[0] * 1.7, n[1] * 1.7, n[2] * 1.7, 4);
    // Mid-scale terrain bumps: ridges, plateaus, valleys
    let detail    = fbm(n[0] * 4.3 + 17.3, n[1] * 4.3 + 11.7, n[2] * 4.3 + 8.1, 5);
    // Fine mountain ridgelines
    let ridge     = fbm(n[0] * 9.1 + 33.1, n[1] * 9.1 + 21.4, n[2] * 9.1 + 44.7, 4);

    let base = continent * 0.60 + detail * 0.28 + ridge * 0.12;

    // Sharpen peaks: quadratic boost above the highland threshold
    let base = if base > 0.58 { base + (base - 0.58) * (base - 0.58) * 3.0 } else { base };
    // Flatten shallow ocean floors (soft continental shelf)
    let base = if base < 0.40 { 0.40 - (0.40 - base) * 0.6 } else { base };

    // Map to absolute height: sea level at 0.40
    let raw = base - 0.40;
    PLANET_RADIUS + raw.clamp(-1.0, 1.0) * if raw < 0.0 { OCEAN_DEPTH } else { TERRAIN_SCALE }
}

// ─────────────────────────────────────────────────────────────────────────────
// ICOSPHERE
// ─────────────────────────────────────────────────────────────────────────────

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    [v[0]/l, v[1]/l, v[2]/l]
}
fn sub3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn midpoint_vertex(
    verts: &mut Vec<[f32; 3]>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
    a: u32, b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = cache.get(&key) { return i; }
    let va = verts[a as usize]; let vb = verts[b as usize];
    let mid = normalize3([(va[0]+vb[0])*0.5, (va[1]+vb[1])*0.5, (va[2]+vb[2])*0.5]);
    let i = verts.len() as u32;
    verts.push(mid); cache.insert(key, i); i
}

struct Icosphere { vertices: Vec<[f32; 3]>, triangles: Vec<[u32; 3]> }

fn icosphere(subs: u32) -> Icosphere {
    let phi = (1.0_f32 + 5.0_f32.sqrt()) * 0.5;
    let mut v: Vec<[f32; 3]> = [
        [-1.0, phi,  0.0], [ 1.0, phi,  0.0], [-1.0,-phi,  0.0], [ 1.0,-phi,  0.0],
        [ 0.0,-1.0,  phi], [ 0.0, 1.0,  phi], [ 0.0,-1.0, -phi], [ 0.0, 1.0, -phi],
        [ phi, 0.0, -1.0], [ phi, 0.0,  1.0], [-phi, 0.0, -1.0], [-phi, 0.0,  1.0],
    ].iter().map(|&x| normalize3(x)).collect();
    let mut t: Vec<[u32; 3]> = vec![
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];
    for _ in 0..subs {
        let mut nt = Vec::with_capacity(t.len() * 4);
        let mut cache = std::collections::HashMap::<(u32,u32),u32>::new();
        for &[a,b,c] in &t {
            let ab = midpoint_vertex(&mut v, &mut cache, a, b);
            let bc = midpoint_vertex(&mut v, &mut cache, b, c);
            let ca = midpoint_vertex(&mut v, &mut cache, c, a);
            nt.push([a,ab,ca]); nt.push([b,bc,ab]);
            nt.push([c,ca,bc]); nt.push([ab,bc,ca]);
        }
        t = nt;
    }
    Icosphere { vertices: v, triangles: t }
}

fn sphere_uv(n: [f32; 3]) -> [f32; 2] {
    [0.5 + n[2].atan2(n[0]) / (2.0 * std::f32::consts::PI),
     0.5 - n[1].asin() / std::f32::consts::PI]
}

// ─────────────────────────────────────────────────────────────────────────────
// PLANET MESH BUILDER  (one MeshUpload per biome zone)
// ─────────────────────────────────────────────────────────────────────────────

fn build_planet_meshes(renderer: &mut Renderer, mats: [MaterialId; ZONE_COUNT]) {
    eprintln!("[SDF Planet] Generating icosphere (subdivisions = {SUBDIVISIONS})…");
    let ico = icosphere(SUBDIVISIONS);
    eprintln!("[SDF Planet]   {} vertices, {} triangles", ico.vertices.len(), ico.triangles.len());

    let heights:   Vec<f32>    = ico.vertices.iter().map(|&n| terrain_height(n)).collect();
    let displaced: Vec<[f32;3]> = ico.vertices.iter().zip(&heights)
        .map(|(&n, &h)| [n[0]*h, n[1]*h, n[2]*h]).collect();

    // Partition triangles by centroid height → biome zone
    let mut buckets: Vec<Vec<[u32; 3]>> = vec![Vec::new(); ZONE_COUNT];
    for &[a,b,c] in &ico.triangles {
        let h = (heights[a as usize] + heights[b as usize] + heights[c as usize]) / 3.0;
        buckets[height_to_zone(h) as usize].push([a, b, c]);
    }

    let bound_r = PLANET_RADIUS + TERRAIN_SCALE + 1.0;
    for (zi, tris) in buckets.iter().enumerate() {
        if tris.is_empty() { continue; }
        let mut verts = Vec::with_capacity(tris.len() * 3);
        let mut idxs  = Vec::with_capacity(tris.len() * 3);
        for (i, &[a,b,c]) in tris.iter().enumerate() {
            let pa = displaced[a as usize];
            let pb = displaced[b as usize];
            let pc = displaced[c as usize];
            // Flat shading: normal perpendicular to the triangle face
            let normal  = normalize3(cross3(sub3(pb, pa), sub3(pc, pa)));
            let up_ref  = if normal[1].abs() < 0.9 { [0.0_f32,1.0,0.0] } else { [1.0,0.0,0.0] };
            let tangent = normalize3(cross3(up_ref, normal));
            let base = (i * 3) as u32;
            for (&pos, &nd) in [
                (&pa, &ico.vertices[a as usize]),
                (&pb, &ico.vertices[b as usize]),
                (&pc, &ico.vertices[c as usize]),
            ] {
                verts.push(PackedVertex::from_components(pos, normal, sphere_uv(nd), tangent, 1.0));
            }
            idxs.extend_from_slice(&[base, base+1, base+2]);
        }
        let mesh = renderer.insert_mesh(MeshUpload { vertices: verts, indices: idxs });
        let _ = insert_object(renderer, mesh, mats[zi], glam::Mat4::IDENTITY, bound_r);
        eprintln!("[SDF Planet]   Zone {zi}: {} tris", tris.len());
    }
}

fn build_moon_mesh(renderer: &mut Renderer, mat: MaterialId, offset: glam::Vec3) {
    let ico    = icosphere(3);
    let moon_r = 14.0_f32;
    let mut verts = Vec::with_capacity(ico.triangles.len() * 3);
    let mut idxs  = Vec::with_capacity(ico.triangles.len() * 3);
    for (i, &[a,b,c]) in ico.triangles.iter().enumerate() {
        let pa: [f32;3] = { let n=ico.vertices[a as usize]; [n[0]*moon_r,n[1]*moon_r,n[2]*moon_r] };
        let pb: [f32;3] = { let n=ico.vertices[b as usize]; [n[0]*moon_r,n[1]*moon_r,n[2]*moon_r] };
        let pc: [f32;3] = { let n=ico.vertices[c as usize]; [n[0]*moon_r,n[1]*moon_r,n[2]*moon_r] };
        let normal  = normalize3(cross3(sub3(pb, pa), sub3(pc, pa)));
        let up_ref  = if normal[1].abs() < 0.9 { [0.0_f32,1.0,0.0] } else { [1.0,0.0,0.0] };
        let tangent = normalize3(cross3(up_ref, normal));
        let base = (i * 3) as u32;
        for (&pos, &nd) in [
            (&pa, &ico.vertices[a as usize]),
            (&pb, &ico.vertices[b as usize]),
            (&pc, &ico.vertices[c as usize]),
        ] {
            verts.push(PackedVertex::from_components(pos, normal, sphere_uv(nd), tangent, 1.0));
        }
        idxs.extend_from_slice(&[base, base+1, base+2]);
    }
    let mesh = renderer.insert_mesh(MeshUpload { vertices: verts, indices: idxs });
    let _ = insert_object(renderer, mesh, mat, glam::Mat4::from_translation(offset), moon_r + 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// APP BOILERPLATE
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    EventLoop::new().expect("event loop")
        .run_app(&mut App { state: None })
        .expect("run");
}

struct App { state: Option<AppState> }

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,
    // Camera
    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    orbit_mode:     bool,
    elapsed:        f32,
    // Input
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),
    // Scene
    sun_id:         LightId,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("Helio SDF Planet Demo (v3)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
        ).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits:   required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
            None,
        )).expect("device");
        device.on_uncaptured_error(Box::new(|e| panic!("[GPU] {:?}", e)));
        let device = Arc::new(device);
        let queue  = Arc::new(queue);
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:  size.width, height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format),
        );
        // Deep-space ambient: very dim cool blue
        renderer.set_clear_color([0.01, 0.01, 0.03, 1.0]);
        renderer.set_ambient([0.04, 0.06, 0.12], 0.18);

        // ── Materials (indexed by Zone discriminant) ─────────────────────
        let mats: [MaterialId; ZONE_COUNT] = [
            // 0 DeepOcean    — dark blue, very smooth (mirror-like sea)
            renderer.insert_material(make_material([0.04, 0.12, 0.34, 1.0], 0.06, 0.0, [0.02, 0.04, 0.10], 0.08)),
            // 1 ShallowWater — lighter teal, slightly rougher
            renderer.insert_material(make_material([0.08, 0.28, 0.55, 1.0], 0.15, 0.0, [0.01, 0.02, 0.06], 0.04)),
            // 2 Beach        — warm sand
            renderer.insert_material(make_material([0.80, 0.74, 0.50, 1.0], 0.92, 0.0, [0.0, 0.0, 0.0], 0.0)),
            // 3 Grassland    — lush green plains
            renderer.insert_material(make_material([0.22, 0.52, 0.17, 1.0], 0.94, 0.0, [0.0, 0.0, 0.0], 0.0)),
            // 4 Highland     — mixed green/brown forested hills
            renderer.insert_material(make_material([0.24, 0.36, 0.14, 1.0], 0.90, 0.0, [0.0, 0.0, 0.0], 0.0)),
            // 5 Mountain     — bare grey rock
            renderer.insert_material(make_material([0.46, 0.42, 0.38, 1.0], 0.78, 0.04, [0.0, 0.0, 0.0], 0.0)),
            // 6 Snow         — bright ice cap, slightly specular
            renderer.insert_material(make_material([0.92, 0.95, 1.00, 1.0], 0.32, 0.0, [0.04, 0.05, 0.08], 0.12)),
        ];

        // ── Planet geometry ──────────────────────────────────────────────
        build_planet_meshes(&mut renderer, mats);

        // ── Moon ─────────────────────────────────────────────────────────
        let mat_moon = renderer.insert_material(make_material(
            [0.58, 0.55, 0.52, 1.0], 0.85, 0.02, [0.0, 0.0, 0.0], 0.0,
        ));
        build_moon_mesh(&mut renderer, mat_moon, glam::Vec3::new(190.0, 35.0, 75.0));

        // ── Lights ───────────────────────────────────────────────────────
        // Primary star (sun) — warm white, will be rotated every frame
        let sun_id = renderer.insert_light(directional_light(
            [-0.6, -0.55, -0.55], [1.00, 0.96, 0.82], 9.5,
        ));
        // Secondary fill — cold blue from the opposite hemisphere (bounce)
        let _ = renderer.insert_light(directional_light(
            [0.5, 0.3, 0.5], [0.35, 0.55, 0.90], 1.2,
        ));
        // Atmospheric rim glow near the equator
        let _ = renderer.insert_light(point_light([160.0, 0.0, 0.0], [1.0, 0.60, 0.25], 6.0, 200.0));
        // Moon back-light
        let _ = renderer.insert_light(point_light([-130.0, 40.0, -100.0], [0.7, 0.8, 1.0], 3.0, 180.0));

        eprintln!("[SDF Planet] Scene ready.  Auto-orbiting — press WASD to fly freely.");

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 40.0, 280.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.14,
            orbit_mode: true, elapsed: 0.0,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            sun_id,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. }, ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, ..
            } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed, button: MouseButton::Left, ..
            } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok { state.window.set_cursor_visible(false); state.cursor_grabbed = true; }
                }
            }
            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage:  wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width:  sz.width, height: sz.height,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    alpha_mode:   wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                });
                state.renderer.set_render_size(sz.width, sz.height);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt  = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if s.cursor_grabbed { s.mouse_delta.0 += dx as f32; s.mouse_delta.1 += dy as f32; }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state { s.window.request_redraw(); }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 25.0;
        const SENS:  f32 = 0.0020;

        self.elapsed += dt;

        // Exit orbit mode the moment the player touches a movement key
        if self.keys.contains(&KeyCode::KeyW) || self.keys.contains(&KeyCode::KeyS)
        || self.keys.contains(&KeyCode::KeyA) || self.keys.contains(&KeyCode::KeyD)
        || self.keys.contains(&KeyCode::Space)
        {
            self.orbit_mode = false;
        }

        // Apply mouse look
        self.cam_yaw   += self.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.55, 1.55);
        self.mouse_delta = (0.0, 0.0);

        if self.orbit_mode {
            // Smooth circle at varying elevation
            let angle  = self.elapsed * 0.07;
            let radius = 255.0_f32;
            let height = 45.0 + 25.0 * (self.elapsed * 0.022).sin();
            self.cam_pos = glam::Vec3::new(radius * angle.cos(), height, radius * angle.sin());
            // Always face the planet centre
            let dir = (-self.cam_pos).normalize();
            self.cam_yaw   = dir.z.atan2(dir.x);
            self.cam_pitch = dir.y.asin();
        } else {
            let (sy, cy) = self.cam_yaw.sin_cos();
            let (sp, cp) = self.cam_pitch.sin_cos();
            let fwd   = glam::Vec3::new(sy * cp, sp, -cy * cp);
            let right = glam::Vec3::new(cy, 0.0, sy);
            if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += fwd   * SPEED * dt; }
            if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= fwd   * SPEED * dt; }
            if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right * SPEED * dt; }
            if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right * SPEED * dt; }
            if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
            if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }
        }

        // Animate the sun: slow rotation around the planet
        let sun_a = self.elapsed * 0.035;
        let _ = self.renderer.update_light(
            self.sun_id,
            directional_light(
                [-(sun_a.cos() * 0.7), -0.5, -(sun_a.sin() * 0.7)],
                [1.00, 0.96, 0.82],
                9.5,
            ),
        );

        // Build camera from current yaw/pitch (orbit mode sets these too)
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let sz  = self.window.inner_size();
        let asp = sz.width as f32 / sz.height.max(1) as f32;
        let camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + fwd, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, asp, 0.5, 3000.0,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}

