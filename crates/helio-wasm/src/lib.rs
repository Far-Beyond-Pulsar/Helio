//! `helio-wasm` — cross-platform WASM wrapper for helio renderer examples.
//!
//! # Usage
//!
//! Implement [`HelioWasmApp`] for your demo type, then call [`launch`]:
//!
//! ```ignore
//! use helio_wasm::{HelioWasmApp, InputState, launch};
//! use helio::{Camera, Renderer};
//! use std::sync::Arc;
//!
//! struct MyDemo { /* ... */ }
//!
//! impl HelioWasmApp for MyDemo {
//!     fn title() -> &'static str { "My Demo" }
//!
//!     fn init(renderer: &mut Renderer, _device: Arc<wgpu::Device>,
//!             _queue: Arc<wgpu::Queue>, width: u32, height: u32) -> Self {
//!         /* build scene */
//!         MyDemo { /* ... */ }
//!     }
//!
//!     fn update(&mut self, renderer: &mut Renderer, dt: f32,
//!               elapsed: f32, input: &InputState) -> Camera {
//!         Camera::perspective_look_at(/* ... */)
//!     }
//! }
//!
//! // Native entry point
//! fn main() { launch::<MyDemo>(); }
//!
//! // WASM entry point (in helio-web-demos)
//! #[cfg(target_arch = "wasm32")]
//! #[wasm_bindgen::prelude::wasm_bindgen(start)]
//! pub fn run() { launch::<MyDemo>(); }
//! ```

mod runner;
pub use runner::launch;

use std::collections::HashSet;
pub use winit::keyboard::KeyCode;
pub use winit::event::MouseButton;

// ── Public API ────────────────────────────────────────────────────────────────

/// Per-frame input snapshot passed to [`HelioWasmApp::update`].
pub struct InputState {
    /// Current drawable viewport size in physical pixels.
    pub viewport_size: (u32, u32),
    /// All currently pressed keyboard keys.
    pub keys: HashSet<KeyCode>,
    /// Mouse movement since last frame (dx, dy in pixels).
    /// Only populated while the cursor is grabbed.
    pub mouse_delta: (f32, f32),
    /// Whether the cursor is currently captured/locked.
    pub cursor_grabbed: bool,
    /// Current cursor position in logical pixels (x, y).
    /// Updated every `CursorMoved` event; most useful when the cursor is free.
    pub cursor_pos: (f32, f32),
    /// True for exactly one frame after the left mouse button was pressed.
    pub mouse_left_just_pressed: bool,
    /// True for exactly one frame after the left mouse button was released.
    pub mouse_left_just_released: bool,
}

impl InputState {
    /// Current viewport aspect ratio, safe while the window is minimized.
    pub fn aspect_ratio(&self) -> f32 {
        self.viewport_size.0.max(1) as f32 / self.viewport_size.1.max(1) as f32
    }
}

/// Implement this trait to create a helio demo that runs on both native and web.
pub trait HelioWasmApp: Sized + 'static {
    /// Window/page title.
    fn title() -> &'static str {
        "Helio Demo"
    }

    /// Which mouse button grabs (locks) the cursor for fly-camera mode.
    ///
    /// Defaults to `Left` (the original behaviour). Override to `Right` for
    /// editor-style demos where left-click is used for object picking.
    fn grab_cursor_button() -> winit::event::MouseButton {
        winit::event::MouseButton::Left
    }

    /// If `true`, releasing the grab button also releases the cursor.
    ///
    /// Defaults to `false`: cursor stays grabbed until `Escape` is pressed.
    /// Override to `true` for "hold-to-fly" right-click behaviour.
    fn release_cursor_on_grab_button_release() -> bool {
        false
    }

    /// Render scale for the demo's color targets (1.0 = full canvas
    /// resolution). Helio's default graph upscales from a scaled buffer via
    /// TAA, so it defaults to `0.75`. Custom graphs that have no TAA upscale
    /// step (e.g. a plain FXAA blit) must return `1.0`, or the scaled depth
    /// buffer will mismatch the full-resolution color attachment.
    fn render_scale() -> f32 {
        0.75
    }

    /// Optionally build a custom render graph for this demo.
    ///
    /// Return `None` (the default) to use helio's standard deferred graph.
    /// Override to insert custom passes — voxel meshing, injected post-process
    /// effects, etc. Called once, before [`init`](HelioWasmApp::init); the
    /// scene is still empty at this point, so populate meshes/lights/volumes in
    /// `init` (passes bind scene resources at render time) and only assemble
    /// the pass pipeline here.
    ///
    /// `config` already carries this demo's [`render_scale`](HelioWasmApp::render_scale)
    /// and the current `width`/`height`; use `config.width` / `config.height`
    /// when locking the graph.
    fn build_graph(
        _device: &std::sync::Arc<wgpu::Device>,
        _queue: &std::sync::Arc<wgpu::Queue>,
        _scene: &helio::Scene,
        _config: helio::RendererConfig,
        _debug_state: std::sync::Arc<std::sync::Mutex<helio::DebugDrawState>>,
        _debug_camera_buf: &wgpu::Buffer,
        _cull_stats_buf: &wgpu::Buffer,
    ) -> Option<helio::RenderGraph> {
        None
    }

    /// Called once after the wgpu device and renderer are ready.
    /// Build your scene (meshes, materials, lights) here.
    fn init(
        renderer: &mut helio::Renderer,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        width: u32,
        height: u32,
    ) -> Self;

    /// Called every frame. Return the camera to render from.
    ///
    /// `dt` — delta time in seconds since the last frame.  
    /// `elapsed` — total seconds since the demo started.  
    /// `input` — keyboard / mouse snapshot for this frame.
    fn update(
        &mut self,
        renderer: &mut helio::Renderer,
        dt: f32,
        elapsed: f32,
        input: &InputState,
    ) -> helio::Camera;

    /// Called when the window is resized. Override to update projection state.
    fn on_resize(&mut self, _renderer: &mut helio::Renderer, _width: u32, _height: u32) {}
}
