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

// ── Public API ────────────────────────────────────────────────────────────────

/// Per-frame input snapshot passed to [`HelioWasmApp::update`].
pub struct InputState {
    /// All currently pressed keyboard keys.
    pub keys: HashSet<KeyCode>,
    /// Mouse movement since last frame (dx, dy in pixels).
    /// Only populated while the cursor is grabbed.
    pub mouse_delta: (f32, f32),
    /// Whether the cursor is currently captured/locked.
    pub cursor_grabbed: bool,
}

/// Implement this trait to create a helio demo that runs on both native and web.
pub trait HelioWasmApp: Sized + 'static {
    /// Window/page title.
    fn title() -> &'static str {
        "Helio Demo"
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
