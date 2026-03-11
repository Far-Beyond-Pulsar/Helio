//! Debug visualization system.
//!
//! [`DebugVizSystem`] manages a set of [`DebugRenderer`] implementations.
//! Each frame, when [`DebugVizSystem::enabled`] is `true`, every active renderer
//! appends [`DebugShape`]s that are consumed by `DebugDrawPass` on top of the
//! scene.
//!
//! # Quick start
//!
//! ```no_run
//! // In your event loop, bind F3:
//! renderer.debug_viz_mut().enabled ^= true;
//!
//! // Enable editor mode so light positions get billboard icons:
//! renderer.set_editor_mode(true);
//!
//! // Toggle individual overlays:
//! renderer.debug_viz_mut().set_enabled("grid", true);
//! renderer.debug_viz_mut().set_enabled("mesh_bounds", false);
//! ```
//!
//! # Built-in renderers (all enabled when F3 is pressed)
//!
//! | Name               | Description                                         |
//! |--------------------|-----------------------------------------------------|
//! | `"light_range"`    | Wireframe spheres at light attenuation extents      |
//! | `"light_direction"`| Direction arrows (cones) for spot/directional lights|
//! | `"mesh_bounds"`    | Wireframe bounding spheres around scene objects     |
//! | `"grid"`           | Snapped world-space XZ grid at Y=0                  |
//!
//! Register custom overlays via [`DebugVizSystem::register`].

use glam::Vec3;
use crate::debug_draw::DebugShape;
use crate::scene::SceneLight;
use crate::features::LightType;

// ── Per-frame context ─────────────────────────────────────────────────────────

/// World-space bounding sphere of a single scene object.
#[derive(Clone, Copy, Debug)]
pub struct ObjectBounds {
    pub center: Vec3,
    pub radius: f32,
}

/// Read-only per-frame context passed to every active [`DebugRenderer`].
pub struct DebugRenderContext<'a> {
    /// All currently active lights (same slice the GPU sees).
    pub lights: &'a [SceneLight],
    /// World-space bounding spheres of all registered scene objects.
    pub object_bounds: &'a [ObjectBounds],
    /// Camera world-space position.
    pub camera_pos: Vec3,
    /// Camera forward unit-vector.
    pub camera_forward: Vec3,
    /// Frame delta time in seconds.
    pub dt: f32,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A pluggable debug overlay.
///
/// Implement this trait and call [`DebugVizSystem::register`] to add custom
/// visualizations.  The renderer calls [`render`] once per frame when both the
/// master switch _and_ this renderer's own `enabled` flag are set.
pub trait DebugRenderer: Send + Sync {
    /// Unique name in `lowercase_snake_case`, e.g. `"my_overlay"`.
    fn name(&self) -> &str;

    /// Whether this specific overlay is currently active.
    fn is_enabled(&self) -> bool;

    /// Enable or disable this overlay independently of the master switch.
    fn set_enabled(&mut self, enabled: bool);

    /// Append [`DebugShape`]s for this frame to `out`.
    ///
    /// This is called only when `is_enabled()` returns `true` **and** the
    /// master [`DebugVizSystem::enabled`] switch is on.
    fn render(&self, ctx: &DebugRenderContext, out: &mut Vec<DebugShape>);
}

// ── System ────────────────────────────────────────────────────────────────────

/// Central debug visualization manager.
///
/// Owned by [`Renderer`].  Access via [`Renderer::debug_viz`] /
/// [`Renderer::debug_viz_mut`].
pub struct DebugVizSystem {
    /// Master on/off switch.  Bind to **F3** in your event loop.
    /// When `false`, [`collect`] returns immediately with no allocation.
    pub enabled: bool,
    renderers: Vec<Box<dyn DebugRenderer>>,
}

impl Default for DebugVizSystem {
    fn default() -> Self {
        let mut s = Self { enabled: false, renderers: Vec::new() };
        // All built-ins are registered enabled; the master switch is off until
        // the user presses F3.
        s.register(Box::new(LightRangeRenderer::new()));
        s.register(Box::new(LightDirectionRenderer::new()));
        s.register(Box::new(MeshBoundsRenderer::new()));
        s.register(Box::new(GridRenderer::new()));
        s
    }
}

impl DebugVizSystem {
    /// Register a renderer.  Replaces any existing entry with the same name.
    pub fn register(&mut self, renderer: Box<dyn DebugRenderer>) {
        if let Some(pos) = self.renderers.iter().position(|r| r.name() == renderer.name()) {
            self.renderers[pos] = renderer;
        } else {
            self.renderers.push(renderer);
        }
    }

    /// Enable or disable a renderer by name.  No-op if name not found.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(r) = self.renderers.iter_mut().find(|r| r.name() == name) {
            r.set_enabled(enabled);
        }
    }

    /// Toggle a renderer by name.  Returns the new enabled state, or `false`
    /// if no renderer with that name is registered.
    pub fn toggle(&mut self, name: &str) -> bool {
        if let Some(r) = self.renderers.iter_mut().find(|r| r.name() == name) {
            let next = !r.is_enabled();
            r.set_enabled(next);
            next
        } else {
            false
        }
    }

    /// Returns `(name, enabled)` pairs for all registered renderers.
    pub fn renderers(&self) -> impl Iterator<Item = (&str, bool)> {
        self.renderers.iter().map(|r| (r.name(), r.is_enabled()))
    }

    /// Collect all shapes this frame into `out`.
    ///
    /// Called internally by the renderer inside the `render()` hot path.
    /// Is a no-op (zero allocation) when `enabled` is `false`.
    pub(crate) fn collect(&self, ctx: &DebugRenderContext, out: &mut Vec<DebugShape>) {
        if !self.enabled {
            return;
        }
        for r in &self.renderers {
            if r.is_enabled() {
                r.render(ctx, out);
            }
        }
    }
}

// ── Built-in: LightRangeRenderer ─────────────────────────────────────────────

/// Renders wireframe attenuation spheres and drop-lines to the ground (Y=0)
/// for all point and spot lights.  Colors match each light's own tint.
pub struct LightRangeRenderer {
    enabled: bool,
    /// Alpha multiplier for the sphere wireframe (default `0.30`).
    pub sphere_alpha: f32,
    /// Alpha multiplier for the drop-line (default `0.55`).
    pub line_alpha: f32,
    /// Wireframe thickness for spheres (default `0.025`).
    pub sphere_thickness: f32,
}

impl LightRangeRenderer {
    pub fn new() -> Self {
        Self { enabled: true, sphere_alpha: 0.30, line_alpha: 0.55, sphere_thickness: 0.025 }
    }
}

impl DebugRenderer for LightRangeRenderer {
    fn name(&self) -> &str { "light_range" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, v: bool) { self.enabled = v; }

    fn render(&self, ctx: &DebugRenderContext, out: &mut Vec<DebugShape>) {
        for light in ctx.lights {
            if matches!(light.light_type, LightType::Directional) { continue; }
            let pos   = Vec3::from(light.position);
            let [r, g, b] = light.color;
            let range = light.range;

            // Attenuation sphere wireframe
            out.push(DebugShape::Sphere {
                center:    pos,
                radius:    range,
                color:     [r, g, b, self.sphere_alpha],
                thickness: self.sphere_thickness,
            });

            // Drop-line to ground
            let ground = Vec3::new(pos.x, 0.0, pos.z);
            if pos.y.abs() > 0.01 {
                out.push(DebugShape::Line {
                    start:     pos,
                    end:       ground,
                    color:     [r, g, b, self.line_alpha],
                    thickness: 0.04,
                });
                // Small dot at the ground touch-point
                out.push(DebugShape::Sphere {
                    center:    ground,
                    radius:    0.06,
                    color:     [r, g, b, self.line_alpha],
                    thickness: 0.04,
                });
            }
        }
    }
}

// ── Built-in: LightDirectionRenderer ─────────────────────────────────────────

/// Renders direction arrows (shaft + cone tip) for spot lights and directional
/// lights.  For directionals, three parallel arrows are drawn near the camera
/// to indicate the global sun direction.
pub struct LightDirectionRenderer {
    enabled: bool,
}

impl LightDirectionRenderer {
    pub fn new() -> Self { Self { enabled: true } }
}

impl DebugRenderer for LightDirectionRenderer {
    fn name(&self) -> &str { "light_direction" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, v: bool) { self.enabled = v; }

    fn render(&self, ctx: &DebugRenderContext, out: &mut Vec<DebugShape>) {
        for light in ctx.lights {
            let [r, g, b] = light.color;
            match light.light_type {
                LightType::Spot { .. } => {
                    let pos = Vec3::from(light.position);
                    let dir = Vec3::from(light.direction).normalize();
                    let shaft_len = (light.range * 0.4).max(0.5);
                    let shaft_end = pos + dir * shaft_len;

                    out.push(DebugShape::Line {
                        start:     pos,
                        end:       shaft_end,
                        color:     [r, g, b, 0.85],
                        thickness: 0.05,
                    });
                    out.push(DebugShape::Cone {
                        apex:      shaft_end,
                        direction: dir,
                        height:    shaft_len * 0.25,
                        radius:    shaft_len * 0.12,
                        color:     [r, g, b, 0.95],
                        thickness: 0.04,
                    });
                }
                LightType::Directional => {
                    // Three arrows in a ring near the camera to show sun direction.
                    let dir = Vec3::from(light.direction).normalize();
                    let offsets = [
                        ctx.camera_pos + Vec3::new( 3.0, 2.0,  0.0),
                        ctx.camera_pos + Vec3::new(-3.0, 2.0,  0.0),
                        ctx.camera_pos + Vec3::new( 0.0, 2.0,  3.0),
                    ];
                    for &origin in &offsets {
                        let tip = origin + dir * 1.8;
                        out.push(DebugShape::Line {
                            start:     origin,
                            end:       tip,
                            color:     [r, g, b, 0.70],
                            thickness: 0.04,
                        });
                        out.push(DebugShape::Cone {
                            apex:      tip,
                            direction: dir,
                            height:    0.35,
                            radius:    0.12,
                            color:     [r, g, b, 0.90],
                            thickness: 0.04,
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

// ── Built-in: MeshBoundsRenderer ─────────────────────────────────────────────

/// Renders a semi-transparent wireframe bounding sphere around every registered
/// scene object.
pub struct MeshBoundsRenderer {
    enabled: bool,
    /// Wireframe color (default lime green at low alpha).
    pub color: [f32; 4],
}

impl MeshBoundsRenderer {
    pub fn new() -> Self {
        Self { enabled: true, color: [0.0, 1.0, 0.35, 0.20] }
    }
}

impl DebugRenderer for MeshBoundsRenderer {
    fn name(&self) -> &str { "mesh_bounds" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, v: bool) { self.enabled = v; }

    fn render(&self, ctx: &DebugRenderContext, out: &mut Vec<DebugShape>) {
        for b in ctx.object_bounds {
            out.push(DebugShape::Sphere {
                center:    b.center,
                radius:    b.radius,
                color:     self.color,
                thickness: 0.020,
            });
        }
    }
}

// ── Built-in: GridRenderer ─────────────────────────────────────────────────

/// Renders a world-space XZ grid at Y=0 that snaps to the camera position
/// so it always appears relative to the viewer.
///
/// The X-axis line is red; the Z-axis line is blue; all other lines are grey.
pub struct GridRenderer {
    enabled: bool,
    /// Half-width/depth of the grid in world-units (default `16.0`).
    pub extent: f32,
    /// Distance between grid lines (default `1.0`).
    pub spacing: f32,
    /// Alpha of secondary (non-axis) lines.
    pub grid_alpha: f32,
}

impl GridRenderer {
    pub fn new() -> Self {
        Self { enabled: true, extent: 16.0, spacing: 1.0, grid_alpha: 0.15 }
    }
}

impl DebugRenderer for GridRenderer {
    fn name(&self) -> &str { "grid" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, v: bool) { self.enabled = v; }

    fn render(&self, ctx: &DebugRenderContext, out: &mut Vec<DebugShape>) {
        // Snap grid centre to the nearest line of the spacing grid.
        let cx = (ctx.camera_pos.x / self.spacing).round() * self.spacing;
        let cz = (ctx.camera_pos.z / self.spacing).round() * self.spacing;

        let steps = (self.extent / self.spacing).ceil() as i32;
        let faint   = [0.55_f32, 0.55, 0.55, self.grid_alpha];
        let axis_x  = [0.80_f32, 0.20, 0.20, 0.55];  // red
        let axis_z  = [0.20_f32, 0.20, 0.80, 0.55];  // blue

        for i in -steps..=steps {
            let t = i as f32 * self.spacing;

            // Lines running along Z (varying X position)
            out.push(DebugShape::Line {
                start:     Vec3::new(cx + t, 0.0, cz - self.extent),
                end:       Vec3::new(cx + t, 0.0, cz + self.extent),
                color:     if i == 0 { axis_x } else { faint },
                thickness: if i == 0 { 0.04 } else { 0.012 },
            });

            // Lines running along X (varying Z position)
            out.push(DebugShape::Line {
                start:     Vec3::new(cx - self.extent, 0.0, cz + t),
                end:       Vec3::new(cx + self.extent, 0.0, cz + t),
                color:     if i == 0 { axis_z } else { faint },
                thickness: if i == 0 { 0.04 } else { 0.012 },
            });
        }
    }
}
