#![cfg_attr(not(feature = "winit"), no_std)]

//! Platform-neutral semantic input and camera controllers.
//!
//! Platform adapters translate routed device input into [`NavigationAction`]
//! and look deltas. Controllers consume that semantic state without knowing
//! whether it came from winit, TRUEOS UI4, OpenXR, a gamepad, or automation.

use glam::{Vec2, Vec3};

/// Semantic controls understood by the shared fly-camera controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum NavigationAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    LookUp,
    LookDown,
    LookLeft,
    LookRight,
    Boost,
}

impl NavigationAction {
    const fn bit(self) -> u16 {
        1 << self as u8
    }
}

/// One controller update after platform input has been reduced to semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NavigationInput {
    /// Local movement axes: +X right, +Y up, +Z forward.
    pub movement: Vec3,
    /// Relative pointer/stick delta: +X turns right, +Y looks down.
    pub look_delta: Vec2,
    /// Held look direction: +X turns right, +Y looks down. Unlike
    /// [`Self::look_delta`], this is a velocity and is scaled by frame time.
    pub look_direction: Vec2,
    pub boost: bool,
    /// Unfocused input is inert even if a platform adapter retained stale keys.
    pub focused: bool,
}

/// Held semantic actions plus frame-local look accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationState {
    held: u16,
    look_delta: Vec2,
    focused: bool,
}

impl Default for NavigationState {
    fn default() -> Self {
        Self::new()
    }
}

impl NavigationState {
    pub const fn new() -> Self {
        Self {
            held: 0,
            look_delta: Vec2::ZERO,
            focused: true,
        }
    }

    pub fn set(&mut self, action: NavigationAction, pressed: bool) {
        if pressed && !self.focused {
            return;
        }
        if pressed {
            self.held |= action.bit();
        } else {
            self.held &= !action.bit();
        }
    }

    pub const fn is_held(&self, action: NavigationAction) -> bool {
        self.held & action.bit() != 0
    }

    pub fn add_look_delta(&mut self, delta: Vec2) {
        if self.focused && delta.is_finite() {
            self.look_delta += delta;
        }
    }

    /// Focus loss clears held state and pending motion, preventing drift when
    /// a window regains focus or a virtual input route disappears.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        if !focused {
            self.held = 0;
            self.look_delta = Vec2::ZERO;
        }
    }

    pub const fn focused(&self) -> bool {
        self.focused
    }

    /// Produce one frame of navigation and consume only the relative delta.
    pub fn take_input(&mut self) -> NavigationInput {
        let input = if self.focused {
            NavigationInput {
                movement: Vec3::new(
                    axis(
                        self.is_held(NavigationAction::MoveRight),
                        self.is_held(NavigationAction::MoveLeft),
                    ),
                    axis(
                        self.is_held(NavigationAction::MoveUp),
                        self.is_held(NavigationAction::MoveDown),
                    ),
                    axis(
                        self.is_held(NavigationAction::MoveForward),
                        self.is_held(NavigationAction::MoveBackward),
                    ),
                ),
                look_delta: self.look_delta,
                look_direction: Vec2::new(
                    axis(
                        self.is_held(NavigationAction::LookRight),
                        self.is_held(NavigationAction::LookLeft),
                    ),
                    axis(
                        self.is_held(NavigationAction::LookDown),
                        self.is_held(NavigationAction::LookUp),
                    ),
                ),
                boost: self.is_held(NavigationAction::Boost),
                focused: true,
            }
        } else {
            NavigationInput::default()
        };
        self.look_delta = Vec2::ZERO;
        input
    }
}

const fn axis(positive: bool, negative: bool) -> f32 {
    positive as u8 as f32 - negative as u8 as f32
}

/// Whether forward movement follows pitch or remains on the world XZ plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlyMovement {
    ViewPlane,
    GroundPlane,
}

/// Tuning which used to be repeated as constants in each demo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyCameraConfig {
    pub movement_speed: f32,
    pub boost_multiplier: f32,
    /// Angular velocity in radians per second for held look actions.
    pub keyboard_look_speed: f32,
    pub look_sensitivity: f32,
    pub pitch_min: f32,
    pub pitch_max: f32,
    pub movement: FlyMovement,
    /// Preserve legacy diagonal speed when false; normalize combined axes when
    /// true for controllers which want direction-independent velocity.
    pub normalize_movement: bool,
    /// Caps a delayed frame so refocus or debugger stalls cannot teleport.
    pub max_delta_seconds: f32,
}

impl Default for FlyCameraConfig {
    fn default() -> Self {
        Self {
            // Keep the quick traversal pace available in every demo without
            // hiding it behind a modifier key.
            movement_speed: 15.0,
            boost_multiplier: 1.0,
            keyboard_look_speed: 1.8,
            look_sensitivity: 0.002,
            pitch_min: -1.5,
            pitch_max: 1.5,
            movement: FlyMovement::ViewPlane,
            normalize_movement: false,
            // Desktop demos use a tighter cap so a focus change, window drag,
            // or debugger stop cannot visibly jump the camera through a scene.
            // This is the standard Helio/Linux flycam policy.
            max_delta_seconds: 0.05,
        }
    }
}

/// Derived camera axes using Helio's right-handed, Y-up, forward=-Z basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraBasis {
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

/// Reusable free-flight pose and update policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    config: FlyCameraConfig,
}

impl FlyCamera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32, config: FlyCameraConfig) -> Self {
        let mut camera = Self {
            position,
            yaw,
            pitch,
            config,
        };
        camera.pitch = camera.clamp_pitch(pitch);
        camera
    }

    pub fn look_at(position: Vec3, target: Vec3, config: FlyCameraConfig) -> Self {
        let forward = (target - position).normalize_or_zero();
        let yaw = forward.x.atan2(-forward.z);
        let pitch = forward.y.clamp(-1.0, 1.0).asin();
        Self::new(position, yaw, pitch, config)
    }

    pub const fn position(&self) -> Vec3 {
        self.position
    }

    pub const fn yaw(&self) -> f32 {
        self.yaw
    }

    pub const fn pitch(&self) -> f32 {
        self.pitch
    }

    pub const fn config(&self) -> FlyCameraConfig {
        self.config
    }

    pub fn set_config(&mut self, config: FlyCameraConfig) {
        self.config = config;
        self.pitch = self.clamp_pitch(self.pitch);
    }

    pub fn basis(&self) -> CameraBasis {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        CameraBasis {
            forward: Vec3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch),
            right: Vec3::new(cos_yaw, 0.0, sin_yaw),
            up: Vec3::Y,
        }
    }

    /// Apply one semantic input sample. Returns true when the pose changed.
    pub fn update(&mut self, input: NavigationInput, delta_seconds: f32) -> bool {
        if !input.focused {
            return false;
        }
        let mut changed = false;
        if input.look_delta != Vec2::ZERO && input.look_delta.is_finite() {
            self.yaw += input.look_delta.x * self.config.look_sensitivity;
            self.pitch =
                self.clamp_pitch(self.pitch - input.look_delta.y * self.config.look_sensitivity);
            changed = true;
        }

        let delta_seconds = delta_seconds
            .max(0.0)
            .min(self.config.max_delta_seconds.max(0.0));
        if input.look_direction != Vec2::ZERO
            && input.look_direction.is_finite()
            && delta_seconds != 0.0
        {
            let look_step =
                input.look_direction * self.config.keyboard_look_speed.max(0.0) * delta_seconds;
            self.yaw += look_step.x;
            self.pitch = self.clamp_pitch(self.pitch - look_step.y);
            changed = true;
        }
        if delta_seconds == 0.0 || input.movement == Vec3::ZERO {
            return changed;
        }

        let basis = self.basis();
        let forward = match self.config.movement {
            FlyMovement::ViewPlane => basis.forward,
            FlyMovement::GroundPlane => {
                Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos()).normalize_or_zero()
            }
        };
        let mut direction = basis.right * input.movement.x
            + Vec3::Y * input.movement.y
            + forward * input.movement.z;
        if self.config.normalize_movement && direction.length_squared() > 1.0 {
            direction = direction.normalize();
        }
        let boost = if input.boost {
            self.config.boost_multiplier.max(0.0)
        } else {
            1.0
        };
        self.position += direction * self.config.movement_speed * boost * delta_seconds;
        changed || direction != Vec3::ZERO
    }

    fn clamp_pitch(&self, pitch: f32) -> f32 {
        let low = self.config.pitch_min.min(self.config.pitch_max);
        let high = self.config.pitch_min.max(self.config.pitch_max);
        pitch.clamp(low, high)
    }
}

/// Standard perspective lens kept separate from pose/controller state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveLens {
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for PerspectiveLens {
    fn default() -> Self {
        Self {
            fov_y_radians: core::f32::consts::FRAC_PI_3,
            near: 0.1,
            far: 1_000.0,
        }
    }
}

#[cfg(feature = "winit")]
pub fn navigation_action_from_winit_key(key: winit::keyboard::KeyCode) -> Option<NavigationAction> {
    use winit::keyboard::KeyCode;
    Some(match key {
        KeyCode::KeyW => NavigationAction::MoveForward,
        KeyCode::KeyS => NavigationAction::MoveBackward,
        KeyCode::KeyA => NavigationAction::MoveLeft,
        KeyCode::KeyD => NavigationAction::MoveRight,
        KeyCode::Space => NavigationAction::MoveUp,
        KeyCode::ShiftLeft | KeyCode::ShiftRight => NavigationAction::MoveDown,
        KeyCode::KeyI => NavigationAction::LookUp,
        KeyCode::KeyK => NavigationAction::LookDown,
        KeyCode::KeyJ => NavigationAction::LookLeft,
        KeyCode::KeyL => NavigationAction::LookRight,
        _ => return None,
    })
}

/// Thin desktop adapter. It translates winit input and owns desktop cursor
/// capture policy while leaving camera behavior in [`FlyCamera`].
#[cfg(feature = "winit")]
#[derive(Debug, Default)]
pub struct WinitFlyInput {
    navigation: NavigationState,
    cursor_grabbed: bool,
}

#[cfg(feature = "winit")]
impl WinitFlyInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_key(&mut self, key: winit::keyboard::KeyCode, pressed: bool) -> bool {
        let Some(action) = navigation_action_from_winit_key(key) else {
            return false;
        };
        self.navigation.set(action, pressed);
        true
    }

    pub fn add_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        if self.cursor_grabbed {
            self.navigation
                .add_look_delta(Vec2::new(delta_x as f32, delta_y as f32));
        }
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.navigation.set_focused(focused);
    }

    /// Synchronize window focus and release capture when desktop focus leaves
    /// this application, so a later click can establish a fresh grab.
    pub fn set_window_focused(&mut self, window: &winit::window::Window, focused: bool) {
        self.navigation.set_focused(focused);
        if !focused {
            self.release_cursor(window);
        }
    }

    pub fn take_input(&mut self) -> NavigationInput {
        self.navigation.take_input()
    }

    pub const fn cursor_grabbed(&self) -> bool {
        self.cursor_grabbed
    }

    /// Preserve the demos' existing policy: prefer confinement and fall back
    /// to true relative locking where the window system supports it.
    pub fn grab_cursor(&mut self, window: &winit::window::Window) -> bool {
        use winit::window::CursorGrabMode;

        if self.cursor_grabbed {
            return true;
        }
        let grabbed = window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
            .is_ok();
        if grabbed {
            window.set_cursor_visible(false);
            self.cursor_grabbed = true;
        }
        grabbed
    }

    /// Returns whether a captured cursor was released. Callers can retain the
    /// familiar "first Escape releases, second Escape exits" behavior.
    pub fn release_cursor(&mut self, window: &winit::window::Window) -> bool {
        if !self.cursor_grabbed {
            return false;
        }
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
        window.set_cursor_visible(true);
        self.cursor_grabbed = false;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FlyCameraConfig {
        FlyCameraConfig {
            max_delta_seconds: 1.0,
            ..FlyCameraConfig::default()
        }
    }

    #[test]
    fn yaw_zero_looks_down_negative_z() {
        let camera = FlyCamera::new(Vec3::ZERO, 0.0, 0.0, config());
        assert_eq!(camera.basis().forward, -Vec3::Z);
        assert_eq!(camera.basis().right, Vec3::X);
    }

    #[test]
    fn semantic_state_produces_axes_and_consumes_relative_delta() {
        let mut state = NavigationState::new();
        state.set(NavigationAction::MoveForward, true);
        state.set(NavigationAction::MoveLeft, true);
        state.set(NavigationAction::LookLeft, true);
        state.set(NavigationAction::LookUp, true);
        state.add_look_delta(Vec2::new(3.0, -2.0));
        let input = state.take_input();
        assert_eq!(input.movement, Vec3::new(-1.0, 0.0, 1.0));
        assert_eq!(input.look_delta, Vec2::new(3.0, -2.0));
        assert_eq!(input.look_direction, Vec2::new(-1.0, -1.0));
        assert_eq!(state.take_input().look_delta, Vec2::ZERO);
        assert!(state.is_held(NavigationAction::MoveForward));
    }

    #[test]
    fn focus_loss_clears_held_and_relative_state() {
        let mut state = NavigationState::new();
        state.set(NavigationAction::MoveForward, true);
        state.add_look_delta(Vec2::ONE);
        state.set_focused(false);
        state.set(NavigationAction::MoveRight, true);
        assert_eq!(state.take_input(), NavigationInput::default());
        assert!(!state.is_held(NavigationAction::MoveForward));
        assert!(!state.is_held(NavigationAction::MoveRight));
    }

    #[test]
    fn legacy_diagonal_and_pitch_follow_are_configurable() {
        let mut camera = FlyCamera::new(Vec3::ZERO, 0.0, 0.5, config());
        let changed = camera.update(
            NavigationInput {
                movement: Vec3::new(1.0, 0.0, 1.0),
                focused: true,
                ..NavigationInput::default()
            },
            1.0,
        );
        assert!(changed);
        assert!(camera.position().x > 4.9);
        assert!(camera.position().y > 0.0);
        assert!(camera.position().z < 0.0);
    }

    #[test]
    fn look_at_round_trips_target_direction_and_clamps_pitch() {
        let position = Vec3::new(13.0, 10.0, 15.0);
        let target = Vec3::new(3.5, 1.7, 3.5);
        let camera = FlyCamera::look_at(position, target, config());
        let expected = (target - position).normalize();
        assert!(camera.basis().forward.abs_diff_eq(expected, 1.0e-5));

        let mut camera = FlyCamera::new(Vec3::ZERO, 0.0, 0.0, config());
        camera.update(
            NavigationInput {
                look_delta: Vec2::new(0.0, -10_000.0),
                focused: true,
                ..NavigationInput::default()
            },
            0.0,
        );
        assert_eq!(camera.pitch(), camera.config().pitch_max);
    }

    #[test]
    fn held_look_is_time_based() {
        let mut camera = FlyCamera::new(Vec3::ZERO, 0.0, 0.0, config());
        assert!(camera.update(
            NavigationInput {
                look_direction: Vec2::new(1.0, -1.0),
                focused: true,
                ..NavigationInput::default()
            },
            0.5,
        ));
        assert!((camera.yaw() - 0.9).abs() < 1.0e-6);
        assert!((camera.pitch() - 0.9).abs() < 1.0e-6);
    }

    #[cfg(feature = "winit")]
    #[test]
    fn winit_maps_ijkl_to_held_look_actions() {
        use winit::keyboard::KeyCode;

        assert_eq!(
            navigation_action_from_winit_key(KeyCode::KeyI),
            Some(NavigationAction::LookUp)
        );
        assert_eq!(
            navigation_action_from_winit_key(KeyCode::KeyK),
            Some(NavigationAction::LookDown)
        );
        assert_eq!(
            navigation_action_from_winit_key(KeyCode::KeyJ),
            Some(NavigationAction::LookLeft)
        );
        assert_eq!(
            navigation_action_from_winit_key(KeyCode::KeyL),
            Some(NavigationAction::LookRight)
        );
    }
}
