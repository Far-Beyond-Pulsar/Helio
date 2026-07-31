//! OpenXR controller input: thumbsticks, grip poses and a click, via the action system.
//!
//! # Why this is not just "read the stick"
//!
//! OpenXR has no API for polling a device directly. Input goes through *actions*: you
//! declare semantic actions ("move", "turn") up front, suggest bindings for each
//! interaction profile you know about, attach the action set to the session, and sync
//! once per frame. The runtime then decides which physical control drives which action —
//! which is what lets the same build work on Touch, Index and WMR controllers without
//! knowing which one is plugged in.
//!
//! The consequence worth knowing: **bindings must be suggested before the session is
//! created, and the action set attached exactly once, before the first sync.** Attaching
//! twice, or suggesting a binding for a profile path the runtime does not know, fails the
//! whole call rather than the offending entry.
//!
//! # Profiles
//!
//! Bindings are suggested for the three profiles that cover essentially all desktop
//! hardware, plus the KHR simple controller as a floor. A profile the runtime does not
//! recognise is skipped rather than aborting setup, because a runtime that has never
//! heard of, say, the Index controller should not prevent a Touch user from moving.

use openxr::{Action, ActionSet, Binding, Path, Session};

use crate::{Result, XrError};

/// Thumbstick and button state for one frame, already resolved across whichever
/// controllers the runtime bound.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControllerState {
    /// Left stick, x = right, y = forward. Range roughly [-1, 1] per axis.
    pub left_stick: glam::Vec2,
    /// Right stick, same convention. Conventionally drives turning.
    pub right_stick: glam::Vec2,
    /// Primary click (A / trigger, depending on profile) on either hand.
    pub select: bool,
}

/// The action set plus the actions Helio's demos use.
pub struct XrInput {
    action_set: ActionSet,
    left_stick: Action<openxr::Vector2f>,
    right_stick: Action<openxr::Vector2f>,
    select: Action<bool>,
    grip: Action<openxr::Posef>,
    left_hand: Path,
    right_hand: Path,
    /// Lazily-created per-hand grip action spaces, in `[left, right]` order.
    grip_spaces: Option<[openxr::Space; 2]>,
    /// Stage reference space the grip poses are located relative to.
    grip_stage: Option<openxr::Space>,
}

impl XrInput {
    /// Declare actions and suggest bindings. Call **before** creating the session.
    pub fn new(instance: &openxr::Instance) -> Result<Self> {
        let action_set = instance
            .create_action_set("helio", "Helio Input", 0)
            .map_err(|e| XrError::Platform(format!("create_action_set: {e}")))?;

        let left_hand = instance
            .string_to_path("/user/hand/left")
            .map_err(|e| XrError::Platform(format!("string_to_path: {e}")))?;
        let right_hand = instance
            .string_to_path("/user/hand/right")
            .map_err(|e| XrError::Platform(format!("string_to_path: {e}")))?;
        let hands = [left_hand, right_hand];

        let left_stick = action_set
            .create_action::<openxr::Vector2f>("move", "Move", &hands)
            .map_err(|e| XrError::Platform(format!("create_action(move): {e}")))?;
        let right_stick = action_set
            .create_action::<openxr::Vector2f>("turn", "Turn", &hands)
            .map_err(|e| XrError::Platform(format!("create_action(turn): {e}")))?;
        let select = action_set
            .create_action::<bool>("select", "Select", &hands)
            .map_err(|e| XrError::Platform(format!("create_action(select): {e}")))?;
        let grip = action_set
            .create_action::<openxr::Posef>("grip", "Grip Pose", &hands)
            .map_err(|e| XrError::Platform(format!("create_action(grip): {e}")))?;

        let input = Self {
            action_set,
            left_stick,
            right_stick,
            select,
            grip,
            left_hand,
            right_hand,
            grip_spaces: None,
            grip_stage: None,
        };
        input.suggest_bindings(instance)?;
        Ok(input)
    }

    fn suggest_bindings(&self, instance: &openxr::Instance) -> Result<()> {
        // Bindings suggested per interaction profile. The stick component paths differ by
        // vendor — Touch and WMR call it `thumbstick`, Index calls it `thumbstick` too but
        // reports a `trackpad` as well, and the simple controller has no stick at all
        // (click only, so a headset with bare controllers still reports *something* rather
        // than failing setup). Grip pose paths are vendor-neutral `/input/grip/pose`; only
        // the click-only simple controller has no pose binding.
        struct Profile {
            path: &'static str,
            left_stick: Option<&'static str>,
            right_stick: Option<&'static str>,
            click: &'static str,
            left_grip: Option<&'static str>,
            right_grip: Option<&'static str>,
        }

        const PROFILES: &[Profile] = &[
            Profile {
                path: "/interaction_profiles/oculus/touch_controller",
                left_stick: Some("/user/hand/left/input/thumbstick"),
                right_stick: Some("/user/hand/right/input/thumbstick"),
                click: "/user/hand/right/input/a/click",
                left_grip: Some("/user/hand/left/input/grip/pose"),
                right_grip: Some("/user/hand/right/input/grip/pose"),
            },
            Profile {
                path: "/interaction_profiles/valve/index_controller",
                left_stick: Some("/user/hand/left/input/thumbstick"),
                right_stick: Some("/user/hand/right/input/thumbstick"),
                click: "/user/hand/right/input/a/click",
                left_grip: Some("/user/hand/left/input/grip/pose"),
                right_grip: Some("/user/hand/right/input/grip/pose"),
            },
            Profile {
                path: "/interaction_profiles/microsoft/motion_controller",
                left_stick: Some("/user/hand/left/input/thumbstick"),
                right_stick: Some("/user/hand/right/input/thumbstick"),
                click: "/user/hand/right/input/trigger/value",
                left_grip: Some("/user/hand/left/input/grip/pose"),
                right_grip: Some("/user/hand/right/input/grip/pose"),
            },
            Profile {
                path: "/interaction_profiles/khr/simple_controller",
                left_stick: None,
                right_stick: None,
                click: "/user/hand/right/input/select/click",
                left_grip: None,
                right_grip: None,
            },
        ];

        for profile in PROFILES {
            // Resolve every path first. A runtime that does not know this profile fails
            // here, and that must skip the profile rather than abort input entirely —
            // otherwise one unknown headset makes every other headset unusable.
            let Ok(profile_path) = instance.string_to_path(profile.path) else {
                continue;
            };

            let mut bindings: Vec<Binding> = Vec::new();
            if let Some(left) = profile.left_stick {
                if let Ok(path) = instance.string_to_path(left) {
                    bindings.push(Binding::new(&self.left_stick, path));
                }
            }
            if let Some(right) = profile.right_stick {
                if let Ok(path) = instance.string_to_path(right) {
                    bindings.push(Binding::new(&self.right_stick, path));
                }
            }
            if let Ok(path) = instance.string_to_path(profile.click) {
                bindings.push(Binding::new(&self.select, path));
            }
            if let Some(left) = profile.left_grip {
                if let Ok(path) = instance.string_to_path(left) {
                    bindings.push(Binding::new(&self.grip, path));
                }
            }
            if let Some(right) = profile.right_grip {
                if let Ok(path) = instance.string_to_path(right) {
                    bindings.push(Binding::new(&self.grip, path));
                }
            }
            if bindings.is_empty() {
                continue;
            }

            if let Err(error) = instance.suggest_interaction_profile_bindings(profile_path, &bindings)
            {
                log::debug!("[XR] runtime rejected bindings for {}: {error}", profile.path);
            }
        }
        Ok(())
    }

    /// Attach the action set to the session. Call once, after session creation and
    /// before the first [`Self::sync`]; the runtime rejects a second attach.
    pub fn attach<G: openxr::Graphics>(&self, session: &Session<G>) -> Result<()> {
        session
            .attach_action_sets(&[&self.action_set])
            .map_err(|e| XrError::Platform(format!("attach_action_sets: {e}")))
    }

    /// Sync the action set and read this frame's state.
    ///
    /// Returns the default (all-zero) state when the session is not focused — the
    /// runtime reports actions as inactive then, and treating that as "stick centred" is
    /// what stops the player drifting while the menu is up.
    pub fn sync<G: openxr::Graphics>(&self, session: &Session<G>) -> Result<ControllerState> {
        session
            .sync_actions(&[(&self.action_set).into()])
            .map_err(|e| XrError::Platform(format!("sync_actions: {e}")))?;

        let read_stick = |action: &Action<openxr::Vector2f>, hand: Path| -> glam::Vec2 {
            match action.state(session, hand) {
                // `is_active` false means the runtime has not bound this action on this
                // hand (or the session is not focused). Reporting centred rather than
                // stale is what keeps the player from drifting while a menu is up.
                Ok(value) if value.is_active => {
                    glam::Vec2::new(value.current_state.x, value.current_state.y)
                }
                _ => glam::Vec2::ZERO,
            }
        };

        let select = [self.left_hand, self.right_hand].iter().any(|hand| {
            matches!(self.select.state(session, *hand), Ok(v) if v.is_active && v.current_state)
        });

        Ok(ControllerState {
            left_stick: read_stick(&self.left_stick, self.left_hand),
            right_stick: read_stick(&self.right_stick, self.right_hand),
            select,
        })
    }

    /// Locate both controllers' grip poses and return them as world-space
    /// matrices, ready to use as object transforms.
    ///
    /// `world_from_stage` maps the XR stage space (the room/floor origin) onto
    /// the engine's world space — the same matrix passed to
    /// `Renderer::set_xr_stage_transform`. Each returned matrix is therefore
    /// `world_from_stage * stage_from_grip`. A `None` entry means that hand is
    /// not currently tracked (controller off, session not focused, runtime
    /// hasn't bound the pose action yet, …); callers should keep the previous
    /// transform rather than snapping to the origin.
    ///
    /// The per-hand action spaces (and the stage base space) are created on the
    /// first call and reused, so per-frame cost is two `locate_space` calls.
    /// `time` should be the frame's display time (see
    /// `Renderer::xr_last_display_time`) so the poses match the rendered frame
    /// as closely as the runtime allows.
    pub fn grip_pose_matrices<G: openxr::Graphics>(
        &mut self,
        session: &Session<G>,
        time: openxr::Time,
        world_from_stage: &glam::Mat4,
    ) -> Result<[Option<glam::Mat4>; 2]> {
        if self.grip_spaces.is_none() {
            let identity = openxr::Posef {
                orientation: openxr::Quaternionf {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                position: openxr::Vector3f {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            };
            let stage = session
                .create_reference_space(openxr::ReferenceSpaceType::STAGE, identity)
                .map_err(|e| XrError::Platform(format!("create_reference_space(STAGE): {e}")))?;
            let left = self
                .grip
                .create_space(session, self.left_hand, identity)
                .map_err(|e| XrError::Platform(format!("create_action_space(left grip): {e}")))?;
            let right = self
                .grip
                .create_space(session, self.right_hand, identity)
                .map_err(|e| XrError::Platform(format!("create_action_space(right grip): {e}")))?;
            self.grip_spaces = Some([left, right]);
            self.grip_stage = Some(stage);
        }

        let spaces = self.grip_spaces.as_ref().expect("grip spaces created");
        let stage = self.grip_stage.as_ref().expect("grip stage created");

        let mut out = [None, None];
        for (i, space) in spaces.iter().enumerate() {
            if let Ok(location) = space.locate(stage, time) {
                if location
                    .location_flags
                    .contains(openxr::SpaceLocationFlags::POSITION_VALID)
                    && location
                        .location_flags
                        .contains(openxr::SpaceLocationFlags::ORIENTATION_VALID)
                {
                    out[i] = Some(*world_from_stage * pose_to_mat4(&location.pose));
                }
            }
        }
        Ok(out)
    }
}

fn pose_to_mat4(pose: &openxr::Posef) -> glam::Mat4 {
    let q = pose.orientation;
    let p = pose.position;
    // OpenXR quaternions are stored (x, y, z, w); glam uses (x, y, z, w) too.
    glam::Mat4::from_quat(glam::Quat::from_xyzw(q.x, q.y, q.z, q.w))
        * glam::Mat4::from_translation(glam::Vec3::new(p.x, p.y, p.z))
}
