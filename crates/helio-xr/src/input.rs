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
    left_hand: Path,
    right_hand: Path,
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

        let input = Self {
            action_set,
            left_stick,
            right_stick,
            select,
            left_hand,
            right_hand,
        };
        input.suggest_bindings(instance)?;
        Ok(input)
    }

    fn suggest_bindings(&self, instance: &openxr::Instance) -> Result<()> {
        // (profile path, left stick, right stick, click)
        //
        // The stick component paths differ by vendor — Touch and WMR call it
        // `thumbstick`, Index calls it `thumbstick` too but reports a `trackpad` as well,
        // and the simple controller has no stick at all (click only, so a headset with
        // bare controllers still reports *something* rather than failing setup).
        const PROFILES: &[(&str, Option<&str>, Option<&str>, &str)] = &[
            (
                "/interaction_profiles/oculus/touch_controller",
                Some("/user/hand/left/input/thumbstick"),
                Some("/user/hand/right/input/thumbstick"),
                "/user/hand/right/input/a/click",
            ),
            (
                "/interaction_profiles/valve/index_controller",
                Some("/user/hand/left/input/thumbstick"),
                Some("/user/hand/right/input/thumbstick"),
                "/user/hand/right/input/a/click",
            ),
            (
                "/interaction_profiles/microsoft/motion_controller",
                Some("/user/hand/left/input/thumbstick"),
                Some("/user/hand/right/input/thumbstick"),
                "/user/hand/right/input/trigger/value",
            ),
            (
                "/interaction_profiles/khr/simple_controller",
                None,
                None,
                "/user/hand/right/input/select/click",
            ),
        ];

        for (profile, left, right, click) in PROFILES {
            // Resolve every path first. A runtime that does not know this profile fails
            // here, and that must skip the profile rather than abort input entirely —
            // otherwise one unknown headset makes every other headset unusable.
            let Ok(profile_path) = instance.string_to_path(profile) else {
                continue;
            };

            let mut bindings: Vec<Binding> = Vec::new();
            if let Some(left) = left {
                if let Ok(path) = instance.string_to_path(left) {
                    bindings.push(Binding::new(&self.left_stick, path));
                }
            }
            if let Some(right) = right {
                if let Ok(path) = instance.string_to_path(right) {
                    bindings.push(Binding::new(&self.right_stick, path));
                }
            }
            if let Ok(path) = instance.string_to_path(click) {
                bindings.push(Binding::new(&self.select, path));
            }
            if bindings.is_empty() {
                continue;
            }

            if let Err(error) = instance.suggest_interaction_profile_bindings(profile_path, &bindings)
            {
                log::debug!("[XR] runtime rejected bindings for {profile}: {error}");
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
}
