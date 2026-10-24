//! Semantic actions, suggested before session creation, with per-hand poses.

use super::rig::{Buttons, FrameInput};
use glam::{Mat4, Quat, Vec2, Vec3};
use helio_xr::{Result, XrSession};
use openxr::{Action, ActionSet, Binding, Path, Space};

pub struct Input {
    set: ActionSet,
    hands: [Path; 2],
    stick: Action<openxr::Vector2f>,
    trigger: Action<bool>,
    squeeze: Action<bool>,
    primary: Action<bool>,
    secondary: Action<bool>,
    grip: Action<openxr::Posef>,
    aim: Action<openxr::Posef>,
    spaces: Option<Spaces>,
}

struct Spaces {
    head: Space,
    grip: [Space; 2],
    aim: Space,
}

impl Input {
    pub fn new(instance: &openxr::Instance) -> Result<Self> {
        let set = instance.create_action_set("voxel_craft", "Voxel Craft", 0)?;
        let hands = [
            instance.string_to_path("/user/hand/left")?,
            instance.string_to_path("/user/hand/right")?,
        ];
        let input = Self {
            stick: set.create_action("move_turn", "Move / turn / hotbar", &hands)?,
            trigger: set.create_action("dig_build", "Dig / build", &hands)?,
            squeeze: set.create_action("sneak_sprint", "Sneak / sprint", &hands)?,
            primary: set.create_action("pick_jump", "Pick block / jump", &hands)?,
            secondary: set.create_action("recenter_fly", "Recenter / toggle flight", &hands)?,
            grip: set.create_action("grip", "Hand position", &hands)?,
            aim: set.create_action("aim", "Point at blocks", &hands)?,
            set,
            hands,
            spaces: None,
        };
        input.bind(instance)?;
        Ok(input)
    }

    fn bind(&self, instance: &openxr::Instance) -> Result<()> {
        for profile in [
            "oculus/touch_controller",
            "valve/index_controller",
            "microsoft/motion_controller",
            "htc/vive_controller",
            "khr/simple_controller",
        ] {
            let mut bindings = Vec::new();
            for (hand, side) in ["left", "right"].into_iter().enumerate() {
                let path = |component: &str| {
                    instance.string_to_path(&format!("/user/hand/{side}/input/{component}"))
                };
                bindings.push(Binding::new(&self.grip, path("grip/pose")?));
                if hand == 1 {
                    bindings.push(Binding::new(&self.aim, path("aim/pose")?));
                }
                if profile == "khr/simple_controller" {
                    bindings.push(Binding::new(&self.trigger, path("select/click")?));
                    bindings.push(Binding::new(&self.primary, path("menu/click")?));
                    continue;
                }
                let vive = profile == "htc/vive_controller";
                let motion = profile == "microsoft/motion_controller";
                bindings.push(Binding::new(
                    &self.stick,
                    path(if vive { "trackpad" } else { "thumbstick" })?,
                ));
                bindings.push(Binding::new(&self.trigger, path("trigger/value")?));
                bindings.push(Binding::new(
                    &self.squeeze,
                    path(if vive || motion {
                        "squeeze/click"
                    } else {
                        "squeeze/value"
                    })?,
                ));
                let primary = match profile {
                    "oculus/touch_controller" if hand == 0 => "x/click",
                    "oculus/touch_controller" | "valve/index_controller" => "a/click",
                    "htc/vive_controller" => "trackpad/click",
                    _ => "thumbstick/click",
                };
                let secondary = match profile {
                    "oculus/touch_controller" if hand == 0 => "y/click",
                    "oculus/touch_controller" | "valve/index_controller" => "b/click",
                    _ => "menu/click",
                };
                bindings.push(Binding::new(&self.primary, path(primary)?));
                bindings.push(Binding::new(&self.secondary, path(secondary)?));
            }
            let profile_path =
                instance.string_to_path(&format!("/interaction_profiles/{profile}"))?;
            // Profiles unknown to a runtime do not disable the other controllers.
            if let Err(error) =
                instance.suggest_interaction_profile_bindings(profile_path, &bindings)
            {
                log::debug!("[XR] {profile} bindings unavailable: {error}");
            }
        }
        Ok(())
    }

    pub fn attach(&mut self, session: &XrSession) -> Result<()> {
        session.session.attach_action_sets(&[&self.set])?;
        self.spaces = Some(Spaces {
            head: session.session.create_reference_space(
                openxr::ReferenceSpaceType::VIEW,
                openxr::Posef::IDENTITY,
            )?,
            grip: [
                self.grip
                    .create_space(&session.session, self.hands[0], openxr::Posef::IDENTITY)?,
                self.grip
                    .create_space(&session.session, self.hands[1], openxr::Posef::IDENTITY)?,
            ],
            aim: self
                .aim
                .create_space(&session.session, self.hands[1], openxr::Posef::IDENTITY)?,
        });
        Ok(())
    }

    pub fn sample(&self, session: &XrSession, time: openxr::Time) -> Result<Option<FrameInput>> {
        let Some(spaces) = &self.spaces else {
            return Ok(None);
        };
        let focused = session.session_state == openxr::SessionState::FOCUSED;
        let Some(head) = locate(&spaces.head, &session.stage, time) else {
            return Ok(None);
        };
        if focused {
            session.session.sync_actions(&[(&self.set).into()])?;
        }
        let button = |action: &Action<bool>, hand: usize| {
            focused
                && matches!(action.state(&session.session, self.hands[hand]), Ok(state) if state.is_active && state.current_state)
        };
        let stick = |hand: usize| {
            if focused {
                if let Ok(state) = self.stick.state(&session.session, self.hands[hand]) {
                    if state.is_active {
                        return Vec2::new(state.current_state.x, state.current_state.y);
                    }
                }
            }
            Vec2::ZERO
        };
        let tracked = |action: &Action<openxr::Posef>, hand: usize, space: &Space| {
            if focused
                && action
                    .is_active(&session.session, self.hands[hand])
                    .unwrap_or(false)
            {
                locate(space, &session.stage, time)
            } else {
                None
            }
        };
        Ok(Some(FrameInput {
            focused,
            head,
            grips: [
                tracked(&self.grip, 0, &spaces.grip[0]),
                tracked(&self.grip, 1, &spaces.grip[1]),
            ],
            aim: tracked(&self.aim, 1, &spaces.aim),
            movement: stick(0),
            turn: stick(1),
            buttons: Buttons {
                mine: button(&self.trigger, 1),
                place: button(&self.trigger, 0),
                jump: button(&self.primary, 1),
                pick: button(&self.primary, 0),
                fly: button(&self.secondary, 1),
                recenter: button(&self.secondary, 0),
                sneak: button(&self.squeeze, 0),
                sprint: button(&self.squeeze, 1),
            },
        }))
    }
}

fn locate(space: &Space, stage: &Space, time: openxr::Time) -> Option<Mat4> {
    let location = space.locate(stage, time).ok()?;
    let valid =
        openxr::SpaceLocationFlags::POSITION_VALID | openxr::SpaceLocationFlags::ORIENTATION_VALID;
    if !location.location_flags.contains(valid) {
        return None;
    }
    let p = location.pose.position;
    let q = location.pose.orientation;
    let position = Vec3::new(p.x, p.y, p.z);
    let rotation = Quat::from_xyzw(q.x, q.y, q.z, q.w);
    if !position.is_finite() || !rotation.is_finite() || rotation.length_squared() < 0.5 {
        return None;
    }
    Some(Mat4::from_rotation_translation(
        rotation.normalize(),
        position,
    ))
}
