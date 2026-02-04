pub struct AnimationState {
    pub name: String,
    pub clip_index: usize,
    pub transitions: Vec<StateTransition>,
}

pub struct StateTransition {
    pub target_state: usize,
    pub condition: TransitionCondition,
    pub blend_duration: f32,
}

pub enum TransitionCondition {
    Always,
    Parameter(String, f32),
}

pub struct StateMachine {
    pub states: Vec<AnimationState>,
    pub current_state: usize,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            current_state: 0,
        }
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}
