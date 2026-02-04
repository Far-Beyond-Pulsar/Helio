use glam::{Quat, Vec3};

pub struct KeyFrame<T> {
    pub time: f32,
    pub value: T,
}

pub struct AnimationChannel {
    pub bone_index: usize,
    pub position_keys: Vec<KeyFrame<Vec3>>,
    pub rotation_keys: Vec<KeyFrame<Quat>>,
    pub scale_keys: Vec<KeyFrame<Vec3>>,
}

pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
    pub looping: bool,
}

impl AnimationClip {
    pub fn new(name: String, duration: f32) -> Self {
        Self {
            name,
            duration,
            channels: Vec::new(),
            looping: true,
        }
    }
    
    pub fn sample(&self, time: f32) -> Vec<(usize, Vec3, Quat, Vec3)> {
        let mut transforms = Vec::new();
        let t = if self.looping {
            time % self.duration
        } else {
            time.min(self.duration)
        };
        
        for channel in &self.channels {
            let position = Self::sample_vec3(&channel.position_keys, t);
            let rotation = Self::sample_quat(&channel.rotation_keys, t);
            let scale = Self::sample_vec3(&channel.scale_keys, t);
            
            transforms.push((channel.bone_index, position, rotation, scale));
        }
        
        transforms
    }
    
    fn sample_vec3(keys: &[KeyFrame<Vec3>], time: f32) -> Vec3 {
        if keys.is_empty() {
            return Vec3::ZERO;
        }
        
        if keys.len() == 1 || time <= keys[0].time {
            return keys[0].value;
        }
        
        if time >= keys[keys.len() - 1].time {
            return keys[keys.len() - 1].value;
        }
        
        for i in 0..keys.len() - 1 {
            if time >= keys[i].time && time < keys[i + 1].time {
                let t = (time - keys[i].time) / (keys[i + 1].time - keys[i].time);
                return keys[i].value.lerp(keys[i + 1].value, t);
            }
        }
        
        keys[0].value
    }
    
    fn sample_quat(keys: &[KeyFrame<Quat>], time: f32) -> Quat {
        if keys.is_empty() {
            return Quat::IDENTITY;
        }
        
        if keys.len() == 1 || time <= keys[0].time {
            return keys[0].value;
        }
        
        if time >= keys[keys.len() - 1].time {
            return keys[keys.len() - 1].value;
        }
        
        for i in 0..keys.len() - 1 {
            if time >= keys[i].time && time < keys[i + 1].time {
                let t = (time - keys[i].time) / (keys[i + 1].time - keys[i].time);
                return keys[i].value.slerp(keys[i + 1].value, t);
            }
        }
        
        keys[0].value
    }
}
