use glam::Vec2;

pub struct TAA {
    pub enabled: bool,
    pub jitter_sequence: JitterSequence,
    pub history_blend: f32,
    pub sharpness: f32,
    pub variance_clipping: bool,
    pub motion_blur_scale: f32,
}

impl Default for TAA {
    fn default() -> Self {
        Self {
            enabled: true,
            jitter_sequence: JitterSequence::Halton,
            history_blend: 0.95,
            sharpness: 0.5,
            variance_clipping: true,
            motion_blur_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterSequence {
    Halton,
    R2,
    UniformRandom,
}

impl JitterSequence {
    pub fn get_jitter(&self, frame: u64, resolution: (u32, u32)) -> Vec2 {
        match self {
            JitterSequence::Halton => Self::halton_jitter(frame, resolution),
            JitterSequence::R2 => Self::r2_jitter(frame, resolution),
            JitterSequence::UniformRandom => Vec2::ZERO,
        }
    }
    
    fn halton_jitter(frame: u64, resolution: (u32, u32)) -> Vec2 {
        let x = Self::halton(frame, 2);
        let y = Self::halton(frame, 3);
        
        Vec2::new(
            (x - 0.5) / resolution.0 as f32,
            (y - 0.5) / resolution.1 as f32,
        )
    }
    
    fn r2_jitter(frame: u64, resolution: (u32, u32)) -> Vec2 {
        let g = 1.32471795724474602596;
        let a1 = 1.0 / g;
        let a2 = 1.0 / (g * g);
        
        let x = (0.5 + a1 * frame as f32) % 1.0;
        let y = (0.5 + a2 * frame as f32) % 1.0;
        
        Vec2::new(
            (x - 0.5) / resolution.0 as f32,
            (y - 0.5) / resolution.1 as f32,
        )
    }
    
    fn halton(index: u64, base: u64) -> f32 {
        let mut result = 0.0;
        let mut f = 1.0;
        let mut i = index;
        
        while i > 0 {
            f /= base as f32;
            result += f * (i % base) as f32;
            i /= base;
        }
        
        result
    }
}
