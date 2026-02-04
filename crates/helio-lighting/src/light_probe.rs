use glam::{Vec3, Mat4};

pub struct LightProbe {
    pub position: Vec3,
    pub sh_coefficients: [Vec3; 9],
    pub radius: f32,
}

impl LightProbe {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            sh_coefficients: [Vec3::ZERO; 9],
            radius: 10.0,
        }
    }
    
    pub fn sample(&self, normal: Vec3) -> Vec3 {
        // Sample spherical harmonics
        let mut result = Vec3::ZERO;
        
        result += self.sh_coefficients[0] * 0.282095;
        result += self.sh_coefficients[1] * 0.488603 * normal.y;
        result += self.sh_coefficients[2] * 0.488603 * normal.z;
        result += self.sh_coefficients[3] * 0.488603 * normal.x;
        
        result
    }
}

pub struct LightProbeVolume {
    pub probes: Vec<LightProbe>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub spacing: Vec3,
}

impl LightProbeVolume {
    pub fn new(bounds_min: Vec3, bounds_max: Vec3, spacing: Vec3) -> Self {
        let mut probes = Vec::new();
        
        let mut pos = bounds_min;
        while pos.x < bounds_max.x {
            pos.y = bounds_min.y;
            while pos.y < bounds_max.y {
                pos.z = bounds_min.z;
                while pos.z < bounds_max.z {
                    probes.push(LightProbe::new(pos));
                    pos.z += spacing.z;
                }
                pos.y += spacing.y;
            }
            pos.x += spacing.x;
        }
        
        Self {
            probes,
            bounds_min,
            bounds_max,
            spacing,
        }
    }
    
    pub fn sample(&self, position: Vec3, normal: Vec3) -> Vec3 {
        // Find nearest probes and interpolate
        let mut result = Vec3::ZERO;
        let mut weight_sum = 0.0;
        
        for probe in &self.probes {
            let dist = (probe.position - position).length();
            if dist < probe.radius {
                let weight = 1.0 - (dist / probe.radius);
                result += probe.sample(normal) * weight;
                weight_sum += weight;
            }
        }
        
        if weight_sum > 0.0 {
            result / weight_sum
        } else {
            Vec3::ZERO
        }
    }
}
