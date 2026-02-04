use glam::Vec3;

pub struct Heightmap {
    pub width: u32,
    pub height: u32,
    pub heights: Vec<f32>,
    pub min_height: f32,
    pub max_height: f32,
}

impl Heightmap {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            heights: vec![0.0; (width * height) as usize],
            min_height: 0.0,
            max_height: 100.0,
        }
    }
    
    pub fn get_height(&self, x: u32, y: u32) -> f32 {
        let index = (y * self.width + x) as usize;
        self.heights.get(index).copied().unwrap_or(0.0)
    }
    
    pub fn set_height(&mut self, x: u32, y: u32, height: f32) {
        let index = (y * self.width + x) as usize;
        if let Some(h) = self.heights.get_mut(index) {
            *h = height;
        }
    }
    
    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        let x = u * (self.width - 1) as f32;
        let y = v * (self.height - 1) as f32;
        
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        
        let h00 = self.get_height(x0, y0);
        let h10 = self.get_height(x1, y0);
        let h01 = self.get_height(x0, y1);
        let h11 = self.get_height(x1, y1);
        
        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;
        
        h0 * (1.0 - fy) + h1 * fy
    }
    
    pub fn compute_normal(&self, x: u32, y: u32) -> Vec3 {
        let h_l = self.get_height(x.saturating_sub(1), y);
        let h_r = self.get_height((x + 1).min(self.width - 1), y);
        let h_d = self.get_height(x, y.saturating_sub(1));
        let h_u = self.get_height(x, (y + 1).min(self.height - 1));
        
        let dx = h_r - h_l;
        let dy = h_u - h_d;
        
        Vec3::new(-dx, 2.0, -dy).normalize()
    }
}
