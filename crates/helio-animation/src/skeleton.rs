use glam::{Mat4, Quat, Vec3};

pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    pub local_transform: Mat4,
    pub inverse_bind_pose: Mat4,
}

pub struct Skeleton {
    pub bones: Vec<Bone>,
    pub bone_names: std::collections::HashMap<String, usize>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self {
            bones: Vec::new(),
            bone_names: std::collections::HashMap::new(),
        }
    }
    
    pub fn add_bone(&mut self, name: String, parent: Option<usize>, local_transform: Mat4) -> usize {
        let index = self.bones.len();
        self.bone_names.insert(name.clone(), index);
        self.bones.push(Bone {
            name,
            parent,
            local_transform,
            inverse_bind_pose: Mat4::IDENTITY,
        });
        index
    }
    
    pub fn compute_bone_matrices(&self) -> Vec<Mat4> {
        let mut matrices = vec![Mat4::IDENTITY; self.bones.len()];
        
        for (i, bone) in self.bones.iter().enumerate() {
            let parent_matrix = bone.parent
                .map(|p| matrices[p])
                .unwrap_or(Mat4::IDENTITY);
            
            matrices[i] = parent_matrix * bone.local_transform;
        }
        
        matrices
    }
}

impl Default for Skeleton {
    fn default() -> Self {
        Self::new()
    }
}
