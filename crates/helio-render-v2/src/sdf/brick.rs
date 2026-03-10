//! Sparse Brick Map — CPU-side classification and GPU buffer management
//!
//! Divides the SDF volume into a grid of 8^3 bricks. Only bricks whose AABB
//! intersects at least one edit's bounding sphere are allocated atlas slots
//! and evaluated on the GPU.

use super::edit_list::SdfEdit;
use super::primitives::SdfShapeType;
use std::sync::Arc;

/// Voxels per brick edge
pub const DEFAULT_BRICK_SIZE: u32 = 8;

/// Sentinel value in the brick index meaning "no brick allocated"
pub const EMPTY_BRICK: u32 = 0xFFFF_FFFF;

/// CPU-side brick map managing classification and atlas slot allocation.
pub struct BrickMap {
    brick_size: u32,
    brick_grid_dim: u32,
    atlas_bricks_per_axis: u32,

    // CPU-side index state
    brick_index: Vec<u32>,
    active_bricks: Vec<u32>,
    atlas_next_slot: u32,
    atlas_free_list: Vec<u32>,

    // GPU buffers (created in create_gpu_resources)
    pub brick_index_buffer: Option<Arc<wgpu::Buffer>>,
    pub active_bricks_buffer: Option<Arc<wgpu::Buffer>>,
    pub atlas_texture: Option<Arc<wgpu::Texture>>,
    pub atlas_view: Option<Arc<wgpu::TextureView>>,
}

impl BrickMap {
    pub fn new(grid_dim: u32, brick_size: u32, atlas_bricks_per_axis: u32) -> Self {
        let brick_grid_dim = grid_dim / brick_size;
        let total_bricks = (brick_grid_dim * brick_grid_dim * brick_grid_dim) as usize;

        Self {
            brick_size,
            brick_grid_dim,
            atlas_bricks_per_axis,
            brick_index: vec![EMPTY_BRICK; total_bricks],
            active_bricks: Vec::new(),
            atlas_next_slot: 0,
            atlas_free_list: Vec::new(),
            brick_index_buffer: None,
            active_bricks_buffer: None,
            atlas_texture: None,
            atlas_view: None,
        }
    }

    /// Create GPU resources. Called once from `SdfFeature::register()`.
    pub fn create_gpu_resources(&mut self, device: &wgpu::Device) {
        let bgd = self.brick_grid_dim;
        let total_bricks = bgd * bgd * bgd;
        let max_atlas_bricks = self.atlas_bricks_per_axis.pow(3);

        // Brick index buffer: one u32 per brick cell
        let brick_index_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Brick Index"),
            size: (total_bricks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Active bricks buffer: worst case = all bricks active
        let active_bricks_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Active Bricks"),
            size: (max_atlas_bricks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Atlas texture: atlas_bricks_per_axis * brick_size per dimension
        let atlas_dim = self.atlas_bricks_per_axis * self.brick_size;
        let atlas_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Brick Atlas"),
            size: wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let atlas_view = Arc::new(atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        }));

        self.brick_index_buffer = Some(brick_index_buffer);
        self.active_bricks_buffer = Some(active_bricks_buffer);
        self.atlas_texture = Some(atlas_texture);
        self.atlas_view = Some(atlas_view);
    }

    /// Classify which bricks are active based on edit bounding spheres.
    /// Returns `true` if the active set changed.
    pub fn classify(
        &mut self,
        edits: &[SdfEdit],
        volume_min: [f32; 3],
        volume_max: [f32; 3],
    ) -> bool {
        // Compute bounding spheres for all edits
        let bounds: Vec<(glam::Vec3, f32)> = edits
            .iter()
            .map(|edit| Self::edit_bounding_sphere(edit))
            .collect();

        let bgd = self.brick_grid_dim;
        let brick_world_size = [
            (volume_max[0] - volume_min[0]) / bgd as f32,
            (volume_max[1] - volume_min[1]) / bgd as f32,
            (volume_max[2] - volume_min[2]) / bgd as f32,
        ];

        let old_count = self.active_bricks.len();
        self.active_bricks.clear();

        for bz in 0..bgd {
            for by in 0..bgd {
                for bx in 0..bgd {
                    let linear = bx + by * bgd + bz * bgd * bgd;
                    let brick_min = glam::Vec3::new(
                        volume_min[0] + bx as f32 * brick_world_size[0],
                        volume_min[1] + by as f32 * brick_world_size[1],
                        volume_min[2] + bz as f32 * brick_world_size[2],
                    );
                    let brick_max = glam::Vec3::new(
                        brick_min.x + brick_world_size[0],
                        brick_min.y + brick_world_size[1],
                        brick_min.z + brick_world_size[2],
                    );

                    let is_active = bounds.iter().any(|&(center, radius)| {
                        Self::sphere_aabb_intersect(center, radius, brick_min, brick_max)
                    });

                    if is_active {
                        // Allocate atlas slot if needed
                        if self.brick_index[linear as usize] == EMPTY_BRICK {
                            let slot = self.alloc_atlas_slot();
                            self.brick_index[linear as usize] = slot;
                        }
                        self.active_bricks.push(linear);
                    } else {
                        // Recycle atlas slot if brick was previously active
                        if self.brick_index[linear as usize] != EMPTY_BRICK {
                            self.atlas_free_list.push(self.brick_index[linear as usize]);
                            self.brick_index[linear as usize] = EMPTY_BRICK;
                        }
                    }
                }
            }
        }

        // Check if the active set changed
        self.active_bricks.len() != old_count
    }

    /// Upload brick_index and active_bricks to GPU.
    pub fn upload(&self, queue: &wgpu::Queue) {
        if let Some(buf) = &self.brick_index_buffer {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.brick_index));
        }
        if let Some(buf) = &self.active_bricks_buffer {
            if !self.active_bricks.is_empty() {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.active_bricks));
            }
        }
    }

    /// Number of currently active bricks.
    pub fn active_count(&self) -> u32 {
        self.active_bricks.len() as u32
    }

    pub fn brick_size(&self) -> u32 {
        self.brick_size
    }

    pub fn brick_grid_dim(&self) -> u32 {
        self.brick_grid_dim
    }

    pub fn atlas_bricks_per_axis(&self) -> u32 {
        self.atlas_bricks_per_axis
    }

    /// Raw brick index slice (for concatenation in clip maps).
    pub fn brick_index_slice(&self) -> &[u32] {
        &self.brick_index
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn alloc_atlas_slot(&mut self) -> u32 {
        if let Some(slot) = self.atlas_free_list.pop() {
            slot
        } else {
            let slot = self.atlas_next_slot;
            let max_slots = self.atlas_bricks_per_axis.pow(3);
            assert!(
                slot < max_slots,
                "Brick atlas overflow: tried to allocate slot {} but max is {}",
                slot,
                max_slots
            );
            self.atlas_next_slot += 1;
            slot
        }
    }

    /// Compute a world-space bounding sphere for an edit.
    fn edit_bounding_sphere(edit: &SdfEdit) -> (glam::Vec3, f32) {
        // Center is the translation column of the transform
        let center = glam::Vec3::new(
            edit.transform.w_axis.x,
            edit.transform.w_axis.y,
            edit.transform.w_axis.z,
        );

        // Local-space extent depends on shape type
        let local_extent = match edit.shape {
            SdfShapeType::Sphere => edit.params.param0,
            SdfShapeType::Cube => {
                let hx = edit.params.param0;
                let hy = edit.params.param1;
                let hz = edit.params.param2;
                (hx * hx + hy * hy + hz * hz).sqrt()
            }
            SdfShapeType::Capsule => edit.params.param0 + edit.params.param1,
            SdfShapeType::Torus => edit.params.param0 + edit.params.param1,
            SdfShapeType::Cylinder => {
                let r = edit.params.param0;
                let h = edit.params.param1;
                (r * r + h * h).sqrt()
            }
        };

        // Account for non-uniform scaling
        let sx = edit.transform.x_axis.truncate().length();
        let sy = edit.transform.y_axis.truncate().length();
        let sz = edit.transform.z_axis.truncate().length();
        let max_scale = sx.max(sy).max(sz);

        let world_radius = local_extent * max_scale + edit.blend_radius;
        (center, world_radius)
    }

    /// Sphere-AABB intersection test.
    fn sphere_aabb_intersect(
        center: glam::Vec3,
        radius: f32,
        aabb_min: glam::Vec3,
        aabb_max: glam::Vec3,
    ) -> bool {
        // Find the closest point on the AABB to the sphere center
        let closest = center.clamp(aabb_min, aabb_max);
        let dist_sq = (closest - center).length_squared();
        dist_sq <= radius * radius
    }
}
