use wgpu;

pub const HLFS_LEVELS: u32 = 4;
pub const HLFS_RES: u32 = 128;
pub const HLFS_NEAR_FIELD: f32 = 50.0;
pub const HLFS_CASCADE_SCALE: f32 = 2.0;

/// Per-level clip-stack state
pub struct ClipStackLevel {
    /// 3D texture for this level (RGBA16F, 128³)
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    /// Toroidal origin in voxel-space
    pub origin: [i32; 3],
    pub half_extent: f32,
    pub voxel_size: f32,
}

pub struct ClipStack {
    pub levels: [ClipStackLevel; 4],
    /// Double-buffered: read (accumulated from prev frames) and write (current injection target)
    pub read: Vec<wgpu::TextureView>,
    pub write: Vec<wgpu::TextureView>,
    /// Secondary textures backing the read views
    _read_textures: Vec<wgpu::Texture>,
}

impl ClipStack {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let per_level = |label: String| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&label),
                size: wgpu::Extent3d {
                    width: HLFS_RES,
                    height: HLFS_RES,
                    depth_or_array_layers: HLFS_RES,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, texture_view)
        };

        // Write set: injected into this frame (owned by levels)
        let mut levels: [std::mem::MaybeUninit<ClipStackLevel>; 4] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut write = Vec::with_capacity(4);

        for i in 0..4 {
            let label = format!("HLFS Clip-Stack Write Level {}", i);
            let half_extent = HLFS_NEAR_FIELD * HLFS_CASCADE_SCALE.powi(i as i32);
            let voxel_size = 2.0 * half_extent / HLFS_RES as f32;
            let (texture, texture_view) = per_level(label);
            write.push(texture_view.clone());
            levels[i] = std::mem::MaybeUninit::new(ClipStackLevel {
                texture,
                texture_view,
                origin: [0; 3],
                half_extent,
                voxel_size,
            });
        }
        let levels = unsafe { std::mem::transmute::<_, [ClipStackLevel; 4]>(levels) };

        // Read set: accumulated from previous frames
        let mut read_textures = Vec::with_capacity(4);
        let mut read = Vec::with_capacity(4);
        for i in 0..4 {
            let label = format!("HLFS Clip-Stack Read Level {}", i);
            let (tex, view) = per_level(label);
            read_textures.push(tex);
            read.push(view);
        }

        ClipStack {
            levels,
            read,
            write,
            _read_textures: read_textures,
        }
    }

    /// Toroidal ring-buffer shift placeholder.
    ///
    /// When the camera moves beyond a voxel-sized threshold, this shifts the
    /// clip-stack origins and issues copy commands to wrap around the 3D
    /// texture toroidally, reusing existing data instead of discarding it.
    #[allow(unused)]
    pub fn toroidal_shift(
        &mut self,
        _device: &wgpu::Device,
        _encoder: &mut wgpu::CommandEncoder,
        _camera_pos: [f32; 3],
    ) {
        // Phase 1: no-op — origins remain at zero.
        // Phase 2 will implement the actual compute-based copy.
    }

    /// Swaps read/write roles so that this frame's accumulation becomes
    /// next frame's history.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
    }
}
