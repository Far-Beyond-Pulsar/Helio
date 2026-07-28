pub mod pipeline;
pub mod simulation;

use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use wgpu::util::DeviceExt;
use std::f32::consts::PI;

/// Simple fullscreen blit: copies a texture to the render target as-is.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:  texture_2d<f32>;
@group(0) @binding(1) var blit_samp: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return V(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}
@fragment fn fs(in: V) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_samp, in.uv);
}
";

const SIM_SIZE: u32 = 256;
const CAUSTICS_SIZE: u32 = 256;
const MAX_DROPS_BUFFERED: usize = 16;
pub(crate) const CASCADE_COUNT: usize = 3;
pub(crate) const MAX_SIM_VOLUMES: u32 = 8;
pub(crate) const CASCADE_PATCH_SIZES: [f32; 3] = [30.0, 90.0, 270.0];

// ---- Clipmap ring structure --------------------------------------------------------

const CLIPMAP_GRID_SNAP: f32 = 0.25;
const MAX_CLIPMAP_VERTS: u64 = 2048;
const MAX_CLIPMAP_INDICES: u64 = 12288;

struct ClipmapRingDef {
    inner_radius: f32,
    outer_radius: f32,
    inner_divs: u32,
    outer_divs: u32,
    level_divs: u32,
}

const CLIPMAP_RINGS: [ClipmapRingDef; 5] = [
    ClipmapRingDef { inner_radius: 0.0,  outer_radius: 3.0,   inner_divs: 1,  outer_divs: 12, level_divs: 4 },
    ClipmapRingDef { inner_radius: 3.0,  outer_radius: 10.0,  inner_divs: 12, outer_divs: 16, level_divs: 4 },
    ClipmapRingDef { inner_radius: 10.0, outer_radius: 30.0,  inner_divs: 16, outer_divs: 24, level_divs: 3 },
    ClipmapRingDef { inner_radius: 30.0, outer_radius: 90.0,  inner_divs: 24, outer_divs: 32, level_divs: 3 },
    ClipmapRingDef { inner_radius: 90.0, outer_radius: 250.0, inner_divs: 32, outer_divs: 48, level_divs: 2 },
];

// ---- Mesh helpers ----------------------------------------------------------------

/// Static box: 5 side/bottom faces (4×4 each, w = 1).  No top face — the top
/// is provided each frame by the camera-centred clipmap.
fn make_static_box_mesh(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    const FACE_DETAIL: u32 = 4;
    let fn1 = FACE_DETAIL + 1;
    let side_vert_count = (fn1 * fn1) as usize;
    let side_idx_count  = (FACE_DETAIL * FACE_DETAIL * 6) as usize;
    let total_verts = 4 * side_vert_count;
    let total_indices = 4 * side_idx_count;

    let mut verts: Vec<[f32; 4]> = Vec::with_capacity(total_verts);
    let mut indices: Vec<u32> = Vec::with_capacity(total_indices);

    let mut add_face = |verts: &mut Vec<[f32; 4]>,
                        indices: &mut Vec<u32>,
                        make_vert: fn(u32, u32) -> [f32; 4]| {
        let voffset = verts.len() as u32;
        for j in 0..fn1 {
            for i in 0..fn1 {
                verts.push(make_vert(i, j));
            }
        }
        for j in 0..FACE_DETAIL {
            for i in 0..FACE_DETAIL {
                let tl = voffset + j * fn1 + i;
                let tr = voffset + j * fn1 + (i + 1);
                let bl = voffset + (j + 1) * fn1 + i;
                let br = voffset + (j + 1) * fn1 + (i + 1);
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }
    };

    // Front face (z = -1)
    add_face(&mut verts, &mut indices, |i, j| {
        let x = i as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        let y = j as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        [x, y, -1.0, 1.0]
    });
    // Back face (z = 1)
    add_face(&mut verts, &mut indices, |i, j| {
        let x = i as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        let y = j as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        [x, y, 1.0, 1.0]
    });
    // Left face (x = -1)
    add_face(&mut verts, &mut indices, |i, j| {
        let z = i as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        let y = j as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        [-1.0, y, z, 1.0]
    });
    // Right face (x = 1)
    add_face(&mut verts, &mut indices, |i, j| {
        let z = i as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        let y = j as f32 / FACE_DETAIL as f32 * 2.0 - 1.0;
        [1.0, y, z, 1.0]
    });

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Static Box VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Static Box IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

/// Small static grid for the caustics projection pass (uses the old normalized
/// [-1,1] coordinate convention that the caustics vertex shader expects).
fn make_caustics_grid(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    const DETAIL: u32 = 32;
    let n = DETAIL + 1;
    let total_verts = (n * n) as usize;
    let total_indices = (DETAIL * DETAIL * 6) as usize;

    let mut verts: Vec<[f32; 4]> = Vec::with_capacity(total_verts);
    let mut indices: Vec<u32> = Vec::with_capacity(total_indices);

    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / DETAIL as f32 * 2.0 - 1.0;
            let y = j as f32 / DETAIL as f32 * 2.0 - 1.0;
            verts.push([x, y, 0.0, 0.0]);
        }
    }
    for j in 0..DETAIL {
        for i in 0..DETAIL {
            let tl = j * n + i;
            let tr = j * n + (i + 1);
            let bl = (j + 1) * n + i;
            let br = (j + 1) * n + (i + 1);
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Caustics VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Caustics IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

/// Static top-face grid in normalized [-1,1] space.
/// The vertex shader maps these to world XZ via water_sim_uv_to_world_xz.
fn make_top_grid(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    const DETAIL: u32 = 128;
    let n = DETAIL + 1;
    let mut verts: Vec<[f32; 4]> = Vec::with_capacity((n * n) as usize);
    let mut indices: Vec<u32> = Vec::with_capacity((DETAIL * DETAIL * 6) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / DETAIL as f32 * 2.0 - 1.0;
            let y = j as f32 / DETAIL as f32 * 2.0 - 1.0;
            verts.push([x, y, 0.0, 0.0]);
        }
    }
    for j in 0..DETAIL {
        for i in 0..DETAIL {
            let tl = j * n + i;
            let tr = j * n + (i + 1);
            let bl = (j + 1) * n + i;
            let br = (j + 1) * n + (i + 1);
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Top Grid VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Top Grid IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

/// Build one concentric ring of the clipmap, connecting adjacent levels with
/// triangles.  Level 0 uses `inner_divs` (to match the previous ring's outer
/// edge); all remaining levels use `outer_divs`.
fn generate_ring(
    verts: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    ring: &ClipmapRingDef,
    cx: f32,
    cz: f32,
) {
    let n_levels = ring.level_divs + 1;
    let mut level_starts: Vec<u32> = Vec::with_capacity(n_levels as usize);
    let mut level_div_counts: Vec<u32> = Vec::with_capacity(n_levels as usize);

    for level in 0..n_levels {
        level_starts.push(verts.len() as u32);
        let t = level as f32 / ring.level_divs as f32;
        let radius = ring.inner_radius + (ring.outer_radius - ring.inner_radius) * t;

        let divs = if level == 0 {
            if ring.inner_divs == 1 {
                verts.push([cx, cz, 0.0, 0.0]);
                level_div_counts.push(1);
                continue;
            }
            ring.inner_divs
        } else {
            ring.outer_divs
        };

        for s in 0..divs {
            let angle = s as f32 / divs as f32 * 2.0 * PI;
            let x = cx + radius * angle.cos();
            let z = cz + radius * angle.sin();
            verts.push([x, z, 0.0, 0.0]);
        }
        level_div_counts.push(divs);
    }

    for level in 0..ring.level_divs {
        let next = level + 1;
        let start_a = level_starts[level as usize];
        let start_b = level_starts[next as usize];
        let divs_a = level_div_counts[level as usize];
        let divs_b = level_div_counts[next as usize];

        if divs_a == 1 {
            // Center point → first circle: triangle fan
            for s in 0..divs_b {
                let s_next = (s + 1) % divs_b;
                indices.extend_from_slice(&[start_a, start_b + s, start_b + s_next]);
            }
        } else if divs_a == divs_b {
            // Same vertex count on both levels: regular quads
            for s in 0..divs_a {
                let s_next = (s + 1) % divs_a;
                let tl = start_a + s;
                let tr = start_a + s_next;
                let bl = start_b + s;
                let br = start_b + s_next;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        } else {
            // Different vertex counts: generalised triangulation that walks
            // both circles simultaneously, advancing whichever is "behind" in
            // angular space.
            let mut i = 0u32;
            let mut j = 0u32;
            while i < divs_a || j < divs_b {
                if i >= divs_a {
                    j += 1;
                } else if j >= divs_b {
                    i += 1;
                } else {
                    let next_ang_i = (i + 1) as f64 / divs_a as f64;
                    let next_ang_j = (j + 1) as f64 / divs_b as f64;

                    if next_ang_i <= next_ang_j + 1e-10 {
                        indices.push(start_a + i % divs_a);
                        indices.push(start_a + (i + 1) % divs_a);
                        indices.push(start_b + j % divs_b);
                        i += 1;
                    }
                    if next_ang_j <= next_ang_i + 1e-10 {
                        indices.push(start_a + i % divs_a);
                        indices.push(start_b + (j + 1) % divs_b);
                        indices.push(start_b + j % divs_b);
                        j += 1;
                    }
                }
            }
        }
    }
}

fn vec4_vbl() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        }],
    }
}

// ---- Pass struct ----------------------------------------------------------------

pub struct WaterSimPass {
    pub(crate) sim_bgl: wgpu::BindGroupLayout,
    pub(crate) hitbox_bgl: wgpu::BindGroupLayout,

    pub(crate) drop_pipeline: wgpu::RenderPipeline,
    pub(crate) update_pipeline: wgpu::RenderPipeline,
    pub(crate) normal_pipeline: wgpu::RenderPipeline,
    pub(crate) hitbox_pipeline: wgpu::RenderPipeline,

    pub(crate) sim_tex_a: wgpu::Texture,
    pub(crate) sim_tex_b: wgpu::Texture,
    pub(crate) sim_array_view_a: wgpu::TextureView,
    pub(crate) sim_array_view_b: wgpu::TextureView,
    pub(crate) sim_layer_views_a: Vec<wgpu::TextureView>,
    pub(crate) sim_layer_views_b: Vec<wgpu::TextureView>,
    pub(crate) front_per_layer: Vec<bool>,

    pub(crate) sampler: wgpu::Sampler,
    pub(crate) output_sampler: wgpu::Sampler,
    pub(crate) depth_sampler: wgpu::Sampler,

    pub(crate) drop_buf: wgpu::Buffer,
    pub(crate) update_bufs: [wgpu::Buffer; 3],
    pub(crate) normal_bufs: [wgpu::Buffer; 3],
    pub(crate) hitbox_count_buf: wgpu::Buffer,

    pub(crate) pending_drops: std::collections::VecDeque<simulation::DropUniform>,
    pub(crate) drop_staged: bool,

    /// Static box: 5 side/bottom faces (unchanging AABB walls, w = 1)
    pub(crate) static_box_vbuf: wgpu::Buffer,
    pub(crate) static_box_ibuf: wgpu::Buffer,
    pub(crate) static_box_index_count: u32,

    /// Static grid top face (normalized [-1,1] mapped to world by vertex shader)
    pub(crate) top_vbuf: wgpu::Buffer,
    pub(crate) top_ibuf: wgpu::Buffer,
    pub(crate) top_index_count: u32,

    /// Small static grid for the caustics projection pass (normalized coords)
    pub(crate) caustics_vbuf: wgpu::Buffer,
    pub(crate) caustics_ibuf: wgpu::Buffer,
    pub(crate) caustics_index_count: u32,

    pub(crate) caustics_sampler: wgpu::Sampler,

    pub(crate) caustics_render_bgl: wgpu::BindGroupLayout,
    pub(crate) render_bgl: wgpu::BindGroupLayout,
    pub(crate) render_bg: Option<wgpu::BindGroup>,
    pub(crate) render_bg_key: Option<(usize, usize, usize, usize, usize)>,
    pub(crate) normal_bgs: Vec<Option<wgpu::BindGroup>>,
    pub(crate) normal_bg_keys: Vec<Option<usize>>,

    pub(crate) hitbox_bg: Option<wgpu::BindGroup>,
    pub(crate) hitbox_bg_key: Option<(usize, usize)>,
    pub(crate) drop_bg: Option<wgpu::BindGroup>,
    pub(crate) drop_bg_key: Option<usize>,
    pub(crate) update_bgs: Vec<Option<wgpu::BindGroup>>,
    pub(crate) update_bg_keys: Vec<Option<usize>>,
    pub(crate) underwater_tint_bg: Option<wgpu::BindGroup>,
    pub(crate) underwater_tint_bg_key: Option<(usize, usize, usize)>,

    pub(crate) caustics_pipeline: wgpu::RenderPipeline,
    pub(crate) surface_pipeline: wgpu::RenderPipeline,

    pub(crate) _pre_aa_fallback_tex: wgpu::Texture,
    pub(crate) pre_aa_fallback_view: wgpu::TextureView,

    pub(crate) _gbuffer_fallback_tex: wgpu::Texture,
    pub(crate) gbuffer_fallback_view: wgpu::TextureView,

    pub(crate) internal_width: u32,
    pub(crate) internal_height: u32,
    pub(crate) surface_format: wgpu::TextureFormat,
    pub(crate) viewport_buf: wgpu::Buffer,

    pub(crate) blit_bgl: wgpu::BindGroupLayout,
    pub(crate) blit_pipeline: wgpu::RenderPipeline,
    pub(crate) blit_bg: Option<wgpu::BindGroup>,
    pub(crate) blit_bg_key: Option<usize>,

    pub(crate) water_output_view: Option<wgpu::TextureView>,

    pub(crate) caustics_bg_key: Option<(usize, usize)>,
    pub(crate) caustics_bg: Option<wgpu::BindGroup>,

    pub(crate) _tint_scratch_tex: wgpu::Texture,
    pub(crate) tint_scratch_view: wgpu::TextureView,
    pub(crate) underwater_tint_bgl: wgpu::BindGroupLayout,
    pub(crate) underwater_tint_pipeline: wgpu::RenderPipeline,

    pub(crate) wave_spring: f32,
    pub(crate) wave_damping: f32,

    pub(crate) wind_direction: [f32; 2],
    pub(crate) wind_strength: f32,
    pub(crate) wave_scale: f32,
    pub(crate) wave_speed: f32,
    pub(crate) sim_time: f32,
}

// ---- Public API ----------------------------------------------------------------

impl WaterSimPass {
    pub fn set_sim_dynamics(&mut self, spring: f32, damping: f32) {
        self.wave_spring = spring.clamp(0.1, 2.0);
        self.wave_damping = damping.clamp(0.0, 1.0);
    }

    pub fn set_wind(&mut self, direction: [f32; 2], strength: f32) {
        let len = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        self.wind_direction = if len > 1e-6 {
            [direction[0] / len, direction[1] / len]
        } else {
            [0.0, 0.0]
        };
        self.wind_strength = strength.max(0.0);
    }

    pub fn set_wave_scale(&mut self, scale: f32) {
        self.wave_scale = scale.max(0.01);
    }

    pub fn set_wave_speed(&mut self, speed: f32) {
        self.wave_speed = speed.max(0.0);
    }

    pub fn add_drop(&mut self, center_x: f32, center_z: f32, radius: f32, strength: f32) {
        if self.pending_drops.len() < MAX_DROPS_BUFFERED {
            self.pending_drops.push_back(simulation::DropUniform {
                center: [center_x, center_z],
                radius,
                strength,
            });
        }
    }

    pub fn resize_internal(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let tint_scratch_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Tint Scratch"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.tint_scratch_view =
            tint_scratch_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self._tint_scratch_tex = tint_scratch_tex;

        self.viewport_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Viewport"),
            contents: bytemuck::cast_slice(&[
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        self.internal_width = width;
        self.internal_height = height;

        self.water_output_view = None;
        self.render_bg = None;
        self.render_bg_key = None;
        self.blit_bg = None;
        self.blit_bg_key = None;
    }
}

// ---- RenderPass impl ----------------------------------------------------------------

impl RenderPass for WaterSimPass {
    fn name(&self) -> &'static str {
        "WaterSim"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {}

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
        builder.write_color(
            "water_output",
            ResourceFormat::from(self.surface_format),
            ResourceSize::MatchSurface,
        );
        builder.write_color_raw(
            "water_caustics",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::Absolute {
                width: CAUSTICS_SIZE,
                height: CAUSTICS_SIZE,
            },
        );
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "gbuffer",
            "depth",
            "pre_aa",
            "water_hitbox_count",
            "water_hitboxes",
            "water_volume_count",
            "water_volumes",
            "water_caustics",
            // Min-reduced depth pyramid, marched for water reflections. Built
            // by HiZBuildPass from `depth` alone, so it is available here.
            "hiz_min",
        ]
    }
    fn writes(&self) -> &'static [&'static str] {
        &[
            "water_sim_texture",
            "water_sim_sampler",
            "water_caustics",
            "pre_aa",
        ]
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        let view = if self.front_per_layer[0] {
            &self.sim_layer_views_a[0]
        } else {
            &self.sim_layer_views_b[0]
        };
        frame.water_sim_texture.write(view, "WaterSim");
        frame
            .water_sim_sampler
            .write(&self.output_sampler, "WaterSim");
        if let Some(view) = &self.water_output_view {
            frame.pre_aa.write(view, "WaterSim");
        }
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.sim_time += self.wave_speed / 60.0;
        let step_dt = 1.0 / 120.0;
        for ci in 0..CASCADE_COUNT {
            let delta = simulation::DeltaUniform {
                delta: [1.0 / SIM_SIZE as f32, 1.0 / SIM_SIZE as f32],
                spring: self.wave_spring,
                damping: self.wave_damping,
                wind_dir: self.wind_direction,
                wind_strength: self.wind_strength,
                time: self.sim_time,
                wave_scale: self.wave_scale,
                time_step: step_dt,
                cascade_patch_size: CASCADE_PATCH_SIZES[ci],
                cascade_id: ci as u32,
            };
            ctx.write_buffer(&self.update_bufs[ci], 0, bytemuck::bytes_of(&delta));
            ctx.write_buffer(&self.normal_bufs[ci], 0, bytemuck::bytes_of(&delta));
        }

        let count = ctx.frame_resources.water_hitbox_count;
        ctx.write_buffer(
            &self.hitbox_count_buf,
            0,
            bytemuck::bytes_of(&simulation::HitboxCountUniform {
                count,
                _pad: [0; 3],
            }),
        );

        self.drop_staged = false;
        if let Some(drop) = self.pending_drops.pop_front() {
            ctx.write_buffer(&self.drop_buf, 0, bytemuck::bytes_of(&drop));
            self.drop_staged = true;
        }



        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let volume_count = ctx.resources.water_volume_count.max(1);
        let total_layers = volume_count * CASCADE_COUNT as u32;

        let layer_view = |layer: usize, front: bool| -> &wgpu::TextureView {
            if front {
                &self.sim_layer_views_a[layer]
            } else {
                &self.sim_layer_views_b[layer]
            }
        };

        // ---- 1. Hitbox displacement (cascade 0 for all volumes) ------------
        if ctx.resources.water_hitbox_count > 0 {
            if let Some(hitboxes_buf) = ctx.resources.water_hitboxes.get() {
                for vol_idx in 0..volume_count {
                    let layer = (vol_idx * CASCADE_COUNT as u32) as usize;
                    let src = layer_view(layer, self.front_per_layer[layer]);
                    let dst = if self.front_per_layer[layer] {
                        &self.sim_layer_views_b[layer]
                    } else {
                        &self.sim_layer_views_a[layer]
                    };

                    let src_key = src as *const wgpu::TextureView as usize;
                    let hitboxes_key = hitboxes_buf as *const wgpu::Buffer as usize;
                    let new_key = (src_key, hitboxes_key);
                    if self.hitbox_bg_key != Some(new_key) {
                        self.hitbox_bg =
                            Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("WaterSim Hitbox BG"),
                                layout: &self.hitbox_bgl,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(src),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: self.hitbox_count_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: hitboxes_buf.as_entire_binding(),
                                    },
                                ],
                            }));
                        self.hitbox_bg_key = Some(new_key);
                    }
                    let bg = self.hitbox_bg.as_ref().unwrap();

                    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: dst,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let desc = wgpu::RenderPassDescriptor {
                        label: Some("WaterSim Hitbox"),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    };
                    let mut pass = ctx.begin_render_pass(&desc);
                    pass.set_pipeline(&self.hitbox_pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..1);
                    drop(pass);
                    self.front_per_layer[layer] = !self.front_per_layer[layer];
                }
            }
        }

        // ---- 2. Drop ripple (cascade 0 for all volumes) --------------------
        if self.drop_staged {
            for vol_idx in 0..volume_count {
                let layer = (vol_idx * CASCADE_COUNT as u32) as usize;
                let src = layer_view(layer, self.front_per_layer[layer]);
                let dst = if self.front_per_layer[layer] {
                    &self.sim_layer_views_b[layer]
                } else {
                    &self.sim_layer_views_a[layer]
                };

                let src_key = src as *const wgpu::TextureView as usize;
                if self.drop_bg_key != Some(src_key) {
                    self.drop_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("WaterSim Drop BG"),
                        layout: &self.sim_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.drop_buf.as_entire_binding(),
                            },
                        ],
                    }));
                    self.drop_bg_key = Some(src_key);
                }
                let bg = self.drop_bg.as_ref().unwrap();

                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("WaterSim Drop"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.drop_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..6, 0..1);
                drop(pass);
                self.front_per_layer[layer] = !self.front_per_layer[layer];
            }
        }

        // ---- 3 & 4. Cascade wave propagation + normal recomputation --------
        if ctx.resources.water_volume_count > 0 || ctx.resources.water_hitbox_count > 0 {
        for vol_idx in 0..volume_count {
            let base = (vol_idx * CASCADE_COUNT as u32) as usize;
        for ci in 0..CASCADE_COUNT {
            let layer = base + ci;

        // ---- 3. Wave propagation (2 steps per layer) ----------------------
        for i in 0..2u32 {
            let src = layer_view(layer, self.front_per_layer[layer]);
            let dst = if self.front_per_layer[layer] {
                &self.sim_layer_views_b[layer]
            } else {
                &self.sim_layer_views_a[layer]
            };

            let src_key = src as *const wgpu::TextureView as usize;
            let bg_label = format!("WaterSim Update BG V{} C{}", vol_idx, ci);
            if self.update_bg_keys[layer] != Some(src_key) {
                self.update_bgs[layer] = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&bg_label),
                    layout: &self.sim_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.update_bufs[ci].as_entire_binding(),
                        },
                    ],
                }));
                self.update_bg_keys[layer] = Some(src_key);
            }
            let bg = self.update_bgs[layer].as_ref().unwrap();

            let label_str = format!("WaterSim Update {} V{} C{}", i + 1, vol_idx, ci);
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some(&label_str),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front_per_layer[layer] = !self.front_per_layer[layer];
        }

        // ---- 4. Normal recomputation (per layer) --------------------------
        {
            let src = layer_view(layer, self.front_per_layer[layer]);
            let dst = if self.front_per_layer[layer] {
                &self.sim_layer_views_b[layer]
            } else {
                &self.sim_layer_views_a[layer]
            };

            let src_key = src as *const wgpu::TextureView as usize;
            let nrm_label = format!("WaterSim Normal BG V{} C{}", vol_idx, ci);
            if self.normal_bg_keys[layer] != Some(src_key) {
                self.normal_bgs[layer] = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&nrm_label),
                    layout: &self.sim_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.normal_bufs[ci].as_entire_binding(),
                        },
                    ],
                }));
                self.normal_bg_keys[layer] = Some(src_key);
            }
            let bg = self.normal_bgs[layer].as_ref().unwrap();

            let nrm_pass_label = format!("WaterSim Normal V{} C{}", vol_idx, ci);
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some(&nrm_pass_label),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.normal_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front_per_layer[layer] = !self.front_per_layer[layer];
        }
        }
        }
        }

        // ---- Consolidate: copy any layers still on tex_b to tex_a --------
        // After simulation, ensure all layers are on tex_a for rendering.
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        for layer in 0..total_layers as usize {
            if !self.front_per_layer[layer] {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.sim_tex_b,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.sim_tex_a,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: SIM_SIZE,
                        height: SIM_SIZE,
                        depth_or_array_layers: 1,
                    },
                );
                self.front_per_layer[layer] = true;
            }
        }

        // ---- 5. Caustics projection (cascade 0, volume 0) -----------------
        if ctx.resources.water_volume_count > 0 {
            if let Some(vols_buf) = ctx.resources.water_volumes.get() {
                let vols_key = vols_buf as *const wgpu::Buffer as usize;
                let sim_key = &self.sim_array_view_a as *const wgpu::TextureView as usize;
                let new_key = (vols_key, sim_key);

                if self.caustics_bg_key != Some(new_key) {
                    self.caustics_bg =
                        Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Water Caustics BG"),
                            layout: &self.caustics_render_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: vols_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(&self.sim_array_view_a),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                                },
                            ],
                        }));
                    self.caustics_bg_key = Some(new_key);
                }

                let caustics_view = ctx.resources.water_caustics.read("WaterSim").unwrap();
                let cau_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: caustics_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("Water Caustics"),
                    color_attachments: &cau_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.caustics_pipeline);
                pass.set_bind_group(0, self.caustics_bg.as_ref().unwrap(), &[]);
                pass.set_vertex_buffer(0, self.caustics_vbuf.slice(..));
                pass.set_index_buffer(self.caustics_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.caustics_index_count, 0, 0..1);
                drop(pass);
            }
        }

        // ---- 6. Blit pre_aa -> water_output (scene baseline) -----------------
        if self.water_output_view.is_none() {
            self.water_output_view = ctx.resource_pool.get_view("water_output").cloned();
        }
        let water_output_view = self
            .water_output_view
            .as_ref()
            .expect("water_output view from graph");
        let scene_view: &wgpu::TextureView = ctx
            .resources
            .pre_aa
            .get()
            .unwrap_or(&self.pre_aa_fallback_view);
        let blit_key = scene_view as *const _ as usize;
        if self.blit_bg_key != Some(blit_key) {
            self.blit_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Blit BG"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                    },
                ],
            }));
            self.blit_bg_key = Some(blit_key);
        }
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: water_output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("Water Blit"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, self.blit_bg.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        // ---- 7. Water surface render -> water_output --------------------------
        if ctx.resources.water_volume_count > 0 {
            if let Some(vols_buf) = ctx.resources.water_volumes.get() {
                let gbuffer_normal_view = ctx
                    .resources
                    .gbuffer
                    .get()
                    .map(|gb| gb.normal)
                    .unwrap_or(&self.gbuffer_fallback_view);
                let depth_view = ctx.depth;

                let hiz_min_view = match ctx.resource_pool.get_view("hiz_min") {
                    Some(v) => v.clone(),
                    None => {
                        return Err(helio_core::Error::InvalidPassConfig(
                            "WaterSim reflections require hiz_min; HiZBuildPass must run \
                             before WaterSim in the graph"
                                .to_string(),
                        ))
                    }
                };

                let scene_key = scene_view as *const wgpu::TextureView as usize;
                let gbuffer_key = gbuffer_normal_view as *const wgpu::TextureView as usize;
                let new_key = (
                    vols_buf as *const wgpu::Buffer as usize,
                    &self.sim_array_view_a as *const wgpu::TextureView as usize,
                    scene_key,
                    gbuffer_key,
                    depth_view as *const wgpu::TextureView as usize,
                );
                if self.render_bg_key != Some(new_key) {
                    self.render_bg =
                        Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Water Render BG"),
                            layout: &self.render_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: ctx.scene.camera.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: vols_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(&self.sim_array_view_a),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        ctx.resources.water_caustics.read("WaterSim").unwrap(),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.caustics_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::TextureView(scene_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: self.viewport_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::TextureView(depth_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 9,
                                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 10,
                                    resource: wgpu::BindingResource::TextureView(
                                        gbuffer_normal_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 11,
                                    resource: wgpu::BindingResource::TextureView(&hiz_min_view),
                                },
                            ],
                        }));
                    self.render_bg_key = Some(new_key);
                }
                let render_bg = self.render_bg.as_ref().unwrap();

                let depth_attachment = wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: None,
                    stencil_ops: None,
                };
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: water_output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })];

                // Multi-volume surface: draw each volume as an instance.
                // The vertex shader uses @builtin(instance_index) as the
                // volume index for per-volume parameter lookups and sim
                // texture layer selection.
                {
                    let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Surface"),
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: Some(depth_attachment.clone()),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    pass.set_pipeline(&self.surface_pipeline);
                    pass.set_bind_group(0, render_bg, &[]);
                    // Top face: instance_count = water_volume_count
                    pass.set_vertex_buffer(0, self.top_vbuf.slice(..));
                    pass.set_index_buffer(self.top_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.top_index_count, 0, 0..ctx.resources.water_volume_count);

                    // Static box sides/bottom
                    pass.set_vertex_buffer(0, self.static_box_vbuf.slice(..));
                    pass.set_index_buffer(self.static_box_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.static_box_index_count, 0, 0..ctx.resources.water_volume_count);
                }

                // 3. Underwater effect
                {
                    let vols_key = vols_buf as *const wgpu::Buffer as usize;
                    let water_output_key = water_output_view as *const wgpu::TextureView as usize;
                    let new_tint_key = (
                        vols_key,
                        water_output_key,
                        depth_view as *const wgpu::TextureView as usize,
                    );
                    if self.underwater_tint_bg_key != Some(new_tint_key) {
                        self.underwater_tint_bg =
                            Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Water Underwater Tint BG"),
                                layout: &self.underwater_tint_bgl,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: ctx.scene.camera.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: vols_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::TextureView(
                                            water_output_view,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: wgpu::BindingResource::Sampler(
                                            &self.output_sampler,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: wgpu::BindingResource::TextureView(depth_view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 5,
                                        resource: wgpu::BindingResource::Sampler(
                                            &self.depth_sampler,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 6,
                                        resource: wgpu::BindingResource::TextureView(&self.sim_array_view_a),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 7,
                                        resource: wgpu::BindingResource::Sampler(
                                            &self.output_sampler,
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 8,
                                        resource: wgpu::BindingResource::TextureView(
                                            ctx.resources
                                                .water_caustics
                                                .read("WaterSim")
                                                .unwrap(),
                                        ),
                                    },
                                ],
                            }));
                        self.underwater_tint_bg_key = Some(new_tint_key);
                    }
                        let tint_bg = self.underwater_tint_bg.as_ref().unwrap();
                    let tint_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: &self.tint_scratch_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let mut tint_pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Underwater Tint"),
                            color_attachments: &tint_attachments,
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    tint_pass.set_pipeline(&self.underwater_tint_pipeline);
                    tint_pass.set_bind_group(0, tint_bg, &[]);
                    tint_pass.draw(0..3, 0..1);
                    drop(tint_pass);

                    let scratch_blit_bg =
                        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Water Tint Blit BG"),
                            layout: &self.blit_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.tint_scratch_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.output_sampler),
                                },
                            ],
                        });
                    let blit_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: water_output_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let mut blit_pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Water Tint Blit Back"),
                            color_attachments: &blit_attachments,
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    blit_pass.set_pipeline(&self.blit_pipeline);
                    blit_pass.set_bind_group(0, &scratch_blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }
        }

        Ok(())
    }
}
