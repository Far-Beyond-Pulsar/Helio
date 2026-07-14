use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};

pub const CHAR_W: u32 = 14;
pub const ROW_H: u32 = 24;
const DEFAULT_COLS: u32 = 80;
const DEFAULT_ROWS: u32 = 30;
const MAX_COLS: u32 = 280;
const MAX_ROWS: u32 = 90;
const MAX_BARS: u32 = 512;
const MAX_PIES: u32 = 64;
const MAX_LINES: u32 = 128;

const FONT8X8: [u8; 95 * 8] = {
    let mut data = [0u8; 95 * 8];
    let raw: &[u8] = &[
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18,
        0x00, 0x66, 0x66, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C,
        0x6C, 0x00, 0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00, 0x62, 0x66, 0x0C, 0x18, 0x30,
        0x66, 0x46, 0x00, 0x3C, 0x66, 0x3C, 0x38, 0x67, 0x66, 0x3F, 0x00, 0x18, 0x18, 0x30, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00, 0x30, 0x18, 0x0C,
        0x0C, 0x0C, 0x18, 0x30, 0x00, 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x18,
        0x18, 0x7E, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30, 0x00,
        0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00,
        0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00, 0x00, 0x3C, 0x66, 0x6E, 0x7E, 0x76, 0x66, 0x3C,
        0x00, 0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00, 0x3C, 0x66, 0x06, 0x1C, 0x30, 0x60,
        0x7E, 0x00, 0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00, 0x0C, 0x1C, 0x3C, 0x6C, 0x7E,
        0x0C, 0x0C, 0x00, 0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00, 0x3C, 0x60, 0x60, 0x7C,
        0x66, 0x66, 0x3C, 0x00, 0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00, 0x3C, 0x66, 0x66,
        0x3C, 0x66, 0x66, 0x3C, 0x00, 0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00, 0x00, 0x18,
        0x18, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30, 0x0C,
        0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00,
        0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x00, 0x3C, 0x66, 0x06, 0x1C, 0x18, 0x00, 0x18,
        0x00, 0x3C, 0x66, 0x6E, 0x6E, 0x60, 0x3E, 0x00, 0x00, 0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66,
        0x66, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00, 0x3C, 0x66, 0x60, 0x60, 0x60,
        0x66, 0x3C, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x00, 0x7E, 0x60, 0x60, 0x7C,
        0x60, 0x60, 0x7E, 0x00, 0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00, 0x3C, 0x66, 0x60,
        0x6E, 0x66, 0x66, 0x3C, 0x00, 0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00, 0x3C, 0x18,
        0x18, 0x18, 0x18, 0x18, 0x3C, 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00, 0x66,
        0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00,
        0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00, 0x66, 0x76, 0x7E, 0x7E, 0x6C, 0x64, 0x60,
        0x00, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60,
        0x60, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x0E, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x78,
        0x6C, 0x66, 0x00, 0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00, 0x7E, 0x18, 0x18, 0x18,
        0x18, 0x18, 0x18, 0x00, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x3C, 0x18, 0x00, 0x63, 0x63, 0x6B, 0x7F, 0x7F, 0x77, 0x63, 0x00, 0x66, 0x66,
        0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00, 0x7E,
        0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00, 0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00,
        0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00, 0x00, 0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C,
        0x00, 0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFE, 0x00, 0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3C, 0x06, 0x3E,
        0x66, 0x3E, 0x00, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00, 0x00, 0x00, 0x3C, 0x60,
        0x60, 0x60, 0x3C, 0x00, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00, 0x00, 0x00, 0x3C,
        0x66, 0x7E, 0x60, 0x3C, 0x00, 0x1C, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x30, 0x00, 0x00, 0x00,
        0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00, 0x18,
        0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00, 0x0C, 0x00, 0x1C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38,
        0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00, 0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C,
        0x00, 0x00, 0x00, 0x76, 0x7F, 0x6B, 0x63, 0x63, 0x00, 0x00, 0x00, 0x7C, 0x66, 0x66, 0x66,
        0x66, 0x00, 0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00, 0x00, 0x00, 0x7C, 0x66, 0x66,
        0x7C, 0x60, 0x60, 0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x00, 0x00, 0x7C, 0x66,
        0x60, 0x60, 0x60, 0x00, 0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00, 0x30, 0x30, 0x7C,
        0x30, 0x30, 0x30, 0x1C, 0x00, 0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00, 0x00, 0x00,
        0x66, 0x66, 0x66, 0x3C, 0x18, 0x00, 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00, 0x00,
        0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C,
        0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00, 0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E,
        0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x70, 0x18, 0x18, 0x0E, 0x18, 0x18,
        0x70, 0x00, 0x00, 0x00, 0x00, 0x76, 0xDC, 0x00, 0x00, 0x00,
    ];
    let mut i = 0;
    while i < raw.len() {
        data[i] = raw[i];
        i += 1;
    }
    data
};

fn clamp01(v: f32) -> f32 {
    v.max(0.0).min(1.0)
}

pub struct DebugOverlayState {
    pub enabled: bool,
    grid_cols: u32,
    grid_rows: u32,
    char_grid: Vec<u32>,
    small_cols: u32,
    small_rows: u32,
    small_grid: Vec<u32>,
    bars: Vec<f32>,
    bar_colors: Vec<f32>,
    pies: Vec<f32>,
    pie_colors: Vec<f32>,
    lines: Vec<f32>,
    line_colors: Vec<f32>,
}

impl DebugOverlayState {
    pub fn new() -> Arc<Mutex<Self>> {
        let cols = DEFAULT_COLS;
        let rows = DEFAULT_ROWS;
        Arc::new(Mutex::new(Self {
            enabled: false,
            grid_cols: DEFAULT_COLS,
            grid_rows: DEFAULT_ROWS,
            char_grid: vec![0u32; (DEFAULT_COLS * DEFAULT_ROWS) as usize],
            small_cols: DEFAULT_COLS * 14 / 8,
            small_rows: DEFAULT_ROWS * 24 / 12,
            small_grid: vec![0u32; ((DEFAULT_COLS * 14 / 8) * (DEFAULT_ROWS * 24 / 12)) as usize],
            bars: Vec::with_capacity(MAX_BARS as usize * 4),
            bar_colors: Vec::with_capacity(MAX_BARS as usize * 4),
            pies: Vec::with_capacity(MAX_PIES as usize * 4),
            pie_colors: Vec::with_capacity(MAX_PIES as usize * 4),
            lines: Vec::with_capacity(MAX_LINES as usize * 4),
            line_colors: Vec::with_capacity(MAX_LINES as usize * 4),
        }))
    }

    pub fn set_grid_size(&mut self, cols: u32, rows: u32) {
        if cols != self.grid_cols || rows != self.grid_rows {
            self.grid_cols = cols;
            self.grid_rows = rows;
            self.char_grid = vec![0u32; (cols * rows) as usize];
            self.small_cols = cols * 14 / 8;
            self.small_rows = rows * 24 / 12;
            self.small_grid = vec![0u32; (self.small_cols * self.small_rows) as usize];
        }
    }

    pub fn grid_cols(&self) -> u32 {
        self.grid_cols
    }
    pub fn grid_rows(&self) -> u32 {
        self.grid_rows
    }
    pub fn small_cols(&self) -> u32 {
        self.small_cols
    }
    pub fn small_rows(&self) -> u32 {
        self.small_rows
    }

    pub fn clear(&mut self) {
        self.char_grid.fill(0);
        self.small_grid.fill(0);
        self.bars.clear();
        self.bar_colors.clear();
        self.pies.clear();
        self.pie_colors.clear();
        self.lines.clear();
        self.line_colors.clear();
    }

    pub fn set_char(&mut self, col: u32, row: u32, ch: char) {
        if col < self.grid_cols && row < self.grid_rows {
            let idx = (row * self.grid_cols + col) as usize;
            if idx < self.char_grid.len() {
                self.char_grid[idx] = ch as u32;
            }
        }
    }

    pub fn write_text(&mut self, col: u32, row: u32, text: &str) {
        for (i, ch) in text.chars().enumerate() {
            self.set_char(col + i as u32, row, ch);
        }
    }

    pub fn write_text_right(&mut self, row: u32, text: &str) {
        let width = text.chars().count() as u32;
        let col = self.grid_cols.saturating_sub(width);
        self.write_text(col, row, text);
    }

    pub fn set_char_small(&mut self, col: u32, row: u32, ch: char) {
        if col < self.small_cols && row < self.small_rows {
            let idx = (row * self.small_cols + col) as usize;
            if idx < self.small_grid.len() {
                self.small_grid[idx] = ch as u32;
            }
        }
    }

    pub fn write_small(&mut self, col: u32, row: u32, text: &str) {
        for (i, ch) in text.chars().enumerate() {
            self.set_char_small(col + i as u32, row, ch);
        }
    }

    /// Write a table row with columns separated by `|`. Each column is padded to `col_widths`.
    /// e.g. write_table_row(row, &[15, 8, 10], &["name", "width", "format"])
    pub fn write_table_row(&mut self, row: u32, col_widths: &[u32], values: &[&str]) {
        let mut x = 0u32;
        for (i, val) in values.iter().enumerate() {
            let w = col_widths.get(i).copied().unwrap_or(12);
            self.write_text(x, row, val);
            // pad with spaces up to column width
            let pad = w.saturating_sub(val.chars().count() as u32);
            for _ in 0..pad {
                x += 1;
                if x >= self.grid_cols {
                    break;
                }
            }
            x += val.chars().count() as u32;
            if i + 1 < values.len() {
                if x < self.grid_cols {
                    self.set_char(x, row, ' ');
                }
                x += 1;
            }
        }
    }

    pub fn add_bar(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, g: f32, b: f32, a: f32) {
        if self.bars.len() / 4 >= MAX_BARS as usize {
            return;
        }
        self.bars.push(x);
        self.bars.push(y);
        self.bars.push(w);
        self.bars.push(h);
        self.bar_colors.push(clamp01(r));
        self.bar_colors.push(clamp01(g));
        self.bar_colors.push(clamp01(b));
        self.bar_colors.push(clamp01(a));
    }

    pub fn add_line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, r: f32, g: f32, b: f32, a: f32) {
        if self.lines.len() / 4 >= MAX_LINES as usize {
            return;
        }
        self.lines.push(x1);
        self.lines.push(y1);
        self.lines.push(x2);
        self.lines.push(y2);
        self.line_colors.push(clamp01(r));
        self.line_colors.push(clamp01(g));
        self.line_colors.push(clamp01(b));
        self.line_colors.push(clamp01(a));
    }

    pub fn add_pie_slice(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        end_angle: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) {
        if self.pies.len() / 4 >= MAX_PIES as usize {
            return;
        }
        self.pies.push(cx);
        self.pies.push(cy);
        self.pies.push(radius);
        self.pies.push(end_angle);
        self.pie_colors.push(clamp01(r));
        self.pie_colors.push(clamp01(g));
        self.pie_colors.push(clamp01(b));
        self.pie_colors.push(clamp01(a));
    }

    pub fn bars_data(&self) -> &[f32] {
        &self.bars
    }
    pub fn bar_colors_data(&self) -> &[f32] {
        &self.bar_colors
    }
    pub fn pies_data(&self) -> &[f32] {
        &self.pies
    }
    pub fn pie_colors_data(&self) -> &[f32] {
        &self.pie_colors
    }
    pub fn lines_data(&self) -> &[f32] {
        &self.lines
    }
    pub fn line_colors_data(&self) -> &[f32] {
        &self.line_colors
    }
    pub fn char_grid_slice(&self) -> &[u32] {
        &self.char_grid
    }
    pub fn small_grid_slice(&self) -> &[u32] {
        &self.small_grid
    }
}

pub struct DebugOverlayPass {
    shared: Arc<Mutex<DebugOverlayState>>,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    char_buf: wgpu::Buffer,
    small_grid_buf: wgpu::Buffer,
    bar_buf: wgpu::Buffer,
    pie_buf: wgpu::Buffer,
    line_buf: wgpu::Buffer,
    font_texture: wgpu::Texture,
    font_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_dirty: bool,
    screen_w: u32,
    screen_h: u32,
}

impl DebugOverlayPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shared: Arc<Mutex<DebugOverlayState>>,
        target_format: wgpu::TextureFormat,
        screen_w: u32,
        screen_h: u32,
    ) -> Self {
        let font_tex = create_font_texture(device, queue);
        let font_view = font_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DebugOverlay Font Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DebugOverlay Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/debug_overlay.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DebugOverlay BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DebugOverlay PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DebugOverlay Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DebugOverlay Params"),
            size: std::mem::size_of::<[u32; 9]>() as u64, // screen_w, screen_h, big_cols, big_rows, small_cols, small_rows, num_bars, num_pies, num_lines
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let char_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DebugOverlay Char Grid"),
            size: (MAX_COLS * MAX_ROWS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bar_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DebugOverlay Bar Data"),
            size: (MAX_BARS as u64) * 4 * 2 * 4, // bars + colors
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pie_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DebugOverlay Pie Data"),
            size: (MAX_PIES as u64) * 4 * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DebugOverlay Line Data"),
            size: (MAX_LINES as u64) * 4 * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Small grid buffer (8x12 font) — needs ~3.5x bigger than big grid
        let small_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DebugOverlay Small Grid"),
            size: (MAX_COLS * 2 * MAX_ROWS * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shared,
            pipeline,
            bgl,
            params_buf,
            char_buf,
            small_grid_buf,
            bar_buf,
            pie_buf,
            line_buf,
            font_texture: font_tex,
            font_view,
            sampler,
            bind_group: None,
            bind_group_dirty: true,
            screen_w,
            screen_h,
        }
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.shared.lock().unwrap().enabled = enabled;
    }

    pub fn resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.screen_w = width;
        self.screen_h = height;
        self.bind_group_dirty = true;
    }
}

impl RenderPass for DebugOverlayPass {
    fn name(&self) -> &'static str {
        "DebugOverlay"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let mut shared = self.shared.lock().unwrap();
        if !shared.enabled {
            return Ok(());
        }

        // Update grid size from shared state
        let grid_cols = shared.grid_cols();
        let grid_rows = shared.grid_rows();
        let buf_size = (grid_cols * grid_rows * 4) as u64;
        if self.char_buf.size() < buf_size {
            // Re-create if needed (rare - only on resize)
        }

        let params = [
            self.screen_w,
            self.screen_h,
            grid_cols,
            grid_rows,
            shared.small_cols(),
            shared.small_rows(),
            shared.bars.len() as u32 / 4,
            shared.pies.len() as u32 / 4,
            shared.lines.len() as u32 / 4,
        ];
        ctx.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Write char grids
        let grid_data = shared.char_grid_slice();
        ctx.write_buffer(&self.char_buf, 0, bytemuck::cast_slice(grid_data));
        let small_data = shared.small_grid_slice();
        ctx.write_buffer(&self.small_grid_buf, 0, bytemuck::cast_slice(small_data));

        // Write bar data: [x,y,w,h for each bar] then [r,g,b,a for each bar]
        let bars = shared.bars_data();
        if !bars.is_empty() {
            let num = bars.len() as u64 * 4; // byte offset of bar geometry (4 floats per bar × 4 bytes)
            ctx.write_buffer(&self.bar_buf, 0, bytemuck::cast_slice(bars));
            let colors = shared.bar_colors_data();
            ctx.write_buffer(&self.bar_buf, num, bytemuck::cast_slice(colors));
        }

        // Write pie data: [cx,cy,radius,end_angle for each slice] then [r,g,b,a for each slice]
        let pies = shared.pies_data();
        if !pies.is_empty() {
            let num = pies.len() as u64 * 4;
            ctx.write_buffer(&self.pie_buf, 0, bytemuck::cast_slice(pies));
            let pie_colors = shared.pie_colors_data();
            ctx.write_buffer(&self.pie_buf, num, bytemuck::cast_slice(pie_colors));
        }

        // Write line data: [x1,y1,x2,y2 for each line] then [r,g,b,a for each line]
        let lines = shared.lines_data();
        if !lines.is_empty() {
            let num = lines.len() as u64 * 4;
            ctx.write_buffer(&self.line_buf, 0, bytemuck::cast_slice(lines));
            let line_colors = shared.line_colors_data();
            ctx.write_buffer(&self.line_buf, num, bytemuck::cast_slice(line_colors));
        }

        if self.bind_group.is_none() || self.bind_group_dirty {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DebugOverlay BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.char_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.font_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.bar_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.pie_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.line_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.small_grid_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_dirty = false;
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let enabled = self.shared.lock().unwrap().enabled;
        if !enabled {
            return Ok(());
        }
        let Some(bg) = &self.bind_group else {
            return Ok(());
        };

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("DebugOverlay"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut rp = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&desc);
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, bg, &[]);
        rp.draw(0..3, 0..1);
        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.resize(device, width, height);
    }
}

fn create_font_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let tex_w: u32 = 128;
    let tex_h: u32 = 64;
    let mut pixels = vec![0u8; (tex_w * tex_h * 4) as usize];

    for char_idx in 0..95 {
        let atlas_col = char_idx % 16;
        let atlas_row = char_idx / 16;
        for row in 0..8 {
            let byte = FONT8X8[char_idx * 8 + row];
            for col in 0..8 {
                let bit = (byte >> (7 - col)) & 1;
                if bit != 0 {
                    let px = atlas_col * 8 + col;
                    let py = atlas_row * 8 + row;
                    let pi = ((py as usize) * (tex_w as usize) + (px as usize)) * 4;
                    if pi + 3 < pixels.len() {
                        pixels[pi] = 0xFF;
                        pixels[pi + 1] = 0xFF;
                        pixels[pi + 2] = 0xFF;
                        pixels[pi + 3] = 0xFF;
                    }
                }
            }
        }
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DebugOverlay Font Atlas"),
        size: wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex_w * 4),
            rows_per_image: Some(tex_h),
        },
        wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
    );

    texture
}
