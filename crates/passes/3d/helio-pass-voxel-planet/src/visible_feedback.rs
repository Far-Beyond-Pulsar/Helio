use glam::DVec3;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

pub(super) const CAPACITY: usize = 256;
const COPY_BYTES: u64 = 16 + CAPACITY as u64 * 8;
const STORAGE_BYTES: u64 = COPY_BYTES + 4096 * 4;
// Frame.screen.w is an exactly represented integer f32. Its existing low
// bits 1/2 control the horizon/fail-safe; bit 3 enables this diagnostic and
// bits 8..12 carry the CPU-selected stride exponent. No ABI grows.
const SAMPLE_ENABLED: u32 = 1 << 3;
const SAMPLE_SHIFT: u32 = 8;
const SAMPLE_MASK: u32 = 31 << SAMPLE_SHIFT;

pub(super) fn sample_stride(size: [u32; 2]) -> u32 {
    let mut stride = 1u32;
    while u64::from(size[0]).div_ceil(u64::from(stride))
        * u64::from(size[1]).div_ceil(u64::from(stride)) > 255 {
        stride <<= 1;
    }
    stride
}

pub(super) fn sample_flags(flags: u32, stride: u32) -> u32 {
    debug_assert!(stride.is_power_of_two());
    (flags & !SAMPLE_MASK) | SAMPLE_ENABLED | (stride.trailing_zeros() << SAMPLE_SHIFT)
}

fn sample_count(size: [u32; 2], stride: u32, frame: u32) -> u32 {
    let phase = frame & 0xffffff;
    let x = phase & (stride - 1);
    let y = (phase >> stride.trailing_zeros()) & (stride - 1);
    let axis = |extent: u32, offset: u32| {
        if offset >= extent { 0 } else { 1 + (extent - 1 - offset) / stride }
    };
    axis(size[0], x) * axis(size[1], y)
}

#[derive(Clone, Copy)]
pub(super) struct View {
    pub eye: DVec3,
    pub forward: DVec3,
    pub up: DVec3,
    pub projection_y: f32,
    pub size: [u32; 2],
    pub id: u32,
    pub frame: u32,
    pub encoded_frame: u64,
    pub diagnostic_stride: Option<u32>,
    pub max_eye_delta: f64,
    pub captured_at: std::time::Instant,
}

#[derive(Default)]
pub(super) struct Batch {
    pub blocks: Vec<(u32, u32)>,
    pub source: Option<View>,
    pub sampled_primary: Option<super::SampledPrimaryStats>,
}

impl View {
    fn accepts(self, old: Self) -> bool {
        self.size == old.size
            && self.id == old.id
            && self.frame.wrapping_sub(old.frame) <= 8
            && self.eye.distance(old.eye) <= self.max_eye_delta
            && self.forward.dot(old.forward) >= 0.75
    }

    fn accepts_diagnostic(self, old: Self) -> bool {
        self.accepts(old) && self.up.dot(old.up) >= 0.75
            && self.projection_y == old.projection_y
            && self.diagnostic_stride == old.diagnostic_stride
            && self.captured_at.checked_duration_since(old.captured_at)
                .is_some_and(|age| age <= std::time::Duration::from_millis(500))
    }
}

fn decode_sample(word: u32, source: View, current: View) -> Option<super::SampledPrimaryStats> {
    let stride = source.diagnostic_stride?;
    if !stride.is_power_of_two() || !current.accepts_diagnostic(source)
        || !source.projection_y.is_finite() || source.projection_y <= 0.0 { return None; }
    let sampled_rays = sample_count(source.size, stride, source.frame);
    let [terrain_hits, coarse_over_2px, coarse_over_4px, unresolved] =
        std::array::from_fn(|index| (word >> (index * 8)) & 255);
    if sampled_rays == 0 || sampled_rays > 255 || terrain_hits + unresolved > sampled_rays
        || coarse_over_2px > terrain_hits || coarse_over_4px > coarse_over_2px { return None; }
    Some(super::SampledPrimaryStats {
        encoded_frame: source.encoded_frame, source_frame: source.frame,
        age_frames: current.frame.wrapping_sub(source.frame), captured_at: source.captured_at,
        eye: source.eye, forward: source.forward, up: source.up, view_id: source.id,
        viewport: source.size, projection_y: source.projection_y, stride, sampled_rays,
        terrain_hits, coarse_over_2px, coarse_over_4px, unresolved,
    })
}

struct Readback {
    buffer: wgpu::Buffer,
    // 0 pending, 1 mapped, 2 mapping failed.
    state: Arc<AtomicU8>,
    stage: u8,
    requests: bool,
    view: Option<View>,
}

pub(super) struct Feedback {
    pub buffer: wgpu::Buffer,
    readbacks: Vec<Readback>,
    pub counts: [u32; 3],
    last_view: Option<View>,
}

impl Feedback {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet visible block requests"),
            size: STORAGE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readbacks = (0..3)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("planet visible request readback"),
                    size: COPY_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(0)),
                stage: 0,
                requests: false,
                view: None,
            })
            .collect();
        Self {
            buffer,
            readbacks,
            counts: [0; 3],
            last_view: None,
        }
    }

    pub fn bytes(&self) -> u64 {
        STORAGE_BYTES + COPY_BYTES * self.readbacks.len() as u64
    }

    pub fn pending(&self) -> bool {
        // Diagnostic-only copies never keep the renderer scheduling frames.
        self.readbacks.iter().any(|r| r.stage != 0 && r.requests)
    }

    pub fn view_changed(&self, current: View) -> bool {
        self.last_view.is_none_or(|old| {
            old.size != current.size
                || old.id != current.id
                || old.eye.distance(current.eye) > 0.01
                || old.forward.dot(current.forward) < 0.99999
        })
    }

    // The caller already polls the device without waiting. Mapping begins
    // only on the next encode, after the copy has been submitted.
    pub fn poll(&mut self, current: View) -> Batch {
        let mut blocks = Vec::new();
        let mut source = None;
        let mut sampled_primary = None;
        let mut newest_age = u32::MAX;
        for r in &mut self.readbacks {
            let state = r.state.load(Ordering::Acquire);
            if r.stage == 2 && state != 0 {
                if state == 1 {
                    if r.view.is_some_and(|old| {
                        current.accepts(old) && current.frame.wrapping_sub(old.frame) < newest_age
                    }) {
                        newest_age = current.frame.wrapping_sub(r.view.unwrap().frame);
                        let data = r.buffer.slice(..).get_mapped_range().unwrap();
                        for (n, count) in self.counts.iter_mut().enumerate() {
                            *count = u32::from_le_bytes(data[n * 4..n * 4 + 4].try_into().unwrap());
                        }
                        blocks = decode(&data);
                        source = r.view;
                        sampled_primary = decode_sample(u32::from_le_bytes(data[12..16].try_into().unwrap()),
                            r.view.unwrap(), current);
                    }
                    r.buffer.unmap();
                }
                r.stage = 0;
                r.view = None;
                r.state.store(0, Ordering::Release);
            }
        }
        for r in &mut self.readbacks {
            if r.stage == 1 {
                let state = r.state.clone();
                r.buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        state.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release);
                    });
                r.stage = 2;
            }
        }
        // Prefer the newest completed frame, rather than replaying old views.
        blocks.sort_unstable();
        blocks.dedup();
        // No cached diagnostic is replayed when no accepted fresh readback
        // arrived. Request counters retain their historical API separately.
        Batch { blocks, source, sampled_primary }
    }

    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, view: View, requests: bool) -> Option<usize> {
        let at = self.readbacks.iter().position(|r| r.stage == 0)?;
        encoder.clear_buffer(&self.buffer, 0, None);
        let r = &mut self.readbacks[at];
        r.view = Some(view);
        r.stage = 1;
        r.requests = requests;
        if requests { self.last_view = Some(view); }
        Some(at)
    }

    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder, at: usize) {
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.readbacks[at].buffer, 0, COPY_BYTES);
    }
}

fn decode(data: &[u8]) -> Vec<(u32, u32)> {
    if data.len() < COPY_BYTES as usize {
        return Vec::new();
    }
    let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    let count = (word(0) as usize).min(CAPACITY);
    (0..count)
        .map(|i| (word(16 + i * 8), word(20 + i * 8)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampled_view() -> View {
        View { eye: DVec3::ZERO, forward: DVec3::Z, up: DVec3::Y,
            projection_y: 2.0, size: [1196, 729], id: 1, frame: 7,
            encoded_frame: 123, diagnostic_stride: Some(64), max_eye_delta: 50.0,
            captured_at: std::time::Instant::now() }
    }

    #[test]
    fn primary_sample_stride_bounds_every_phase_and_preserves_flags() {
        for size in [[1,1], [17,16], [1196,729], [3840,2160], [8192,4320]] {
            let stride = sample_stride(size);
            assert!(stride.is_power_of_two());
            let flags = sample_flags(6, stride);
            assert_eq!(flags & 6, 6, "horizon and fail-safe flags must survive");
            assert_eq!(flags as f32 as u32, flags, "uniform integer must be exact");
            assert_eq!(1 << ((flags & SAMPLE_MASK) >> SAMPLE_SHIFT), stride);
            let mut total = 0u64;
            for phase in 0..stride * stride {
                let count = sample_count(size, stride, phase);
                assert!(count <= 255, "packed counters must never carry");
                total += u64::from(count);
            }
            assert_eq!(total, u64::from(size[0]) * u64::from(size[1]));
        }
        assert_eq!(sample_count([0,12], sample_stride([0,12]), 0), 0);
        let large = [u32::MAX, u32::MAX];
        assert!(sample_count(large, sample_stride(large), u32::MAX) <= 255);
    }

    #[test]
    fn primary_sample_packing_preserves_source_and_unresolved_denominators() {
        let source = sampled_view();
        let current = View { frame: 10, encoded_frame: 126, eye: DVec3::X, ..source };
        let word = 100 | (30 << 8) | (20 << 16) | (40 << 24);
        let sample = decode_sample(word, source, current).unwrap();
        assert_eq!((sample.terrain_hits, sample.coarse_over_2px, sample.coarse_over_4px, sample.unresolved), (100,30,20,40));
        assert_eq!(sample.encoded_frame, 123);
        assert_eq!(sample.source_frame, 7);
        assert_eq!(sample.age_frames, 3);
        assert_eq!(sample.eye, source.eye);
        assert_ne!(sample.eye, current.eye);
        assert_eq!(sample.viewport, source.size);
        assert_eq!(sample.sampled_rays, sample_count(source.size, 64, 7));
        assert!(decode_sample(1 | (2 << 8), source, current).is_none());
        assert!(decode_sample(255 | (1 << 24), source, current).is_none());
        // The maximal allowed whole grid packs without adjacent carries.
        let full = View { size: [255,1], frame: 0, diagnostic_stride: Some(1), ..source };
        let all_large = (0..255).fold(0u32, |word, _| word + 1 + (1 << 8) + (1 << 16));
        let sample = decode_sample(all_large, full, full).unwrap();
        assert_eq!((sample.terrain_hits, sample.coarse_over_2px, sample.coarse_over_4px, sample.unresolved), (255,255,255,0));
        assert_eq!(decode_sample(255 << 24, full, full).unwrap().unresolved, 255);
    }

    #[test]
    fn primary_samples_reject_stale_pose_projection_and_disabled_capture() {
        let source = sampled_view();
        assert!(decode_sample(0, source, source).is_some(), "fresh empty sky is a measured sample");
        for current in [View { id: 2, ..source }, View { size: [729,1196], ..source },
            View { frame: 16, ..source }, View { up: -DVec3::Y, ..source },
            View { projection_y: 1.0, ..source },
            View { captured_at: source.captured_at + std::time::Duration::from_millis(501), ..source },
            View { diagnostic_stride: None, ..source }] {
            assert!(decode_sample(0, source, current).is_none());
        }
        assert!(decode_sample(0, View { diagnostic_stride: None, ..source }, source).is_none());
        assert!(Batch::default().sampled_primary.is_none(), "no fresh batch cannot replay zero counts");
    }

    #[test]
    fn primary_classifier_gpu_matches_packed_estimates_and_default_is_disabled() {
        use wgpu::util::DeviceExt;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let production = include_str!("../shaders/visible_feedback.wgsl");
        let classifier = &production[production.find("fn sample_primary_hit").unwrap()..];
        let source = format!(r#"
override PRIMARY_SAMPLES: bool = false;
const ST_HIT:u32=1u; const ST_EXHAUSTED:u32=2u; const ST_LOADING:u32=3u;
struct Frame {{ screen:vec4<f32>, layer:vec4<f32>, hints:vec4<u32> }}
struct Camera {{ proj:mat4x4<f32> }}
struct Hit {{ t:f32, info:u32 }}
struct Case {{ words:vec4<u32>, values:vec4<f32>, phase:vec4<u32> }}
struct Requests {{ count:atomic<u32>, overflow:atomic<u32>, attempts:atomic<u32>, sampled:atomic<u32> }}
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(21) var<storage,read_write> visible_requests:Requests;
var<private> frame:Frame; var<private> camera:Camera;
{classifier}
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
    let c=cases[id.x]; frame.screen=vec4<f32>(0.0,c.values.w,0.0,f32(c.words.w));
    frame.layer=vec4<f32>(0.0,c.values.y,0.0,0.0); frame.hints.w=c.phase.x<<8u;
    camera.proj[1][1]=c.values.z;
    sample_primary_hit(c.words.xy,Hit(c.values.x,c.words.z));
}}
"#);
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Case { words:[u32;4], values:[f32;4], phase:[u32;4] }
        let mut cases = Vec::new();
        let mut expected = 0u32;
        // Real Hit status/level packing, distinct FOVs, both size thresholds,
        // the base-level exclusion, sample-phase rejection and flag gating.
        for (status, level, width, proj, xy, enabled) in [
            (0,0,8.0,1.0,[7,0],true), (3,0,8.0,1.0,[7,0],true),
            (2,0,8.0,2.0,[7,0],true), (1,0,8.0,1.0,[7,0],true),
            (1,1,1.9,1.0,[7,0],true), (1,4,2.1,2.0,[7,0],true),
            (1,12,3.9,0.5,[7,0],true), (1,8,4.1,1.7,[7,0],true),
            (1,8,8.0,1.0,[8,0],true), (3,0,8.0,1.0,[7,0],false),
        ] {
            let flags = if enabled { sample_flags(6,64) } else { 6 };
            let pixel = 2.0 * 50.0 / (proj * 729.0);
            cases.push(Case { words:[xy[0],xy[1],status | (level<<5),flags],
                values:[50.0,width*pixel/(1u32<<level) as f32,proj,729.0], phase:[7,0,0,0] });
            if enabled && xy == [7,0] {
                if status==2 || status==3 { expected += 1<<24; }
                if status==1 { expected += 1;
                    if level>0 && width>2.0 { expected += 1<<8; }
                    if level>0 && width>4.0 { expected += 1<<16; }
                }
            }
        }
        let inputs=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:None,contents:bytemuck::cast_slice(&cases),usage:wgpu::BufferUsages::STORAGE });
        let feedback=Feedback::new(&device);
        let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:Some("actual primary classifier probe"),source:wgpu::ShaderSource::Wgsl(source.into()) });
        for enabled in [false,true] {
            let constants=if enabled { vec![("PRIMARY_SAMPLES",1.0)] } else { vec![] };
            let pipeline=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:None,layout:None,module:&shader,entry_point:Some("probe"),
                compilation_options:wgpu::PipelineCompilationOptions { constants:&constants,..Default::default() },cache:None });
            let group=device.create_bind_group(&wgpu::BindGroupDescriptor { label:None,
                layout:&pipeline.get_bind_group_layout(0),entries:&[
                    wgpu::BindGroupEntry { binding:0,resource:inputs.as_entire_binding() },
                    wgpu::BindGroupEntry { binding:21,resource:feedback.buffer.as_entire_binding() }] });
            let staging=device.create_buffer(&wgpu::BufferDescriptor { label:None,size:16,
                usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false });
            let mut encoder=device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&feedback.buffer,0,None);
            { let mut pass=encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline); pass.set_bind_group(0,&group,&[]);
                pass.dispatch_workgroups(cases.len() as u32,1,1); }
            encoder.copy_buffer_to_buffer(&feedback.buffer,0,&staging,0,16);
            queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read,|result|result.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let data=staging.slice(..).get_mapped_range().unwrap();
            assert_eq!(u32::from_le_bytes(data[12..16].try_into().unwrap()),if enabled {expected}else{0});
        }
        let mut feedback=feedback;
        let mut encoder=device.create_command_encoder(&Default::default());
        assert!(feedback.begin(&mut encoder,sampled_view(),false).is_some());
        assert!(!feedback.pending(),"diagnostics must not schedule more frames");
        assert!(feedback.begin(&mut encoder,sampled_view(),true).is_some());
        assert!(feedback.pending(),"ordinary requests still wait for admission");
    }

    #[test]
    fn feedback_decodes_bounded_full_keys_and_signed_indices() {
        let mut data = vec![0u8; COPY_BYTES as usize];
        data[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let key = 12u32 | (2 << 24) | (7 << 27);
        data[16..20].copy_from_slice(&key.to_le_bytes());
        data[20..24].copy_from_slice(&(-8i32).to_le_bytes());
        let keys = decode(&data);
        assert_eq!(keys.len(), CAPACITY);
        assert_eq!(keys[0], (key, (-8i32) as u32));
        assert!(decode(&data[..16]).is_empty());
    }

    #[test]
    fn feedback_rejects_old_camera_and_resize_without_double_frame_wrap() {
        let old = View {
            eye: DVec3::ZERO,
            forward: DVec3::Z,
            up: DVec3::Y,
            projection_y: 2.0,
            size: [1280, 720],
            id: 1,
            frame: u32::MAX - 1,
            encoded_frame: 1,
            diagnostic_stride: None,
            max_eye_delta: 50.0,
            captured_at: std::time::Instant::now(),
        };
        let now = View {
            frame: 1,
            eye: DVec3::X * 2.0,
            ..old
        };
        assert!(now.accepts(old));
        assert!(!View {
            size: [720, 1280],
            ..now
        }
        .accepts(old));
        assert!(!View { id: 2, ..now }.accepts(old));
        assert!(!View {
            eye: DVec3::X * 100.0,
            ..now
        }
        .accepts(old));
        assert!(!View {
            forward: -DVec3::Z,
            ..now
        }
        .accepts(old));
        assert!(!View { frame: 10, ..now }.accepts(old));
    }
}
