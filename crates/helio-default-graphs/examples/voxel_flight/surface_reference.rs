//! Offline spatial sampling of canonical terrain through the real lighting graph.
//! The capture point precedes AA/postprocessing. Samples are accumulated in
//! linear light, never by averaging normals and then shading that average.
use super::*;

pub struct ReferencePass {
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Buffer,
    camera_output: wgpu::Buffer,
    cpu_camera: Vec<u8>,
    size: [u32; 2],
    captures: u64,
    last_frame: Option<u64>,
}
impl ReferencePass {
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("linear surface reference capture"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var lighting:texture_2d<f32>;
@group(0) @binding(1) var albedo:texture_2d<f32>;
@group(0) @binding(2) var sunlight:texture_2d<f32>;
struct Sample { lighting:vec4<f32>, albedo:vec4<f32>, sunlight:vec4<f32> }
@group(0) @binding(3) var<storage,read_write> output:array<Sample>;
@group(0) @binding(4) var<storage,read> camera:array<vec4<u32>>;
@group(0) @binding(5) var<storage,read_write> camera_copy:array<vec4<u32>>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(lighting);
    if any(id.xy>=size) {return;}
    let p=vec2<i32>(id.xy);
    if all(id.xy==vec2<u32>(0u)) {
        for(var i=0u;i<23u;i++) {camera_copy[i]=camera[i];}
    }
    output[id.x+id.y*size.x]=Sample(textureLoad(lighting,p,0),
        textureLoad(albedo,p,0),textureLoad(sunlight,p,0));
}"#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reference pre-AA capture"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear reference pixels"),
            size: u64::from(size[0]) * u64::from(size[1]) * 48,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let camera_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference camera snapshot"),
            size: std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            output,
            camera_output,
            cpu_camera: Vec::new(),
            size,
            captures: 0,
            last_frame: None,
        }
    }
}
impl helio_core::RenderPass for ReferencePass {
    fn name(&self) -> &'static str {
        "VoxelSurfaceReference"
    }
    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa", "gbuffer", "directional_visibility"]
    }
    fn execute(&mut self, ctx: &mut helio_core::PassContext) -> helio_core::Result<()> {
        let Some(sunlight) = ctx
            .registry
            .texture_view(helio_core::ResourceKey::new("directional_visibility"))
        else {
            return Ok(());
        }; // Initial source loading.
        let lighting = ctx
            .registry
            .texture_view(helio_core::ResourceKey::new("pre_aa"))
            .expect("reference requires linear lighting");
        let gbuffer = ctx
            .registry
            .get::<helio_core::ViewGroup<'_, 4>>(helio_core::ResourceKey::new("gbuffer"))
            .expect("reference requires the GBuffer view group");
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(lighting),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(gbuffer.views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(sunlight),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ctx.camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: self.camera_output.as_entire_binding(),
            },
        ];
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let descriptor = wgpu::ComputePassDescriptor::default();
        let mut pass = ctx.begin_graphics_compute_pass(&descriptor);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        drop(pass);
        self.captures += 1;
        self.last_frame = Some(ctx.frame_num);
        let c = ctx.camera_data;
        self.cpu_camera = [
            c.view.as_slice(),
            c.proj.as_slice(),
            c.view_proj.as_slice(),
            c.inv_view_proj.as_slice(),
            c.position_near.as_slice(),
            c.forward_far.as_slice(),
            c.jitter_frame.as_slice(),
            c.prev_view_proj.as_slice(),
        ]
        .into_iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
        Ok(())
    }
}

pub fn target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reference HDR graph target"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn read(flight: &Flight, source: &wgpu::Buffer) -> Vec<u8> {
    let staging = flight.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reference readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = flight.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    flight.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    flight
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    bytes
}

struct Frame {
    lighting: Vec<u8>,
    hits: Vec<u8>,
    camera: Vec<u8>,
}
fn capture(flight: &Flight) -> Frame {
    let tap = flight.renderer.find_pass::<ReferencePass>().unwrap();
    assert!(tap.captures > 0, "reference capture did not execute");
    assert_eq!(
        tap.last_frame,
        Some(flight.frame as u64 - 1),
        "stale reference capture"
    );
    let terrain = flight.renderer.find_pass::<LazyEngineVoxelPass>().unwrap();
    assert!(
        !terrain.needs_frame(),
        "reference requires settled canonical residency"
    );
    assert_eq!(terrain.primary_hit_extent().unwrap(), flight.size);
    let camera = read(flight, &tap.camera_output);
    assert!(
        camera == tap.cpu_camera,
        "uploaded camera differs from graph CPU camera"
    );
    Frame {
        lighting: read(flight, &tap.output),
        hits: read(flight, terrain.primary_hit_buffer().unwrap()),
        camera,
    }
}
fn word(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}
fn scalar(bytes: &[u8], index: usize) -> f64 {
    f64::from(f32::from_bits(word(bytes, index)))
}

#[derive(Clone)]
struct Pixel {
    rgb: [f64; 3],
    rgb2: [f64; 3],
    albedo: [f64; 3],
    faces: [u32; 6],
    face_rgb: [[f64; 3]; 6],
    materials: [u32; 24],
    solid: u32,
    cached: u32,
    depth: [f64; 2],
    sun: f64,
}
impl Default for Pixel {
    fn default() -> Self {
        Self {
            rgb: [0.0; 3],
            rgb2: [0.0; 3],
            albedo: [0.0; 3],
            faces: [0; 6],
            face_rgb: [[0.0; 3]; 6],
            materials: [0; 24],
            solid: 0,
            cached: 0,
            depth: [f64::INFINITY, f64::NEG_INFINITY],
            sun: 0.0,
        }
    }
}
fn accumulate(pixels: &mut [Pixel], frame: &Frame) {
    assert_eq!(frame.lighting.len(), pixels.len() * 48);
    assert_eq!(frame.hits.len(), pixels.len() * 32);
    for (i, p) in pixels.iter_mut().enumerate() {
        let lit = &frame.lighting[i * 48..(i + 1) * 48];
        let hit = &frame.hits[i * 32..(i + 1) * 32];
        let status = word(hit, 3);
        p.cached += u32::from(status & 0x08000000 != 0);
        assert!(
            (status & 3) <= 1,
            "invalid primary reference hit at pixel {i}: {status}"
        );
        for a in 0..3 {
            let value = scalar(lit, a);
            assert!(
                value.is_finite() && value >= 0.0,
                "invalid radiance {value}"
            );
            p.rgb[a] += value;
            p.rgb2[a] += value * value;
        }
        if status & 3 == 1 {
            let face = ((status >> 28) & 7) as usize;
            assert!((1..=6).contains(&face));
            let material = ((status >> 8) & 3) as usize;
            p.solid += 1;
            p.faces[face - 1] += 1;
            p.materials[(face - 1) * 4 + material] += 1;
            let depth = scalar(hit, 7);
            assert!(depth.is_finite() && depth >= 0.0);
            p.depth[0] = p.depth[0].min(depth);
            p.depth[1] = p.depth[1].max(depth);
            let sun = scalar(lit, 8);
            assert!((0.0..=1.0).contains(&sun), "invalid sunlight {sun}");
            p.sun += sun;
            for a in 0..3 {
                p.albedo[a] += scalar(lit, 4 + a);
                p.face_rgb[face - 1][a] += scalar(lit, a);
            }
        }
    }
}
fn preview(path: &Path, pixels: &[[f64; 3]], size: [u32; 2]) {
    // Presentation only. Numerical comparisons use the un-tonemapped CSV/f32.
    let rgba: Vec<u8> = pixels
        .iter()
        .flat_map(|p| {
            let mut v = [255u8; 4];
            for a in 0..3 {
                v[a] = ((p[a] / (1.0 + p[a])).powf(1.0 / 2.2) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            v
        })
        .collect();
    image::save_buffer(path, &rgba, size[0], size[1], image::ColorType::Rgba8).unwrap();
}

fn save(output: &Path, name: &str, pixels: &[Pixel], center: &Frame, size: [u32; 2], samples: u32) {
    let divisor = f64::from(samples);
    let mut csv = std::io::BufWriter::new(
        fs::File::create(output.join(format!("{name}.pixels.csv"))).unwrap(),
    );
    write!(csv,"x,y,coverage,r,g,b,variance_r,variance_g,variance_b,albedo_r,albedo_g,albedo_b,sampled_depth_min_m,sampled_depth_max_m,sunlit_coverage").unwrap();
    for face in 0..6 {
        write!(
            csv,
            ",face_{face}_coverage,face_{face}_r,face_{face}_g,face_{face}_b"
        )
        .unwrap();
    }
    for face in 0..6 {
        for material in 0..4 {
            write!(csv, ",face_{face}_material_{material}").unwrap();
        }
    }
    writeln!(csv).unwrap();
    let mut mean = Vec::new();
    let mut central = Vec::new();
    let mut raw = Vec::new();
    let mut mse = 0.0;
    let mut mixed = 0usize;
    let mut partial = 0usize;
    for (i, p) in pixels.iter().enumerate() {
        assert_eq!(p.faces.iter().sum::<u32>(), p.solid);
        assert_eq!(p.materials.iter().sum::<u32>(), p.solid);
        let rgb = p.rgb.map(|v| v / divisor);
        let at_center = std::array::from_fn(|a| scalar(&center.lighting[i * 48..], a));
        for a in 0..3 {
            mse += (rgb[a] - at_center[a]).powi(2);
            raw.extend((rgb[a] as f32).to_le_bytes());
        }
        mean.push(rgb);
        central.push(at_center);
        mixed += usize::from(p.faces.iter().filter(|n| **n > 0).count() > 1);
        partial += usize::from(p.solid > 0 && p.solid < samples);
        write!(
            csv,
            "{},{},{}",
            i % size[0] as usize,
            i / size[0] as usize,
            f64::from(p.solid) / divisor
        )
        .unwrap();
        for v in rgb {
            write!(csv, ",{v:.9}").unwrap();
        }
        for a in 0..3 {
            write!(
                csv,
                ",{:.9}",
                (p.rgb2[a] / divisor - rgb[a] * rgb[a]).max(0.0)
            )
            .unwrap();
        }
        for v in p.albedo {
            write!(csv, ",{:.9}", v / divisor).unwrap();
        }
        if p.solid > 0 {
            write!(csv, ",{:.9},{:.9}", p.depth[0], p.depth[1]).unwrap();
        } else {
            write!(csv, ",,").unwrap();
        }
        write!(csv, ",{:.9}", p.sun / divisor).unwrap();
        for face in 0..6 {
            write!(csv, ",{:.9}", f64::from(p.faces[face]) / divisor).unwrap();
            for v in p.face_rgb[face] {
                write!(csv, ",{:.9}", v / divisor).unwrap();
            }
        }
        for n in p.materials {
            write!(csv, ",{:.9}", f64::from(n) / divisor).unwrap();
        }
        writeln!(csv).unwrap();
    }
    fs::write(output.join(format!("{name}.linear.f32")), raw).unwrap();
    fs::write(
        output.join(format!("{name}.center.lighting.bin")),
        &center.lighting,
    )
    .unwrap();
    fs::write(output.join(format!("{name}.center.hits.bin")), &center.hits).unwrap();
    fs::write(
        output.join(format!("{name}.center.camera.bin")),
        &center.camera,
    )
    .unwrap();
    csv.flush().unwrap();
    preview(&output.join(format!("{name}.mean.png")), &mean, size);
    preview(&output.join(format!("{name}.center.png")), &central, size);
    eprintln!("VOXEL_SURFACE_REFERENCE name={name} samples={samples} mixed_face_pixels={mixed} partial_coverage_pixels={partial} center_vs_mean_linear_rmse={:.9}",(mse/(pixels.len()*3) as f64).sqrt());
    eprintln!("VOXEL_SURFACE_CACHE_USE name={name} cached_primary_samples={} total_primary_samples={}",pixels.iter().map(|p|u64::from(p.cached)).sum::<u64>(),pixels.len() as u64*u64::from(samples));
}

fn edited(world: &World, edits: &[(DVec3, f32, u32)]) -> World {
    let mut result = world.clone();
    for &(point, radius, material) in edits {
        result
            .apply_edit(helio_pass_tiny_voxel::world::Edit {
                cell: std::array::from_fn(|a| (point[a] * 10.0).floor() as i32),
                radius,
                material,
            })
            .unwrap();
    }
    result
}

pub(super) fn cases(initial: World) -> [(&'static str, World, DVec3, Vec3); 8] {
    let ground = initial.ground_spawn(0.0, 0.0, 3.0);
    let cave = ground + DVec3::new(0.0, 0.0, -12.0);
    let cave_world = edited(
        &initial,
        &[(cave, 4.0, 3), (cave + DVec3::new(0.0, -0.3, 2.0), 2.8, 0)],
    );
    let shell = ground + DVec3::new(0.0, 4.0, -14.0);
    let thin = edited(
        &initial,
        &[
            (shell, 3.0, 3),
            (shell, (3.0 - initial.voxel_size()) as f32, 0),
        ],
    );
    let destroyed = edited(&thin, &[(shell + DVec3::Z * 2.8, 1.2, 0)]);
    [
        (
            "slope",
            initial.clone(),
            ground + DVec3::Y * 20.0,
            Vec3::new(0.0, -0.8, -1.0),
        ),
        ("ridge", initial, ground, Vec3::new(0.0, -0.02, -1.0)),
        (
            "cave",
            cave_world.clone(),
            ground,
            Vec3::new(0.0, 0.0, -1.0),
        ),
        (
            "thin-wall",
            thin.clone(),
            ground,
            (shell - ground).as_vec3(),
        ),
        (
            "destroyed-wall",
            destroyed.clone(),
            ground,
            (shell - ground).as_vec3(),
        ),
        ("cave-close", cave_world, cave + DVec3::Z * 5.0, -Vec3::Z),
        ("thin-wall-close", thin, shell + DVec3::Z * 5.0, -Vec3::Z),
        (
            "destroyed-wall-close",
            destroyed,
            shell + DVec3::Z * 5.0,
            -Vec3::Z,
        ),
    ]
}

pub fn run(flight: &mut Flight, output: &Path, grid: u32) {
    assert!(grid.is_power_of_two() && (2..=16).contains(&grid));
    assert!(
        u64::from(flight.size[0]) * u64::from(flight.size[1]) <= 65536,
        "use reference crops, at most 65536 pixels"
    );
    flight.renderer.set_jitter_enabled(false);
    flight.renderer.set_camera_jitter_override(Some([0.0, 0.0]));
    flight.renderer.set_frame_delta_override(Some(1.0 / 60.0));
    let cases = cases((*flight.world).clone());
    let selected = std::env::var("HELIO_VOXEL_REFERENCE_CASES").ok();
    let cases: Vec<_> = cases
        .into_iter()
        .filter(|(name, _, _, _)| {
            selected
                .as_ref()
                .is_none_or(|list| list.split(',').any(|case| case == *name))
        })
        .collect();
    assert!(!cases.is_empty(), "no reference cases selected");
    let case_count = cases.len() * 4;
    let mut poses = std::io::BufWriter::new(fs::File::create(output.join("poses.csv")).unwrap());
    writeln!(poses,"name,eye_x,eye_y,eye_z,forward_x,forward_y,forward_z,sun_x,sun_y,sun_z,grid_metres,samples_per_pixel").unwrap();
    for (case, world, eye, forward) in cases {
        flight.world = Arc::new(world);
        for (light, sun) in [Vec3::new(0.4, 0.8, 0.3), Vec3::new(-0.7, 0.4, -0.2)]
            .into_iter()
            .enumerate()
        {
            flight.sunlight = sun;
            (flight.update_sun)(sun);
            for movement in 0..2 {
                let eye = eye + DVec3::X * (movement as f64 * 0.025);
                let name = format!("{case}-light{light}-move{movement}");
                flight.renderer.set_camera_jitter_override(Some([0.0, 0.0]));
                flight.settle("ground_load", eye, forward);
                #[cfg(feature = "voxel-surface-cache")]
                {
                    let stats=flight.renderer.find_pass::<LazyEngineVoxelPass>().unwrap().surface_patch_stats().unwrap();
                    eprintln!("VOXEL_SURFACE_PATCH name={name} stats={stats:?}");
                    assert_eq!(stats.ready,stats.requested);
                    assert_eq!(stats.ready>0,stats.enabled);
                }
                flight.draw("reference", eye, forward);
                let center = capture(flight);
                flight.draw("reference", eye, forward);
                let repeated = capture(flight);
                for (offset, len) in [(0, 128), (288, 8)] {
                    assert_eq!(
                        &center.camera[offset..offset + len],
                        &repeated.camera[offset..offset + len],
                        "same-pose view/projection/jitter changed"
                    );
                }
                for (kind, a, b) in [
                    ("hits", &center.hits, &repeated.hits),
                    ("lighting", &center.lighting, &repeated.lighting),
                ] {
                    if a != b {
                        fs::write(output.join(format!("{name}.repeat-a.{kind}.bin")), a).unwrap();
                        fs::write(output.join(format!("{name}.repeat-b.{kind}.bin")), b).unwrap();
                        panic!("same-pose {kind} changed for {name}; raw pair saved");
                    }
                }
                if light == 0 {
                    let frame = flight.source.lock().unwrap().as_ref().unwrap().clone();
                    assert!(
                        canonical::save(
                            &center.hits,
                            flight.size,
                            &frame,
                            &output.join(format!("{name}-center"))
                        ),
                        "canonical center audit failed"
                    );
                }
                let mut pixels = vec![Pixel::default(); (flight.size[0] * flight.size[1]) as usize];
                let mut raw = std::io::BufWriter::new(
                    fs::File::create(output.join(format!("{name}.samples.bin"))).unwrap(),
                );
                for y in 0..grid {
                    for x in 0..grid {
                        let jitter = [
                            (x as f32 + 0.5) / grid as f32 - 0.5,
                            (y as f32 + 0.5) / grid as f32 - 0.5,
                        ];
                        flight.renderer.set_camera_jitter_override(Some(jitter));
                        flight.draw("reference", eye, forward);
                        let frame = capture(flight);
                        accumulate(&mut pixels, &frame);
                        // Per sample: frame-major 48-byte lighting records, then
                        // 32-byte canonical hit records. Explicit schema below.
                        raw.write_all(&frame.lighting).unwrap();
                        raw.write_all(&frame.hits).unwrap();
                    }
                }
                raw.flush().unwrap();
                let forward = forward.normalize();
                writeln!(
                    poses,
                    "{name},{:.9},{:.9},{:.9},{},{},{},{},{},{},{},{}",
                    eye.x,
                    eye.y,
                    eye.z,
                    forward.x,
                    forward.y,
                    forward.z,
                    sun.x,
                    sun.y,
                    sun.z,
                    flight.world.voxel_size(),
                    grid * grid
                )
                .unwrap();
                save(output, &name, &pixels, &center, flight.size, grid * grid);
            }
        }
    }
    poses.flush().unwrap();
    fs::write(output.join("capture.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema": 2,
        "lighting_stream": "graphics-current-frame",
        "appearance_filter": std::env::var_os("HELIO_VOXEL_APPEARANCE_FILTER").is_some(),
        "size": flight.size,
        "samples_per_pixel": grid * grid,
    })).unwrap()).unwrap();
    fs::write(output.join("README.txt"),format!(
        "Canonical surface reference; {}x{} pixels; {}x{} regular spatial samples per pixel.\nCaptured linear Rgba16Float lighting before AA/postprocess; SSR/environment/planar reflections disabled. Exact repeated-frame comparison rejects history dependence.\nRaw *.samples.bin: samples in y-major subpixel order; each has width*height records of 12 little-endian f32 values (lighting RGBA, albedo RGBA, sunlight RGBA), followed by width*height 32-byte Hit records (cell i32x3,status u32,ray f32x3,distance f32).\n*.linear.f32 is interleaved RGB mean before tone mapping. Preview uses x/(1+x), then gamma 1/2.2.\nDepth minima/maxima and coverage are sampled estimates, not conservative bounds. {} samples per pixel are not a convergence proof. Face IDs0..5 are +X,-X,+Y,-Y,+Z,-Z.\nNo production cache or performance qualification is claimed.\n",flight.size[0],flight.size[1],grid,grid,grid*grid)).unwrap();
    flight.csv.flush().unwrap();
    eprintln!(
        "VOXEL_SURFACE_REFERENCE_COMPLETE cases={case_count} samples_per_pixel={}",
        grid * grid
    );
}
