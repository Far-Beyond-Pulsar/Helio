//! ╔═══════════════════════════════════════════════════════════════════════════╗
//! ║  T4 — THE STATELESSNESS GATE (Helio#238 issue test 4).                    ║
//! ║                                                                           ║
//! ║  Helio is a STATELESS FRAME FUNCTION. This test is the enforcement of     ║
//! ║  that sentence, and it is deliberately ruthless:                          ║
//! ║                                                                           ║
//! ║    Tear down ALL Helio-side frame structures between renders — the        ║
//! ║    graph, every pass, every bind group, the frame-transient meta          ║
//! ║    buffer, the density target, the staging buckets — by DROPPING the      ║
//! ║    entire Renderer and rebuilding a fresh one from scratch. Then render   ║
//! ║    the identical frame again. The pixels must be BYTE-IDENTICAL           ║
//! ║    (hash-compared). Three independent renderer lifetimes, three           ║
//! ║    hashes, all equal — twice consecutively, as the issue demands.         ║
//! ║                                                                           ║
//! ║  Anything Helio smuggled across that boundary — a cached bind group       ║
//! ║  keyed to a dead allocation, residency remembered in a shader-side        ║
//! ║  latch, a lazily-retained meta row, an off-by-one frame seed — shows up   ║
//! ║  here as a hash mismatch, not as a subtle visual bug in someone's game.   ║
//! ║  All PERSISTENT streaming state lives outside the renderer (in SceneDB's  ║
//! ║  tier substrate); everything inside must be re-derivable from scene       ║
//! ║  content + config, which is exactly what this rebuild exercises:          ║
//! ║                                                                           ║
//! ║    - the VT meta table is rebuilt frame-transiently from component rows   ║
//! ║      (no Renderer field survives the drop);                               ║
//! ║    - residency columns come from whatever the host last published — here, ║
//! ║      nothing, so every slot samples its full chain and the `//!use        ║
//! ║      helio_vt` contract DEGENERATES to pre-VT sampling (the golden        ║
//! ║      property T1 pins);                                                   ║
//! ║    - the demand-feedback pair (clear → five-writer feedback → compact →   ║
//! ║      fenced readback) runs identically in every lifetime, and the         ║
//! ║      parsed snapshot — asserted below against the hand-computed wanted    ║
//! ║      mip — reproduces bit-for-bit.                                        ║
//! ╚═══════════════════════════════════════════════════════════════════════════╝
//!
//! Headless machines skip gracefully (no adapter ⇒ no assertion), matching
//! `helio-default-graphs/tests/limited_native.rs` conventions.

use std::sync::{Arc, Mutex};

use glam::Vec3;
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, DebugCameraUniform, DebugDrawState,
    GpuLight, GpuMaterial, GroupMask, LightType, ObjectDescriptor, Renderer, RendererConfig, Scene,
    SceneActor,
};
use helio_default_graphs::build_default_graph_external;
use libhelio::{wanted_mip_from_derivatives, VtTextureMeta};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
/// Checkerboard side in texels — also the base-mip dimension the demand math
/// divides through.
const TEX: u32 = 256;
/// Quad world size (from -HALF to +HALF on XY, facing +Z).
const HALF: f32 = 1.0;
const FOV_DEG: f32 = 60.0;

fn main() {}

fn device_or_skip(label: &str) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("skipping: no GPU adapter (headless)");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: required_wgpu_features(adapter.features()),
        required_limits: required_wgpu_limits(adapter.limits()),
        ..Default::default()
    })) {
        Ok((d, q)) => Some((Arc::new(d), Arc::new(q))),
        Err(_) => {
            eprintln!("skipping: no device");
            None
        }
    }
}

/// FNV-1a 64-bit over a byte slice — stable, dependency-free, order-fixed.
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn checkerboard_rgba(size: u32) -> Vec<u8> {
    const CELL: u32 = 16;
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x / CELL) + (y / CELL)) % 2 == 0;
            data.extend_from_slice(if on { &[235, 235, 235, 255] } else { &[20, 20, 20, 255] });
        }
    }
    data
}

fn quad_mesh() -> helio::MeshUpload {
    // CCW when viewed from +Z; v increases upward so |dpdy(v)| = 1/height_px.
    let corners: [([f32; 3], [f32; 2]); 4] = [
        ([-HALF, -HALF, 0.0], [0.0, 0.0]),
        ([HALF, -HALF, 0.0], [1.0, 0.0]),
        ([HALF, HALF, 0.0], [1.0, 1.0]),
        ([-HALF, HALF, 0.0], [0.0, 1.0]),
    ];
    let vertices = corners
        .iter()
        .map(|(pos, uv)| {
            helio::PackedVertex::from_components(*pos, [0.0, 0.0, 1.0], *uv, [1.0, 0.0, 0.0], 1.0)
        })
        .collect();
    helio::MeshUpload {
        vertices,
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// One FULL renderer lifetime: fresh scene content, fresh graph, fresh
/// renderer. Nothing is shared with any other lifetime except the device —
/// which is the point: same GPU, same inputs, so ONLY smuggled renderer state
/// could ever make outputs differ.
fn build_lifetime(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    tag: &str,
) -> Renderer {
    let mut scene = Scene::new(Arc::clone(device), Arc::clone(queue));

    // ── Content: checkerboard-textured quad, one light ────────────────────
    let tex_id = scene
        .insert_texture(helio::TextureUpload {
            label: Some("Gate Checkerboard".to_string()),
            width: TEX,
            height: TEX,
            format: wgpu::TextureFormat::Rgba8Unorm,
            data: checkerboard_rgba(TEX),
            // Streamed-asset metadata WITH residency unpublished: the row
            // starts at the unrestricted sentinel, i.e. the pre-VT sampling
            // degenerate case. (Helio#238: residency is published BETWEEN
            // frames by the engine; this gate pins the never-published path.)
            vt: Some(VtTextureMeta {
                width: TEX,
                height: TEX,
                mip_count: 9,
                format_discriminant: u32::MAX, // uncompressed upload
                srgb: false,
                block_bytes: 16,
            }),
            sampler: Default::default(),
        })
        .expect("checkerboard inserts");

    let material_id = scene.insert_material(GpuMaterial {
        base_color: [1.0, 1.0, 1.0, 1.0],
        emissive: [0.0; 4],
        roughness_metallic: [0.9, 0.0, 1.5, 0.0],
        tex_base_color: tex_id.slot(),
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    });

    let mesh_id = scene.insert_mesh(quad_mesh());

    scene.insert_actor(SceneActor::object(ObjectDescriptor {
        mesh: mesh_id,
        material: material_id,
        transform: glam::Mat4::IDENTITY,
        bounds: [0.0, 0.0, 0.0, HALF * std::f32::consts::SQRT_2],
        flags: 3,
        groups: GroupMask::NONE,
        movability: None,
        user_tag: 0,
    }));
    scene.insert_actor(SceneActor::light(GpuLight {
        position_range: [0.0, 0.0, 0.0, f32::MAX],
        direction_outer: [-0.40824828, -0.40824828, -0.81649661, 0.0],
        color_intensity: [1.0, 0.98, 0.95, 3.0],
        shadow_index: u32::MAX, // no shadow map: one less stateful surface
        light_type: LightType::Directional as u32,
        inner_angle: 0.0,
        _pad: 0,
        god_rays_enabled: 0,
        god_rays_density: 1.0,
        god_rays_weight: 0.6,
        god_rays_decay: 1.0,
        god_rays_exposure: 0.7,
        flare_enabled: 0,
        flare_type: 0,
        flare_intensity: 0.0,
        flare_scale: 0.0,
        flare_tint_r: 0.0,
        flare_tint_g: 0.0,
        flare_tint_b: 0.0,
        ies_profile_index: -1,
        light_function_index: -1,
        ies_angle_scale: 0.0,
        ies_angle_offset: 0.0,
    }));
    scene.flush();

    // ── Graph + renderer, exactly as a host would build them ──────────────
    let mut config = RendererConfig::new(WIDTH, HEIGHT, wgpu::TextureFormat::Rgba8Unorm);
    config.render_scale = 1.0;
    config.enable_portals = false;
    config.enable_foliage = false;
    let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
    let debug_camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gate Debug Camera"),
        size: core::mem::size_of::<DebugCameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cull_stats = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gate Cull Stats"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let graph = build_default_graph_external(
        device,
        queue,
        &scene,
        config,
        Arc::clone(&debug_state),
        &debug_camera,
        &cull_stats,
        None,
    );
    #[allow(deprecated)]
    let renderer = Renderer::new_with_external_device(
        Arc::clone(device),
        Arc::clone(queue),
        config.surface_format,
        config.width,
        config.height,
        config.render_scale,
        config,
        scene,
        graph,
        debug_state,
        debug_camera,
        cull_stats,
    );
    let _ = tag;
    renderer
}

fn target_view(device: &wgpu::Device, tag: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(tag),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// The one camera every lifetime shares: straight-on at the quad, which fills
/// EXACTLY half the viewport height (see the wanted-mip derivation below).
fn camera() -> Camera {
    let distance = HALF / (FOV_DEG.to_radians() * 0.5).tan(); // g = 0.5
    Camera::perspective_look_at(
        Vec3::new(0.0, 0.0, distance),
        Vec3::ZERO,
        Vec3::Y,
        FOV_DEG.to_radians(),
        1.0,
        0.1,
        100.0,
    )
}

/// Renders ONE frame in a lifetime and hashes the packed RGBA rows.
fn render_and_hash(renderer: &mut Renderer, device: &wgpu::Device, queue: &Arc<wgpu::Queue>) -> u64 {
    let (_texture, view) = target_view(device, "Gate Target");
    let cam = camera();
    renderer.render(&cam, &view).expect("gate frame renders");
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    // Read back packed rows (256-byte stride stripped).
    let bytes_per_row = (WIDTH * 4).div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gate Staging"),
        size: (bytes_per_row * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Gate Copy") });
    encoder.copy_texture_to_buffer(
        _texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).expect("map callback channel");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map result").expect("staging maps");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let mut hasher_input = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in 0..HEIGHT {
        let start = (row * bytes_per_row) as usize;
        hasher_input.extend_from_slice(&mapped[start..start + (WIDTH * 4) as usize]);
    }
    drop(mapped);
    staging.unmap();
    hash_bytes(&hasher_input)
}

#[test]
fn tearing_down_every_frame_structure_between_frames_renders_byte_identically() {
    let Some((device, queue)) = device_or_skip("T4 statelessness gate") else { return };

    // ── Lifetime 1 ──────────────────────────────────────────────────────────
    let hash1 = {
        let mut r = build_lifetime(&device, &queue, "gate-lifetime-1");
        render_and_hash(&mut r, &device, &queue)
        // `r` DROPS HERE: graph, passes, bind groups, transient buffers — gone.
    };

    // ── Lifetime 2 (consecutive teardown #1) ────────────────────────────────
    // Also drives the demand-feedback plumbing to completion: frame 1
    // enqueues the fenced readback, frame 2's opening poll consumes it, and
    // the parsed snapshot must name the HAND-COMPUTED wanted mip.
    let (hash2, feedback_snapshot) = {
        let mut r = build_lifetime(&device, &queue, "gate-lifetime-2");
        let h_first = render_and_hash(&mut r, &device, &queue);
        // Second frame: consume frame 1's fenced readback…
        let h_second = render_and_hash(&mut r, &device, &queue);
        assert_eq!(
            h_first, h_second,
            "two identical frames inside ONE lifetime must also hash identically"
        );
        let snapshot = r.take_vt_feedback().expect("frame-1 feedback consumed by frame 2");
        (h_first, snapshot)
    };
    assert_eq!(
        hash1, hash2,
        "STATELESSNESS VIOLATION: lifetime 2 rendered differently after total teardown"
    );

    // ── Lifetime 3 (consecutive teardown #2 — the issue's "twice") ──────────
    let hash3 = {
        let mut r = build_lifetime(&device, &queue, "gate-lifetime-3");
        render_and_hash(&mut r, &device, &queue)
    };
    assert_eq!(hash1, hash3, "lifetime 3 diverged: state survived teardown again");

    // ── Demand feedback roundtrip inside lifetime 2 (full-frame T3 half) ────
    //
    // Geometry derivation: the quad spans NDC y ∈ [-0.5, +0.5] (distance
    // chosen so g = 0.5), i.e. exactly 64 px of the 128 px viewport.
    // Perspective-correct UV derivatives are CONSTANT across a planar quad:
    // v sweeps 0→1 over 64 px ⇒ |dpdy(v)| = 1/64 per pixel; u likewise.
    // Footprint in texels = TEX × derivative = 256/64 = 4 ⇒ wanted mip
    // = round(log2(4)) = 2 — the value every fragment records and compaction
    // max-reduces to, immune to intra-cell races because every writer agrees.
    let expected = wanted_mip_from_derivatives(
        [1.0 / 64.0, 0.0],
        [0.0, 1.0 / 64.0],
        TEX as f32,
        TEX as f32,
        9, // the uploaded meta row's mip_count (chain-top clamp; expected 2 < 8)
    );
    let snapshot = feedback_snapshot;
    assert!(
        snapshot.touched_stores > 0,
        "a textured quad must produce feedback stores"
    );
    assert_eq!(
        snapshot.wanted_mips.first().map(|&(s, m)| (s, m)),
        Some((0, expected)),
        "slot 0's compacted demand must equal the hand-computed wanted mip"
    );

    println!(
        "T4 STATELESSNESS GATE: three independent renderer lifetimes produced \
         identical hash {hash1:#018x}; feedback snapshot reproduced slot 0 → mip {expected}"
    );
}
