//! Scene CPU preparation: light sorting, shadow matrix computation, sky uniforms.
//!
//! The old per-frame batch-diffing / instancing logic has been removed.  Geometry
//! is now registered once via `Renderer::add_object` and lives in the persistent
//! proxy registry (`registered_objects`).  This file only handles the *environment*
//! side: lights, shadows, billboards, sky uniforms.

use super::*;
use super::uniforms::{GpuShadowMatrix, SkyUniform};
use super::shadow_math::{CSM_SPLITS, compute_point_light_matrices, compute_directional_cascades, compute_spot_matrix};
use super::helpers::estimate_sky_ambient;
use super::portal::build_portal_scene_layout;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3};

impl Renderer {
    // ─────────────────────────────────────────────────────────────────────────
    // PRIMARY PATH — called from render() via pending_env
    // ─────────────────────────────────────────────────────────────────────────

    /// Process a [`SceneEnv`]: upload lights, shadow matrices, billboards and sky
    /// uniforms.  Called at the top of `render()` whenever `pending_env` is `Some`.
    pub(crate) fn prepare_env(&mut self, env: SceneEnv, camera: &Camera) {
        let t_start = std::time::Instant::now();

        // ── Temporal light sort cache ────────────────────────────────────────
        let mut light_pos_hash: u64 = 0xcbf29ce484222325;
        for light in &env.lights {
            let bits = light.position[0].to_bits() as u64
                     ^ light.position[1].to_bits() as u64
                     ^ light.position[2].to_bits() as u64;
            light_pos_hash ^= bits;
            light_pos_hash  = light_pos_hash.wrapping_mul(0x100000001b3);
        }
        let camera_moved = {
            let dx = camera.position[0] - self.cached_camera_pos[0];
            let dy = camera.position[1] - self.cached_camera_pos[1];
            let dz = camera.position[2] - self.cached_camera_pos[2];
            let d  = (dx*dx + dy*dy + dz*dz).sqrt();
            d > self.camera_move_threshold
        };
        let should_re_sort =
            env.lights.len() != self.cached_light_count
            || light_pos_hash  != self.cached_light_position_hash
            || camera_moved;

        self.cached_light_count         = env.lights.len();
        self.cached_light_position_hash = light_pos_hash;
        self.cached_camera_pos          = camera.position.into();

        // ── Sort light indices by importance ─────────────────────────────────
        let total_lights = env.lights.len();
        self.scratch_sorted_light_indices.clear();
        self.scratch_sorted_light_indices.reserve(total_lights);
        for i in 0..total_lights { self.scratch_sorted_light_indices.push(i); }

        if should_re_sort {
            let _t = std::time::Instant::now();
            self.scratch_sorted_light_indices.sort_by(|&ia, &ib| {
                fn score(light: &crate::scene::SceneLight, cam: glam::Vec3) -> f32 {
                    match light.light_type {
                        crate::features::LightType::Directional => f32::MAX,
                        crate::features::LightType::Point => {
                            let lp = glam::Vec3::from(light.position);
                            let d  = cam.distance(lp).max(0.25);
                            let r  = light.range.max(0.001);
                            let ang = (r / d).min(8.0);
                            let prox = 1.0 / (1.0 + (d/r)*(d/r));
                            light.intensity.max(0.0) * ang * ang * prox
                        }
                        crate::features::LightType::Spot { inner_angle, outer_angle } => {
                            let lp  = glam::Vec3::from(light.position);
                            let d   = cam.distance(lp).max(0.25);
                            let r   = light.range.max(0.001);
                            let ang = (r / d).min(8.0);
                            let prox = 1.0 / (1.0 + (d/r)*(d/r));
                            let to_cam  = (cam - lp).normalize_or_zero();
                            let dir     = glam::Vec3::from(light.direction).normalize_or_zero();
                            let cos_a   = dir.dot(to_cam);
                            let ic = inner_angle.cos(); let oc = outer_angle.cos();
                            let denom = (ic - oc).max(1e-6);
                            let t = ((cos_a - oc) / denom).clamp(0.0, 1.0);
                            let cone = t*t*(3.0 - 2.0*t);
                            light.intensity.max(0.0) * ang * ang * prox * (0.25 + 0.75*cone)
                        }
                    }
                }
                score(&env.lights[ib], camera.position)
                    .partial_cmp(&score(&env.lights[ia], camera.position))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let ms = _t.elapsed().as_secs_f32() * 1000.0;
            if ms > 0.5 { eprintln!("⚠️ [Env] Light sort: {} lights — {:.2}ms", total_lights, ms); }
        }

        // ── Upload GPU lights ─────────────────────────────────────────────────
        let count = self.scratch_sorted_light_indices.len();
        self.ensure_light_buffer_capacity(count as u32);
        self.scratch_gpu_lights.clear();
        self.scratch_gpu_lights.reserve(count);
        for &idx in &self.scratch_sorted_light_indices {
            let l = &env.lights[idx];
            let light_type = match l.light_type {
                crate::features::LightType::Directional  => 0.0,
                crate::features::LightType::Point        => 1.0,
                crate::features::LightType::Spot { .. }  => 2.0,
            };
            let (cos_inner, cos_outer) = match l.light_type {
                crate::features::LightType::Spot { inner_angle, outer_angle } =>
                    (inner_angle.cos(), outer_angle.cos()),
                _ => (0.0, 0.0),
            };
            let dir_len = (l.direction[0]*l.direction[0]
                         + l.direction[1]*l.direction[1]
                         + l.direction[2]*l.direction[2]).sqrt();
            let direction = if dir_len > 1e-6 {
                [l.direction[0]/dir_len, l.direction[1]/dir_len, l.direction[2]/dir_len]
            } else { [0.0, -1.0, 0.0] };
            self.scratch_gpu_lights.push(GpuLight {
                position: l.position, light_type,
                direction, range: l.range,
                color: l.color, intensity: l.intensity,
                cos_inner, cos_outer, _pad: [0.0; 2],
            });
        }
        if !self.scratch_gpu_lights.is_empty() {
            self.queue.write_buffer(&self.light_buffer, 0,
                bytemuck::cast_slice(&self.scratch_gpu_lights));
        }

        // ── Shadow matrices ────────────────────────────────────────────────────
        let identity = Mat4::IDENTITY;
        self.scratch_shadow_mats.clear();
        self.scratch_shadow_mats.reserve(MAX_LIGHTS as usize * 6);
        self.scratch_shadow_matrix_hashes.clear();
        self.scratch_shadow_matrix_hashes.reserve(count);
        for &idx in &self.scratch_sorted_light_indices {
            let l = &env.lights[idx];
            let six: [Mat4; 6] = match l.light_type {
                crate::features::LightType::Point => {
                    compute_point_light_matrices(l.position, l.range)
                }
                crate::features::LightType::Directional => {
                    let [c0,c1,c2,c3] = compute_directional_cascades(
                        camera.position, camera.view_proj_inv, l.direction);
                    [c0, c1, c2, c3, identity, identity]
                }
                crate::features::LightType::Spot { outer_angle, .. } => {
                    let m = compute_spot_matrix(l.position, l.direction, l.range, outer_angle);
                    [m, identity, identity, identity, identity, identity]
                }
            };
            let mat_hash = {
                let mut h: u64 = 0xcbf29ce484222325;
                for m in &six {
                    for f in m.to_cols_array() {
                        h ^= f.to_bits() as u64;
                        h  = h.wrapping_mul(0x100000001b3);
                    }
                }
                h
            };
            self.scratch_shadow_matrix_hashes.push(mat_hash);
            for m in &six { self.scratch_shadow_mats.push(GpuShadowMatrix { mat: m.to_cols_array() }); }
        }
        self.scratch_shadow_mats.resize(MAX_LIGHTS as usize * 6, GpuShadowMatrix::zeroed());
        self.queue.write_buffer(&self.shadow_matrix_buffer, 0,
            bytemuck::cast_slice(&self.scratch_shadow_mats));

        // ── Light count / face counts / cull lights ────────────────────────────
        self.light_count_arc.store(count as u32, Ordering::Relaxed);
        {
            let mut fc = self.light_face_counts.lock().unwrap();
            fc.clear();
            for &idx in &self.scratch_sorted_light_indices {
                fc.push(match env.lights[idx].light_type {
                    crate::features::LightType::Point       => 6,
                    crate::features::LightType::Directional => 4,
                    crate::features::LightType::Spot { .. } => 1,
                });
            }
        }
        {
            let mut cull = self.shadow_cull_lights.lock().unwrap();
            cull.clear();
            for (slot, &li) in self.scratch_sorted_light_indices.iter().enumerate() {
                let l = &env.lights[li];
                let shadow_cull_range = match l.light_type {
                    crate::features::LightType::Directional => l.range * 2.2,
                    _                                        => l.range * 5.0,
                };
                cull.push(ShadowCullLight {
                    position: l.position, direction: l.direction,
                    range: shadow_cull_range,
                    is_directional: matches!(l.light_type, crate::features::LightType::Directional),
                    is_point:       matches!(l.light_type, crate::features::LightType::Point),
                    matrix_hash: self.scratch_shadow_matrix_hashes.get(slot).copied().unwrap_or(0),
                });
            }
        }

        // ── Billboards ──────────────────────────────────────────────────────────
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.set_billboards_slice(&env.billboards);
        }

        // ── Ambient + sky storage ───────────────────────────────────────────────
        self.scene_ambient_color     = env.ambient_color;
        self.scene_ambient_intensity = env.ambient_intensity;
        self.scene_light_count       = count as u32;
        self.scene_sky_color         = env.sky_color;

        // ── Temporal sky-LUT cache ──────────────────────────────────────────────
        let sky_state_changed = {
            let has_sky = env.sky_atmosphere.is_some();
            let color_changed = self.cached_sky_color != env.sky_color;
            if has_sky != self.cached_sky_has_sky || color_changed {
                true
            } else if let Some(ref atm) = env.sky_atmosphere {
                let sun_dir = env.lights.iter()
                    .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                    .map(|l| { let d = Vec3::from(l.direction).normalize(); [-d.x, -d.y, -d.z] })
                    .unwrap_or([0.0, 1.0, 0.0]);
                let sun_moved = (sun_dir[0] - self.cached_sky_sun_direction[0]).abs() > 0.01
                             || (sun_dir[1] - self.cached_sky_sun_direction[1]).abs() > 0.01
                             || (sun_dir[2] - self.cached_sky_sun_direction[2]).abs() > 0.01;
                sun_moved || (atm.sun_intensity - self.cached_sky_sun_intensity).abs() > 0.01
            } else { false }
        };
        self.sky_state_changed  = sky_state_changed;
        self.cached_sky_has_sky = env.sky_atmosphere.is_some();
        self.cached_sky_color   = env.sky_color;
        self.scene_has_sky      = env.sky_atmosphere.is_some();
        self.scene_csm_splits   = CSM_SPLITS;

        if let Some(ref atm) = env.sky_atmosphere {
            let sun_dir = env.lights.iter()
                .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                .map(|l| { let d = Vec3::from(l.direction).normalize(); [-d.x, -d.y, -d.z] })
                .unwrap_or([0.0, 1.0, 0.0]);
            self.cached_sky_sun_direction = sun_dir;
            self.cached_sky_sun_intensity = atm.sun_intensity;

            let clouds = atm.clouds.as_ref();
            let sky_uni = SkyUniform {
                sun_direction:    sun_dir,
                sun_intensity:    atm.sun_intensity,
                rayleigh_scatter: atm.rayleigh_scatter,
                rayleigh_h_scale: atm.rayleigh_h_scale,
                mie_scatter:      atm.mie_scatter,
                mie_h_scale:      atm.mie_h_scale,
                mie_g:            atm.mie_g,
                sun_disk_cos:     atm.sun_disk_angle.cos(),
                earth_radius:     atm.earth_radius,
                atm_radius:       atm.atm_radius,
                exposure:         atm.exposure,
                clouds_enabled:   clouds.is_some() as u32,
                cloud_coverage:   clouds.map(|c| c.coverage).unwrap_or(0.5),
                cloud_density:    clouds.map(|c| c.density).unwrap_or(0.8),
                cloud_base:       clouds.map(|c| c.base_height).unwrap_or(800.0),
                cloud_top:        clouds.map(|c| c.top_height).unwrap_or(1800.0),
                cloud_wind_x:     clouds.map(|c| c.wind_direction[0]).unwrap_or(1.0),
                cloud_wind_z:     clouds.map(|c| c.wind_direction[1]).unwrap_or(0.0),
                cloud_speed:      clouds.map(|c| c.wind_speed).unwrap_or(0.3),
                time_sky:         self.frame_count as f32 / 60.0,
                skylight_intensity: env.skylight.as_ref().map(|sl| sl.intensity).unwrap_or(1.0),
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            };
            if self.sky_state_changed {
                self.queue.write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&sky_uni));
            }
            if let Some(ref skylight) = env.skylight {
                let sun_elev = sun_dir[1].clamp(-1.0, 1.0);
                let sky_amb  = estimate_sky_ambient(sun_elev, &atm.rayleigh_scatter);
                let tint     = skylight.color_tint;
                self.scene_ambient_color = [
                    env.ambient_color[0] + sky_amb[0] * tint[0],
                    env.ambient_color[1] + sky_amb[1] * tint[1],
                    env.ambient_color[2] + sky_amb[2] * tint[2],
                ];
                self.scene_ambient_intensity = env.ambient_intensity.max(skylight.intensity);
            }
        } else {
            self.sky_state_changed = true;
        }

        // Feed sky/ambient into the RC feature
        if let Some(rc) = self.features.get_typed_mut::<RadianceCascadesFeature>("radiance_cascades") {
            let c = self.scene_ambient_color;
            let i = self.scene_ambient_intensity;
            rc.set_sky_color([c[0]*i, c[1]*i, c[2]*i]);
        }

        let ms = t_start.elapsed().as_secs_f32() * 1000.0;
        if self.debug_printout && self.frame_count % 60 == 0 {
            eprintln!("🔧 prepare_env: {:.2}ms (lights={}, sky={})", ms, count, self.scene_has_sky);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // COMPAT SHIM — for callers that haven't migrated to add_object yet
    // ─────────────────────────────────────────────────────────────────────────

    /// Legacy API: provide a [`Scene`] snapshot each frame.  Internally converts
    /// to `set_env_from_scene` + `render()`.  Geometry in `scene.objects` is pushed
    /// directly onto the draw list (one-shot, no GPU buffer retained across frames).
    ///
    /// Prefer `add_object` / `set_scene_env` / `render()` for new code.
    pub fn render_scene(&mut self, scene: &Scene, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        // Capture scene layout for the live-portal thread.
        if self.live_portal.is_some() {
            let current_key = (scene.objects.len(), scene.lights.len(), scene.billboards.len());
            if self.portal_scene_key != current_key || self.latest_scene_layout.is_none() {
                self.latest_scene_layout = Some(build_portal_scene_layout(scene, camera));
                self.portal_scene_key    = current_key;
                self.pending_layout_changed = true;
            } else {
                if let Some(ref mut layout) = self.latest_scene_layout {
                    let forward = camera.forward();
                    layout.camera = Some(PortalSceneCamera {
                        position: [camera.position.x, camera.position.y, camera.position.z],
                        forward:  [forward.x, forward.y, forward.z],
                    });
                }
                self.pending_layout_changed = false;
            }
        }

        // Push scene objects directly onto the draw list (legacy one-shot path).
        // Callers that want zero per-frame GPU alloc should migrate to add_object().
        {
            let mut dl  = self.draw_list.lock().unwrap();
            let mut sdl = self.shadow_draw_list.lock().unwrap();
            sdl.clear();
            for obj in &scene.objects {
                let (bind_group, transparent) = match obj.material.as_ref() {
                    Some(mat) => (Arc::clone(&mat.bind_group), mat.transparent_blend),
                    None      => (Arc::clone(&self.default_material_bind_group), false),
                };
                let mut dc = DrawCall::new(&obj.mesh, bind_group, transparent);
                // Single-instance: pack the transform directly on the DrawCall.
                let cols = obj.transform.to_cols_array();
                let instance_buf = Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("legacy_inst"),
                        contents: bytemuck::bytes_of(&cols),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
                dc.instance_buffer = Some(Arc::clone(&instance_buf));
                dc.instance_count  = 1;
                dl.push(dc.clone());
                sdl.push(dc);
                // NOTE: instance_buf drops here; DrawCall holds the Arc so it lives
                // until the draw list is consumed by render().
            }
        }

        // Convert scene lights/sky/billboards into a SceneEnv and process immediately.
        self.set_env_from_scene(scene);
        if let Some(env) = self.pending_env.take() {
            self.prepare_env(env, camera);
        }

        self.render(camera, target, delta_time)
    }

    /// Submit a mesh using per-instance transforms.
    pub fn draw_mesh_instanced(&mut self,
        mesh: &GpuMesh,
        material: Arc<wgpu::BindGroup>,
        instance_buf: Arc<wgpu::Buffer>,
        count: u32,
    ) {
        let mut dc = DrawCall::new(mesh, material, false);
        dc.instance_buffer = Some(instance_buf);
        dc.instance_count = count;
        self.draw_list.lock().unwrap().push(dc);
    }
}
