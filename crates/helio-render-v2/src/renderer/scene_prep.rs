//! Scene CPU preparation: batching, light sorting, shadow matrix computation, sky uniforms.

use super::*;
use super::uniforms::{GpuShadowMatrix, SkyUniform};
use super::shadow_math::{CSM_SPLITS, compute_point_light_matrices, compute_directional_cascades, compute_spot_matrix};
use super::helpers::estimate_sky_ambient;
use super::portal::build_portal_scene_layout;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3};

/// FNV-1a hash of a flat slice of `[f32; 16]` transform matrices.
/// Used to skip redundant `write_buffer` calls when data hasn't changed.
#[inline]
fn fnv_mats(mats: &[[f32; 16]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for mat in mats {
        for &f in mat {
            h ^= f.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

impl Renderer {
    /// Render the full scene. Everything in the scene is drawn; nothing else.
    pub fn render_scene(&mut self, scene: &Scene, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        let render_scene_start = std::time::Instant::now();

        // Capture scene layout for the portal thread before render mutates state.
        if self.live_portal.is_some() {
            let current_key = (scene.objects.len(), scene.lights.len(), scene.billboards.len());
            if self.portal_scene_key != current_key || self.latest_scene_layout.is_none() {
                self.latest_scene_layout = Some(build_portal_scene_layout(scene, camera));
                self.portal_scene_key = current_key;
                self.pending_layout_changed = true;
            } else {
                // Scene structure unchanged — only update camera (O(1)).
                if let Some(ref mut layout) = self.latest_scene_layout {
                    let forward = camera.forward();
                    layout.camera = Some(PortalSceneCamera {
                        position: [camera.position.x, camera.position.y, camera.position.z],
                        forward: [forward.x, forward.y, forward.z],
                    });
                }
                self.pending_layout_changed = false;
            }
        }

        // ---------- automatic batching/instancing ----------
        // Proxy lifetime: keep GPU buffers alive this many frames after a batch was
        // last frustum-visible.  Camera rotation can cycle 500-1000 chunks out of the
        // frustum and back within 1-2 frames; without this window every rotation
        // triggers hundreds of create_buffer_init calls.  120 frames @ 60 fps = 2s,
        // large enough to absorb any realistic camera sweep.
        const BATCH_EVICT_FRAMES: u64 = 120;

        let t1_queue_draws;
        {
            // ── Persistent proxy model (Unreal FPrimitiveSceneProxy lifetime) ─────
            //
            // scene.objects = frustum-visible chunks ONLY (the app already frustum-culled).
            // We separate three orthogonal concepts:
            //   1. Proxy alive  — GPU buffer exists; evicted after BATCH_EVICT_FRAMES absent.
            //   2. Visible now  — in scene.objects this frame → goes into draw_list.
            //   3. Shadow caster — all alive proxies → shadow_draw_list (shadow culls itself).
            //
            // Per-frame cost: O(N_visible) partitioning + O(delta) GPU alloc/upload.
            // At steady-state (no new chunks, just camera rotation): zero GPU alloc,
            // zero draw_list_generation bumps, zero shadow_draw_list rebuilds.

            // Step 1: partition VISIBLE objects into (mesh_ptr, mat_ptr) batches.
            self.scratch_batches.clear();
            self.scratch_example_idx.clear();
            for (i_obj, obj) in scene.objects.iter().enumerate() {
                let mesh_ptr = Arc::as_ptr(&obj.mesh.vertex_buffer) as usize;
                let mat_ptr  = obj.material.as_ref()
                    .map(|m| Arc::as_ptr(&m.bind_group) as usize)
                    .unwrap_or(0);
                let key = (mesh_ptr, mat_ptr);
                self.scratch_batches.entry(key).or_default().push(obj.transform.to_cols_array());
                self.scratch_example_idx.entry(key).or_insert(i_obj);
            }

            // Step 2: evict proxies absent for > BATCH_EVICT_FRAMES.
            // Proxies merely outside the frustum this frame are NOT evicted.
            let stale_keys: Vec<(usize, usize)> = self.persistent_batch_draws
                .keys()
                .filter(|&&k| {
                    let last = self.batch_last_seen.get(&k).copied().unwrap_or(0);
                    self.frame_count.saturating_sub(last) > BATCH_EVICT_FRAMES
                })
                .copied()
                .collect();
            let eviction_count = stale_keys.len();
            for key in stale_keys {
                self.persistent_batch_draws.remove(&key);
                self.batch_last_seen.remove(&key);
                self.shadow_batch_buffers.remove(&key);
                self.shadow_batch_capacities.remove(&key);
                self.shadow_batch_transform_hashes.remove(&key);
            }

            // Step 3: add new proxies / refresh buffers for currently-visible batches.
            let mut addition_count: u32 = 0;
            let mut instance_total: u32 = 0;
            for (&key, mats) in &self.scratch_batches {
                if mats.is_empty() { continue; }
                let count = mats.len() as u32;
                instance_total += count;

                // Stamp visibility so the eviction clock resets on re-appearance.
                self.batch_last_seen.insert(key, self.frame_count);

                let prev_cap = self.shadow_batch_capacities.get(&key).copied().unwrap_or(0);
                if count != prev_cap {
                    // New proxy or instance count changed: (re)allocate buffer.
                    let new_buf = Arc::new(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Batch Inst"),
                            contents: bytemuck::cast_slice(mats),
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        },
                    ));
                    let buf_for_dc = Arc::clone(&new_buf);
                    self.shadow_batch_buffers.insert(key, new_buf);
                    self.shadow_batch_capacities.insert(key, count);
                    self.shadow_batch_transform_hashes.insert(key, fnv_mats(mats));

                    let is_new = !self.persistent_batch_draws.contains_key(&key);
                    if is_new { addition_count += 1; }

                    let i_obj = self.scratch_example_idx[&key];
                    let obj   = &scene.objects[i_obj];
                    let (bind_group, transparent) = match obj.material.as_ref() {
                        Some(mat) => (Arc::clone(&mat.bind_group), mat.transparent_blend),
                        None      => (Arc::clone(&self.default_material_bind_group), false),
                    };
                    let mut dc = DrawCall::new(&obj.mesh, bind_group, transparent);
                    dc.instance_buffer        = Some(buf_for_dc);
                    dc.instance_buffer_offset = 0;
                    dc.instance_count         = count;
                    self.persistent_batch_draws.insert(key, dc);
                } else {
                    // Stable count: write_buffer only when transform data changed.
                    if let Some(buf) = self.shadow_batch_buffers.get(&key) {
                        let h      = fnv_mats(mats);
                        let prev_h = self.shadow_batch_transform_hashes.get(&key).copied().unwrap_or(0);
                        if h != prev_h {
                            self.queue.write_buffer(buf, 0, bytemuck::cast_slice(mats));
                            self.shadow_batch_transform_hashes.insert(key, h);
                        }
                    }
                }
            }

            // Step 4: draw_list = VISIBLE proxies only (frustum-culled set).
            // Culled-but-alive proxies are not submitted for rasterization this frame.
            {
                let mut dl = self.draw_list.lock().unwrap();
                for &key in self.scratch_batches.keys() {
                    if let Some(dc) = self.persistent_batch_draws.get(&key) {
                        dl.push(dc.clone());
                    }
                }
            }

            // shadow_draw_list = ALL alive proxies (shadow does per-face frustum culling).
            // Only rebuild when proxies are added or evicted — not on frustum changes.
            if eviction_count > 0 || addition_count > 0 {
                *self.shadow_draw_list.lock().unwrap() =
                    self.persistent_batch_draws.values().cloned().collect();
                self.draw_list_generation = self.draw_list_generation.wrapping_add(1);
                eprintln!(
                    "⚠️ [Scene] proxy +{} evict {} | alive {} | visible {} | {:.1} KB",
                    addition_count, eviction_count,
                    self.persistent_batch_draws.len(),
                    self.scratch_batches.len(),
                    instance_total as f32 * 64.0 / 1024.0,
                );
            }

            t1_queue_draws = render_scene_start.elapsed().as_secs_f32() * 1000.0;
            self.pending_scene_stage_ms[0] = t1_queue_draws;
        } // end batch update block

        // ────────────────────────────────────────────────────────────────────
        // TEMPORAL LIGHT SORTING CACHE
        // ────────────────────────────────────────────────────────────────────

        let mut light_pos_hash: u64 = 0xcbf29ce484222325;  // FNV-1a offset basis
        for light in &scene.lights {
            let bits = light.position[0].to_bits() as u64
                     ^ light.position[1].to_bits() as u64
                     ^ light.position[2].to_bits() as u64;
            light_pos_hash ^= bits;
            light_pos_hash = light_pos_hash.wrapping_mul(0x100000001b3);
        }

        let camera_moved = {
            let dx = camera.position[0] - self.cached_camera_pos[0];
            let dy = camera.position[1] - self.cached_camera_pos[1];
            let dz = camera.position[2] - self.cached_camera_pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist > self.camera_move_threshold
        };

        let should_re_sort =
            scene.lights.len() != self.cached_light_count
            || light_pos_hash != self.cached_light_position_hash
            || camera_moved;

        self.cached_light_count = scene.lights.len();
        self.cached_light_position_hash = light_pos_hash;
        self.cached_camera_pos = camera.position.into();

        let total_lights = scene.lights.len();

        self.scratch_sorted_light_indices.clear();
        self.scratch_sorted_light_indices.reserve(total_lights);
        for i in 0..total_lights {
            self.scratch_sorted_light_indices.push(i);
        }

        if should_re_sort {
            let _light_sort_t = std::time::Instant::now();
            self.scratch_sorted_light_indices.sort_by(|&ia, &ib| {
                fn score(light: &crate::scene::SceneLight, camera_pos: glam::Vec3) -> f32 {
                    match light.light_type {
                        crate::features::LightType::Directional => {
                            f32::MAX
                        }
                        crate::features::LightType::Point => {
                            let lp = glam::Vec3::from(light.position);
                            let d  = camera_pos.distance(lp).max(0.25);
                            let r  = light.range.max(0.001);
                            let intensity = light.intensity.max(0.0);
                            let angular = (r / d).min(8.0);
                            let proximity = 1.0 / (1.0 + (d / r) * (d / r));
                            intensity * (angular * angular) * proximity
                        }
                        crate::features::LightType::Spot { inner_angle, outer_angle } => {
                            let lp = glam::Vec3::from(light.position);
                            let d  = camera_pos.distance(lp).max(0.25);
                            let r  = light.range.max(0.001);
                            let intensity = light.intensity.max(0.0);
                            let angular = (r / d).min(8.0);
                            let proximity = 1.0 / (1.0 + (d / r) * (d / r));
                            let to_camera = (camera_pos - lp).normalize_or_zero();
                            let dir = glam::Vec3::from(light.direction).normalize_or_zero();
                            let cos_a = dir.dot(to_camera);
                            let inner_cos = inner_angle.cos();
                            let outer_cos = outer_angle.cos();
                            let denom = (inner_cos - outer_cos).max(1e-6);
                            let t = ((cos_a - outer_cos) / denom).clamp(0.0, 1.0);
                            let cone = t * t * (3.0 - 2.0 * t);
                            intensity * (angular * angular) * proximity * (0.25 + 0.75 * cone)
                        }
                    }
                }
                let sb = score(&scene.lights[ib], camera.position);
                let sa = score(&scene.lights[ia], camera.position);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });            let _lsort_ms = _light_sort_t.elapsed().as_secs_f32() * 1000.0;
            if _lsort_ms > 0.5 {
                eprintln!("⚠️ [Scene] Light sort: {} lights — {:.2}ms", total_lights, _lsort_ms);
            }        } else if self.frame_count % 60 == 0 {
            log::trace!("⚡ Light sort cache HIT — skipping re-sort");
        }

        let count = self.scratch_sorted_light_indices.len();
        if self.frame_count % 120 == 0 && total_lights > count {
            log::trace!("Light culling: {} visible of {} total", count, total_lights);
        }

        self.ensure_light_buffer_capacity(count as u32);

        self.scratch_gpu_lights.clear();
        self.scratch_gpu_lights.reserve(count);
        for &idx in &self.scratch_sorted_light_indices {
            let l = &scene.lights[idx];
            let light_type = match l.light_type {
                crate::features::LightType::Directional => 0.0,
                crate::features::LightType::Point => 1.0,
                crate::features::LightType::Spot { .. } => 2.0,
            };
            let (cos_inner, cos_outer) = match l.light_type {
                crate::features::LightType::Spot { inner_angle, outer_angle } => {
                    (inner_angle.cos(), outer_angle.cos())
                }
                _ => (0.0, 0.0),
            };
            let dir_len = (l.direction[0] * l.direction[0]
                + l.direction[1] * l.direction[1]
                + l.direction[2] * l.direction[2]).sqrt();
            let direction = if dir_len > 1e-6 {
                [l.direction[0] / dir_len, l.direction[1] / dir_len, l.direction[2] / dir_len]
            } else {
                [0.0, -1.0, 0.0]
            };

            self.scratch_gpu_lights.push(GpuLight {
                position: l.position,
                light_type,
                direction,
                range: l.range,
                color: l.color,
                intensity: l.intensity,
                cos_inner,
                cos_outer,
                _pad: [0.0; 2],
            });
        }

        if !self.scratch_gpu_lights.is_empty() {
            self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&self.scratch_gpu_lights));
        }
        let t2_lights = render_scene_start.elapsed().as_secs_f32() * 1000.0;

        // Compute and upload light-space matrices — 6 per light (cube faces).
        let identity = Mat4::IDENTITY;
        self.scratch_shadow_mats.clear();
        self.scratch_shadow_mats.reserve(MAX_LIGHTS as usize * 6);
        self.scratch_shadow_matrix_hashes.clear();
        self.scratch_shadow_matrix_hashes.reserve(count);
        for &idx in &self.scratch_sorted_light_indices {
            let l = &scene.lights[idx];
            let six: [Mat4; 6] = match l.light_type {
                crate::features::LightType::Point => {
                    compute_point_light_matrices(l.position, l.range)
                }
                crate::features::LightType::Directional => {
                    let [c0, c1, c2, c3] = compute_directional_cascades(
                        camera.position,
                        camera.view_proj_inv,
                        l.direction,
                    );
                    [c0, c1, c2, c3, identity, identity]
                }
                crate::features::LightType::Spot { outer_angle, .. } => {
                    let m0 = compute_spot_matrix(l.position, l.direction, l.range, outer_angle);
                    [m0, identity, identity, identity, identity, identity]
                }
            };
            let mat_hash = {
                let mut h: u64 = 0xcbf29ce484222325;
                for m in &six {
                    let cols = m.to_cols_array();
                    for f in cols {
                        h ^= f.to_bits() as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                }
                h
            };
            self.scratch_shadow_matrix_hashes.push(mat_hash);
            for m in &six {
                self.scratch_shadow_mats.push(GpuShadowMatrix { mat: m.to_cols_array() });
            }
        }
        self.scratch_shadow_mats.resize(MAX_LIGHTS as usize * 6, GpuShadowMatrix::zeroed());
        self.queue.write_buffer(&self.shadow_matrix_buffer, 0, bytemuck::cast_slice(&self.scratch_shadow_mats));
        let t3_shadows = render_scene_start.elapsed().as_secs_f32() * 1000.0;

        // Update shared light count and per-light face counts (ShadowPass reads these)
        self.light_count_arc.store(count as u32, Ordering::Relaxed);
        {
            let mut fc = self.light_face_counts.lock().unwrap();
            fc.clear();
            for &idx in &self.scratch_sorted_light_indices {
                let l = &scene.lights[idx];
                let faces: u8 = match l.light_type {
                    crate::features::LightType::Point       => 6,
                    crate::features::LightType::Directional => 4,
                    crate::features::LightType::Spot { .. } => 1,
                };
                fc.push(faces);
            }
        }
        {
            let mut cull = self.shadow_cull_lights.lock().unwrap();
            cull.clear();
            for (slot, &li) in self.scratch_sorted_light_indices.iter().enumerate() {
                let l = &scene.lights[li];
                let shadow_cull_range = match l.light_type {
                    crate::features::LightType::Directional => l.range * 2.2,
                    _ => l.range * 5.0,
                };
                cull.push(ShadowCullLight {
                    position:       l.position,
                    direction:      l.direction,
                    range:          shadow_cull_range,
                    is_directional: matches!(l.light_type, crate::features::LightType::Directional),
                    is_point:       matches!(l.light_type, crate::features::LightType::Point),
                    matrix_hash:    self.scratch_shadow_matrix_hashes.get(slot).copied().unwrap_or(0),
                });
            }
        }

        // Update billboard instances from scene
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.set_billboards_slice(&scene.billboards);
        }
        let t4_billboards = render_scene_start.elapsed().as_secs_f32() * 1000.0;

        // Store scene ambient + sky for globals upload in render()
        self.scene_ambient_color    = scene.ambient_color;
        self.scene_ambient_intensity = scene.ambient_intensity;
        self.scene_light_count = count as u32;
        self.scene_sky_color = scene.sky_color;

        // ────────────────────────────────────────────────────────────────────
        // TEMPORAL SKY LUT CACHING
        // ────────────────────────────────────────────────────────────────────
        let sky_state_changed = {
            let has_sky_now = scene.sky_atmosphere.is_some();
            let sky_color_changed = self.cached_sky_color != scene.sky_color;

            if has_sky_now != self.cached_sky_has_sky || sky_color_changed {
                true
            } else {
                if let Some(atm) = &scene.sky_atmosphere {
                    let sun_dir = scene.lights.iter()
                        .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                        .map(|l| {
                            let d = Vec3::from(l.direction).normalize();
                            [-d.x, -d.y, -d.z]
                        })
                        .unwrap_or([0.0, 1.0, 0.0]);

                    let sun_moved = (sun_dir[0] - self.cached_sky_sun_direction[0]).abs() > 0.01
                                 || (sun_dir[1] - self.cached_sky_sun_direction[1]).abs() > 0.01
                                 || (sun_dir[2] - self.cached_sky_sun_direction[2]).abs() > 0.01;
                    let intensity_changed = (atm.sun_intensity - self.cached_sky_sun_intensity).abs() > 0.01;

                    sun_moved || intensity_changed
                } else {
                    false
                }
            }
        };

        self.sky_state_changed = sky_state_changed;
        self.cached_sky_has_sky = scene.sky_atmosphere.is_some();
        self.cached_sky_color = scene.sky_color;

        self.scene_has_sky = scene.sky_atmosphere.is_some();
        self.scene_csm_splits = CSM_SPLITS;

        // Build and upload sky uniforms
        if let Some(atm) = &scene.sky_atmosphere {
            let sun_dir = scene.lights.iter()
                .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                .map(|l| {
                    let d = Vec3::from(l.direction).normalize();
                    [-d.x, -d.y, -d.z]
                })
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
                skylight_intensity: scene.skylight.as_ref().map(|sl| sl.intensity).unwrap_or(1.0),
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            };

            if self.sky_state_changed {
                self.queue.write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&sky_uni));
            } else if self.frame_count % 60 == 0 {
                log::trace!("⚡ Sky LUT cache HIT — skipping re-render");
            }

            if let Some(skylight) = &scene.skylight {
                let sun_elev = sun_dir[1].clamp(-1.0, 1.0);
                let sky_amb  = estimate_sky_ambient(sun_elev, &atm.rayleigh_scatter);
                let tint     = skylight.color_tint;
                self.scene_ambient_color = [
                    scene.ambient_color[0] + sky_amb[0] * tint[0],
                    scene.ambient_color[1] + sky_amb[1] * tint[1],
                    scene.ambient_color[2] + sky_amb[2] * tint[2],
                ];
                self.scene_ambient_intensity = scene.ambient_intensity.max(skylight.intensity);
            }
        } else {
            self.sky_state_changed = true;
        }

        // Feed the computed sky/ambient colour into the RC feature
        if let Some(rc) = self.features.get_typed_mut::<RadianceCascadesFeature>("radiance_cascades") {
            let c = self.scene_ambient_color;
            let i = self.scene_ambient_intensity;
            rc.set_sky_color([c[0] * i, c[1] * i, c[2] * i]);
        }

        let t5_sky = render_scene_start.elapsed().as_secs_f32() * 1000.0;

        if self.debug_printout && self.frame_count % 60 == 0 {
            eprintln!("🔧 render_scene CPU breakdown: queue={:.2}ms, lights={:.2}ms, shadow={:.2}ms, bb={:.2}ms, sky={:.2}ms",
                t1_queue_draws, t2_lights - t1_queue_draws,
                t3_shadows - t2_lights, t4_billboards - t3_shadows, t5_sky - t4_billboards);
        }

        // Stash per-stage timings so render() can include them in the portal snapshot.
        // [0]=draws, [1]=lights, [2]=shadows, [3]=billboards, [4]=sky
        self.pending_scene_stage_ms = [
            t1_queue_draws,
            t2_lights       - t1_queue_draws,
            t3_shadows      - t2_lights,
            t4_billboards   - t3_shadows,
            t5_sky          - t4_billboards,
        ];

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
