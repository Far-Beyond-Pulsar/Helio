use web_time::Instant;

use arrayvec::ArrayVec;
use helio_pass_debug::DebugCameraUniform;
use helio_v3::Result as HelioResult;

use crate::groups::GroupId;
use crate::scene::Camera;

use super::renderer_impl::Renderer;

impl Renderer {
    /// Presents a rendered surface texture on this renderer's queue.
    pub fn present(&self, texture: wgpu::SurfaceTexture) {
        self.queue.present(texture);
    }

    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView) -> HelioResult<()> {
        if let Some((w, h)) = self.pending_resize.take() {
            self.apply_resize_now(w, h);
        }

        let now = Instant::now();
        let dt = now
            .duration_since(self.last_render_time)
            .as_secs_f32()
            .min(0.1);
        self.last_render_time = now;
        self.delta_time = dt;
        self.frame_times[self.frame_times_cursor] = dt;
        self.frame_times_cursor = (self.frame_times_cursor + 1) % self.frame_times.len();
        self.graph.set_delta_time(dt);

        let internal_w = (((self.output_width as f32) * self.render_scale).ceil() as u32).max(1);
        let internal_h = (((self.output_height as f32) * self.render_scale).ceil() as u32).max(1);
        let frame_idx = self.scene.gpu_scene().frame_count;
        let [jx, jy] = libhelio::temporal_jitter_ndc(frame_idx, internal_w, internal_h);
        let jitter_mat = glam::Mat4::from_translation(glam::Vec3::new(jx, jy, 0.0));
        let jittered_m = jitter_mat * camera.proj * camera.view;
        let col = jittered_m.to_cols_array();
        let debug_camera_uniform = DebugCameraUniform {
            view_proj: [
                [col[0], col[1], col[2], col[3]],
                [col[4], col[5], col[6], col[7]],
                [col[8], col[9], col[10], col[11]],
                [col[12], col[13], col[14], col[15]],
            ],
        };
        self.queue.write_buffer(
            &self.debug_camera_buffer,
            0,
            bytemuck::bytes_of(&debug_camera_uniform),
        );

        let mut jittered_camera = *camera;
        jittered_camera.proj = jitter_mat * camera.proj;
        jittered_camera.jitter = [jx, jy];
        self.scene.update_camera(jittered_camera);
        self.scene.flush();

        let editor_hidden = self.scene.is_group_hidden(GroupId::EDITOR);
        let light_count = self.scene.gpu_scene().lights.len();
        let light_gen = self.scene.gpu_scene().movable_lights_generation;
        let corona_gen = self.corona_emitter_generation;
        if self.billboard_dirty
            || light_count != self.billboard_cached_light_count
            || light_gen != self.billboard_cached_light_gen
            || editor_hidden != self.billboard_cached_editor_hidden
            || corona_gen != self.billboard_cached_corona_gen
        {
            self.billboard_scratch.clear();
            self.billboard_scratch
                .extend_from_slice(&self.billboard_instances);
            if !editor_hidden {
                for light in self.scene.gpu_scene().lights.as_slice() {
                    if light.light_type == libhelio::LightType::Point as u32
                        || light.light_type == libhelio::LightType::Spot as u32
                    {
                        let [x, y, z, _] = light.position_range;
                        let [r, g, b, _] = light.color_intensity;
                        self.billboard_scratch
                            .push(helio_pass_billboard::BillboardInstance {
                                world_pos: [x, y, z, 0.0],
                                scale_flags: [0.25, 0.25, 0.0, 0.0],
                                color: [r, g, b, 1.0],
                            });
                    }
                }
                for emitter in &self.corona_emitters {
                    let [x, y, z, _] = emitter.transform[3];
                    self.billboard_scratch
                        .push(helio_pass_billboard::BillboardInstance {
                            world_pos: [x, y, z, 0.0],
                            scale_flags: [0.25, 0.25, 0.0, 0.0],
                            color: [0.2, 0.8, 1.0, 1.0],
                        });
                }
            }
            self.billboard_generation = self.billboard_generation.wrapping_add(1);
            self.billboard_dirty = false;
            self.billboard_cached_light_count = light_count;
            self.billboard_cached_light_gen = light_gen;
            self.billboard_cached_editor_hidden = editor_hidden;
            self.billboard_cached_corona_gen = corona_gen;
        }

        let water_volume_count = self.scene.water_volumes_count();
        if water_volume_count > 0 && self.scene.water_volumes_dirty() {
            let water_volumes = self.scene.get_water_volumes_gpu_slice();
            let water_volume_dirty_range = self.scene.water_volumes_dirty_range();
            if let Some(pass) = self
                .graph
                .find_pass_mut::<helio_pass_water_sim::WaterSimPass>()
            {
                let vol = &water_volumes[0];
                pass.set_sim_dynamics(vol.sim_dynamics[0], vol.sim_dynamics[1]);
                pass.set_wave_scale(vol.sim_dynamics[2]);
                pass.set_wave_speed(vol.wave_params[2]);
                pass.set_wind([vol.wind_params[0], vol.wind_params[1]], vol.wind_params[2]);
            }
            if let Some((start, end)) = water_volume_dirty_range {
                self.queue.write_buffer(
                    &self.water_volumes_buffer,
                    (start * std::mem::size_of::<libhelio::GpuWaterVolume>()) as u64,
                    bytemuck::cast_slice(&water_volumes[start..end]),
                );
            }
            self.scene.clear_water_volumes_dirty();
        }

        let water_hitbox_count = self.scene.water_hitboxes_count();
        if water_hitbox_count > 0 && self.scene.water_hitboxes_dirty() {
            let water_hitboxes = self.scene.get_water_hitboxes_gpu_slice();
            let water_hitbox_dirty_range = self.scene.water_hitboxes_dirty_range();
            if let Some((start, end)) = water_hitbox_dirty_range {
                self.queue.write_buffer(
                    &self.water_hitboxes_buffer,
                    (start * std::mem::size_of::<libhelio::GpuWaterHitbox>()) as u64,
                    bytemuck::cast_slice(&water_hitboxes[start..end]),
                );
            }
            self.scene.clear_water_hitboxes_dirty();
        }

        let mut texture_views =
            ArrayVec::<&wgpu::TextureView, { crate::material::MAX_TEXTURES }>::new();
        let mut samplers = ArrayVec::<&wgpu::Sampler, { crate::material::MAX_TEXTURES }>::new();
        for slot in 0..crate::material::MAX_TEXTURES {
            texture_views.push(self.scene.texture_view_for_slot(slot));
            samplers.push(self.scene.texture_sampler_for_slot(slot));
        }

        let mesh_buffers = self.scene.mesh_buffers();
        let dynamic_mesh_buffers = self.scene.dynamic_mesh_buffers();
        if let Ok(mut state) = self.debug_state.lock() {
            state.camera_position = camera.position;
        }
        let mut frame_resources = libhelio::FrameResources::empty();
        frame_resources.main_scene.write(
            libhelio::MainSceneResources {
                mesh_buffers: libhelio::MeshBuffers {
                    vertices: mesh_buffers.vertices,
                    indices: mesh_buffers.indices,
                    dynamic_vertices: dynamic_mesh_buffers.vertices,
                    dynamic_indices: dynamic_mesh_buffers.indices,
                },
                material_textures: libhelio::MaterialTextureBindings {
                    material_textures: self.scene.material_texture_buffer(),
                    texture_views: texture_views.as_slice(),
                    samplers: samplers.as_slice(),
                    version: self.scene.texture_binding_version(),
                },
                clear_color: self.clear_color,
                ambient_color: self.ambient_color,
                ambient_intensity: self.ambient_intensity,
            },
            "Renderer",
        );
        if !self.billboard_scratch.is_empty() {
            frame_resources.billboards.write(
                libhelio::BillboardFrameData {
                    instances: bytemuck::cast_slice(&self.billboard_scratch),
                    count: self.billboard_scratch.len() as u32,
                    generation: self.billboard_generation,
                },
                "Renderer",
            );
        }

        if !self.corona_emitters.is_empty() {
            frame_resources.corona_emitters.write(
                libhelio::CoronaEmitterFrameData {
                    emitters: bytemuck::cast_slice(&self.corona_emitters),
                    count: self.corona_emitters.len() as u32,
                    generation: self.corona_emitter_generation,
                    max_particles: libhelio::CORONA_MAX_PARTICLES,
                },
                "Renderer",
            );
        }
        if water_volume_count > 0 {
            frame_resources
                .water_volumes
                .write(&self.water_volumes_buffer, "Renderer");
        }
        frame_resources.water_volume_count = water_volume_count;
        if water_hitbox_count > 0 {
            frame_resources
                .water_hitboxes
                .write(&self.water_hitboxes_buffer, "Renderer");
        }
        frame_resources.water_hitbox_count = water_hitbox_count;
        frame_resources
            .depth_texture
            .write(&self.depth_texture, "Renderer");
        if let Some(v) = self
            .full_res_depth_view
            .as_ref()
            .map(|v| v as &wgpu::TextureView)
        {
            frame_resources.full_res_depth.write(v, "Renderer");
        }
        if let Some(t) = self
            .full_res_depth_texture
            .as_ref()
            .map(|t| t as &wgpu::Texture)
        {
            frame_resources.full_res_depth_texture.write(t, "Renderer");
        }
        if let Some(vg_data) = self.scene.vg_frame_data() {
            frame_resources.vg.write(vg_data, "Renderer");
        }
        frame_resources.sky = self.scene.sky_context();

        if self.clear_target_next_frame {
            let clear = wgpu::Color {
                r: self.clear_color[0] as f64,
                g: self.clear_color[1] as f64,
                b: self.clear_color[2] as f64,
                a: self.clear_color[3] as f64,
            };
            let mut clear_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Renderer Resize Target Clear"),
                    });
            {
                let _pass = clear_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Renderer Resize Target Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            self.queue.submit(std::iter::once(clear_encoder.finish()));
            self.clear_target_next_frame = false;
        }

        {
            if let Ok(mut state) = self.debug_overlay_shared.lock() {
                if state.enabled {
                    state.clear();

                    let cols = (self.output_width / 14).min(280);
                    let rows = (self.output_height / 24).min(90);
                    state.set_grid_size(cols, rows);
                    let half = cols / 2;
                    let sw = self.output_width as f32;
                    let sh = self.output_height as f32;

                    let debug_data = self.graph.collect_frame_debug_data();
                    let fps = if self.delta_time > 0.0 {
                        (1.0 / self.delta_time) as u32
                    } else {
                        0
                    };
                    let frame_ms = self.delta_time * 1000.0;

                    let mut l = 0u32;
                    let other_ms = (frame_ms - self.graph_time_ms).max(0.0);
                    let (timings, total_cpu, total_gpu) = self.graph.profiler().export_timings();
                    state.write_text(
                        0,
                        l,
                        &format!(
                            "Helio  FPS: {}  Frame: {:.1} ms  Graph: {:.2} ms  Other: {:.2} ms",
                            fps, frame_ms, self.graph_time_ms, other_ms
                        ),
                    );
                    l += 1;
                    l += 1;

                    if !timings.is_empty() {
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "  CPU-prepare: {:.2} ms  GPU-compute: {:.2} ms",
                                total_cpu, total_gpu
                            ),
                        );
                        l += 1;
                        for pt in &timings {
                            if l >= rows {
                                break;
                            }
                            state.write_text(
                                0,
                                l,
                                &format!("  {:.3}c/{:.3}g ms  {}", pt.cpu_ms, pt.gpu_ms, pt.name),
                            );
                            l += 1;
                        }
                        l += 1;
                    }
                    l += 1;

                    state.write_text(
                        0,
                        l,
                        &format!(
                            "Graph VRAM: {} KB ({} MB)  Subpass chains: {}",
                            debug_data.total_vram_kb,
                            debug_data.total_vram_kb / 1024,
                            debug_data.subpass_chains.len()
                        ),
                    );
                    l += 1;
                    for ch in &debug_data.subpass_chains {
                        if l >= rows {
                            break;
                        }
                        state.write_text(0, l, &format!("  {}", ch));
                        l += 1;
                    }

                    l += 1;
                    if l < rows {
                        let cs = self.cull_stats;
                        let total = cs[0];
                        let frustum = cs[1];
                        let subpixel = cs[2];
                        let frustum_visible = cs[3];
                        let occ_raw = cs[4];
                        let occ = occ_raw.min(frustum_visible);
                        let visible = frustum_visible - occ;
                        let sh_total = cs[5];
                        let sh_visible = cs[6];
                        let sh_occ_raw = cs[7];
                        let sh_occ = sh_occ_raw.min(sh_visible);
                        let sh_frustum = sh_total.saturating_sub(sh_visible + sh_occ);
                        state.write_text(0, l, &format!("── Culling Stats ──────────────────────"));
                        l += 1;
                        state.write_text(0, l, &format!("  Total draws:     {:>6}", total));
                        l += 1;
                        let pct = |n: u32, d: u32| -> f64 {
                            if d == 0 {
                                0.0
                            } else {
                                n as f64 / d as f64 * 100.0
                            }
                        };
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "  Frustum culled:  {:>6}  {:>5.1}%",
                                frustum,
                                pct(frustum, total)
                            ),
                        );
                        l += 1;
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "  Sub-pixel culled:{:>6}  {:>5.1}%",
                                subpixel,
                                pct(subpixel, total)
                            ),
                        );
                        l += 1;
                        state.write_text(
                            0,
                            l,
                            &format!("  Occlusion culled:{:>6}  {:>5.1}%", occ, pct(occ, total)),
                        );
                        l += 1;
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "  Visible:         {:>6}  {:>5.1}%",
                                visible,
                                pct(visible, total)
                            ),
                        );
                        l += 1;
                        l += 1;
                        let sh_vis_final = sh_visible.saturating_sub(sh_occ);
                        state.write_text(0, l, &format!("  Shadow casters:  {:>6}", sh_total));
                        l += 1;
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "    Visible:       {:>6}  {:>5.1}%",
                                sh_vis_final,
                                pct(sh_vis_final, sh_total)
                            ),
                        );
                        l += 1;
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "    Frustum culled:{:>6}  {:>5.1}%",
                                sh_frustum,
                                pct(sh_frustum, sh_total)
                            ),
                        );
                        l += 1;
                        state.write_text(
                            0,
                            l,
                            &format!(
                                "    Occlusion cull:{:>6}  {:>5.1}%",
                                sh_occ,
                                pct(sh_occ, sh_total)
                            ),
                        );
                        l += 1;
                    }

                    let mut table_rows: Vec<Vec<String>> = Vec::new();
                    for res in &debug_data.resources {
                        let chain_tag = if res.chain_local {
                            format!("tile[{}→{}]", res.first_write_pass, res.last_read_pass)
                        } else {
                            String::new()
                        };
                        let wr = format!("W{}→R{}", res.first_write_pass, res.last_read_pass);
                        table_rows.push(vec![
                            res.name.clone(),
                            format!("{}x{}", res.width, res.height),
                            res.format_name.clone(),
                            format!("{}KB", res.size_kb),
                            wr,
                            chain_tag,
                            res.alias.clone(),
                        ]);
                    }
                    let mut col_widths = vec![4u32; 7];
                    for row in &table_rows {
                        for (i, val) in row.iter().enumerate() {
                            col_widths[i] = col_widths[i].max(val.chars().count() as u32);
                        }
                    }
                    let header = ["name", "size", "format", "KB", "W→R", "chain", "alias"];
                    for (i, h) in header.iter().enumerate() {
                        col_widths[i] = col_widths[i].max(h.chars().count() as u32);
                    }
                    let total_table_w: u32 =
                        col_widths.iter().sum::<u32>() + (col_widths.len() as u32 - 1);
                    let right_x = cols.saturating_sub(total_table_w);

                    let mut t = 0u32;
                    let mut x = right_x;
                    for (i, h) in header.iter().enumerate() {
                        state.write_text(x, t, h);
                        x += col_widths[i] + 1;
                    }
                    t += 1;
                    let mut sep = String::new();
                    for w in &col_widths {
                        for _ in 0..*w {
                            sep.push('-');
                        }
                        sep.push(' ');
                    }
                    state.write_text(right_x, t, &sep);
                    t += 1;

                    for row in &table_rows {
                        if t >= rows {
                            break;
                        }
                        let mut x = right_x;
                        for (i, val) in row.iter().enumerate() {
                            let w = col_widths[i] as usize;
                            let display: String = val.chars().take(w).collect();
                            state.write_text(x, t, &display);
                            x += col_widths[i] + 1;
                        }
                        t += 1;
                    }

                    t += 1;
                    if t < rows {
                        let mut pass_rows: Vec<Vec<String>> = Vec::new();
                        for pi in &debug_data.passes {
                            if pi.index == 999 {
                                continue;
                            }
                            let ws = if pi.writes.is_empty() {
                                String::new()
                            } else {
                                pi.writes.join(", ")
                            };
                            pass_rows.push(vec![
                                pi.index.to_string(),
                                pi.kind.clone(),
                                pi.chain_marker.clone(),
                                pi.name.clone(),
                                ws,
                            ]);
                        }
                        let mut pw = vec![2u32, 1, 6, 12, 0];
                        for row in &pass_rows {
                            for (i, val) in row.iter().enumerate() {
                                pw[i] = pw[i].max(val.chars().count() as u32);
                            }
                        }
                        for (i, h) in ["#", "", "chain", "pass", "writes"].iter().enumerate() {
                            pw[i] = pw[i].max(h.chars().count() as u32);
                        }
                        let pass_total: u32 = pw.iter().sum::<u32>() + (pw.len() as u32 - 1);
                        let pass_x = cols.saturating_sub(pass_total);

                        state.write_text(pass_x, t, "Pass pipeline:");
                        t += 1;
                        let mut px = pass_x;
                        for (i, h) in ["#", "", "chain", "pass", "writes"].iter().enumerate() {
                            state.write_text(px, t, h);
                            px += pw[i] + 1;
                        }
                        t += 1;

                        for row in &pass_rows {
                            if t >= rows {
                                break;
                            }
                            let mut px = pass_x;
                            for (i, val) in row.iter().enumerate() {
                                let display: String = val.chars().take(pw[i] as usize).collect();
                                state.write_text(px, t, &display);
                                px += pw[i] + 1;
                            }
                            t += 1;
                        }
                    }

                    let chart_y = sh - 150.0;
                    let graph_w = 220.0;
                    let graph_h = 110.0;
                    let graph_x = 10.0;
                    let pie_r = 80.0;
                    let pie_cx = sw - pie_r - 60.0;
                    let pie_cy = chart_y + graph_h * 0.5;

                    let num_samples = self.frame_times.len();
                    let bar_w = graph_w / num_samples as f32;
                    let max_dt = 0.05;

                    for (ms, y_frac, label) in [
                        (0.050, 0.0, "50ms"),
                        (0.033, 0.34, "33ms"),
                        (0.016, 0.66, "16ms"),
                    ] {
                        let dy = chart_y + graph_h * (1.0 - y_frac);
                        state.add_bar(graph_x, dy, graph_w, 1.0, 0.5, 0.5, 0.5, 0.5);
                        let lcol = ((graph_x + graph_w + 4.0) / 8.0) as u32;
                        let lrow = ((dy - 5.0) / 12.0) as u32;
                        if lcol < state.small_cols() && lrow < state.small_rows() {
                            state.write_small(lcol, lrow, label);
                        }
                    }

                    for (i, &ft) in self.frame_times.iter().enumerate() {
                        let bar_h = (ft / max_dt * graph_h).min(graph_h);
                        let bx = graph_x + i as f32 * bar_w;
                        let by = chart_y + graph_h - bar_h;
                        let color = if ft < 0.016 {
                            (0.3, 0.8, 0.3, 0.8)
                        } else if ft < 0.033 {
                            (0.9, 0.9, 0.2, 0.8)
                        } else {
                            (0.9, 0.3, 0.3, 0.8)
                        };
                        state.add_bar(
                            bx,
                            by,
                            bar_w.max(2.0),
                            bar_h.max(1.0),
                            color.0,
                            color.1,
                            color.2,
                            color.3,
                        );
                    }

                    if debug_data.total_vram_kb > 0 {
                        let vram_total = debug_data.total_vram_kb as f32;
                        let mut angle = 0.0f32;
                        let pie_colors = [
                            (0.3, 0.6, 1.0, 0.9),
                            (0.3, 1.0, 0.6, 0.9),
                            (1.0, 0.6, 0.3, 0.9),
                            (1.0, 0.3, 0.6, 0.9),
                            (0.6, 0.3, 1.0, 0.9),
                            (0.6, 1.0, 0.3, 0.9),
                        ];

                        for (i, res) in debug_data.resources.iter().enumerate() {
                            let frac = res.size_kb as f32 / vram_total;
                            let end = angle + frac * std::f32::consts::TAU;
                            let ci = pie_colors[i % pie_colors.len()];
                            state.add_pie_slice(pie_cx, pie_cy, pie_r, end, ci.0, ci.1, ci.2, ci.3);

                            let mid = angle + frac * std::f32::consts::PI;
                            let edge_x = pie_cx + mid.cos() * pie_r;
                            let edge_y = pie_cy + mid.sin() * pie_r;
                            let pct = (frac * 100.0) as u32;
                            let label = format!("{} {}%", res.name, pct);
                            let lw = label.chars().count() as u32;

                            let prefer_left = mid.cos() >= 0.0;
                            let min_gap = if !prefer_left {
                                (lw as f32 + 2.0) * 8.0
                            } else {
                                20.0
                            };
                            let gap = min_gap.max(20.0).min(200.0);
                            let lx = pie_cx + mid.cos() * (pie_r + gap);
                            let ly = pie_cy + mid.sin() * (pie_r + gap);
                            state.add_line(edge_x, edge_y, lx, ly, 1.0, 1.0, 1.0, 0.7);

                            let sm_cols = state.small_cols();
                            let sm_rows = state.small_rows();
                            let lrow = ((ly - 4.0) / 12.0) as u32;

                            let tip_col = lx / 8.0;
                            let left_col = tip_col + 1.0;
                            let right_col = tip_col - lw as f32 - 1.0;
                            let left_ok = left_col + lw as f32 <= sm_cols as f32;
                            let right_ok = right_col >= 0.0;

                            let lcol = if prefer_left && left_ok {
                                left_col as u32
                            } else if !prefer_left && right_ok {
                                right_col as u32
                            } else if left_ok {
                                left_col as u32
                            } else if right_ok {
                                right_col as u32
                            } else {
                                0u32
                            };
                            if lrow < sm_rows {
                                let max_w = sm_cols.saturating_sub(lcol);
                                let truncated: String =
                                    label.chars().take(max_w as usize).collect();
                                state.write_small(lcol, lrow, &truncated);
                            }
                            angle = end;
                        }
                    }
                }
            }
        }

        {
            let mut clear_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CullStats Clear"),
                    });
            clear_encoder.clear_buffer(&self.cull_stats_buffer, 0, Some(32));
            self.queue.submit(std::iter::once(clear_encoder.finish()));
        }

        let _graph_start = Instant::now();
        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;
        self.graph_time_ms = _graph_start.elapsed().as_secs_f64() as f32 * 1000.0;

        drop(texture_views);
        drop(samplers);
        self.scene.advance_frame();
        Ok(())
    }
}
