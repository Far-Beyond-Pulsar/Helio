#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use arrayvec::ArrayVec;
use helio_core::Result as HelioResult;

use crate::groups::GroupId;
use crate::scene::Camera;

use super::renderer_impl::{
    CullStatsReadbackState, DebugCameraUniform, Renderer, HALTON_JITTER,
};

impl Renderer {
    fn poll_cull_stats_readback(&mut self) {
        if !self.owns_device {
            return;
        }

        let _ = self.device.poll(wgpu::PollType::Poll);
        let result = match &self.cull_stats_readback_state {
            CullStatsReadbackState::Idle | CullStatsReadbackState::Disabled => return,
            CullStatsReadbackState::Mapping(completion) => completion
                .lock()
                .ok()
                .and_then(|mut completion| completion.take()),
        };

        match result {
            Some(Ok(())) => {
                let read_succeeded = match self.cull_stats_staging.slice(..).get_mapped_range() {
                    Ok(mapped) => {
                        if mapped.len() >= 32 {
                            let ptr = mapped.as_ptr() as *const u32;
                            self.cull_stats = unsafe { std::ptr::read_unaligned(ptr.cast()) };
                        }
                        drop(mapped);
                        true
                    }
                    Err(_) => false,
                };
                self.cull_stats_staging.unmap();
                self.cull_stats_readback_state = if read_succeeded {
                    CullStatsReadbackState::Idle
                } else {
                    CullStatsReadbackState::Disabled
                };
            }
            Some(Err(_)) => {
                self.cull_stats_staging.unmap();
                self.cull_stats_readback_state = CullStatsReadbackState::Disabled;
            }
            None => {}
        }
    }

    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView) -> HelioResult<()> {
        // Browser WebGPU buffer mapping is asynchronous. Consume the previous
        // frame's completed readback before recording a new copy.
        self.poll_cull_stats_readback();

        if let Some((w, h)) = self.pending_resize.take() {
            self.apply_resize_now(w, h);
        }

        #[cfg(feature = "bake")]
        if let Some(request) = self.bake_pending.take() {
            let obj_count = request.scene.meshes.len();
            let light_count = request.scene.lights.len();

            log::info!(
                "[helio-bake] Starting pre-frame-1 bake for scene '{}' (cache: {})…",
                request.config.scene_name,
                request.config.cache_dir.display(),
            );

            let bake_start = Instant::now();
            let baked = helio_bake::run_bake_blocking(
                &self.device,
                &self.queue,
                &request.scene,
                &request.config,
            )
            .map_err(|e| helio_core::Error::InvalidPassConfig(e.to_string()))?;
            let bake_duration = bake_start.elapsed();

            let baked = std::sync::Arc::new(baked);

            log::info!(
                "[helio-bake] ✓ Bake complete in {:.2}s — {} objects, {} lights (avg {:.1}ms/obj)",
                bake_duration.as_secs_f32(),
                obj_count,
                light_count,
                if obj_count > 0 { bake_duration.as_millis() as f32 / obj_count as f32 } else { 0.0 }
            );

            self.baked_data = Some(baked.clone());

            self.scene.update_lightmap_indices(baked.lightmap_atlas_regions());
        }

        #[cfg(feature = "bake")]
        if self.baked_data.is_some() && self.scene.is_bake_invalidated() {
            log::warn!(
                "[helio-bake] ⚠️  Static geometry or lights have been added since the last bake!\n\
                 The baked lighting is now out of date. Call renderer.auto_bake() again to rebake the scene."
            );
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32().min(0.1);
        self.last_render_time = now;
        self.delta_time = dt;
        self.frame_times[self.frame_times_cursor] = dt;
        self.frame_times_cursor = (self.frame_times_cursor + 1) % self.frame_times.len();
        self.graph.set_delta_time(dt);

        let internal_w = (((self.output_width as f32) * self.render_scale).ceil() as u32).max(1);
        let internal_h = (((self.output_height as f32) * self.render_scale).ceil() as u32).max(1);
        if internal_w != self.jitter_cache_width || internal_h != self.jitter_cache_height {
            self.jitter_matrices = Self::compute_jitter_matrices(internal_w, internal_h);
            self.jitter_cache_width = internal_w;
            self.jitter_cache_height = internal_h;
        }

        let frame_idx = self.scene.gpu_scene().frame_count;
        let (jitter_mat, jx, jy) = if self.enable_jitter {
            let jitter_mat = self.jitter_matrices[(frame_idx % 16) as usize];
            let raw = HALTON_JITTER[(frame_idx % 16) as usize];
            let jx = ((raw[0] - 0.5) * 2.0) / (internal_w as f32);
            let jy = ((raw[1] - 0.5) * 2.0) / (internal_h as f32);
            (jitter_mat, jx, jy)
        } else {
            (glam::Mat4::IDENTITY, 0.0, 0.0)
        };
        let jittered_m = jitter_mat * camera.proj * camera.view;
        let col = jittered_m.to_cols_array();
        let debug_camera_uniform = DebugCameraUniform {
            view_proj: [
                [col[0],  col[1],  col[2],  col[3]],
                [col[4],  col[5],  col[6],  col[7]],
                [col[8],  col[9],  col[10], col[11]],
                [col[12], col[13], col[14], col[15]],
            ],
        };
        self.queue.write_buffer(
            &self.debug_camera_buffer,
            0,
            bytemuck::bytes_of(&debug_camera_uniform),
        );

        let mut jittered_camera = camera.clone();
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
            self.billboard_scratch.extend_from_slice(&self.billboard_instances);
            if !editor_hidden {
                for light in self.scene.gpu_scene().lights.as_slice() {
                    if light.light_type == libhelio::LightType::Point as u32
                        || light.light_type == libhelio::LightType::Spot as u32
                    {
                        let [x, y, z, _] = light.position_range;
                        let [r, g, b, _] = light.color_intensity;
                        self.billboard_scratch.push(super::renderer_impl::BillboardInstance {
                            world_pos: [x, y, z, 0.0],
                            scale_flags: [0.25, 0.25, 0.0, 0.0],
                            color: [r, g, b, 1.0],
                        });
                    }
                }
                for emitter in &self.corona_emitters {
                    let [x, y, z, _] = emitter.transform[3];
                    self.billboard_scratch.push(super::renderer_impl::BillboardInstance {
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

        let pp_count = self.scene.post_process_volumes_count();
        if pp_count > 0 && self.scene.post_process_volumes_dirty() {
            let range = self.scene.consume_post_process_volumes_dirty_range();
            if let Some((start, end)) = range {
                let volumes = self.scene.get_post_process_volumes_gpu_slice();
                self.queue.write_buffer(
                    &self.pp_volumes_buffer,
                    (start * std::mem::size_of::<libhelio::GpuPostProcessVolume>()) as u64,
                    bytemuck::cast_slice(&volumes[start..end]),
                );
            }
            self.scene.clear_post_process_volumes_dirty();
        }

        {
            // Upload camera defaults as base; GPU volume blending (in PostProcessPass)
            // will blend toward active volumes if any are present.
            let pp = camera.postprocess_settings.to_gpu();
            self.queue.write_buffer(&self.postprocess_buffer, 0, bytemuck::bytes_of(&pp));

            // Gate bloom: conservative when volumes exist since a volume may enable it.
            let bloom_visible = if pp_count > 0 {
                true
            } else {
                pp.bloom_intensity > 0.001 && pp.bloom_enabled != 0
            };
            if let Some(pp_pass) = self.graph.find_pass_mut::<helio_pass_postprocess::PostProcessPass>() {
                pp_pass.set_bloom_active(bloom_visible);
            }
        }

        let mut texture_views = ArrayVec::<&wgpu::TextureView, { crate::material::MAX_TEXTURES }>::new();
        let mut samplers = ArrayVec::<&wgpu::Sampler, { crate::material::MAX_TEXTURES }>::new();
        for slot in 0..crate::material::MAX_TEXTURES {
            texture_views.push(self.scene.texture_view_for_slot(slot));
            samplers.push(self.scene.texture_sampler_for_slot(slot));
        }

        let mesh_buffers = self.scene.mesh_buffers();
        let dynamic_mesh_buffers = self.scene.dynamic_mesh_buffers();
        if let Ok(mut state) = self.debug_state.lock() {
            state.camera_position = camera.position;
            // Volume bounds track whatever the scene currently holds. The
            // generation only moves when the geometry actually differs, so a
            // static scene keeps the pass's cached upload instead of re-sending
            // every frame while the camera moves.
            if state.editor_enabled {
                let lines = self.scene.editor_volume_debug_lines();
                if lines != state.editor_volume_lines {
                    state.editor_volume_lines = lines;
                    state.editor_volume_generation =
                        state.editor_volume_generation.wrapping_add(1);
                }
            } else if !state.editor_volume_lines.is_empty() {
                state.editor_volume_lines = Vec::new();
                state.editor_volume_generation = state.editor_volume_generation.wrapping_add(1);
            }
        }
        let rc_radius = self.gi_config.rc_radius;
        let rc_min = [camera.position.x - rc_radius, camera.position.y - rc_radius, camera.position.z - rc_radius];
        let rc_max = [camera.position.x + rc_radius, camera.position.y + rc_radius, camera.position.z + rc_radius];

        #[cfg(feature = "bake")]
        let baked_ao = self.baked_data.as_deref().and_then(|d| d.ao_view_ref());
        #[cfg(not(feature = "bake"))]
        let baked_ao = None;
        #[cfg(feature = "bake")]
        let baked_ao_sampler = self.baked_data.as_deref().and_then(|d| d.ao_sampler_ref());
        #[cfg(not(feature = "bake"))]
        let baked_ao_sampler = None;
        #[cfg(feature = "bake")]
        let baked_lightmap = self.baked_data.as_deref().and_then(|d| d.lightmap_view_ref());
        #[cfg(not(feature = "bake"))]
        let baked_lightmap = None;
        #[cfg(feature = "bake")]
        let baked_lightmap_sampler = self.baked_data.as_deref().and_then(|d| d.lightmap_sampler_ref());
        #[cfg(not(feature = "bake"))]
        let baked_lightmap_sampler = None;
        #[cfg(feature = "bake")]
        let baked_reflection = self.baked_data.as_deref().and_then(|d| d.reflection_view_ref());
        #[cfg(not(feature = "bake"))]
        let baked_reflection = None;
        #[cfg(feature = "bake")]
        let baked_reflection_sampler = self.baked_data.as_deref().and_then(|d| d.reflection_sampler_ref());
        #[cfg(not(feature = "bake"))]
        let baked_reflection_sampler = None;
        #[cfg(feature = "bake")]
        let baked_irradiance_sh = self.baked_data.as_deref().and_then(|d| d.irradiance_sh_buf_ref());
        #[cfg(not(feature = "bake"))]
        let baked_irradiance_sh = None;
        #[cfg(feature = "bake")]
        let baked_pvs = self.baked_data.as_deref().and_then(|d| d.pvs_ref());
        #[cfg(not(feature = "bake"))]
        let baked_pvs = None;

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
                rc_world_min: rc_min,
                rc_world_max: rc_max,
                tlas: self.scene.tlas(),
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
            frame_resources.water_volumes.write(&self.water_volumes_buffer, "Renderer");
        }
        frame_resources.water_volume_count = water_volume_count;
        if water_hitbox_count > 0 {
            frame_resources.water_hitboxes.write(&self.water_hitboxes_buffer, "Renderer");
        }
        frame_resources.water_hitbox_count = water_hitbox_count;
        frame_resources.pp_volumes.write(&self.pp_volumes_buffer, "Renderer");
        frame_resources.pp_volume_count = pp_count;
        frame_resources.postprocess_uniforms.write(&self.postprocess_buffer, "Renderer");
        frame_resources.depth_texture.write(&self.depth_texture, "Renderer");
        if let Some(v) = self.full_res_depth_view.as_ref().map(|v| v as &wgpu::TextureView) {
            frame_resources.full_res_depth.write(v, "Renderer");
        }
        if let Some(t) = self.full_res_depth_texture.as_ref().map(|t| t as &wgpu::Texture) {
            frame_resources.full_res_depth_texture.write(t, "Renderer");
        }
        if let Some(vg_data) = self.scene.vg_frame_data() {
            frame_resources.vg.write(vg_data, "Renderer");
        }
        frame_resources.sky = self.scene.sky_context();
        if let Some(ao) = baked_ao {
            frame_resources.baked_ao.write(ao, "Renderer");
        }
        if let Some(ao_sampler) = baked_ao_sampler {
            frame_resources.baked_ao_sampler.write(ao_sampler, "Renderer");
        }
        if let Some(lightmap) = baked_lightmap {
            frame_resources.baked_lightmap.write(lightmap, "Renderer");
        }
        if let Some(lightmap_sampler) = baked_lightmap_sampler {
            frame_resources.baked_lightmap_sampler.write(lightmap_sampler, "Renderer");
        }
        if let Some(reflection) = baked_reflection {
            frame_resources.baked_reflection.write(reflection, "Renderer");
        }
        if let Some(reflection_sampler) = baked_reflection_sampler {
            frame_resources.baked_reflection_sampler.write(reflection_sampler, "Renderer");
        }
        if let Some(irradiance_sh) = baked_irradiance_sh {
            frame_resources.baked_irradiance_sh.write(irradiance_sh, "Renderer");
        }
        if let Some(pvs) = baked_pvs {
            frame_resources.baked_pvs.write(pvs, "Renderer");
        }

        if self.clear_target_next_frame {
            let clear = wgpu::Color {
                r: self.clear_color[0] as f64,
                g: self.clear_color[1] as f64,
                b: self.clear_color[2] as f64,
                a: self.clear_color[3] as f64,
            };
            let mut clear_encoder = self
                .device
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
            let mut clear_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        if self.owns_device
            && matches!(
                self.cull_stats_readback_state,
                CullStatsReadbackState::Idle
            )
        {
            let mut read_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CullStats Readback"),
            });
            read_encoder.copy_buffer_to_buffer(
                &self.cull_stats_buffer, 0,
                &self.cull_stats_staging, 0,
                32,
            );
            self.queue.submit(std::iter::once(read_encoder.finish()));

            let completion = std::sync::Arc::new(std::sync::Mutex::new(None));
            let callback_completion = std::sync::Arc::clone(&completion);
            self.cull_stats_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if let Ok(mut completion) = callback_completion.lock() {
                        *completion = Some(result);
                    }
                });
            self.cull_stats_readback_state = CullStatsReadbackState::Mapping(completion);
        }

        drop(texture_views);
        drop(samplers);
        self.scene.advance_frame();
        Ok(())
    }
}
