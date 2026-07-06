//! Scene frame-lifecycle methods.
//!
//! This module contains methods that advance the scene through its frame lifecycle,
//! such as [`advance_frame`](Scene::advance_frame), [`set_render_size`](Scene::set_render_size),
//! and [`clear`](Scene::clear).

use glam::Mat4;
use libhelio::sky::SkyContext;

use crate::scene::Scene;
use crate::scene::SceneActorTrait;

impl Scene {
    /// Remove every object, light, mesh, material, and texture from the scene.
    ///
    /// This is the efficient path for batch renderers that swap the entire scene
    /// between frames. Objects are removed first, which cascades through the full
    /// reference-count chain:
    ///   `remove_object` → `remove_mesh` + `remove_material` → `remove_texture`
    /// so no manual ID tracking is required by the caller.
    ///
    /// Calls `flush()` before returning so GPU buffers are synchronised.
    pub fn clear(&mut self) {
        // Collect all handles before mutating — iterators are invalidated by removal.
        let object_ids: Vec<_> = self.objects.iter_with_handles().map(|(id, _)| id).collect();
        let light_ids:  Vec<_> = self.lights.iter_with_handles().map(|(id, _)| id).collect();

        // Objects first: the cascade frees meshes, materials, and textures.
        for id in object_ids {
            let _ = self.remove_object(id);
        }

        for id in light_ids {
            let _ = self.remove_light(id);
        }

        // Drop all boxed actors.  Each actor holds any state it accumulated
        // during its lifetime (e.g. the MeshActor's Option<MeshUpload> is
        // already None after on_attach, but other actor types may hold data).
        self.custom_actors.clear();

        self.flush();
    }

    /// Insert a custom trait-based scene actor.
    ///
    /// This can be e.g. `SceneActor::Sky`, `MeshActor`, `LightActor`, or other custom actors.
    pub fn insert_actor<A: SceneActorTrait + 'static>(&mut self, mut actor: A) -> crate::scene::actor::SceneActorId {
        actor.on_attach(self);
        let id = actor.inserted_id();
        self.custom_actors.push(Box::new(actor));
        id
    }

    /// Returns effective sky context for the current frame.
    pub fn sky_context(&self) -> SkyContext {
        // First preference: explicit sky actor.
        for actor in self.custom_actors.iter() {
            if let Some(sky) = actor.sky_context() {
                return sky;
            }
        }

        SkyContext::default()
    }

    /// Set the render target size for camera calculations.
    ///
    /// Updates the internal width/height used for aspect ratio calculations
    /// and viewport-dependent effects.
    ///
    /// # Parameters
    /// - `width`: Render target width in pixels
    /// - `height`: Render target height in pixels
    ///
    /// # Example
    /// ```ignore
    /// scene.set_render_size(1920, 1080);
    /// ```
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.gpu_scene.width = width;
        self.gpu_scene.height = height;
    }

    /// Advance the frame counter.
    ///
    /// Increments the internal frame counter used for temporal effects and shader logic.
    /// Call this once per frame after rendering.
    ///
    /// # Frame Counter Uses
    /// - Temporal anti-aliasing (TAA) - jitter pattern sequencing
    /// - Temporal dithering - noise pattern variation
    /// - Shader debugging - frame-dependent visualization
    ///
    /// # Example
    /// ```ignore
    /// loop {
    ///     scene.update_camera(camera);
    ///     scene.flush();
    ///     renderer.render(&scene, target)?;
    ///     scene.advance_frame();
    /// }
    /// ```
    pub fn advance_frame(&mut self) {
        // Tick custom trait-based actors.
        let scene_ptr: *mut Scene = self;
        for actor in self.custom_actors.iter_mut() {
            if actor.is_active() {
                unsafe { actor.on_tick(&mut *scene_ptr) };
            }
        }

        self.gpu_scene.frame_count = self.gpu_scene.frame_count.wrapping_add(1);
    }

    /// Build a [`SceneGeometry`](helio_bake::SceneGeometry) from all static objects and lights.
    ///
    /// Automatically extracts all objects and lights marked as Static or Stationary
    /// (i.e., not Movable) and converts them to bake-ready geometry. This eliminates
    /// the need to manually duplicate scene information for baking.
    ///
    /// # Returns
    /// A `SceneGeometry` containing:
    /// - All static object meshes with their world transforms applied
    /// - All static lights configured for baking
    ///
    /// # Example
    /// ```ignore
    /// // After building your scene normally...
    /// let bake_scene = scene.build_static_bake_scene();
    /// renderer.configure_bake(BakeRequest {
    ///     scene: bake_scene,
    ///     config: BakeConfig::fast("my_scene"),
    /// });
    /// ```
    #[cfg(feature = "bake")]
    pub fn build_static_bake_scene(&mut self) -> helio_bake::SceneGeometry {
        use helio_bake::{LightSource, LightSourceKind, SceneGeometry};
        use libhelio::{LightType, Movability};

        let mut bake_scene = SceneGeometry::new();
        let mut static_object_count = 0;
        let mut static_light_count = 0;

        // Extract all static objects
        for i in 0..self.objects.dense_len() {
            let Some(object_record) = self.objects.get_dense(i) else {
                continue;
            };

            // Skip movable objects - only bake static and stationary geometry
            if object_record.movability == Movability::Movable {
                continue;
            }

            // Extract mesh data from the pool
            let Some(mesh_upload) = self.mesh_pool.extract_mesh_data(object_record.mesh) else {
                continue;
            };

            // Convert to bake mesh with world transform applied
            // Pass mesh slot to generate deterministic UUID for lightmap region mapping
            let transform = Mat4::from_cols_array(&object_record.instance.model);
            let mesh_slot = object_record.mesh.slot();
            let bake_mesh = crate::mesh_upload_to_bake(&mesh_upload, transform, Some(mesh_slot));
            bake_scene.add_mesh(bake_mesh);
            static_object_count += 1;
        }

        // Extract all static lights
        for i in 0..self.lights.dense_len() {
            let Some(light_record) = self.lights.get_dense(i) else {
                continue;
            };

            // Include ALL lights in the bake regardless of movability.
            // Lights default to Movable even for static scenes; filtering them out
            // would result in a zero-light bake and an all-black lightmap.
            // If a user wants a light to be purely dynamic (never baked), they
            // should set bake_enabled = false on the BakeMesh's LightSource.
            let gpu_light = &light_record.gpu;
            let light_type = gpu_light.light_type;

            // Determine light kind from type
            let kind = if light_type == LightType::Directional as u32 {
                LightSourceKind::Directional {
                    direction: [
                        gpu_light.direction_outer[0],
                        gpu_light.direction_outer[1],
                        gpu_light.direction_outer[2],
                    ],
                }
            } else if light_type == LightType::Point as u32 {
                LightSourceKind::Point {
                    position: [
                        gpu_light.position_range[0],
                        gpu_light.position_range[1],
                        gpu_light.position_range[2],
                    ],
                    range: gpu_light.position_range[3],
                }
            } else if light_type == LightType::Spot as u32 {
                LightSourceKind::Spot {
                    position: [
                        gpu_light.position_range[0],
                        gpu_light.position_range[1],
                        gpu_light.position_range[2],
                    ],
                    direction: [
                        gpu_light.direction_outer[0],
                        gpu_light.direction_outer[1],
                        gpu_light.direction_outer[2],
                    ],
                    range: gpu_light.position_range[3],
                    inner_angle: gpu_light.inner_angle.acos(),
                    outer_angle: gpu_light.direction_outer[3].acos(),
                }
            } else {
                continue; // Unknown light type
            };

            bake_scene.add_light(LightSource {
                kind,
                color: [
                    gpu_light.color_intensity[0],
                    gpu_light.color_intensity[1],
                    gpu_light.color_intensity[2],
                ],
                intensity: gpu_light.color_intensity[3],
                bake_enabled: true,
                casts_shadows: gpu_light.shadow_index != u32::MAX,
            });
            static_light_count += 1;
        }

        // ── Transform lightmap UVs into atlas space ────────────────────────────
        //
        // Nebula's `build_atlas_regions` assigns each mesh an equal-area cell in
        // the atlas using a ceil(sqrt(N)) × ceil(sqrt(N)) grid.  The bake WGSL
        // shader at each texel searches ALL mesh triangles to find which triangle
        // contains that atlas-space `lm_uv`.  For correctness, vertex `lm_uv`
        // values must therefore be in ATLAS UV space, NOT in per-mesh [0,1]² UV
        // space.
        //
        // Without this transform every mesh's UV0 covers [0,1]², so for every
        // texel all N meshes' triangles match — mesh 0 always wins (listed first),
        // its lighting bleeds into every other mesh's atlas cell, and meshes 1…N-1
        // all show mesh 0's lighting at runtime.  Three-way correctness chain:
        //   bake:    `lm_uv_atlas = uv_offset + UV0 * uv_scale`  → unique range per mesh
        //   runtime: `atlas_uv   = uv_offset + UV0 * uv_scale`   → same atlas address
        //   result:  runtime UV  == bake UV                        → correct texel lookup
        let n = bake_scene.meshes.len();
        if n > 1 {
            let cols = (n as f64).sqrt().ceil() as u32;
            let rows = (n as u32).div_ceil(cols);
            let cell_w = 1.0_f32 / cols as f32;
            let cell_h = 1.0_f32 / rows as f32;
            for (i, mesh) in bake_scene.meshes.iter_mut().enumerate() {
                let col = (i as u32) % cols;
                let row = (i as u32) / cols;
                let uo = col as f32 * cell_w;
                let vo = row as f32 * cell_h;
                if let Some(uvs) = mesh.lightmap_uvs.as_mut() {
                    for uv in uvs.iter_mut() {
                        uv[0] = uo + uv[0] * cell_w;
                        uv[1] = vo + uv[1] * cell_h;
                    }
                }
            }
            log::debug!(
                "[helio-bake] Transformed lightmap UVs to atlas space: {} meshes → {}×{} grid ({:.4}×{:.4} cells)",
                n, cols, rows, cell_w, cell_h
            );
        }

        log::info!(
            "[helio-bake] Auto-extracted {} static/stationary objects and {} lights for baking",
            static_object_count,
            static_light_count
        );

        // Clear the invalidation flag - scene is now synced with bake data
        self.bake_invalidated = false;

        bake_scene
    }
}

#[cfg(test)]
mod tests {
    use crate::SceneActor;

    use super::*;
    use libhelio::{SkyActor, VolumetricClouds};

    fn create_test_device() -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        ))
        .expect("Failed to create device");

        (std::sync::Arc::new(device), std::sync::Arc::new(queue))
    }

    #[test]
    fn test_sky_actor_detection_default() {
        let (device, queue) = create_test_device();
        let scene = Scene::new(device, queue);

        let sky_ctx = scene.sky_context();
        assert!(!sky_ctx.has_sky, "Default scene should have no sky");
        assert!(sky_ctx.clouds.is_none(), "Default scene should have no clouds");
    }

    #[test]
    fn test_sky_actor_detection_with_clouds() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // Insert sky actor with clouds
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new()
                .with_sky_color([0.5, 0.7, 1.0])
                .with_clouds(VolumetricClouds {
                    coverage: 0.6,
                    density: 0.8,
                    ..Default::default()
                })
        ));

        let sky_ctx = scene.sky_context();
        assert!(sky_ctx.has_sky, "Sky actor should be detected");
        assert!(sky_ctx.clouds.is_some(), "Cloud settings should be detected");

        if let Some(clouds) = sky_ctx.clouds {
            assert!((clouds.coverage - 0.6).abs() < 0.01, "Coverage should match");
            assert!((clouds.density - 0.8).abs() < 0.01, "Density should match");
        }
    }

    #[test]
    fn test_multiple_sky_actors_first_wins() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // Insert first sky actor
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new().with_sky_color([1.0, 0.0, 0.0])
        ));

        // Insert second sky actor (should be ignored)
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new().with_sky_color([0.0, 1.0, 0.0])
        ));

        let sky_ctx = scene.sky_context();
        // First actor wins
        assert!((sky_ctx.sky_color[0] - 1.0).abs() < 0.01, "Should use first actor's color");
    }
}
