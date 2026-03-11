//! Scene CPU preparation: shadow matrix updates, sky uniform upload, ambient sync.
//!
//! `flush_scene_state` replaces the old `prepare_env(SceneEnv)` snapshot path.
//! It runs once per frame and does work **proportional to what actually changed**:
//!
//! | Condition                         | Work done                        |
//! |-----------------------------------|----------------------------------|
//! | Nothing changed, camera static    | O(N) shadow-mat hash check only  |
//! | Camera moved (directional lights) | CSM matrix recompute + hash check|
//! | Light moved / added / removed     | Shadow matrices + GPU upload     |
//! | Sky atmosphere changed            | SkyUniform upload                |
//! | Ambient / skylight changed        | RC ambient color update          |
//!
//! At steady state (no lights moving, camera static, static sky) the GPU
//! upload cost is **zero** — every delta path is gated on a dirty flag.

use super::*;
use super::uniforms::SkyUniform;
use super::shadow_math::CSM_SPLITS;
use super::helpers::estimate_sky_ambient;
use glam::Vec3;

impl Renderer {
    // ─────────────────────────────────────────────────────────────────────────
    // PRIMARY PATH — called from render() every frame
    // ─────────────────────────────────────────────────────────────────────────

    /// Process all pending scene-state changes and flush dirty data to the GPU.
    ///
    /// At steady state this is near-zero CPU work:
    /// * Shadow matrices: O(N) FNV hash checks, GPU write only on hash mismatch.
    /// * Light buffer: single coalesced `write_buffer` over dirty range, no-op
    ///   when range is empty.
    /// * Sky uniform: skipped when `sky_lut_dirty` is false.
    /// * Ambient: skipped when `ambient_dirty` is false.
    pub(crate) fn flush_scene_state(&mut self, camera: &Camera) -> bool {
        // ── Shadow matrices + GPU light flush ──────────────────────────────
        let camera_moved = {
            crate::profile_scope!("Scene::Lights");
            let moved = self.gpu_light_scene.update_shadow_matrices(camera);
            self.gpu_light_scene.flush(&self.queue);
            moved
        };

        let count = self.gpu_light_scene.active_count;
        self.scene_light_count = count;
        self.light_count_arc.store(count, Ordering::Relaxed);

        // Update shadow cull data every frame (contains position/direction that changes when lights move)
        {
            let mut cull = self.shadow_cull_lights.lock().unwrap();
            *cull = self.gpu_light_scene.shadow_cull_lights.clone();
        }

        // Only update face counts when structure changes (zero-cost at steady state)
        if self.gpu_light_scene.structure_changed {
            let mut fc = self.light_face_counts.lock().unwrap();
            fc.clear();
            fc.extend_from_slice(&self.gpu_light_scene.face_counts);
            self.gpu_light_scene.structure_changed = false;
        }

        // ── Sky / ambient state ────────────────────────────────────────────
        {
            crate::profile_scope!("Scene::Sky");

            let ss = &mut self.scene_state;

            self.sky_state_changed = ss.sky_lut_dirty;
            self.scene_has_sky     = ss.sky_atmosphere.is_some();
            self.scene_sky_color   = ss.sky_color;
            self.scene_csm_splits  = CSM_SPLITS;

            let mut amb_color     = ss.ambient_color;
            let mut amb_intensity = ss.ambient_intensity;

            if let Some(ref atm) = ss.sky_atmosphere {
                let sun_dir = self.gpu_light_scene.cached_scene_lights.iter()
                    .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                    .map(|l| {
                        let d = Vec3::from(l.direction).normalize();
                        [-d.x, -d.y, -d.z]
                    })
                    .unwrap_or([0.0, 1.0, 0.0]);

                let sun_moved =
                    (sun_dir[0] - ss.cached_sun_direction[0]).abs() > 0.01
                    || (sun_dir[1] - ss.cached_sun_direction[1]).abs() > 0.01
                    || (sun_dir[2] - ss.cached_sun_direction[2]).abs() > 0.01;
                if sun_moved
                    || (atm.sun_intensity - ss.cached_sun_intensity).abs() > 0.01
                {
                    ss.sky_lut_dirty = true;
                    self.sky_state_changed = true;
                    ss.cached_sun_direction = sun_dir;
                    ss.cached_sun_intensity = atm.sun_intensity;
                }

                if ss.sky_lut_dirty {
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
                        skylight_intensity: ss.skylight.as_ref().map(|sl| sl.intensity).unwrap_or(1.0),
                        _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
                    };
                    self.queue.write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&sky_uni));
                }

                if let Some(ref skylight) = ss.skylight {
                    let sun_elev = sun_dir[1].clamp(-1.0, 1.0);
                    let sky_amb  = estimate_sky_ambient(sun_elev, &atm.rayleigh_scatter);
                    let tint     = skylight.color_tint;
                    amb_color = [
                        ss.ambient_color[0] + sky_amb[0] * tint[0],
                        ss.ambient_color[1] + sky_amb[1] * tint[1],
                        ss.ambient_color[2] + sky_amb[2] * tint[2],
                    ];
                    amb_intensity = ss.ambient_intensity.max(skylight.intensity);
                }
            } else if ss.sky_lut_dirty {
                self.sky_state_changed = true;
            }

            ss.sky_lut_dirty = false;

            self.scene_ambient_color     = amb_color;
            self.scene_ambient_intensity = amb_intensity;

            if ss.ambient_dirty {
                if let Some(rc) = self.features.get_typed_mut::<RadianceCascadesFeature>("radiance_cascades") {
                    let c = amb_color;
                    let i = amb_intensity;
                    rc.set_sky_color([c[0]*i, c[1]*i, c[2]*i]);
                }
                ss.ambient_dirty = false;
            }
        } // profile_scope!("Scene::Sky") drops here

        camera_moved
    }
}
