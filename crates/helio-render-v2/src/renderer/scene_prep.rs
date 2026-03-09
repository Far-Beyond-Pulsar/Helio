//! Scene CPU preparation: light sync, shadow matrix updates, sky uniforms.
//!
//! Geometry lives permanently in `GpuScene` (delta uploads, zero cost when
//! static).  Lights and shadow matrices now live in `GpuLightScene` — the
//! same persistent delta-upload pattern.  `prepare_env` drives those systems
//! and handles sky/ambient/billboard uniforms.

use super::*;
use super::uniforms::SkyUniform;
use super::shadow_math::CSM_SPLITS;
use super::helpers::estimate_sky_ambient;
use glam::Vec3;

impl Renderer {
    // ─────────────────────────────────────────────────────────────────────────
    // PRIMARY PATH — called from render() via pending_env
    // ─────────────────────────────────────────────────────────────────────────

    /// Process a [`SceneEnv`]: sync lights to the GPU, update shadow matrices,
    /// upload billboards, and write sky uniforms.
    ///
    /// Light and shadow data is uploaded in **delta mode** — only dirty slots
    /// (lights whose data actually changed) incur GPU writes.  A scene with no
    /// moving lights produces zero uploads for lights and shadows.
    pub(crate) fn prepare_env(&mut self, env: SceneEnv, camera: &Camera) {
        let t_start = std::time::Instant::now();

        // ── Lights + shadow matrices (GPU-resident, delta upload) ─────────────
        self.gpu_light_scene.sync_lights(&env.lights);
        self.gpu_light_scene.update_shadow_matrices(camera);
        self.gpu_light_scene.flush(&self.queue);

        // Forward per-frame state to ShadowPass via Arc<Mutex<>> channels.
        let count = self.gpu_light_scene.active_count;
        self.scene_light_count = count;
        self.light_count_arc.store(count, Ordering::Relaxed);
        {
            let mut fc = self.light_face_counts.lock().unwrap();
            fc.clear();
            fc.extend_from_slice(&self.gpu_light_scene.face_counts);
        }
        {
            let mut cull = self.shadow_cull_lights.lock().unwrap();
            *cull = self.gpu_light_scene.shadow_cull_lights.clone();
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
        log::trace!("prepare_env: {:.2}ms (lights={}, sky={})", ms, count, self.scene_has_sky);
    }
}
