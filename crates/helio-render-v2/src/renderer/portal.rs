//! Live portal scene layout building and delta computation.

use crate::camera::Camera;
use crate::scene::Scene;
use helio_live_portal::{
    PortalSceneLayout,
    PortalSceneLayoutDelta,
    PortalSceneObject,
    PortalSceneLight,
    PortalSceneBillboard,
    PortalSceneCamera,
};
use std::collections::HashMap;

pub(super) fn build_portal_scene_layout(scene: &Scene, camera: &Camera) -> PortalSceneLayout {
    let objects = scene.objects.iter().enumerate().map(|(id, obj)| PortalSceneObject {
        id: id as u32,
        bounds_center: obj.mesh.bounds_center,
        bounds_radius: obj.mesh.bounds_radius,
        has_material: obj.material.is_some(),
    }).collect::<Vec<_>>();

    let lights = scene.lights.iter().enumerate().map(|(id, l)| PortalSceneLight {
        id: id as u32,
        position: l.position,
        color: l.color,
        intensity: l.intensity,
        range: l.range,
    }).collect::<Vec<_>>();

    let billboards = scene.billboards.iter().enumerate().map(|(id, b)| PortalSceneBillboard {
        id: id as u32,
        position: b.position,
        scale: b.scale,
    }).collect::<Vec<_>>();

    let forward = camera.forward();
    let cam = PortalSceneCamera {
        position: [camera.position.x, camera.position.y, camera.position.z],
        forward: [forward.x, forward.y, forward.z],
    };

    PortalSceneLayout {
        objects,
        lights,
        billboards,
        camera: Some(cam),
    }
}

/// Compute delta between current and previous scene layouts
pub(super) fn compute_scene_delta(current: &PortalSceneLayout, previous: Option<&PortalSceneLayout>) -> PortalSceneLayoutDelta {
    let mut delta = PortalSceneLayoutDelta::default();

    // Build maps of current and previous objects by ID
    let curr_obj_map: HashMap<u32, _> = current.objects.iter().map(|o| (o.id, o)).collect();
    let prev_obj_map: HashMap<u32, _> = previous.map_or(HashMap::new(), |p| {
        p.objects.iter().map(|o| (o.id, o)).collect()
    });

    // Find objects to add or update, track moved ids separately
    for obj in &current.objects {
        if let Some(&prev_obj) = prev_obj_map.get(&obj.id) {
            // Object exists: check if changed
            if prev_obj != obj {
                delta.object_changes.push(obj.clone());
                if prev_obj.bounds_center != obj.bounds_center {
                    delta.moved_object_ids.push(obj.id);
                }
            }
        } else {
            // New object
            delta.object_changes.push(obj.clone());
            delta.moved_object_ids.push(obj.id);
        }
    }

    // Find removed objects
    for id in prev_obj_map.keys() {
        if !curr_obj_map.contains_key(id) {
            delta.removed_object_ids.push(*id);
        }
    }

    // Same for lights
    let curr_light_map: HashMap<u32, _> = current.lights.iter().map(|l| (l.id, l)).collect();
    let prev_light_map: HashMap<u32, _> = previous.map_or(HashMap::new(), |p| {
        p.lights.iter().map(|l| (l.id, l)).collect()
    });

    for light in &current.lights {
        if let Some(&prev_light) = prev_light_map.get(&light.id) {
            if prev_light != light {
                delta.light_changes.push(light.clone());
            }
        } else {
            delta.light_changes.push(light.clone());
        }
    }

    for id in prev_light_map.keys() {
        if !curr_light_map.contains_key(id) {
            delta.removed_light_ids.push(*id);
        }
    }

    // Same for billboards
    let curr_bb_map: HashMap<u32, _> = current.billboards.iter().map(|b| (b.id, b)).collect();
    let prev_bb_map: HashMap<u32, _> = previous.map_or(HashMap::new(), |p| {
        p.billboards.iter().map(|b| (b.id, b)).collect()
    });

    for bb in &current.billboards {
        if let Some(&prev_bb) = prev_bb_map.get(&bb.id) {
            if prev_bb != bb {
                delta.billboard_changes.push(bb.clone());
            }
        } else {
            delta.billboard_changes.push(bb.clone());
        }
    }

    for id in prev_bb_map.keys() {
        if !curr_bb_map.contains_key(id) {
            delta.removed_billboard_ids.push(*id);
        }
    }

    // Check camera
    if previous.map_or(true, |p| p.camera != current.camera) {
        delta.camera = Some(current.camera.clone());
    }

    delta
}

pub(super) fn open_url_in_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }

    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}
