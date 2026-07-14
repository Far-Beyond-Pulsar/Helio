use libhelio::sky::SkyContext;

use crate::scene::{Scene, SceneActorTrait};

impl Scene {
    pub fn clear(&mut self) {
        let object_ids: Vec<_> = self.objects.iter_with_handles().map(|(id, _)| id).collect();
        let light_ids: Vec<_> = self.lights.iter_with_handles().map(|(id, _)| id).collect();

        for id in object_ids {
            let _ = self.remove_object(id);
        }
        for id in light_ids {
            let _ = self.remove_light(id);
        }

        self.custom_actors.clear();
        self.flush();
    }

    pub fn insert_actor<A: SceneActorTrait + 'static>(
        &mut self,
        mut actor: A,
    ) -> crate::scene::actor::SceneActorId {
        actor.on_attach(self);
        let id = actor.inserted_id();
        self.custom_actors.push(Box::new(actor));
        id
    }

    pub fn sky_context(&self) -> SkyContext {
        for actor in &self.custom_actors {
            if let Some(sky) = actor.sky_context() {
                return sky;
            }
        }
        SkyContext::default()
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.gpu_scene.width = width;
        self.gpu_scene.height = height;
    }

    pub fn advance_frame(&mut self) {
        let scene_ptr: *mut Scene = self;
        for actor in &mut self.custom_actors {
            if actor.is_active() {
                unsafe { actor.on_tick(&mut *scene_ptr) };
            }
        }
        self.gpu_scene.frame_count = self.gpu_scene.frame_count.wrapping_add(1);
    }
}
