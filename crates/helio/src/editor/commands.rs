use super::EditorState;
use crate::renderer::Renderer;
use crate::scene::{Scene, SceneActorId};

impl EditorState {
    /// Delete the selected object from the scene and clear the selection.
    ///
    /// Returns `true` if an object was deleted. Rebuild `ScenePicker` afterwards
    /// so the deleted object can no longer be picked.
    pub fn delete_selected(&mut self, scene: &mut Scene) -> bool {
        self.clear_interaction_state();
        match self.take_selected() {
            Some(SceneActorId::Object(id)) => scene.remove_object(id).is_ok(),
            Some(SceneActorId::SectionedObject(id)) => scene.remove_sectioned_object(id).is_ok(),
            _ => false,
        }
    }

    /// Duplicate the selected object at the same transform, select the new copy,
    /// and return its [`ObjectId`].
    ///
    /// Pass a mutable reference to the renderer so the new object can be inserted.
    /// Rebuild `ScenePicker` afterwards so the copy is immediately pickable.
    pub fn duplicate_selected(
        &mut self,
        renderer: &mut Renderer,
    ) -> Option<SceneActorId> {
        let prev_selected = self.selected()?;
        match prev_selected {
            SceneActorId::Object(id) => {
                let desc = renderer.scene().get_object_descriptor(id).ok()?;
                let new_actor = renderer.scene_mut().insert_actor(
                    crate::scene::SceneActor::object(desc)
                );
                let new_id = new_actor.as_object()?;
                self.replace_selected(Some(SceneActorId::Object(new_id)));
                self.clear_interaction_state();
                Some(SceneActorId::Object(new_id))
            }
            SceneActorId::SectionedObject(id) => {
                let new_id = renderer.scene_mut().duplicate_sectioned_object(id).ok()?;
                self.replace_selected(Some(SceneActorId::SectionedObject(new_id)));
                self.clear_interaction_state();
                Some(SceneActorId::SectionedObject(new_id))
            }
            _ => None,
        }
    }
}
