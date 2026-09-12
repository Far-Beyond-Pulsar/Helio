use super::EditorState;
use crate::renderer::Renderer;
use crate::scene::{Scene, SceneEntityId};

impl EditorState {
    /// Delete the selected object from the scene and clear the selection.
    ///
    /// Returns `true` if an object was deleted. Rebuild `ScenePicker` afterwards
    /// so the deleted object can no longer be picked.
    pub fn delete_selected(&mut self, scene: &mut Scene) -> bool {
        self.clear_interaction_state();
        match self.take_selected() {
            Some(SceneEntityId::Object(id)) => scene.remove_object(id).is_ok(),
            Some(SceneEntityId::SectionedObject(id)) => scene.remove_sectioned_object(id).is_ok(),
            _ => false,
        }
    }

    /// Duplicate the selected object at the same transform, select the new copy,
    /// and return its [`ObjectId`].
    ///
    /// Pass a mutable reference to the renderer so the new object can be inserted.
    /// Rebuild `ScenePicker` afterwards so the copy is immediately pickable.
    pub fn duplicate_selected(&mut self, renderer: &mut Renderer) -> Option<SceneEntityId> {
        let prev_selected = self.selected()?;
        match prev_selected {
            SceneEntityId::Object(id) => {
                let desc = renderer.scene().get_object_descriptor(id).ok()?;
                let new_actor = renderer
                    .transient_scene_mut()
                    .insert_entity(crate::scene::SceneEntity::object(desc));
                let new_id = new_actor.as_object()?;
                self.replace_selected(Some(SceneEntityId::Object(new_id)));
                self.clear_interaction_state();
                Some(SceneEntityId::Object(new_id))
            }
            SceneEntityId::SectionedObject(id) => {
                let new_id = renderer
                    .transient_scene_mut()
                    .duplicate_sectioned_object(id)
                    .ok()?;
                self.replace_selected(Some(SceneEntityId::SectionedObject(new_id)));
                self.clear_interaction_state();
                Some(SceneEntityId::SectionedObject(new_id))
            }
            _ => None,
        }
    }
}
