use crate::entity::Entity;

pub trait Actor {
    fn entity(&self) -> Entity;
    fn set_entity(&mut self, entity: Entity);
}
