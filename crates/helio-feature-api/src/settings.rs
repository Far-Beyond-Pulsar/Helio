//! Per-feature settings: each feature defines its own settings type, and a
//! renderer config carries them without knowing any of them.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Settings of the features a renderer runs, one value per type.
///
/// A feature defines `FooSettings: Clone + Default` and reads it with
/// [`Self::get_or_default`] (or, in a node build,
/// `BuildEnv::setting::<FooSettings>()`); a host edits it with
/// [`Self::edit`]. Cheap to clone: values are shared until edited.
#[derive(Clone, Default)]
pub struct FeatureSettings(HashMap<TypeId, Arc<dyn Any + Send + Sync>>);

impl FeatureSettings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the value of type `T`.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) {
        self.0.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// [`Self::insert`], by value.
    pub fn with<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.insert(value);
        self
    }

    pub fn get<T: Any>(&self) -> Option<&T> {
        self.0
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// The value of type `T`, or its default when none was set.
    pub fn get_or_default<T: Any + Clone + Default>(&self) -> T {
        self.get::<T>().cloned().unwrap_or_default()
    }

    /// The value of type `T` to edit in place, starting from its default.
    pub fn edit<T: Any + Send + Sync + Clone + Default>(&mut self) -> &mut T {
        let entry = self
            .0
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Arc::new(T::default()));
        if Arc::get_mut(entry).is_none() {
            let value: T = entry
                .downcast_ref::<T>()
                .expect("settings are stored under their own type id")
                .clone();
            *entry = Arc::new(value);
        }
        Arc::get_mut(entry)
            .and_then(|value| value.downcast_mut::<T>())
            .expect("a freshly made Arc is unique")
    }

    pub(crate) fn entries(
        &self,
    ) -> impl Iterator<Item = (TypeId, Arc<dyn Any + Send + Sync>)> + '_ {
        self.0.iter().map(|(id, value)| (*id, Arc::clone(value)))
    }
}

impl std::fmt::Debug for FeatureSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FeatureSettings({} values)", self.0.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct Foo {
        n: u32,
    }

    #[test]
    fn edit_starts_from_the_default_and_copies_on_write() {
        let mut a = FeatureSettings::new();
        assert_eq!(a.get_or_default::<Foo>(), Foo { n: 0 });
        a.edit::<Foo>().n = 3;
        let b = a.clone();
        a.edit::<Foo>().n = 4;
        assert_eq!(a.get::<Foo>(), Some(&Foo { n: 4 }));
        assert_eq!(b.get::<Foo>(), Some(&Foo { n: 3 }));
    }
}
