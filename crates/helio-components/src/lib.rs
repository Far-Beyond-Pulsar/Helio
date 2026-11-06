//! What Helio's standalone hosts (examples, tests) and the plugins' host
//! rows share: tying a row component to its contract row ([`same_layout!`],
//! [`RowComponent`], [`WorldRows`]) and the lightmap bake. Helio has no kind,
//! so it has no components: each plugin brings its own host components (the
//! meshes' `helio-mesh-host`, the lights' and the materials' api
//! `host_rows`).
//!
//! Helio defines no components (docs/architecture/layered-rewrite.md,
//! phase 1): its passes read buffers under [`helio_core::BufferContract`]s,
//! and whoever produces the data registers the components that fill them.
//! Pulsar registers its own; this crate is the same thing for Helio's own
//! hosts. Nothing below the host layer may depend on it.
//!
//! Row components carry their contract row's exact fields, so
//! struct literals read like the row; `same_layout!` fails the build if they
//! drift apart. Insert a bare row with [`WorldRows::insert_row`]: SceneDB
//! accepts any `Send + Sync` value as a component, so `world.insert(e, row)`
//! compiles but is never mirrored to the GPU.

use pulsar_scenedb::{Entity, World};

/// A contract row with the component that stores it.
pub trait RowComponent: Sized {
    type Component: From<Self> + Send + Sync + 'static;
}

/// Insert contract rows as their components.
pub trait WorldRows {
    /// Insert `row` as its [`RowComponent::Component`], so SceneDB mirrors it
    /// into the contract's buffer.
    fn insert_row<R: RowComponent>(&mut self, entity: Entity, row: R);
}

impl WorldRows for World {
    fn insert_row<R: RowComponent>(&mut self, entity: Entity, row: R) {
        self.insert(entity, R::Component::from(row));
    }
}

/// Ties a component to its contract row: same size, every field at the same
/// offset, stored under the contract's key. Adds the conversions. (Removal
/// needs nothing: `#[derive(SceneStore)]` already zeroes a component's GPU
/// row when it leaves an entity.)
#[macro_export]
macro_rules! same_layout {
    ($component:ident, $row:path, $contract:path, $key:literal; $($field:ident),* $(,)?) => {
        const _: () = {
            assert!(
                ::core::mem::size_of::<$component>() == ::core::mem::size_of::<$row>(),
                concat!(stringify!($component), " and its row differ in size")
            );
            assert!(
                ::core::mem::align_of::<$component>() == ::core::mem::align_of::<$row>(),
                concat!(stringify!($component), " and its row differ in alignment")
            );
            $(
                assert!(
                    ::core::mem::offset_of!($component, $field)
                        == ::core::mem::offset_of!($row, $field),
                    concat!(stringify!($component), ".", stringify!($field), " moved")
                );
            )*
            $contract.assert_producer::<$component>($key);
        };

        impl ::core::convert::From<$row> for $component {
            fn from(row: $row) -> Self {
                ::bytemuck::cast(row)
            }
        }

        impl ::core::convert::From<$component> for $row {
            fn from(component: $component) -> Self {
                ::bytemuck::cast(component)
            }
        }

        impl $component {
            /// This component as its contract row.
            pub fn row(&self) -> $row {
                ::bytemuck::cast(*self)
            }
        }

        // Same fields, same types, same `repr(C)`: same size *and* alignment,
        // so the component reads as its row in place.
        impl ::core::ops::Deref for $component {
            type Target = $row;
            fn deref(&self) -> &$row {
                ::bytemuck::cast_ref(self)
            }
        }

        impl ::core::ops::DerefMut for $component {
            fn deref_mut(&mut self) -> &mut $row {
                ::bytemuck::cast_mut(self)
            }
        }

        impl $crate::RowComponent for $row {
            type Component = $component;
        }
    };
}

/// Bake `scene` now (blocking) and keep the result rendered: the bake's
/// inject pass is a persistent pass, so it survives graph rebuilds.
#[cfg(feature = "bake")]
pub fn bake(
    renderer: &mut helio::Renderer,
    scene: helio_bake::SceneGeometry,
    config: helio_bake::BakeConfig,
) -> Result<(), helio_bake::BakeError> {
    let started = std::time::Instant::now();
    let (objects, lights) = (scene.meshes.len(), scene.lights.len());
    let baked = std::sync::Arc::new(helio_bake::run_bake_blocking(
        renderer.device(),
        renderer.queue(),
        &scene,
        &config,
    )?);
    log::info!(
        "[helio-bake] bake of '{}' done in {:.2}s: {objects} objects, {lights} lights",
        config.scene_name,
        started.elapsed().as_secs_f32(),
    );
    renderer.install_persistent_pass(std::sync::Arc::new(move || {
        Box::new(helio_bake::BakeInjectPass::new(std::sync::Arc::clone(
            &baked,
        )))
    }));
    Ok(())
}
