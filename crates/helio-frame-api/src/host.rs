//! Host inputs: resources a host supplies every frame (outside any pass),
//! declared by the feature whose passes read them.
//!
//! A host input is a data-only node of kind `host.<name>` placed at the
//! [`HOST_INPUTS`] point: it declares the external input so the passes that
//! read it get a Scene pin for it. A preset adds every host input registered,
//! so a feature that reads one (billboards, a particle system's emitters)
//! brings it along.

use helio_feature_api::{ExtensionPoint, NodeDescriptor};

/// Where host-input nodes are placed.
pub const HOST_INPUTS: ExtensionPoint = ExtensionPoint("frame.host_inputs");

/// The kind of the host node for resource `name`.
pub fn host_kind(name: &str) -> String {
    format!("host.{name}")
}

/// The node declaring host input `name`; register it from the feature that
/// reads it.
pub fn host_input(name: &'static str) -> NodeDescriptor {
    NodeDescriptor::new(
        &host_kind(name),
        &format!("Host::{name}"),
        "Host",
        "A resource the renderer supplies every frame.",
        move |build| {
            build.graph.declare_external_input(name);
            Ok(())
        },
    )
    .with_data_only()
    .with_generated_pins()
    .at(HOST_INPUTS, 0)
}
