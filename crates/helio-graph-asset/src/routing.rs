//! Routing validation of a graph asset: does what its nodes produce reach the
//! Output?
//!
//! The engine of the check is [`helio_core::check_routing`], shared with the
//! renderer's runtime guard (`RenderGraph::routing_issues`), which runs it over
//! a built graph's passes. This module is the other front end: it derives the
//! same facts from the pins of a validated, flattened asset, so the editor's
//! Compile reports them before anything is built, naming nodes.
//!
//! One node is one unit, in the order the compiler builds them. What a node
//! reads and writes is what its pins carry (`Scene` pins and SceneDB buffers
//! are the host's; a history or optional input needs no writer). The node
//! whose flow wire enters the Output is the one that writes the final output.

use std::collections::HashSet;

use helio_core::{check_routing, RoutingIssue, RoutingRead, RoutingSeverity, RoutingUnit};
use helio_feature_api::{NodeRegistry, PinKind};

use crate::asset::GraphAsset;
use crate::core_nodes::OUTPUT;
use crate::probe::SCENEDB_PREFIX;
use crate::validate::{Diagnostic, Plan, Severity};

/// Routing findings of the flattened `asset`, built in `plan`'s order.
///
/// A graph with no executing node (only the start and the Output) has nothing
/// to route and reports nothing.
pub fn check_asset_routing(
    asset: &GraphAsset,
    plan: &Plan,
    registry: &NodeRegistry,
) -> Vec<RoutingIssue> {
    let feeds_output: HashSet<&str> = asset
        .connections
        .iter()
        .filter(|wire| {
            asset
                .nodes
                .iter()
                .any(|node| node.id == wire.to.node && node.kind == OUTPUT)
        })
        .map(|wire| wire.from.node.as_str())
        .collect();

    let mut executes = false;
    let mut units = Vec::new();
    for &index in &plan.order {
        let node = &asset.nodes[index];
        let Some(descriptor) = registry.get(&node.kind) else {
            continue;
        };
        let pins = descriptor.pins(&node.params);
        let is_flow = |pin: &helio_feature_api::PinDesc| pin.ty.kind == PinKind::Flow;
        let data = |pin: &helio_feature_api::PinDesc| {
            !is_flow(pin)
                && pin.ty.kind != PinKind::Scene
                && !pin.resource.starts_with(SCENEDB_PREFIX)
        };
        let is_output = node.kind == OUTPUT;
        let runs = pins.inputs.iter().chain(&pins.outputs).any(is_flow);
        // The start and the Output only bound the chain.
        executes |= runs && !is_output && !pins.inputs.is_empty();
        if is_output {
            continue;
        }
        units.push(RoutingUnit {
            id: node.id.clone(),
            reads: pins
                .inputs
                .iter()
                .filter(|pin| data(pin))
                .map(|pin| RoutingRead {
                    resource: pin.resource.clone(),
                    optional: pin.optional || pin.history,
                })
                .collect(),
            writes: pins
                .outputs
                .iter()
                .filter(|pin| data(pin))
                .map(|pin| pin.resource.clone())
                .collect(),
            writes_target: feeds_output.contains(node.id.as_str()),
        });
    }
    if !executes {
        return Vec::new();
    }
    check_routing(&units, &HashSet::new())
}

/// [`check_asset_routing`] as the diagnostics validation reports, each on its
/// first node.
pub fn diagnostics(issues: Vec<RoutingIssue>) -> Vec<Diagnostic> {
    issues
        .into_iter()
        .map(|issue| Diagnostic {
            severity: match issue.severity {
                RoutingSeverity::Error => Severity::Error,
                RoutingSeverity::Warning => Severity::Warning,
            },
            node: issue.units.first().cloned(),
            message: issue.message,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{Connection, NodeInstance, PinRef};
    use crate::core_nodes::ON_FRAME_START;
    use crate::validate::validate;
    use helio_core::RoutingKind;
    use helio_feature_api::{NodeDescriptor, PinDesc, PinType};

    fn unit(id: &str, reads: &[&str], writes: &[&str], target: bool) -> RoutingUnit {
        RoutingUnit {
            id: id.into(),
            reads: reads.iter().map(|r| RoutingRead::required(*r)).collect(),
            writes: writes.iter().map(|w| w.to_string()).collect(),
            writes_target: target,
        }
    }

    fn kinds(issues: &[RoutingIssue]) -> Vec<RoutingKind> {
        issues.iter().map(|issue| issue.kind).collect()
    }

    fn run(units: &[RoutingUnit], host: &[&str]) -> Vec<RoutingIssue> {
        check_routing(units, &host.iter().map(|h| h.to_string()).collect())
    }

    #[test]
    fn a_healthy_graph_reports_nothing() {
        let units = [
            unit("geometry", &[], &["gbuffer", "depth"], false),
            unit("lighting", &["gbuffer", "depth", "sun"], &["scene"], false),
            unit("tonemap", &["scene"], &[], true),
        ];
        assert_eq!(run(&units, &["sun"]), Vec::new());
    }

    #[test]
    fn a_writer_nothing_reads_is_an_orphan_and_a_chain_that_ends_early_cannot_reach() {
        let units = [
            unit("geometry", &[], &["gbuffer", "extra"], false),
            unit("lighting", &["gbuffer"], &["scene"], false),
            unit("tonemap", &["scene"], &[], true),
        ];
        let issues = run(&units, &[]);
        assert_eq!(kinds(&issues), [RoutingKind::OrphanWrite]);
        assert_eq!(issues[0].units, ["geometry"]);
        assert_eq!(issues[0].resource.as_deref(), Some("extra"));
        assert!(issues[0].message.contains("geometry") && issues[0].message.contains("extra"));

        // Nothing past `lighting` reads `scene`: its work never shows.
        let units = [
            unit("geometry", &[], &["gbuffer"], false),
            unit("lighting", &["gbuffer"], &["scene"], false),
            unit("blit", &[], &[], true),
        ];
        let issues = run(&units, &[]);
        let unreachable: Vec<_> = issues
            .iter()
            .filter(|i| i.kind == RoutingKind::CannotReachOutput)
            .flat_map(|i| i.units.clone())
            .collect();
        assert_eq!(unreachable, ["geometry", "lighting"]);
    }

    #[test]
    fn a_write_overwritten_before_any_read_is_reported() {
        let units = [
            unit("a", &[], &["scene"], false),
            unit("b", &[], &["scene"], false),
            unit("tonemap", &["scene"], &[], true),
        ];
        let issues = run(&units, &[]);
        assert_eq!(kinds(&issues), [RoutingKind::CannotReachOutput]);
        assert_eq!(issues[0].units, ["a"]);
    }

    #[test]
    fn a_contributing_unit_with_a_dead_extra_write_names_the_overwriter() {
        let units = [
            unit("a", &[], &["scene", "aux"], false),
            unit("b", &[], &["aux"], false),
            unit("tonemap", &["scene", "aux"], &[], true),
        ];
        let issues = run(&units, &[]);
        assert_eq!(kinds(&issues), [RoutingKind::OrphanWrite]);
        assert_eq!(issues[0].units, ["a", "b"]);
        assert!(issues[0].message.contains("overwrites"));
    }

    #[test]
    fn a_read_nothing_writes_is_reported_unless_host_history_or_optional() {
        let reader = unit("fog", &["nobody_writes"], &["scene"], false);
        let sink = unit("tonemap", &["scene"], &[], true);
        let issues = run(&[reader.clone(), sink.clone()], &[]);
        assert_eq!(kinds(&issues), [RoutingKind::UnwrittenRead]);
        assert_eq!(issues[0].units, ["fog"]);
        assert_eq!(issues[0].resource.as_deref(), Some("nobody_writes"));

        // Supplied by the host.
        assert!(run(&[reader.clone(), sink.clone()], &["nobody_writes"]).is_empty());

        // Optional: the pass binds a fallback.
        let mut optional = reader;
        optional.reads[0].optional = true;
        assert!(run(&[optional, sink], &[]).is_empty());
    }

    #[test]
    fn a_history_read_is_allowed() {
        // Hi-Z reads `depth` before the geometry pass writes it: last frame's.
        let units = [
            unit("hiz", &["depth"], &["hiz"], false),
            unit("geometry", &[], &["depth", "scene"], false),
            unit("cull", &["hiz"], &["visible"], false),
            unit("tonemap", &["scene", "visible"], &[], true),
        ];
        let issues = run(&units, &[]);
        assert_eq!(issues, Vec::new(), "{issues:?}");
    }

    #[test]
    fn an_output_no_pass_writes_is_a_hard_error() {
        let units = [
            unit("geometry", &[], &["gbuffer"], false),
            unit("lighting", &["gbuffer"], &["scene"], false),
        ];
        let issues = run(&units, &[]);
        let error = issues
            .iter()
            .find(|i| i.kind == RoutingKind::OutputNeverWritten)
            .expect("reported");
        assert_eq!(error.severity, RoutingSeverity::Error);
        assert_eq!(issues[0].severity, RoutingSeverity::Error, "errors first");
        assert!(error.message.contains("final output is never written"));
    }

    #[test]
    fn a_later_output_writer_that_ignores_the_first_overwrites_it() {
        let units = [
            unit("fxaa", &["pre_aa"], &[], true),
            unit("tsr", &["history"], &[], true),
        ];
        let issues = run(&units, &["pre_aa", "history"]);
        assert_eq!(kinds(&issues), [RoutingKind::OutputOverwritten]);
        assert_eq!(issues[0].units, ["fxaa", "tsr"]);

        // A later writer that builds on the first is a chain, not an overwrite.
        let units = [
            unit("tonemap", &[], &["ldr"], true),
            unit("overlay", &["ldr"], &[], true),
        ];
        assert!(run(&units, &[]).is_empty());
    }

    fn node(id: &str, kind: &str) -> NodeInstance {
        NodeInstance {
            id: id.into(),
            kind: kind.into(),
            params: Default::default(),
            position: [0.0; 2],
        }
    }

    fn wire(a: &str, ap: &str, b: &str, bp: &str) -> Connection {
        Connection {
            from: PinRef::new(a, ap),
            to: PinRef::new(b, bp),
        }
    }

    fn registry() -> NodeRegistry {
        let mut registry = crate::core_registry();
        registry.register(
            NodeDescriptor::new("t.lit", "Lit", "Test", "", |_| Ok(())).with_pins(
                vec![PinDesc::flow("in")],
                vec![
                    PinDesc::flow("out"),
                    PinDesc::resource(PinType::TEXTURE, "scene"),
                ],
            ),
        );
        registry.register(
            NodeDescriptor::new("t.tonemap", "Tonemap", "Test", "", |_| Ok(())).with_pins(
                vec![
                    PinDesc::flow("in"),
                    PinDesc::resource(PinType::TEXTURE, "scene"),
                ],
                vec![PinDesc::flow("out")],
            ),
        );
        registry
    }

    fn asset(chain: &[&str]) -> GraphAsset {
        let mut asset = GraphAsset::default();
        asset.nodes.push(node("start", ON_FRAME_START));
        asset.nodes.push(node("out", OUTPUT));
        let mut previous = ("start", "out");
        for id in chain {
            let kind = if *id == "lit" { "t.lit" } else { "t.tonemap" };
            asset.nodes.push(node(id, kind));
            asset
                .connections
                .push(wire(previous.0, previous.1, id, "in"));
            previous = (id, "out");
        }
        asset
            .connections
            .push(wire(previous.0, previous.1, "out", "in"));
        if chain == ["lit", "tonemap"] {
            asset
                .connections
                .push(wire("lit", "scene", "tonemap", "scene"));
        }
        asset
    }

    #[test]
    fn validation_reports_routing_of_an_asset() {
        let registry = registry();
        // Healthy: the lit scene is tone-mapped into the output.
        let healthy = validate(&asset(&["lit", "tonemap"]), &registry);
        assert!(healthy.diagnostics.is_empty(), "{:?}", healthy.diagnostics);

        // The producer is the only node: nothing reads `scene`.
        let dead = validate(&asset(&["lit"]), &registry);
        let lit = dead
            .diagnostics
            .iter()
            .find(|d| d.node.as_deref() == Some("lit"))
            .expect("the dead producer is named");
        assert_eq!(lit.severity, Severity::Warning);
        assert!(lit.message.contains("`scene`"), "{}", lit.message);

        // Nothing flows into the Output.
        let mut cut = asset(&["lit", "tonemap"]);
        cut.connections.retain(|wire| wire.to.node != "out");
        let cut = validate(&cut, &registry);
        assert!(cut
            .errors()
            .any(|d| d.message.contains("final output is never written")));
        assert!(cut.plan.is_none());
    }
}
