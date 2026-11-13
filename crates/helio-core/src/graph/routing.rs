//! Routing validation: does what the passes produce reach the presented output?
//!
//! A pipeline once rendered black because the pass that produced the image
//! never fed anything that reached the output. Helio resolves a read by
//! resource name against the passes before it, so such a graph builds, locks
//! and runs: nothing says it is wrong. This module says it, from facts alone:
//! an ordered list of [`RoutingUnit`]s (what each reads, writes, and whether it
//! produces the presented target) and the host-supplied inputs. It touches no
//! GPU state, reads nothing back, and is cheap enough to run once per build.
//!
//! A unit is a pass when the facts come from a built graph
//! ([`RoutingUnit::from_declaration`], [`RenderGraph::routing_issues`]) and a
//! node when they come from a graph asset's pins (`helio-graph-asset`'s
//! `routing` module). Both go through [`check_routing`], so they cannot
//! disagree about what a healthy graph is.
//!
//! The rules, in the order Helio resolves names:
//!
//! * a read resolves to the **latest earlier writer** of the name;
//! * a read with no earlier writer is satisfied by the host
//!   (`declare_external_input`), by *history* (a later unit writes it: last
//!   frame's value) or by being declared optional. Otherwise it is reported;
//! * a write nothing resolves to is a dead write: no read ever sees it,
//!   either because nobody reads the name or because a later unit overwrites
//!   it first;
//! * a unit whose writes cannot lead, through reads, to a unit that produces
//!   the target does not contribute to the output;
//! * no unit producing the target is a hard error: the target is cleared, the
//!   frame is black.
//!
//! [`RenderGraph::routing_issues`]: crate::RenderGraph::routing_issues

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::PassDeclaration;

/// How serious a finding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingSeverity {
    /// The output cannot show the pipeline's result.
    Error,
    /// Wasted or suspicious routing; the output may still be right.
    Warning,
}

/// What a finding is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingKind {
    /// No unit produces the presented target.
    OutputNeverWritten,
    /// A unit produces the target, then a later one overwrites it without
    /// reading what the first one made.
    OutputOverwritten,
    /// A resource is written and no read ever sees it.
    OrphanWrite,
    /// A resource is read and nothing supplies it.
    UnwrittenRead,
    /// A unit writes resources, but none of them leads to the output.
    CannotReachOutput,
}

/// One finding, naming the units and the resource it concerns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingIssue {
    pub severity: RoutingSeverity,
    pub kind: RoutingKind,
    /// The units (pass or node ids) concerned, the culprit first.
    pub units: Vec<String>,
    /// The resource concerned, if one.
    pub resource: Option<String>,
    pub message: String,
}

impl fmt::Display for RoutingIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level = match self.severity {
            RoutingSeverity::Error => "error",
            RoutingSeverity::Warning => "warning",
        };
        write!(f, "{level}: {}", self.message)
    }
}

/// A read of one resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingRead {
    pub resource: String,
    /// The unit has a fallback when nothing supplies the resource.
    pub optional: bool,
}

impl RoutingRead {
    pub fn required(resource: impl Into<String>) -> Self {
        Self {
            resource: resource.into(),
            optional: false,
        }
    }

    pub fn optional(resource: impl Into<String>) -> Self {
        Self {
            resource: resource.into(),
            optional: true,
        }
    }
}

/// What one pass (or node) does to the frame's resources, in build order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoutingUnit {
    /// Names the unit in findings.
    pub id: String,
    pub reads: Vec<RoutingRead>,
    pub writes: Vec<String>,
    /// The unit produces the frame's image in the presented target.
    pub writes_target: bool,
}

impl RoutingUnit {
    /// The unit of pass `index` (its position in build order), from what it
    /// declares (optional reads stay optional).
    pub fn from_declaration(index: usize, declaration: &PassDeclaration) -> Self {
        Self {
            id: format!("{} (pass {index})", declaration.pass),
            reads: declaration
                .reads
                .iter()
                .map(|read| {
                    if read.optional {
                        RoutingRead::optional(read.name)
                    } else {
                        RoutingRead::required(read.name)
                    }
                })
                .collect(),
            writes: declaration
                .writes
                .iter()
                .map(|write| write.name.to_string())
                .collect(),
            writes_target: declaration.writes_target,
        }
    }
}

/// Where a read gets its value.
enum Source {
    /// The latest earlier writer.
    Unit(usize),
    /// Last frame's value: the last writer, which comes later.
    History(usize),
    Host,
    /// Nothing, but the reader has a fallback.
    Fallback,
    Missing,
}

/// Checks the routing of `units` (in build order) against `host_inputs`, the
/// resources the host supplies each frame. Findings are ordered: errors first,
/// then by unit order.
pub fn check_routing(units: &[RoutingUnit], host_inputs: &HashSet<String>) -> Vec<RoutingIssue> {
    let mut issues = Vec::new();

    // Every write position of a name, in order.
    let mut writers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, unit) in units.iter().enumerate() {
        for name in &unit.writes {
            let list = writers.entry(name.as_str()).or_default();
            if list.last() != Some(&index) {
                list.push(index);
            }
        }
    }

    // `edges[consumer]` are the units whose writes it reads; `consumed` the
    // (writer, resource) pairs some read resolves to.
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); units.len()];
    let mut consumed: HashSet<(usize, &str)> = HashSet::new();
    for (index, unit) in units.iter().enumerate() {
        for read in &unit.reads {
            let name = read.resource.as_str();
            let list = writers.get(name).map(Vec::as_slice).unwrap_or(&[]);
            let source = if let Some(&writer) = list.iter().rev().find(|&&writer| writer < index) {
                Source::Unit(writer)
            } else if host_inputs.contains(name) {
                Source::Host
            } else if let Some(&last) = list.last() {
                Source::History(last)
            } else if read.optional {
                Source::Fallback
            } else {
                Source::Missing
            };
            match source {
                Source::Unit(writer) | Source::History(writer) => {
                    edges[index].push(writer);
                    consumed.insert((writer, name));
                }
                Source::Host | Source::Fallback => {}
                Source::Missing => issues.push(RoutingIssue {
                    severity: RoutingSeverity::Warning,
                    kind: RoutingKind::UnwrittenRead,
                    units: vec![unit.id.clone()],
                    resource: Some(read.resource.clone()),
                    message: format!(
                        "`{}` reads `{}`, but no unit writes it and the host does not supply it (it reads a fallback or fails to bind)",
                        unit.id, read.resource
                    ),
                }),
            }
        }
    }

    // Units that contribute to the output: the producers of the target and,
    // walking reads backwards, everything they depend on.
    let targets: Vec<usize> = (0..units.len())
        .filter(|&index| units[index].writes_target)
        .collect();
    let mut reaches_output = vec![false; units.len()];
    let mut pending = targets.clone();
    for &target in &targets {
        reaches_output[target] = true;
    }
    while let Some(unit) = pending.pop() {
        for &producer in &edges[unit] {
            if !reaches_output[producer] {
                reaches_output[producer] = true;
                pending.push(producer);
            }
        }
    }

    match targets.as_slice() {
        [] => issues.push(RoutingIssue {
            severity: RoutingSeverity::Error,
            kind: RoutingKind::OutputNeverWritten,
            units: Vec::new(),
            resource: None,
            message:
                "the final output is never written by any pass: the frame stays cleared (black)"
                    .to_string(),
        }),
        [.., last] => {
            // A producer of the target that the last one does not build on
            // is overwritten: what it drew never shows.
            for &target in &targets[..targets.len() - 1] {
                if !depends_on(&edges, *last, target) {
                    issues.push(RoutingIssue {
                        severity: RoutingSeverity::Warning,
                        kind: RoutingKind::OutputOverwritten,
                        units: vec![units[target].id.clone(), units[*last].id.clone()],
                        resource: None,
                        message: format!(
                            "`{}` writes the final output, but `{}` overwrites it later without reading anything `{}` made",
                            units[target].id, units[*last].id, units[target].id
                        ),
                    });
                }
            }
        }
    }

    for (index, unit) in units.iter().enumerate() {
        if reaches_output[index] || unit.writes.is_empty() {
            continue;
        }
        issues.push(RoutingIssue {
            severity: RoutingSeverity::Warning,
            kind: RoutingKind::CannotReachOutput,
            units: vec![unit.id.clone()],
            resource: None,
            message: format!(
                "`{}` writes {} but its results never reach the output: no chain of reads leads from it to a pass that writes the output",
                unit.id,
                list(&unit.writes)
            ),
        });
    }

    // Dead writes of units that otherwise contribute (the others are already
    // reported whole above).
    for (index, unit) in units.iter().enumerate() {
        if !reaches_output[index] {
            continue;
        }
        for name in &unit.writes {
            if consumed.contains(&(index, name.as_str())) {
                continue;
            }
            let overwriter = writers
                .get(name.as_str())
                .and_then(|list| list.iter().find(|&&writer| writer > index))
                .map(|&writer| &units[writer].id);
            let mut involved = vec![unit.id.clone()];
            let reason = match overwriter {
                Some(other) => {
                    involved.push(other.clone());
                    format!("`{other}` overwrites it before any unit reads it")
                }
                None => "no unit reads it".to_string(),
            };
            issues.push(RoutingIssue {
                severity: RoutingSeverity::Warning,
                kind: RoutingKind::OrphanWrite,
                units: involved,
                resource: Some(name.clone()),
                message: format!("`{}` writes `{name}`, but {reason}", unit.id),
            });
        }
    }

    issues.sort_by_key(|issue| issue.severity != RoutingSeverity::Error);
    issues
}

/// Whether `unit` reads, directly or through other units, what `producer` wrote.
fn depends_on(edges: &[Vec<usize>], unit: usize, producer: usize) -> bool {
    let mut seen = vec![false; edges.len()];
    let mut pending = vec![unit];
    while let Some(current) = pending.pop() {
        for &next in &edges[current] {
            if next == producer {
                return true;
            }
            if !seen[next] {
                seen[next] = true;
                pending.push(next);
            }
        }
    }
    false
}

fn list(names: &[String]) -> String {
    names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}
