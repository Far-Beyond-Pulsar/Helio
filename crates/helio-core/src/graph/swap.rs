//! Selective pass swapping: the pure decision logic behind shader hot reload
//! that replaces only the passes whose shaders changed.
//!
//! `helio-core` names no specific pass here. A graph builder describes which
//! pass types may be swapped through a [`SwapPolicy`] (stored on the graph with
//! [`RenderGraph::set_swap_policy`](super::RenderGraph::set_swap_policy)), and
//! the host decides with [`plan_selective_swap`], which is a pure function over
//! pass identities so it needs no device.
//!
//! # Why a policy exists
//!
//! A pass can only be built by its constructor, so a hot reload still builds a
//! whole replacement graph. A graph builder may create a resource itself (a
//! buffer, a sampler, a shared `Arc`) and hand the *same* handle to several
//! passes. If only one of those passes were taken from the replacement, it
//! would hold the replacement's copy while the others kept the old one, and
//! they would no longer agree on the buffer. So passes that share
//! builder-created resources are declared as a *group* and are swapped
//! together, and a pass the builder does not vouch for is never swapped
//! (the host falls back to rebuilding the whole graph).

use std::any::TypeId;
use std::collections::{BTreeSet, HashSet};

/// What identifies a pass for the purpose of matching it between two graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassIdentity {
    /// `TypeId` of the concrete pass type.
    pub type_id: TypeId,
    /// [`RenderPass::name`](crate::RenderPass::name).
    pub name: &'static str,
    /// [`RenderPass::type_name`](crate::RenderPass::type_name), whose leading
    /// path segment is the crate that defines the pass.
    pub type_name: &'static str,
}

/// The type path of `T`, as [`RenderPass::type_name`](crate::RenderPass::type_name)
/// reports it for a pass of that type. For naming pass types in a
/// [`SwapPolicy`] without an instance.
pub fn type_name_of<T: ?Sized>() -> &'static str {
    std::any::type_name::<T>()
}

/// Whether a pass with `type_name` is defined in the crate `crate_name`
/// (written with underscores, as in a type path).
pub fn type_in_crate(type_name: &str, crate_name: &str) -> bool {
    type_name
        .strip_prefix(crate_name)
        .is_some_and(|rest| rest.starts_with("::"))
}

/// Which pass types a graph builder allows to be swapped on their own or
/// together, declared by the builder that knows what its constructors share.
///
/// Types are identified by [`type_name_of`]. Anything not listed is not
/// swappable.
///
/// # Audit of the default graphs
///
/// The policy shipped with `helio-default-graphs` was derived by reading every
/// constructor call in that crate. Resources the builder creates itself and
/// hands to more than one pass:
///
/// | resource | passes holding it |
/// |---|---|
/// | shadow dirty-flag buffer | `ShadowMatrixPass`, `ShadowDirtyPass` |
/// | shadow face dirty / geometry-count buffers | `ShadowDirtyPass`, `ShadowCullPass`, `ShadowPass` |
/// | shadow face indirect / count buffers | `ShadowCullPass`, `ShadowPass` |
/// | Hi-Z sampler | `HiZBuildPass`, `OcclusionCullPass` |
/// | portal cull output buffers | `PortalCullPass`, `PortalInstancePass` |
/// | foliage arena / tile / blade / indirect buffers | `FoliagePlacePass`, `FoliageGBufferPass` |
/// | perf-overlay shared state | every perf-overlay pass |
///
/// Each row is one group. Handles that come from the renderer (camera, debug
/// camera, cull-stats buffers, the debug-draw state, the SceneDB handle) are the
/// same objects in the live and the replacement graph, so they never make a
/// pass unsafe to swap.
#[derive(Clone, Debug, Default)]
pub struct SwapPolicy {
    independent: Vec<&'static str>,
    groups: Vec<Vec<&'static str>>,
}

impl SwapPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a pass type that shares no builder-created resource with any
    /// other pass, so it may be swapped alone.
    pub fn independent_type(mut self, type_name: &'static str) -> Self {
        self.independent.push(type_name);
        self
    }

    /// [`independent_type`](Self::independent_type) for the type `T`.
    pub fn independent<T: ?Sized>(self) -> Self {
        self.independent_type(type_name_of::<T>())
    }

    /// Declares pass types that share builder-created resources: swapping any
    /// one of them swaps all of them (every instance of each type).
    pub fn group_types(mut self, members: &[&'static str]) -> Self {
        self.groups.push(members.to_vec());
        self
    }

    fn lists(&self, type_name: &str) -> bool {
        self.independent.iter().any(|listed| *listed == type_name)
            || self
                .groups
                .iter()
                .any(|group| group.iter().any(|listed| *listed == type_name))
    }

    /// The pass types that must be swapped to swap `requested`: `requested`
    /// plus, transitively, every member of a group any of them belongs to.
    ///
    /// Fails, naming the type, if any pass in the result is not listed in the
    /// policy.
    pub fn closure(&self, requested: &[&'static str]) -> Result<Vec<&'static str>, String> {
        let mut set: BTreeSet<&'static str> = requested.iter().copied().collect();
        loop {
            let before = set.len();
            for group in &self.groups {
                if group.iter().any(|member| set.contains(member)) {
                    set.extend(group.iter().copied());
                }
            }
            if set.len() == before {
                break;
            }
        }
        if let Some(unlisted) = set.iter().find(|name| !self.lists(name)) {
            return Err(format!(
                "pass type `{unlisted}` is not declared swappable by the graph builder"
            ));
        }
        Ok(set.into_iter().collect())
    }
}

/// Matches the replacement graph's passes to the live graph's, in order.
///
/// Returns, per replacement pass, the index of the live pass of the same type
/// and name, plus the live passes that have no counterpart. A live pass may lack
/// a counterpart only if its type does not occur in the replacement at all (a
/// pass added to the live graph after it was built, such as baked-lighting
/// injection); anything else means the two graphs are not the same pass
/// sequence and the result is an error.
pub fn align_pass_sequences(
    live: &[PassIdentity],
    replacement: &[PassIdentity],
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let replacement_types: HashSet<TypeId> = replacement.iter().map(|pass| pass.type_id).collect();
    let mut mapping = Vec::with_capacity(replacement.len());
    let mut extras = Vec::new();
    for (index, pass) in live.iter().enumerate() {
        let next = replacement.get(mapping.len());
        match next {
            Some(candidate) if candidate.type_id == pass.type_id && candidate.name == pass.name => {
                mapping.push(index);
            }
            _ if !replacement_types.contains(&pass.type_id) => extras.push(index),
            _ => {
                return Err(format!(
                    "pass sequence differs at live pass {index} ('{}')",
                    pass.name
                ));
            }
        }
    }
    if mapping.len() != replacement.len() {
        return Err(format!(
            "the rebuilt graph has {} pass(es) the live graph lacks",
            replacement.len() - mapping.len()
        ));
    }
    Ok((mapping, extras))
}

/// Which passes to move from a replacement graph into the live one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwapPlan {
    /// `(live index, replacement index)` per pass to swap.
    pub picks: Vec<(usize, usize)>,
    /// Names of the passes being swapped, in graph order, without repeats.
    pub pass_names: Vec<&'static str>,
}

/// Decides which passes to swap after shader edits owned by `affected_crates`,
/// or why the whole graph has to be rebuilt instead.
///
/// * the graph must provide a [`SwapPolicy`];
/// * the replacement must be the same pass sequence as the live graph (see
///   [`align_pass_sequences`]);
/// * every affected crate must own at least one pass in the graph, and no
///   live-only pass may belong to one (it could not be refreshed);
/// * the passes owned by affected crates, widened by the policy's groups, must
///   all be swappable.
pub fn plan_selective_swap(
    live: &[PassIdentity],
    replacement: &[PassIdentity],
    affected_crates: &[String],
    policy: Option<&SwapPolicy>,
) -> Result<SwapPlan, String> {
    let policy = policy.ok_or("the graph builder declared no swap policy")?;
    let (mapping, extras) = align_pass_sequences(live, replacement)?;

    let mut requested: Vec<&'static str> = Vec::new();
    for crate_name in affected_crates {
        if let Some(&index) = extras
            .iter()
            .find(|&&index| type_in_crate(live[index].type_name, crate_name))
        {
            return Err(format!(
                "pass `{}` (crate `{crate_name}`) exists only in the live graph and cannot be refreshed",
                live[index].name
            ));
        }
        let before = requested.len();
        requested.extend(
            replacement
                .iter()
                .filter(|pass| type_in_crate(pass.type_name, crate_name))
                .map(|pass| pass.type_name),
        );
        if requested.len() == before {
            return Err(format!(
                "shader owner crate `{crate_name}` matches no pass in the graph"
            ));
        }
    }

    let swap_types = policy.closure(&requested)?;
    let mut picks = Vec::new();
    let mut pass_names: Vec<&'static str> = Vec::new();
    for (replacement_index, pass) in replacement.iter().enumerate() {
        if swap_types.contains(&pass.type_name) {
            picks.push((mapping[replacement_index], replacement_index));
            if !pass_names.contains(&pass.name) {
                pass_names.push(pass.name);
            }
        }
    }
    Ok(SwapPlan { picks, pass_names })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only their `TypeId`s are used.
    #[allow(dead_code)]
    struct A;
    #[allow(dead_code)]
    struct B;
    #[allow(dead_code)]
    struct C;
    #[allow(dead_code)]
    struct D;
    #[allow(dead_code)]
    struct Late;

    fn id<T: 'static>(name: &'static str, type_name: &'static str) -> PassIdentity {
        PassIdentity {
            type_id: TypeId::of::<T>(),
            name,
            type_name,
        }
    }

    fn graph() -> Vec<PassIdentity> {
        vec![
            id::<A>("A", "crate_a::A"),
            id::<B>("B", "crate_b::B"),
            id::<C>("C", "crate_c::C"),
            id::<D>("D", "crate_c::D"),
        ]
    }

    #[test]
    fn type_names_match_by_crate_prefix_only() {
        assert!(type_in_crate("helio_pass_fxaa::FxaaPass", "helio_pass_fxaa"));
        assert!(type_in_crate("helio_pass_fxaa::a::b::Pass<T>", "helio_pass_fxaa"));
        // A crate whose name merely starts with another's is not that crate.
        assert!(!type_in_crate("helio_pass_fxaa2::Pass", "helio_pass_fxaa"));
        assert!(!type_in_crate("helio_pass_fxaa", "helio_pass_fxaa"));
        assert!(!type_in_crate("other::helio_pass_fxaa::Pass", "helio_pass_fxaa"));
    }

    #[test]
    fn type_name_of_matches_the_std_path() {
        assert_eq!(type_name_of::<A>(), std::any::type_name::<A>());
        assert!(type_in_crate(type_name_of::<A>(), module_path!().split("::").next().unwrap()));
    }

    #[test]
    fn closure_widens_to_the_whole_group() {
        let policy = SwapPolicy::new()
            .independent_type("crate_a::A")
            .group_types(&["crate_b::B", "crate_c::C"])
            .group_types(&["crate_c::C", "crate_c::D"]);
        assert_eq!(policy.closure(&["crate_a::A"]).unwrap(), vec!["crate_a::A"]);
        // B pulls in C through the first group and D through the second.
        assert_eq!(
            policy.closure(&["crate_b::B"]).unwrap(),
            vec!["crate_b::B", "crate_c::C", "crate_c::D"]
        );
    }

    #[test]
    fn closure_rejects_unlisted_types() {
        let policy = SwapPolicy::new().independent_type("crate_a::A");
        let error = policy.closure(&["crate_a::A", "crate_x::X"]).unwrap_err();
        assert!(error.contains("crate_x::X"), "{error}");
    }

    #[test]
    fn sequences_align_in_order() {
        let (mapping, extras) = align_pass_sequences(&graph(), &graph()).unwrap();
        assert_eq!(mapping, vec![0, 1, 2, 3]);
        assert!(extras.is_empty());
    }

    #[test]
    fn live_only_pass_types_are_tolerated() {
        let mut live = graph();
        live.push(id::<Late>("Late", "crate_l::Late"));
        let (mapping, extras) = align_pass_sequences(&live, &graph()).unwrap();
        assert_eq!(mapping, vec![0, 1, 2, 3]);
        assert_eq!(extras, vec![4]);
    }

    #[test]
    fn differing_sequences_are_refused() {
        let mut shorter = graph();
        shorter.remove(1);
        assert!(align_pass_sequences(&graph(), &shorter).is_err());
        assert!(align_pass_sequences(&shorter, &graph()).is_err());
        let mut swapped = graph();
        swapped.swap(0, 1);
        assert!(align_pass_sequences(&graph(), &swapped).is_err());
    }

    fn policy() -> SwapPolicy {
        SwapPolicy::new()
            .independent_type("crate_a::A")
            .group_types(&["crate_c::C", "crate_c::D"])
    }

    #[test]
    fn plan_swaps_only_the_owning_pass() {
        let plan = plan_selective_swap(&graph(), &graph(), &["crate_a".into()], Some(&policy())).unwrap();
        assert_eq!(plan.picks, vec![(0, 0)]);
        assert_eq!(plan.pass_names, vec!["A"]);
    }

    #[test]
    fn plan_swaps_a_whole_group() {
        let plan = plan_selective_swap(&graph(), &graph(), &["crate_c".into()], Some(&policy())).unwrap();
        assert_eq!(plan.picks, vec![(2, 2), (3, 3)]);
    }

    #[test]
    fn plan_maps_replacement_indices_past_live_only_passes() {
        let mut live = graph();
        live.insert(1, id::<Late>("Late", "crate_l::Late"));
        let plan = plan_selective_swap(&live, &graph(), &["crate_a".into(), "crate_c".into()], Some(&policy())).unwrap();
        assert_eq!(plan.picks, vec![(0, 0), (3, 2), (4, 3)]);
    }

    #[test]
    fn plan_falls_back_when_something_is_not_swappable() {
        // B is in neither list.
        assert!(plan_selective_swap(&graph(), &graph(), &["crate_b".into()], Some(&policy())).is_err());
        // No policy at all.
        assert!(plan_selective_swap(&graph(), &graph(), &["crate_a".into()], None).is_err());
    }

    #[test]
    fn plan_falls_back_when_the_owner_matches_no_pass() {
        let error = plan_selective_swap(&graph(), &graph(), &["crate_z".into()], Some(&policy())).unwrap_err();
        assert!(error.contains("crate_z"), "{error}");
    }

    #[test]
    fn plan_falls_back_when_a_live_only_pass_is_affected() {
        let mut live = graph();
        live.push(id::<Late>("Late", "crate_l::Late"));
        assert!(plan_selective_swap(&live, &graph(), &["crate_l".into()], Some(&policy())).is_err());
    }
}
