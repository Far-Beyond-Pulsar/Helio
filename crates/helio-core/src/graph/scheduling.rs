use super::execution::RenderGraph;

/// Pre-computed per-pass data populated at graph lock time.
pub(crate) struct CachedPass {
    pub(crate) store_ops: Vec<Option<wgpu::StoreOp>>,
    pub(crate) subpass_index: u32,
    /// Total number of passes in the chain (0 if not in a chain).
    /// Used by the executor to call `vkCmdNextSubpass` between chain members
    /// once wgpu exposes subpass support.
    pub(crate) subpass_count: u32,
    pub(crate) chain_range: std::ops::Range<usize>,
    /// GPU-timing label for the whole chain (`"GBufferPass+PortalInstancePass"`),
    /// `""` when not in a chain. A fused chain is one hardware render pass, so
    /// it is timed as one span: timestamps can't be written into the encoder
    /// while its render pass is open (Helio#298).
    pub(crate) chain_label: &'static str,
}

/// A process-lifetime label for a chain of pass names. Interned so rebuilding
/// the graph reuses the same string instead of leaking a new one each time.
pub(crate) fn chain_label(names: &[&'static str]) -> &'static str {
    static LABELS: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<String, &'static str>>,
    > = std::sync::LazyLock::new(Default::default);
    let joined = names.join("+");
    let mut labels = LABELS.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(label) = labels.get(&joined) {
        return label;
    }
    let label: &'static str = Box::leak(joined.clone().into_boxed_str());
    labels.insert(joined, label);
    label
}

/// An action to perform on the transient resource registry before a pass executes.
#[derive(Clone)]
pub(crate) enum PrePassAction {
    Route {
        name: &'static str,
        view: wgpu::TextureView,
    },
    /// A `write_group` declaration's members, resolved to concrete views and
    /// combined into one action — order-preserving, matching declaration
    /// order. Arity-generic: covers GBuffer's 4-view bundle today and any
    /// future compound resource with zero new code here (see
    /// `docs/helio_3_0_spec.md` §5). The core never interprets `name` or
    /// `members`; it only hands them to the owning pass's
    /// `RenderPass::publish_group`.
    Group {
        name: &'static str,
        members: Vec<(&'static str, wgpu::TextureView)>,
    },
}

/// Pure chain-detection algorithm, factored out of `RenderGraph::detect_subpass_chains`
/// so it can be unit-tested without a GPU device.
///
/// Greedily scans forward: pass `i` fuses into the next pass `j` if `writes[i]`
/// intersects `reads[j]`. Any run of `transparent[k] == true` passes between `i`
/// and `j` is skipped over when looking for `j` (they don't need to declare a
/// dependency to be bridged) but is still folded into the resulting chain
/// range, since ranges are contiguous — the executor keeps the render pass
/// open across them without closing/reopening it (see `chain_transparent` on
/// `RenderPass`).
///
/// `attachments[k]` must be `Some(signature)` only for passes whose
/// `render_pass_descriptor()` actually returns `Some` (a lock-time probe), with
/// `signature` identifying the exact set of texture views used as color/depth
/// attachments. This is load-bearing for two separate reasons:
///
/// 1. The executor only ever opens a chain's render pass from inside the
///    `Some(desc)` branch, at the pass whose index equals `chain_range.start`.
///    A pass that always returns `None` (pure compute, even one with declared
///    writes/reads that happen to satisfy the adjacency check) can therefore
///    never open — or safely be assumed to have already opened — a chain's
///    render pass, so it must never become a chain's start or bridge target.
/// 2. Declared write/read overlap only means two passes have a *data*
///    dependency (e.g. DeferredLight reads the gbuffer as a texture *input*
///    while rendering to a completely different target) — it does NOT mean
///    they render into the same physical attachments. Reusing one pass's open
///    render pass for another whose pipeline expects different attachment
///    formats/counts is a real `wgpu` validation failure (mismatched
///    `RenderPipeline` targets), not just a missed optimization. So fusion
///    additionally requires `attachments[i] == attachments[j]` — the two
///    passes must target the literal same views, e.g. GBuffer and
///    VirtualGeometry both drawing (with `LoadOp::Load`) into the same 5
///    gbuffer textures.
///
/// Only skipped-over `transparent` passes are exempt from both requirements,
/// since they never try to hold, open, or draw into the chain's render pass.
fn compute_chains(
    writes: &[Vec<&str>],
    reads: &[Vec<&str>],
    transparent: &[bool],
    attachments: &[Option<Vec<usize>>],
) -> Vec<std::ops::Range<usize>> {
    let len = writes.len();
    let mut chains = Vec::new();
    let mut i = 0;
    while i < len {
        if attachments[i].is_none() {
            i += 1;
            continue;
        }
        let chain_start = i;
        loop {
            let mut j = i + 1;
            while j < len && transparent[j] {
                j += 1;
            }
            if j >= len {
                break;
            }
            let same_attachments = match (&attachments[i], &attachments[j]) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            if !same_attachments {
                break;
            }
            let can_fuse = writes[i].iter().any(|w| reads[j].contains(w));
            if !can_fuse {
                break;
            }
            i = j;
        }
        let chain_len = i + 1 - chain_start;
        if chain_len >= 2 {
            chains.push(chain_start..i + 1);
        }
        i += 1;
    }
    chains
}

/// Computes deterministic topological recording layers for the render DAG.
/// Passes in one layer have no declared read/write dependency between them and
/// may therefore be recorded concurrently by the executor.
pub(crate) fn compute_parallel_layers(
    writes: &[Vec<&str>],
    reads: &[Vec<&str>],
) -> Vec<Vec<usize>> {
    assert_eq!(writes.len(), reads.len());
    let mut predecessors = vec![Vec::<usize>::new(); writes.len()];
    for current in 0..writes.len() {
        for prior in 0..current {
            // Preserve declared order for every hazard.  In particular, an
            // earlier read must finish before a later pass overwrites the
            // same resource; otherwise both passes would be scheduled in one
            // parallel layer and the GPU could race the read against the
            // overwrite.
            let dependency = writes[prior].iter().any(|resource| {
                reads[current].contains(resource) || writes[current].contains(resource)
            }) || reads[prior]
                .iter()
                .any(|resource| writes[current].contains(resource));
            if dependency {
                predecessors[current].push(prior);
            }
        }
    }

    let mut layers = vec![0usize; writes.len()];
    for current in 0..writes.len() {
        layers[current] = predecessors[current]
            .iter()
            .map(|&prior| layers[prior] + 1)
            .max()
            .unwrap_or(0);
    }
    let layer_count = layers.iter().copied().max().map_or(0, |max| max + 1);
    let mut result = vec![Vec::new(); layer_count];
    for (index, layer) in layers.into_iter().enumerate() {
        result[layer].push(index);
    }
    result
}

/// Groups passes into recording units and layers the units.
///
/// A unit is one pass, or one whole fused chain: a chain shares a single open
/// `wgpu::RenderPass`, so it must be recorded by one thread. A unit's reads and
/// writes are the union of its members', and units in one layer have no
/// declared dependency between them. Each layer lists its units' pass ranges in
/// ascending order, and the units themselves are ordered by first pass index,
/// which is a valid topological order and matches serial recording.
pub(crate) fn compute_unit_layers(
    writes: &[Vec<&str>],
    reads: &[Vec<&str>],
    chains: &[std::ops::Range<usize>],
) -> Vec<Vec<std::ops::Range<usize>>> {
    assert_eq!(writes.len(), reads.len());
    let mut units: Vec<std::ops::Range<usize>> = Vec::new();
    let mut index = 0;
    while index < writes.len() {
        let range = chains
            .iter()
            .find(|chain| chain.start == index && chain.end <= writes.len())
            .cloned()
            .unwrap_or(index..index + 1);
        index = range.end;
        units.push(range);
    }
    let unit_writes: Vec<Vec<&str>> = units
        .iter()
        .map(|range| range.clone().flat_map(|i| writes[i].iter().copied()).collect())
        .collect();
    let unit_reads: Vec<Vec<&str>> = units
        .iter()
        .map(|range| range.clone().flat_map(|i| reads[i].iter().copied()).collect())
        .collect();
    compute_parallel_layers(&unit_writes, &unit_reads)
        .into_iter()
        .map(|layer| layer.into_iter().map(|unit| units[unit].clone()).collect())
        .collect()
}

impl RenderGraph {
    /// Whether the graph records independent units on worker threads: enabled,
    /// every pass allows it, and at least one layer has units to overlap.
    pub(crate) fn parallel_recording_active(&self) -> bool {
        self.parallel_recording
            && !cfg!(target_arch = "wasm32")
            && self.parallel_units.iter().any(|layer| layer.len() > 1)
            && self.passes.iter().all(|pass| pass.supports_parallel_recording())
    }

    /// Recomputes the recording units and their layers from the current
    /// chains. A fused chain is recorded as one unit, so chains and worker
    /// recording coexist. With worker recording switched off, the serial
    /// executor keeps its historical behaviour of recording would-be chain
    /// members as ordinary standalone passes whenever the graph has
    /// independent work.
    pub(crate) fn finish_chain_detection(&mut self) {
        if !self.parallel_recording && self.parallel_layers.iter().any(|layer| layer.len() > 1) {
            self.subpass_chains.clear();
        }
        let units = {
            let (writes, reads, _) = self.chain_read_write_sets();
            compute_unit_layers(&writes, &reads, &self.subpass_chains)
        };
        let mut sequential: Vec<std::ops::Range<usize>> = units.iter().flatten().cloned().collect();
        sequential.sort_by_key(|range| range.start);
        self.sequential_units = sequential.into_iter().map(|range| vec![range]).collect();
        self.parallel_units = units;
        // Units changed, so cached recordings (keyed by unit) are stale.
        self.reset_recording_cache();
    }

    /// Detect chains of adjacent passes where each writes a resource the next
    /// reads. These could be fused into a single render pass with `next_subpass()`
    /// to keep inter-pass data in tile memory.
    pub(crate) fn detect_subpass_chains(&mut self) {
        let (writes_set, reads_set, _transparent) = self.chain_read_write_sets();
        let len = self.passes.len();
        let no_transparent = vec![false; len];
        let dummy_signature: Vec<Option<Vec<usize>>> = vec![Some(vec![0]); len];
        self.subpass_chains =
            compute_chains(&writes_set, &reads_set, &no_transparent, &dummy_signature);
        self.finish_chain_detection();
    }

    /// Same as `detect_subpass_chains`, but `attachments[i]` gives the exact set
    /// of texture views (as a lock-time `render_pass_descriptor` probe) each
    /// pass renders into — `None` if it's compute-only.
    pub(crate) fn detect_subpass_chains_probed(&mut self, attachments: &[Option<Vec<usize>>]) {
        let (writes_set, reads_set, transparent) = self.chain_read_write_sets();
        self.subpass_chains = compute_chains(&writes_set, &reads_set, &transparent, attachments);
        self.finish_chain_detection();
    }

    pub(crate) fn chain_read_write_sets(&self) -> (Vec<Vec<&str>>, Vec<Vec<&str>>, Vec<bool>) {
        let mut writes_set: Vec<Vec<&str>> = Vec::with_capacity(self.passes.len());
        let mut reads_set: Vec<Vec<&str>> = Vec::with_capacity(self.passes.len());
        let mut transparent: Vec<bool> = Vec::with_capacity(self.passes.len());
        for pass in self.passes.iter() {
            let mut w: Vec<&str> = pass.writes().to_vec();
            let mut r: Vec<&str> = pass.reads().to_vec();
            let mut builder = crate::graph::ResourceBuilder::new();
            pass.declare_resources(&mut builder);
            for d in builder.declarations() {
                match d.access {
                    crate::graph::ResourceAccess::Read => {
                        if !r.contains(&d.name) {
                            r.push(d.name);
                        }
                    }
                    crate::graph::ResourceAccess::Write => {
                        if !w.contains(&d.name) {
                            w.push(d.name);
                        }
                    }
                }
            }
            writes_set.push(w);
            reads_set.push(r);
            transparent.push(pass.chain_transparent());
        }
        (writes_set, reads_set, transparent)
    }
}

#[cfg(test)]
mod chain_tests {
    use super::{compute_chains, compute_parallel_layers, compute_unit_layers};

    fn sig(ids: &[usize]) -> Option<Vec<usize>> {
        Some(ids.to_vec())
    }
    fn none() -> Option<Vec<usize>> {
        None
    }

    #[test]
    fn adjacent_pair_fuses() {
        let writes = vec![vec!["a"], vec!["b"]];
        let reads = vec![vec![], vec!["a"]];
        let transparent = vec![false, false];
        let attachments = vec![sig(&[1]), sig(&[1])];
        assert_eq!(
            compute_chains(&writes, &reads, &transparent, &attachments),
            vec![0..2]
        );
    }

    #[test]
    fn no_dependency_means_no_chain() {
        let writes = vec![vec!["a"], vec!["b"]];
        let reads = vec![vec![], vec!["c"]];
        let transparent = vec![false, false];
        let attachments = vec![sig(&[1]), sig(&[1])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn single_transparent_gap_is_bridged() {
        let writes = vec![vec!["a"], vec![], vec![]];
        let reads = vec![vec![], vec![], vec!["a"]];
        let transparent = vec![false, true, false];
        let attachments = vec![sig(&[1]), none(), sig(&[1])];
        assert_eq!(
            compute_chains(&writes, &reads, &transparent, &attachments),
            vec![0..3]
        );
    }

    #[test]
    fn consecutive_transparent_gaps_are_bridged() {
        let writes = vec![vec!["a"], vec![], vec![], vec![]];
        let reads = vec![vec![], vec![], vec![], vec!["a"]];
        let transparent = vec![false, true, true, false];
        let attachments = vec![sig(&[1]), none(), none(), sig(&[1])];
        assert_eq!(
            compute_chains(&writes, &reads, &transparent, &attachments),
            vec![0..4]
        );
    }

    #[test]
    fn transparent_gap_without_real_dependency_forms_no_chain() {
        let writes = vec![vec!["a"], vec![], vec![]];
        let reads = vec![vec![], vec![], vec!["b"]];
        let transparent = vec![false, true, false];
        let attachments = vec![sig(&[1]), none(), sig(&[1])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn transparent_pass_never_starts_a_chain() {
        let writes = vec![vec![], vec!["a"]];
        let reads = vec![vec![], vec![]];
        let transparent = vec![true, false];
        let attachments = vec![none(), sig(&[1])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn trailing_transparent_pass_with_nothing_after_breaks_chain_cleanly() {
        let writes = vec![vec!["a"], vec![]];
        let reads = vec![vec![], vec![]];
        let transparent = vec![false, true];
        let attachments = vec![sig(&[1]), none()];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn non_real_pass_never_starts_or_ends_a_bridge_even_if_it_declares_matching_io() {
        let writes = vec![vec!["a"], vec![], vec![]];
        let reads = vec![vec![], vec![], vec!["a"]];
        let transparent = vec![false, true, false];
        let attachments = vec![none(), none(), sig(&[1])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn non_real_pass_in_the_middle_of_a_would_be_bridge_blocks_it() {
        let writes = vec![vec!["a"], vec![], vec![]];
        let reads = vec![vec![], vec![], vec!["a"]];
        let transparent = vec![false, false, false];
        let attachments = vec![sig(&[1]), none(), sig(&[1])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn differing_attachments_block_fusion_even_with_matching_reads_and_writes() {
        let writes = vec![vec!["color_output"], vec![]];
        let reads = vec![vec![], vec!["color_output"]];
        let transparent = vec![false, false];
        let attachments = vec![sig(&[1, 2, 3, 4, 5]), sig(&[99])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn matching_attachments_and_dependency_still_fuse() {
        let writes = vec![vec!["color_output"], vec![]];
        let reads = vec![vec![], vec!["color_output"]];
        let transparent = vec![false, false];
        let attachments = vec![sig(&[1, 2, 3, 4, 5]), sig(&[1, 2, 3, 4, 5])];
        assert_eq!(
            compute_chains(&writes, &reads, &transparent, &attachments),
            vec![0..2]
        );
    }

    #[test]
    fn independent_passes_share_a_parallel_layer() {
        let writes = vec![vec!["a"], vec!["b"], vec!["c"]];
        let reads = vec![vec![], vec![], vec!["a"]];
        assert_eq!(
            compute_parallel_layers(&writes, &reads),
            vec![vec![0, 1], vec![2]]
        );
    }

    #[test]
    fn dependency_chain_gets_strictly_ordered_layers() {
        let writes = vec![vec!["a"], vec!["b"], vec!["c"]];
        let reads = vec![vec![], vec!["a"], vec!["b"]];
        assert_eq!(
            compute_parallel_layers(&writes, &reads),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn read_before_write_gets_strictly_ordered_layers() {
        let writes = vec![vec![], vec!["history"]];
        let reads = vec![vec!["history"], vec![]];

        assert_eq!(
            compute_parallel_layers(&writes, &reads),
            vec![vec![0], vec![1]],
            "a later overwrite must not race an earlier read"
        );
    }

    #[test]
    fn independent_work_disables_fusion_to_keep_worker_recording_valid() {
        let writes = vec![vec!["a"], vec!["b"], vec!["c"]];
        let reads = vec![vec![], vec!["a"], vec![]];
        let layers = compute_parallel_layers(&writes, &reads);
        assert_eq!(layers, vec![vec![0, 2], vec![1]]);
        assert!(layers.iter().any(|layer| layer.len() > 1));
    }

    #[test]
    fn chain_labels_join_names_and_are_interned() {
        let a = super::chain_label(&["GBufferPass", "PortalInstancePass"]);
        assert_eq!(a, "GBufferPass+PortalInstancePass");
        // Rebuilding the graph asks again: same string, nothing new leaked.
        let b = super::chain_label(&["GBufferPass", "PortalInstancePass"]);
        assert!(std::ptr::eq(a, b));
        assert_ne!(super::chain_label(&["GBufferPass"]), a);
    }

    #[test]
    fn a_fused_chain_is_one_unit_in_one_layer() {
        // 0 -> 1 chained; 2 is independent of both; 3 reads what 1 wrote.
        let writes = vec![vec!["a"], vec!["b"], vec!["c"], vec![]];
        let reads = vec![vec![], vec!["a"], vec![], vec!["b"]];
        let layers = compute_unit_layers(&writes, &reads, &[0..2]);
        assert_eq!(layers, vec![vec![0..2, 2..3], vec![3..4]]);
    }

    #[test]
    fn without_chains_units_match_passes() {
        let writes = vec![vec!["a"], vec!["b"], vec!["c"]];
        let reads = vec![vec![], vec!["a"], vec![]];
        let layers = compute_unit_layers(&writes, &reads, &[]);
        assert_eq!(layers, vec![vec![0..1, 2..3], vec![1..2]]);
    }
}
