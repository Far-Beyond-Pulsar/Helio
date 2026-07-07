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
}

/// An action to perform on FrameResources before a pass executes.
pub(crate) enum PrePassAction {
    Route { name: String, view: wgpu::TextureView },
    Gbuffer {
        albedo: wgpu::TextureView,
        normal: wgpu::TextureView,
        orm: wgpu::TextureView,
        emissive: wgpu::TextureView,
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
            if j >= len { break; }
            let same_attachments = match (&attachments[i], &attachments[j]) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            if !same_attachments { break; }
            let can_fuse = writes[i].iter().any(|w| reads[j].contains(w));
            if !can_fuse { break; }
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

impl RenderGraph {
    /// Detect chains of adjacent passes where each writes a resource the next
    /// reads. These could be fused into a single render pass with `next_subpass()`
    /// to keep inter-pass data in tile memory.
    pub(crate) fn detect_subpass_chains(&mut self) {
        let (writes_set, reads_set, _transparent) = self.chain_read_write_sets();
        let len = self.passes.len();
        let no_transparent = vec![false; len];
        let dummy_signature: Vec<Option<Vec<usize>>> = vec![Some(vec![0]); len];
        self.subpass_chains = compute_chains(&writes_set, &reads_set, &no_transparent, &dummy_signature);
    }

    /// Same as `detect_subpass_chains`, but `attachments[i]` gives the exact set
    /// of texture views (as a lock-time `render_pass_descriptor` probe) each
    /// pass renders into — `None` if it's compute-only.
    pub(crate) fn detect_subpass_chains_probed(&mut self, attachments: &[Option<Vec<usize>>]) {
        let (writes_set, reads_set, transparent) = self.chain_read_write_sets();
        self.subpass_chains = compute_chains(&writes_set, &reads_set, &transparent, attachments);
    }

    fn chain_read_write_sets(&self) -> (Vec<Vec<&str>>, Vec<Vec<&str>>, Vec<bool>) {
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
                    crate::graph::ResourceAccess::Read => { if !r.contains(&d.name) { r.push(d.name); } }
                    crate::graph::ResourceAccess::Write => { if !w.contains(&d.name) { w.push(d.name); } }
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
    use super::compute_chains;

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
        assert_eq!(compute_chains(&writes, &reads, &transparent, &attachments), vec![0..2]);
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
        assert_eq!(compute_chains(&writes, &reads, &transparent, &attachments), vec![0..3]);
    }

    #[test]
    fn consecutive_transparent_gaps_are_bridged() {
        let writes = vec![vec!["a"], vec![], vec![], vec![]];
        let reads = vec![vec![], vec![], vec![], vec!["a"]];
        let transparent = vec![false, true, true, false];
        let attachments = vec![sig(&[1]), none(), none(), sig(&[1])];
        assert_eq!(compute_chains(&writes, &reads, &transparent, &attachments), vec![0..4]);
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
        let writes = vec![vec!["gbuffer"], vec![]];
        let reads = vec![vec![], vec!["gbuffer"]];
        let transparent = vec![false, false];
        let attachments = vec![sig(&[1, 2, 3, 4, 5]), sig(&[99])];
        assert!(compute_chains(&writes, &reads, &transparent, &attachments).is_empty());
    }

    #[test]
    fn matching_attachments_and_dependency_still_fuse() {
        let writes = vec![vec!["gbuffer"], vec![]];
        let reads = vec![vec![], vec!["gbuffer"]];
        let transparent = vec![false, false];
        let attachments = vec![sig(&[1, 2, 3, 4, 5]), sig(&[1, 2, 3, 4, 5])];
        assert_eq!(compute_chains(&writes, &reads, &transparent, &attachments), vec![0..2]);
    }
}
