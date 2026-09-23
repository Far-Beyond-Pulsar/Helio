//! Focused GPU timing views for expensive cathedral-rendering stages.
//!
//! The render graph already records timestamps for every pass.  This module
//! deliberately stays on the diagnostic side of the boundary: it only
//! classifies the completed samples and never adds work to the graph or
//! changes pass scheduling.

use super::RenderTimingSnapshot;

/// A coarse bucket used by the cathedral profiling view.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FocusedTimingGroup {
    /// Hierarchical light finding and sampling.
    Hlfs,
    /// Ray-traced or screen-space reflection trace and composite passes.
    RtReflections,
    /// Shadow matrix, culling, dirty tracking, and atlas rendering.
    Shadows,
    /// Froxel injection and volumetric integration.
    VolumetricFog,
    /// Temporal/spatial reconstruction and denoising stages.
    Denoising,
}

impl FocusedTimingGroup {
    pub const ALL: [Self; 5] = [
        Self::Hlfs,
        Self::RtReflections,
        Self::Shadows,
        Self::VolumetricFog,
        Self::Denoising,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Hlfs => "HLFS",
            Self::RtReflections => "RT reflections",
            Self::Shadows => "Shadows",
            Self::VolumetricFog => "Volumetric fog",
            Self::Denoising => "Denoising",
        }
    }

    /// Classifies the stable graph pass/stage labels used by Helio.
    ///
    /// Internal HLFS stage labels are included as well. They are useful to
    /// callers that feed pass-local timestamp samples into this same report.
    pub fn classify(name: &'static str) -> Option<Self> {
        if name == "HLFS" || name.starts_with("HLFS ") {
            if name == "HLFS temporal filter"
                || name == "HLFS spatial filter"
                || name == "HLFS temporal denoising"
                || name == "HLFS spatial denoising"
            {
                return Some(Self::Denoising);
            }
            return Some(Self::Hlfs);
        }
        if name == "SsrPass"
            || name == "SsrCompositePass"
            || name == "RT Reflection"
            || name == "RT Reflection Composite"
        {
            return Some(Self::RtReflections);
        }
        if name == "VolumetricFogPass" {
            return Some(Self::VolumetricFog);
        }
        if name == "TSR" || name == "TAA" || name == "TemporalResolve" {
            return Some(Self::Denoising);
        }
        if name.starts_with("Shadow") {
            return Some(Self::Shadows);
        }
        None
    }
}

/// One measured pass in the focused view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocusedTiming {
    pub group: FocusedTimingGroup,
    pub name: &'static str,
    pub cpu_ms: Option<f32>,
    pub gpu_ms: Option<f32>,
}

/// A zero-allocation summary of the expensive renderer stages.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FocusedTimingReport {
    pub generation: u64,
    pub gpu_frame_index: Option<u64>,
    pub gpu_availability: super::GpuTimingAvailability,
    pub entries: Vec<FocusedTiming>,
}

impl FocusedTimingReport {
    /// Builds a report from the latest asynchronous profiler snapshot.
    pub fn from_snapshot(snapshot: &RenderTimingSnapshot) -> Self {
        let entries = snapshot
            .passes
            .iter()
            .filter_map(|pass| {
                FocusedTimingGroup::classify(pass.name).map(|group| FocusedTiming {
                    group,
                    name: pass.name,
                    cpu_ms: pass.cpu_ms,
                    gpu_ms: pass.gpu_ms,
                })
            })
            .collect();
        Self {
            generation: snapshot.generation,
            gpu_frame_index: snapshot.gpu_frame_index,
            gpu_availability: snapshot.gpu_availability,
            entries,
        }
    }

    /// Sums the available GPU samples for one group.
    pub fn gpu_ms(&self, group: FocusedTimingGroup) -> Option<f32> {
        let mut total = 0.0;
        let mut found = false;
        for entry in self.entries.iter().filter(|entry| entry.group == group) {
            if let Some(ms) = entry.gpu_ms {
                total += ms;
                found = true;
            }
        }
        found.then_some(total)
    }

    /// Sums the available CPU samples for one group.
    pub fn cpu_ms(&self, group: FocusedTimingGroup) -> Option<f32> {
        let mut total = 0.0;
        let mut found = false;
        for entry in self.entries.iter().filter(|entry| entry.group == group) {
            if let Some(ms) = entry.cpu_ms {
                total += ms;
                found = true;
            }
        }
        found.then_some(total)
    }

    /// Formats a compact line suitable for a diagnostic overlay or log.
    pub fn format_gpu_line(&self) -> String {
        FocusedTimingGroup::ALL
            .into_iter()
            .map(|group| match self.gpu_ms(group) {
                Some(ms) => format!("{}={ms:.2}ms", group.label()),
                None => format!("{}=n/a", group.label()),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}
