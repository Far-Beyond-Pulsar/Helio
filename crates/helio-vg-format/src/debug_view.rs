//! The virtual-geometry debug views: stable ids, names, descriptions and the
//! shared WGSL constants (plan `.agents/plans/virtual-geometry.md` section 12,
//! "Debug views").
//!
//! Every view is a renderer-wide debug mode (`Renderer::set_debug_mode`) that
//! `helio-pass-vg-shade` advertises through `RenderPass::debug_views` under the
//! category [`CATEGORY`]. The ids are stable (never renumbered): `20`/`21` are
//! the original triangle/cluster views, the rest start at [`FIRST_ID`]. The
//! deferred lighting pass treats ids in `100..128` (and `20`/`21`) as "show the
//! G-buffer albedo unlit", which is where the shade pass writes the colours.

/// Category shown by the editor's debug-view menu and the demo's cycler.
pub const CATEGORY: &str = "Virtual Geometry";
/// First id of the views added after the original two.
pub const FIRST_ID: u32 = 100;
/// One past the last id the lighting pass reserves for albedo-passthrough
/// debug views (`FIRST_ID..END_ID`).
pub const END_ID: u32 = 128;
/// Environment variable naming a view (id or [`VgDebugView::key`]) to select
/// at renderer creation; meant for tests and headless captures.
pub const ENV_VAR: &str = "VG_DEBUG_VIEW";

/// One virtual-geometry debug view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum VgDebugView {
    /// Hashed colour per physical triangle.
    TriangleId = 20,
    /// Hashed colour per physical cluster.
    ClusterId = 21,
    /// Cluster coarseness (error-derived DAG depth proxy), heat ramp.
    LodLevel = 100,
    /// Selected cluster error relative to the asset extent.
    DetailRatio = 101,
    /// Projected triangle area in pixels.
    TriangleSize = 102,
    /// Raster tier that produced the pixel.
    RasterTier = 103,
    /// Software-raster overdraw per 16x16 tile.
    SwOverdraw = 104,
    /// Software triangles overlapping each tile (the dense-tile trigger).
    DenseOverlap = 105,
    /// The dense-tile mask.
    DenseMask = 106,
    /// Page residency / streaming pressure.
    PageResidency = 107,
    /// Which cull stage the cluster just survived (phase, frustum, cone, error margins).
    CullSurvival = 108,
    /// Projected cluster error divided by tau.
    ErrorVsTau = 109,
    /// Hashed colour per instance.
    Instance = 110,
    /// Resolved barycentrics.
    Barycentrics = 111,
    /// Normal decoded from the vis-buffer triangle.
    VisNormal = 112,
    /// Material bin.
    MaterialBin = 113,
    /// Which pixels the virtual-geometry shade wrote.
    VgMask = 114,
}

/// Every view in menu order.
pub const ALL: [VgDebugView; 17] = [
    VgDebugView::TriangleId,
    VgDebugView::ClusterId,
    VgDebugView::LodLevel,
    VgDebugView::DetailRatio,
    VgDebugView::TriangleSize,
    VgDebugView::RasterTier,
    VgDebugView::SwOverdraw,
    VgDebugView::DenseOverlap,
    VgDebugView::DenseMask,
    VgDebugView::PageResidency,
    VgDebugView::CullSurvival,
    VgDebugView::ErrorVsTau,
    VgDebugView::Instance,
    VgDebugView::Barycentrics,
    VgDebugView::VisNormal,
    VgDebugView::MaterialBin,
    VgDebugView::VgMask,
];

impl VgDebugView {
    /// The renderer debug mode of this view.
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// The view for a renderer debug mode, if it is one of ours.
    pub fn from_id(id: u32) -> Option<Self> {
        ALL.iter().copied().find(|view| view.id() == id)
    }

    /// Menu name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TriangleId => "Triangle ID",
            Self::ClusterId => "Cluster ID",
            Self::LodLevel => "LOD Level",
            Self::DetailRatio => "Detail vs Root",
            Self::TriangleSize => "Triangle Size",
            Self::RasterTier => "Raster Tier",
            Self::SwOverdraw => "SW Overdraw",
            Self::DenseOverlap => "Dense Tile Overlap",
            Self::DenseMask => "Dense Tile Mask",
            Self::PageResidency => "Page Residency",
            Self::CullSurvival => "Culling Survival",
            Self::ErrorVsTau => "Error vs Tau",
            Self::Instance => "Instance ID",
            Self::Barycentrics => "Barycentrics",
            Self::VisNormal => "Vis Normal",
            Self::MaterialBin => "Material Bin",
            Self::VgMask => "Classic vs VG Mask",
        }
    }

    /// Short lowercase key accepted by [`parse`].
    pub const fn key(self) -> &'static str {
        match self {
            Self::TriangleId => "triangle",
            Self::ClusterId => "cluster",
            Self::LodLevel => "lod",
            Self::DetailRatio => "detail",
            Self::TriangleSize => "tri_size",
            Self::RasterTier => "tier",
            Self::SwOverdraw => "sw_overdraw",
            Self::DenseOverlap => "dense_overlap",
            Self::DenseMask => "dense_mask",
            Self::PageResidency => "residency",
            Self::CullSurvival => "cull",
            Self::ErrorVsTau => "error",
            Self::Instance => "instance",
            Self::Barycentrics => "bary",
            Self::VisNormal => "normal",
            Self::MaterialBin => "material",
            Self::VgMask => "mask",
        }
    }

    /// One-line description.
    pub const fn description(self) -> &'static str {
        match self {
            Self::TriangleId => "Hashed colour per physical triangle of the selected cut",
            Self::ClusterId => "Hashed colour per physical cluster of the selected cut",
            Self::LodLevel => {
                "Coarseness of the selected cluster (error-derived DAG depth): blue finest, red coarsest"
            }
            Self::DetailRatio => {
                "Selected cluster error relative to its asset's extent: blue near full detail, red near the root"
            }
            Self::TriangleSize => {
                "Projected triangle area in pixels: red sub-pixel, green a few pixels, blue large"
            }
            Self::RasterTier => {
                "Raster tier: blue hardware (shade by cluster size), green micro-triangle software, orange dense, magenta ray traced"
            }
            Self::SwOverdraw => {
                "Software-raster depth complexity per 16x16 tile (pixel hits / 256); drives dense-tile selection"
            }
            Self::DenseOverlap => {
                "Software triangles overlapping each 16x16 tile; drives dense-tile selection"
            }
            Self::DenseMask => "Tiles classified dense this frame (orange) over the software-tile coverage (grey)",
            Self::PageResidency => {
                "Streaming: green refined, teal leaf, red fallback parent awaiting its child page (brighter = more urgent)"
            }
            Self::CullSurvival => {
                "Cull survivors: green phase 0, yellow phase-1 recovery after occlusion; cyan frustum edge, magenta backface edge, red near the error limit"
            }
            Self::ErrorVsTau => {
                "Projected error divided by tau: blue far from refining, red about to refine, magenta above tau (fallback)"
            }
            Self::Instance => "Hashed colour per virtual-geometry instance",
            Self::Barycentrics => "Resolved perspective-correct barycentrics as RGB",
            Self::VisNormal => "World-space normal decoded from the vis-buffer triangle",
            Self::MaterialBin => "Material bin the pixel is shaded in",
            Self::VgMask => "Green pixels were shaded by virtual geometry, grey by the classic path",
        }
    }
}

/// `VG_DEBUG_VIEW` style text: a decimal id (`"100"`) or a [`VgDebugView::key`].
pub fn parse(text: &str) -> Option<VgDebugView> {
    let text = text.trim();
    if let Ok(id) = text.parse::<u32>() {
        return VgDebugView::from_id(id);
    }
    let lower = text.to_ascii_lowercase();
    ALL.iter().copied().find(|view| view.key() == lower)
}

/// The view named by [`ENV_VAR`], if set and valid.
pub fn from_env() -> Option<VgDebugView> {
    parse(&std::env::var(ENV_VAR).ok()?)
}

/// `VGDBG_*` WGSL constants generated from the Rust ids so the two cannot drift.
pub fn wgsl_constants() -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for view in ALL {
        let name = view.key().to_ascii_uppercase();
        let _ = writeln!(s, "const VGDBG_{name}: u32 = {}u;", view.id());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_stable_and_in_range() {
        let mut ids: Vec<u32> = ALL.iter().map(|v| v.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ALL.len());
        for view in ALL {
            assert!(
                view.id() == 20 || view.id() == 21 || (FIRST_ID..END_ID).contains(&view.id()),
                "{view:?}"
            );
            assert_eq!(VgDebugView::from_id(view.id()), Some(view));
            assert_eq!(parse(view.key()), Some(view));
            assert_eq!(parse(&view.id().to_string()), Some(view));
            assert!(!view.name().is_empty());
        }
        // The original F3/F4 modes keep their ids.
        assert_eq!(VgDebugView::TriangleId.id(), 20);
        assert_eq!(VgDebugView::ClusterId.id(), 21);
        assert_eq!(parse("nonsense"), None);
        assert_eq!(parse("7"), None);
    }
}
