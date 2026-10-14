//! SceneDB-owned lens records: one [`LensComponent`] per view, and the
//! [`LensElementComponent`] rows it stacks in front of the sensor.
//!
//! A view's lens is an ordered stack of optical elements reduced per pixel
//! to four quantities: a ray map (sensor uv -> scene uv, per wavelength), a
//! focus field `s(uv)`, a pupil function and a transmission. Phase 1 resolves
//! the focus field only; the other three are identity. The lens is owned by
//! the camera: post-process volumes never blend lens settings.
//!
//! Both rows are packed GPU components read by [`crate::LensPass`], matched
//! to a view like `CameraPostProcessComponent`: the first enabled
//! [`LensComponent`] whose `view_id` equals the camera's (u32 bits of
//! `jitter_frame.w`) wins; with none the lens is off and the image passes
//! through untouched.
use pulsar_scenedb_derive::SceneStore;

/// Most elements one lens stacks; rows past this are ignored.
pub const MAX_LENS_ELEMENTS: u32 = 32;

/// `focus_mode`: the lens stays at `focus_distance_m`.
pub const FOCUS_MANUAL: u32 = 0;
/// `focus_mode`: the lens refocuses on the depth around `af_point`, easing
/// at `af_speed` per second.
pub const FOCUS_AUTO_POINT: u32 = 1;

/// Element kinds (`LensElementComponent::kind`).
pub mod element_kind {
    /// Inert row.
    pub const NONE: u32 = 0;
    /// Adds `params[0]` dioptres to the focus inside its region:
    /// `1/s' = 1/s + sum(D)`. `params[1]` darkens the seam (0..1) and
    /// `params[2]` is the filter's distance in front of the lens in metres
    /// (<= 0: 0.08), which sets the physical seam width.
    pub const DIOPTER: u32 = 1;
    /// Lensbaby-style sweet spot: focus is normal inside the region and
    /// `params[0]` pixels of extra defocus apply outside it.
    pub const SWEET_SPOT: u32 = 2;
}

/// Region kinds (`LensElementComponent::region_kind`). Regions live in
/// image-height units centred on the frame: `q = (uv - 0.5) * (aspect, 1)`,
/// y down.
pub mod region_kind {
    /// The whole frame.
    pub const FULL_FRAME: u32 = 0;
    /// Covers `dot(q, (cos a, sin a)) > offset`. `region = [a, offset]`.
    pub const HALF_PLANE: u32 = 1;
    /// `region = [cx, cy, rx, ry, rotation]`.
    pub const ELLIPSE: u32 = 2;
    /// Convex polygon of `region_count` (3..=8) vertices,
    /// `region = [x0, y0, x1, y1, ...]`, either winding.
    pub const POLYGON: u32 = 3;
    /// Annulus `r0 <= |q - c| <= r1`. `region = [r0, r1, cx, cy]`.
    pub const RADIAL_BAND: u32 = 4;
    /// Reserved for a mask-texture channel; covers nothing yet.
    pub const MASK: u32 = 5;
}

/// A camera's lens. One row per view; see the module doc.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "camera_lenses")]
pub struct LensComponent {
    #[gpu]
    pub view_id: u32,
    #[gpu]
    pub enabled: u32,
    /// [`FOCUS_MANUAL`] or [`FOCUS_AUTO_POINT`].
    #[gpu]
    pub focus_mode: u32,
    /// 0 economical, 1 high (more gather taps).
    #[gpu]
    pub quality: u32,
    #[gpu]
    pub focal_length_mm: f32,
    #[gpu]
    pub f_number: f32,
    /// Transmission stop; carried for exposure, not used by the pass yet.
    #[gpu]
    pub t_stop: f32,
    #[gpu]
    pub sensor_width_mm: f32,
    /// Zero derives it from the width at 3:2.
    #[gpu]
    pub sensor_height_mm: f32,
    #[gpu]
    pub focus_distance_m: f32,
    /// Autofocus point in uv (y down).
    #[gpu]
    pub af_u: f32,
    #[gpu]
    pub af_v: f32,
    /// Autofocus ease rate in 1/s (<= 0: instant).
    #[gpu]
    pub af_speed: f32,
    /// Largest circle of confusion diameter in pixels (<= 0: 32).
    #[gpu]
    pub max_coc_px: f32,
    /// First row of this lens's elements in the `lens_elements` buffer.
    #[gpu]
    pub element_first: u32,
    #[gpu]
    pub element_count: u32,
    /// Scheimpflug: focal plane tilt in radians about an axis in the image
    /// plane at `tilt_axis` radians from +x.
    #[gpu]
    pub tilt_angle: f32,
    #[gpu]
    pub tilt_axis: f32,
    /// Petzval field curvature in dioptres at the frame corner; the focus
    /// shifts by `field_curvature * r^2`.
    #[gpu]
    pub field_curvature: f32,
    /// Iris blade count (0 = circular); shapes the flare pupil.
    #[gpu]
    pub aperture_blades: u32,
    /// Iris rotation in radians.
    #[gpu]
    pub aperture_rotation: f32,
    /// Autofocus sampling radius in uv (<= 0: 0.02).
    #[gpu]
    pub af_radius: f32,
    #[gpu]
    pub _pad0: f32,
    #[gpu]
    pub _pad1: f32,
}
const _: () = assert!(std::mem::size_of::<LensComponent>() == 96);
const _: () = assert!(std::mem::offset_of!(LensComponent, focal_length_mm) == 16);
const _: () = assert!(std::mem::offset_of!(LensComponent, element_first) == 56);
const _: () = assert!(std::mem::offset_of!(LensComponent, tilt_angle) == 64);
const _: () = assert!(std::mem::offset_of!(LensComponent, aperture_rotation) == 80);

impl LensComponent {
    /// A 50 mm f/2.8 lens on a 36 x 24 mm sensor focused at 3 m.
    pub fn new(view_id: u32) -> Self {
        Self {
            view_id,
            enabled: 1,
            focus_mode: FOCUS_MANUAL,
            quality: 0,
            focal_length_mm: 50.0,
            f_number: 2.8,
            t_stop: 2.8,
            sensor_width_mm: 36.0,
            sensor_height_mm: 24.0,
            focus_distance_m: 3.0,
            af_u: 0.5,
            af_v: 0.5,
            af_speed: 0.0,
            max_coc_px: 0.0,
            element_first: 0,
            element_count: 0,
            tilt_angle: 0.0,
            tilt_axis: 0.0,
            field_curvature: 0.0,
            aperture_blades: 6,
            aperture_rotation: 0.0,
            af_radius: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// One optical element in front of the sensor. Rows are indexed from
/// [`LensComponent::element_first`]; kind [`element_kind::NONE`] rows are
/// skipped, so a gap in the buffer is inert.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "lens_elements")]
pub struct LensElementComponent {
    /// [`element_kind`].
    #[gpu]
    pub kind: u32,
    /// [`region_kind`].
    #[gpu]
    pub region_kind: u32,
    /// Polygon vertex count.
    #[gpu]
    pub region_count: u32,
    #[gpu]
    pub flags: u32,
    /// Width of the region's soft edge in image-height units. Negative:
    /// derived from the aperture (see [`element_kind::DIOPTER`]).
    #[gpu]
    pub feather: f32,
    #[gpu]
    pub _pad0: f32,
    #[gpu]
    pub _pad1: f32,
    #[gpu]
    pub _pad2: f32,
    /// Kind-specific parameters.
    #[gpu]
    pub params: [f32; 16],
    /// Region-kind-specific geometry.
    #[gpu]
    pub region: [f32; 16],
}
const _: () = assert!(std::mem::size_of::<LensElementComponent>() == 160);
const _: () = assert!(std::mem::offset_of!(LensElementComponent, params) == 32);
const _: () = assert!(std::mem::offset_of!(LensElementComponent, region) == 96);

/// Where an element acts, for building [`LensElementComponent`] rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LensRegion<'a> {
    FullFrame,
    HalfPlane { angle: f32, offset: f32 },
    Ellipse { center: [f32; 2], radii: [f32; 2], rotation: f32 },
    /// Convex, 3 to 8 vertices.
    Polygon(&'a [[f32; 2]]),
    RadialBand { r0: f32, r1: f32, center: [f32; 2] },
    Mask,
}

impl LensElementComponent {
    fn with_region(kind: u32, params: [f32; 16], region: LensRegion<'_>) -> Self {
        let mut row: Self = bytemuck::Zeroable::zeroed();
        row.kind = kind;
        row.params = params;
        row.feather = -1.0;
        let mut set = |region_kind: u32, values: &[f32]| {
            row.region_kind = region_kind;
            row.region[..values.len()].copy_from_slice(values);
        };
        match region {
            LensRegion::FullFrame => set(region_kind::FULL_FRAME, &[]),
            LensRegion::HalfPlane { angle, offset } => {
                set(region_kind::HALF_PLANE, &[angle, offset])
            }
            LensRegion::Ellipse { center, radii, rotation } => set(
                region_kind::ELLIPSE,
                &[center[0], center[1], radii[0], radii[1], rotation],
            ),
            LensRegion::Polygon(vertices) => {
                let vertices = &vertices[..vertices.len().min(8)];
                let flat: Vec<f32> = vertices.iter().flatten().copied().collect();
                set(region_kind::POLYGON, &flat);
                row.region_count = vertices.len() as u32;
            }
            LensRegion::RadialBand { r0, r1, center } => {
                set(region_kind::RADIAL_BAND, &[r0, r1, center[0], center[1]])
            }
            LensRegion::Mask => set(region_kind::MASK, &[]),
        }
        row
    }

    /// A split diopter adding `power_dpt` dioptres inside `region`.
    pub fn diopter(power_dpt: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = power_dpt;
        Self::with_region(element_kind::DIOPTER, params, region)
    }

    /// A sweet spot: `extra_blur_px` of extra defocus outside `region`.
    pub fn sweet_spot(extra_blur_px: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = extra_blur_px;
        Self::with_region(element_kind::SWEET_SPOT, params, region)
    }

    /// Fixes the soft edge width instead of deriving it from the aperture.
    pub fn with_feather(mut self, feather: f32) -> Self {
        self.feather = feather;
        self
    }

    /// Darkens a diopter's seam by `amount` (0..1) at its centre.
    pub fn with_seam_darkening(mut self, amount: f32) -> Self {
        self.params[1] = amount;
        self
    }

    /// Sets how far in front of the lens a diopter sits, in metres.
    pub fn with_mount_distance(mut self, metres: f32) -> Self {
        self.params[2] = metres;
        self
    }
}

// The pinned SceneDB dispatches removals/despawns through this release
// registry, but only generates registrations for variable-length components.
// Packed rows need an explicit tombstone: the shader scans capacity, and a
// stale row would keep a deleted lens or element alive.
pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::world_mirror::VarLenReleaseRegistration {
        component_id: pulsar_scenedb::component::component_id::<LensComponent>,
        release: clear_lens_row,
    }
}
fn clear_lens_row(mirror: &pulsar_scenedb::gpu::GpuMirrorHandle, row: u32) {
    let zero: LensComponent = bytemuck::Zeroable::zeroed();
    mirror.store().mark_gpu_row_dirty(
        pulsar_scenedb::component::component_id::<__ScenedbGpuPacked_LensComponent>(),
        row, bytemuck::bytes_of(&zero),
    );
}

pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::world_mirror::VarLenReleaseRegistration {
        component_id: pulsar_scenedb::component::component_id::<LensElementComponent>,
        release: clear_element_row,
    }
}
fn clear_element_row(mirror: &pulsar_scenedb::gpu::GpuMirrorHandle, row: u32) {
    let zero: LensElementComponent = bytemuck::Zeroable::zeroed();
    mirror.store().mark_gpu_row_dirty(
        pulsar_scenedb::component::component_id::<__ScenedbGpuPacked_LensElementComponent>(),
        row, bytemuck::bytes_of(&zero),
    );
}
