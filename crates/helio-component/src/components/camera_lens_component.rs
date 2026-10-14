//! A camera's lens, as authored in the editor.
//!
//! `CameraLensComponent` is the reflected face of Helio's physically based
//! lens (`helio_pass_lens`): optics (focal length, f-number, sensor, iris),
//! focus (manual or autofocus), the Scheimpflug tilt and field curvature of
//! the focal surface, and an ordered stack of optical elements (split
//! diopters, a Lensbaby sweet spot) with the image region each acts on.
//! Named presets ([`LENS_PRESETS`]) are data over the same fields.
//!
//! Like `CameraPostProcessComponent`, the authored value derives its GPU
//! rows on every insert, write, removal and replay, matched to a render view
//! by `view_id` (the editor viewport is view 0):
//!
//! - [`CameraLensSourceRow`] (`camera_lens_sources`), at the instance
//!   entity's row: the pass's [`LensComponent`] row, bit for bit. The
//!   environment join gates it (attached, instance enabled, owner current)
//!   and publishes it as `camera_lenses`, which the lens pass reads. The
//!   instance's enabled switch in the inspector therefore turns the lens off.
//! - `lens_elements`: a block of [`LENS_ELEMENT_SLOTS`] [`LensElementComponent`]
//!   rows per lens, from a compact allocator, because the pass addresses a
//!   lens's elements as one contiguous range (`element_first .. +
//!   element_count`) and the entity index of a component instance is sparse.
//!   The block is freed, and its rows zeroed, when the component leaves its
//!   object. A lens that is off has no elements to read.
//!
//! The lens is owned by the camera: no post-process volume blends it, and it
//! is not gated by the owner's visibility.
use std::collections::HashMap;
use std::sync::Mutex;

use engine_class_derive::{engine_class, register_world_component};
use helio_pass_lens::{
    element_kind, region_kind, LensComponent, LensElementComponent, LensRegion, FOCUS_AUTO_POINT,
    FOCUS_MANUAL, MAX_LENS_ELEMENTS,
};
use pulsar_reflection::{ReflectError, ReflectResult, Reflectable, pulsar_type};
use pulsar_scenedb::gpu::{GpuMirrorHandle, SceneGpuStore};
use pulsar_scenedb_derive::SceneStore;
use serde::{Deserialize, Deserializer, Serialize};

pub const CAMERA_LENS_CLASS_NAME: &str = "CameraLensComponent";

/// Most optical elements one authored lens stacks (the pass allows
/// [`MAX_LENS_ELEMENTS`]; this is the size of the block each lens owns in
/// the `lens_elements` buffer).
pub const LENS_ELEMENT_SLOTS: usize = 8;
const _: () = assert!(LENS_ELEMENT_SLOTS as u32 <= MAX_LENS_ELEMENTS);

/// The SceneDB buffers this component writes: the lens source rows and the
/// lens pass's element rows.
/// The authored lens's source rows, which the environment join publishes as
/// the `camera_lenses` the pass reads.
pub const CAMERA_LENS_SOURCES_BUFFER: &str = "camera_lens_sources";
pub const LENS_ELEMENTS_BUFFER: &str = "lens_elements";

/// Registers the lens GPU columns on the host's store; the lens stays off
/// without this.
pub fn register_lens_gpu_columns(store: &mut SceneGpuStore, device: &std::sync::Arc<wgpu::Device>) {
    CameraLensSourceRow::register_gpu_columns_growable(store, 4, device);
    LensElementComponent::register_gpu_columns_growable(
        store,
        4 * LENS_ELEMENT_SLOTS as u32,
        device,
    );
}

/// The packed GPU component a lens buffer holds, for a host that checks its
/// registered buffers' schemas; `None` for any other key.
pub fn lens_buffer_component_id(key: &str) -> Option<pulsar_scenedb::component::ComponentId> {
    match key {
        CAMERA_LENS_SOURCES_BUFFER => Some(CameraLensSourceRow::packed_gpu_component_id()),
        LENS_ELEMENTS_BUFFER => Some(LensElementComponent::packed_gpu_component_id()),
        _ => None,
    }
}

/// An enum that reads from its variant name or its index (the reflection
/// JSON form), as `CurveAlgorithm` does.
macro_rules! lens_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Reflectable)]
        pub enum $name {
            $($(#[$vmeta])* $variant),+
        }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(untagged)]
                enum Repr {
                    Name(String),
                    Index(u64),
                }
                let parsed = match Repr::deserialize(deserializer)? {
                    Repr::Name(name) => Self::ALL
                        .iter()
                        .copied()
                        .find(|v| format!("{v:?}").eq_ignore_ascii_case(&name)),
                    Repr::Index(index) => Self::ALL.get(index as usize).copied(),
                };
                parsed.ok_or_else(|| {
                    serde::de::Error::custom(concat!("unknown ", stringify!($name)))
                })
            }
        }
    };
}

/// A camera lens's source row: the pass's [`LensComponent`] row, bit for
/// bit (`enabled` 0 when the authored component is disabled). The
/// environment join places it into `camera_lenses`.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "camera_lens_sources")]
pub struct CameraLensSourceRow {
    /// The 96 bytes of the row, in six four-word groups (reflection
    /// registers only these array shapes).
    #[gpu]
    pub words0: [u32; 4],
    #[gpu]
    pub words1: [u32; 4],
    #[gpu]
    pub words2: [u32; 4],
    #[gpu]
    pub words3: [u32; 4],
    #[gpu]
    pub words4: [u32; 4],
    #[gpu]
    pub words5: [u32; 4],
}
const _: () = assert!(
    std::mem::size_of::<CameraLensSourceRow>() == std::mem::size_of::<LensComponent>()
);

impl CameraLensSourceRow {
    pub fn of(lens: &LensComponent) -> Self {
        bytemuck::pod_read_unaligned(bytemuck::bytes_of(lens))
    }
}

lens_enum! {
    /// How the lens finds its focus distance.
    LensFocusMode {
        /// Stays at the focus distance.
        #[default]
        Manual,
        /// Refocuses on the depth around the autofocus point.
        AutoPoint,
    }
}

lens_enum! {
    /// Gather quality of the defocus.
    LensGatherQuality {
        /// 24 taps.
        #[default]
        Economical,
        /// 56 taps.
        High,
    }
}

lens_enum! {
    /// What an optical element does.
    LensElementKind {
        /// Adds dioptres to the focus inside its region
        /// (`1/s' = 1/s + D`): a split diopter filter.
        #[default]
        Diopter,
        /// Lensbaby-style sweet spot: normal focus inside the region and
        /// extra defocus outside it.
        SweetSpot,
        /// The pupil function inside the region: iris shape, anamorphic
        /// squeeze, cat's eye, swirl, spherical aberration, onion rings,
        /// apodization, central obstruction (bokeh).
        Pupil,
        /// Longitudinal chromatic aberration inside the region: a focus
        /// shift in dioptres per colour channel.
        LongitudinalCa,
    }
}

lens_enum! {
    /// Where on the frame an element acts. Regions live in image-height
    /// units centred on the frame (x right, y down; the frame spans
    /// `-aspect/2 .. aspect/2` by `-0.5 .. 0.5`).
    LensRegionShape {
        /// The whole frame.
        FullFrame,
        /// The side of a line: `dot(p, (cos a, sin a)) > offset`.
        #[default]
        HalfPlane,
        /// An ellipse at the centre with radii and a rotation.
        Ellipse,
        /// A convex polygon of 3 or 4 vertices.
        Polygon,
        /// The ring between an inner and an outer radius.
        RadialBand,
    }
}

/// One optical element of a lens stack. Only the fields of its kind and
/// region shape are read.
#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[serde(default)]
pub struct LensElementProps {
    /// A disabled element stays in the stack but does nothing.
    #[property]
    pub enabled: bool,
    #[property]
    pub kind: LensElementKind,
    /// Diopter: added optical power in dioptres. Positive power focuses
    /// nearer: `1/s' = 1/s + D`.
    #[property(min = -8.0, max = 8.0, step = 0.05)]
    pub power_dpt: f32,
    /// Diopter: how dark the seam is at its centre (0..1).
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub seam_darkening: f32,
    /// Diopter: distance of the filter in front of the lens in metres; it
    /// sets how soft the seam is (0: 0.08 m).
    #[property(min = 0.0, max = 1.0, step = 0.005)]
    pub mount_distance_m: f32,
    /// Pupil: use the lens's own iris (blades and rotation) for the shape.
    #[property]
    pub inherit_iris: bool,
    /// Pupil: iris blades (0: circle, 3..16) and how curved their edges are
    /// (0 straight, 1 a circle); rotation in degrees.
    #[property(min = 0.0, max = 16.0, step = 1.0)]
    pub blades: u32,
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub blade_curvature: f32,
    #[property(min = -180.0, max = 180.0, step = 1.0)]
    pub iris_rotation_deg: f32,
    /// Pupil: f-number at and above which the blades are as curved as the
    /// curvature says; wide open the iris is round (0: off).
    #[property(min = 0.0, max = 32.0, step = 0.1)]
    pub round_until_f_number: f32,
    /// Pupil: anamorphic squeeze (1.33, 1.5, 1.8, 2.0): bokeh this many
    /// times taller than wide.
    #[property(min = 1.0, max = 3.0, step = 0.01)]
    pub squeeze: f32,
    /// Pupil: cat's eye strength (optical vignetting toward the corners)
    /// and the barrel radius clipping it (0: 1).
    #[property(min = 0.0, max = 2.0, step = 0.01)]
    pub cats_eye: f32,
    #[property(min = 0.0, max = 2.0, step = 0.01)]
    pub barrel_radius: f32,
    /// Pupil: swirl, a tangential stretch growing with the field radius.
    #[property(min = -2.0, max = 2.0, step = 0.01)]
    pub swirl: f32,
    /// Pupil: spherical aberration (positive: bright rim; negative: soft
    /// centre-weighted) and how far it flips in front of the focus (0..1).
    #[property(min = -1.0, max = 1.0, step = 0.01)]
    pub spherical: f32,
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub spherical_asymmetry: f32,
    /// Pupil: onion-ring ripple amplitude and ripples across the radius.
    #[property(min = 0.0, max = 1.0, step = 0.01)]
    pub onion_amplitude: f32,
    #[property(min = 0.0, max = 32.0, step = 0.5)]
    pub onion_rings: f32,
    /// Pupil: Gaussian apodization, in pupil radii (0: off).
    #[property(min = 0.0, max = 2.0, step = 0.01)]
    pub apodization: f32,
    /// Pupil: central obstruction in pupil radii (mirror-lens donut).
    #[property(min = 0.0, max = 0.95, step = 0.01)]
    pub obstruction: f32,
    /// Longitudinal chromatic aberration: focus shift per channel in
    /// dioptres (negative red and blue: purple foreground fringes, green
    /// background ones).
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub red_shift_dpt: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub green_shift_dpt: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub blue_shift_dpt: f32,
    /// Sweet spot: extra defocus outside the region, in pixels.
    #[property(min = 0.0, max = 64.0, step = 0.5)]
    pub extra_blur_px: f32,
    #[property]
    pub shape: LensRegionShape,
    /// Half-plane: direction of the covered side in degrees (0: the right
    /// half; 90: the bottom half). Ellipse: its rotation.
    #[property(min = -180.0, max = 180.0, step = 0.5)]
    pub angle_deg: f32,
    /// Half-plane: line offset from the centre in image heights.
    #[property(min = -1.0, max = 1.0, step = 0.005)]
    pub offset: f32,
    /// Ellipse and radial band: centre in image heights.
    #[property(min = -1.0, max = 1.0, step = 0.005)]
    pub center_x: f32,
    #[property(min = -1.0, max = 1.0, step = 0.005)]
    pub center_y: f32,
    /// Ellipse: radii in image heights.
    #[property(min = 0.0, max = 2.0, step = 0.005)]
    pub radius_x: f32,
    #[property(min = 0.0, max = 2.0, step = 0.005)]
    pub radius_y: f32,
    /// Radial band: inner and outer radius in image heights.
    #[property(min = 0.0, max = 2.0, step = 0.005)]
    pub inner_radius: f32,
    #[property(min = 0.0, max = 2.0, step = 0.005)]
    pub outer_radius: f32,
    /// Polygon: 3 or 4 vertices (the pass takes up to 8).
    #[property(min = 3.0, max = 4.0, step = 1.0)]
    pub polygon_vertices: u32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p0_x: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p0_y: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p1_x: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p1_y: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p2_x: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p2_y: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p3_x: f32,
    #[property(min = -2.0, max = 2.0, step = 0.005)]
    pub p3_y: f32,
    /// The region's soft edge is derived from the aperture (a diopter
    /// seam narrows as the lens stops down) unless this is off.
    #[property]
    pub auto_feather: bool,
    /// Soft edge width in image heights when `auto_feather` is off.
    #[property(min = 0.0, max = 1.0, step = 0.002)]
    pub feather: f32,
}

impl Default for LensElementProps {
    fn default() -> Self {
        Self {
            enabled: true,
            kind: LensElementKind::Diopter,
            power_dpt: 1.0,
            seam_darkening: 0.0,
            mount_distance_m: 0.0,
            extra_blur_px: 12.0,
            shape: LensRegionShape::HalfPlane,
            angle_deg: 0.0,
            offset: 0.0,
            center_x: 0.0,
            center_y: 0.0,
            radius_x: 0.25,
            radius_y: 0.25,
            inner_radius: 0.2,
            outer_radius: 0.5,
            polygon_vertices: 3,
            p0_x: -0.3,
            p0_y: -0.3,
            p1_x: 0.3,
            p1_y: -0.3,
            p2_x: 0.0,
            p2_y: 0.3,
            p3_x: 0.0,
            p3_y: 0.0,
            auto_feather: true,
            feather: 0.02,
            inherit_iris: true,
            blades: 0,
            blade_curvature: 0.0,
            iris_rotation_deg: 0.0,
            round_until_f_number: 0.0,
            squeeze: 1.0,
            cats_eye: 0.0,
            barrel_radius: 0.0,
            swirl: 0.0,
            spherical: 0.0,
            spherical_asymmetry: 0.0,
            onion_amplitude: 0.0,
            onion_rings: 0.0,
            apodization: 0.0,
            obstruction: 0.0,
            red_shift_dpt: 0.0,
            green_shift_dpt: 0.0,
            blue_shift_dpt: 0.0,
        }
    }
}

impl LensElementProps {
    /// The element a preset describes.
    pub fn from_preset(element: &PresetElement) -> Self {
        let mut props = Self {
            kind: element.kind,
            power_dpt: if element.kind == LensElementKind::Diopter {
                element.amount
            } else {
                0.0
            },
            extra_blur_px: if element.kind == LensElementKind::SweetSpot {
                element.amount
            } else {
                0.0
            },
            seam_darkening: element.seam_darkening,
            auto_feather: element.feather.is_none(),
            feather: element.feather.unwrap_or(0.02),
            ..Self::default()
        };
        match element.region {
            PresetRegion::FullFrame => props.shape = LensRegionShape::FullFrame,
            PresetRegion::HalfPlane { angle_deg, offset } => {
                props.shape = LensRegionShape::HalfPlane;
                props.angle_deg = angle_deg;
                props.offset = offset;
            }
            PresetRegion::Ellipse { center, radii, rotation_deg } => {
                props.shape = LensRegionShape::Ellipse;
                (props.center_x, props.center_y) = (center[0], center[1]);
                (props.radius_x, props.radius_y) = (radii[0], radii[1]);
                props.angle_deg = rotation_deg;
            }
            PresetRegion::RadialBand { inner, outer, center } => {
                props.shape = LensRegionShape::RadialBand;
                (props.inner_radius, props.outer_radius) = (inner, outer);
                (props.center_x, props.center_y) = (center[0], center[1]);
            }
        }
        props
    }

    fn region(&self) -> ([[f32; 2]; 4], usize) {
        (
            [
                [self.p0_x, self.p0_y],
                [self.p1_x, self.p1_y],
                [self.p2_x, self.p2_y],
                [self.p3_x, self.p3_y],
            ],
            self.polygon_vertices.clamp(3, 4) as usize,
        )
    }

    /// The pass row of this element. A disabled element is an inert row.
    pub fn to_row(&self) -> LensElementComponent {
        if !self.enabled {
            return bytemuck::Zeroable::zeroed();
        }
        let (vertices, count) = self.region();
        let region = match self.shape {
            LensRegionShape::FullFrame => LensRegion::FullFrame,
            LensRegionShape::HalfPlane => LensRegion::HalfPlane {
                angle: self.angle_deg.to_radians(),
                offset: self.offset,
            },
            LensRegionShape::Ellipse => LensRegion::Ellipse {
                center: [self.center_x, self.center_y],
                radii: [self.radius_x, self.radius_y],
                rotation: self.angle_deg.to_radians(),
            },
            LensRegionShape::Polygon => LensRegion::Polygon(&vertices[..count]),
            LensRegionShape::RadialBand => LensRegion::RadialBand {
                r0: self.inner_radius,
                r1: self.outer_radius,
                center: [self.center_x, self.center_y],
            },
        };
        let mut row = match self.kind {
            LensElementKind::Diopter => {
                let mut row = LensElementComponent::diopter(self.power_dpt, region)
                    .with_seam_darkening(self.seam_darkening);
                if self.mount_distance_m > 0.0 {
                    row = row.with_mount_distance(self.mount_distance_m);
                }
                row
            }
            LensElementKind::SweetSpot => {
                LensElementComponent::sweet_spot(self.extra_blur_px, region)
            }
            LensElementKind::Pupil => {
                let mut row = LensElementComponent::pupil(region);
                if !self.inherit_iris {
                    row = row.with_iris(
                        self.blades,
                        self.blade_curvature,
                        self.iris_rotation_deg.to_radians(),
                    );
                }
                row.with_f_number_rounding(self.round_until_f_number)
                    .with_anamorphic(self.squeeze.max(1.0))
                    .with_cats_eye(self.cats_eye, self.barrel_radius)
                    .with_swirl(self.swirl)
                    .with_spherical_aberration(self.spherical, self.spherical_asymmetry)
                    .with_onion_rings(self.onion_amplitude, self.onion_rings)
                    .with_apodization(self.apodization)
                    .with_central_obstruction(self.obstruction)
            }
            LensElementKind::LongitudinalCa => LensElementComponent::longitudinal_ca(
                self.red_shift_dpt,
                self.green_shift_dpt,
                self.blue_shift_dpt,
                region,
            ),
        };
        if !self.auto_feather {
            row = row.with_feather(self.feather.max(0.0));
        }
        debug_assert!(row.kind != element_kind::NONE);
        debug_assert!(row.region_kind != region_kind::MASK);
        row
    }
}

fn serialize_lens_element_json(value: &LensElementProps) -> ReflectResult<serde_json::Value> {
    serde_json::to_value(value).map_err(|e| ReflectError::SerializationFailed(e.to_string()))
}

fn deserialize_lens_element_json(value: serde_json::Value) -> ReflectResult<LensElementProps> {
    serde_json::from_value(value).map_err(|e| ReflectError::DeserializationFailed(e.to_string()))
}

#[pulsar_type(
    serialize_json_with = serialize_lens_element_json,
    deserialize_json_with = deserialize_lens_element_json
)]
pub type RegisteredLensElementProps = LensElementProps;

lens_enum! {
    /// A named lens setup. `Custom` uses the fields below it; any other
    /// preset replaces the optics, tilt, curvature, iris and element stack
    /// (and its focus distance, where it sets one) with its own. Bake a
    /// preset into the fields with the `bake_preset` method to adjust it.
    LensPreset {
        #[default]
        Custom,
        Standard50mmF2_8,
        Portrait85mmF1_4,
        SplitDiopter2Way,
        SplitDiopter3Way,
        TiltShiftMiniature,
        LensbabySweetSpot,
    }
}

impl LensPreset {
    /// The preset's data; `None` for `Custom`.
    pub fn data(self) -> Option<&'static LensPresetData> {
        match self {
            Self::Custom => None,
            other => LENS_PRESETS.iter().find(|p| p.preset == other),
        }
    }
}

/// Where a preset element acts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PresetRegion {
    FullFrame,
    HalfPlane { angle_deg: f32, offset: f32 },
    Ellipse { center: [f32; 2], radii: [f32; 2], rotation_deg: f32 },
    RadialBand { inner: f32, outer: f32, center: [f32; 2] },
}

/// One element of a preset: `amount` is dioptres for a diopter and pixels
/// of extra defocus for a sweet spot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresetElement {
    pub kind: LensElementKind,
    pub region: PresetRegion,
    pub amount: f32,
    /// `None`: derived from the aperture.
    pub feather: Option<f32>,
    pub seam_darkening: f32,
}

/// A named lens setup as data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensPresetData {
    pub preset: LensPreset,
    pub name: &'static str,
    pub focal_length_mm: f32,
    pub f_number: f32,
    pub aperture_blades: u32,
    pub tilt_deg: f32,
    pub tilt_axis_deg: f32,
    pub field_curvature_dpt: f32,
    /// `None` keeps the authored focus distance.
    pub focus_distance_m: Option<f32>,
    pub elements: &'static [PresetElement],
}

const fn diopter(power_dpt: f32, region: PresetRegion) -> PresetElement {
    PresetElement {
        kind: LensElementKind::Diopter,
        region,
        amount: power_dpt,
        feather: None,
        seam_darkening: 0.15,
    }
}

/// The named presets. A split diopter adds `D` dioptres where it covers:
/// a lens focused at `s` then focuses at `s'` with `1/s' = 1/s + D`, so
/// +1 D at a 3 m base focus brings that part of the frame to 0.75 m.
pub const LENS_PRESETS: &[LensPresetData] = &[
    LensPresetData {
        preset: LensPreset::Standard50mmF2_8,
        name: "Standard 50mm f/2.8",
        focal_length_mm: 50.0,
        f_number: 2.8,
        aperture_blades: 6,
        tilt_deg: 0.0,
        tilt_axis_deg: 0.0,
        field_curvature_dpt: 0.0,
        focus_distance_m: None,
        elements: &[],
    },
    LensPresetData {
        preset: LensPreset::Portrait85mmF1_4,
        name: "Portrait 85mm f/1.4",
        focal_length_mm: 85.0,
        f_number: 1.4,
        aperture_blades: 9,
        tilt_deg: 0.0,
        tilt_axis_deg: 0.0,
        field_curvature_dpt: 0.0,
        focus_distance_m: None,
        elements: &[],
    },
    // Right half of the frame focuses nearer; the seam is the filter's edge.
    LensPresetData {
        preset: LensPreset::SplitDiopter2Way,
        name: "2-way split diopter",
        focal_length_mm: 50.0,
        f_number: 5.6,
        aperture_blades: 6,
        tilt_deg: 0.0,
        tilt_axis_deg: 0.0,
        field_curvature_dpt: 0.0,
        focus_distance_m: Some(4.0),
        elements: &[diopter(
            1.5,
            PresetRegion::HalfPlane { angle_deg: 0.0, offset: 0.0 },
        )],
    },
    // Left third nearest, centre at the base focus, right third in between.
    LensPresetData {
        preset: LensPreset::SplitDiopter3Way,
        name: "3-way split diopter",
        focal_length_mm: 50.0,
        f_number: 5.6,
        aperture_blades: 6,
        tilt_deg: 0.0,
        tilt_axis_deg: 0.0,
        field_curvature_dpt: 0.0,
        focus_distance_m: Some(6.0),
        elements: &[
            diopter(
                2.0,
                PresetRegion::HalfPlane { angle_deg: 180.0, offset: 0.3 },
            ),
            diopter(
                0.75,
                PresetRegion::HalfPlane { angle_deg: 0.0, offset: 0.3 },
            ),
        ],
    },
    // A wedge-shaped focal plane: a sharp band across the frame, the rest
    // falling away, which reads as a miniature.
    LensPresetData {
        preset: LensPreset::TiltShiftMiniature,
        name: "Tilt-shift miniature",
        focal_length_mm: 65.0,
        f_number: 2.0,
        aperture_blades: 8,
        tilt_deg: 6.0,
        tilt_axis_deg: 0.0,
        field_curvature_dpt: 0.0,
        focus_distance_m: Some(5.0),
        elements: &[],
    },
    // Sharp in a central ellipse, increasingly soft outside it.
    LensPresetData {
        preset: LensPreset::LensbabySweetSpot,
        name: "Lensbaby sweet spot",
        focal_length_mm: 50.0,
        f_number: 2.0,
        aperture_blades: 6,
        tilt_deg: 0.0,
        tilt_axis_deg: 0.0,
        field_curvature_dpt: 0.0,
        focus_distance_m: None,
        elements: &[PresetElement {
            kind: LensElementKind::SweetSpot,
            region: PresetRegion::Ellipse {
                center: [0.0, 0.0],
                radii: [0.22, 0.22],
                rotation_deg: 0.0,
            },
            amount: 14.0,
            feather: Some(0.14),
            seam_darkening: 0.0,
        }],
    },
];

/// A camera's lens (authored).
#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("Camera", category_color = "#3AA0FF")]
#[category("Optics", category_color = "#E0B040")]
#[category("Focus", category_color = "#50C878")]
#[category("Focal Surface", category_color = "#C080E0")]
#[category("Elements", category_color = "#E07050")]
#[serde(default)]
pub struct CameraLensComponent {
    #[property(category = "Camera")]
    pub enabled: bool,
    /// Render view this lens belongs to (the editor viewport is view 0);
    /// independent of any editor camera type.
    #[property(category = "Camera")]
    pub view_id: u32,
    /// A named setup, or Custom to use the fields below.
    #[property(category = "Camera")]
    pub preset: LensPreset,
    #[property(min = 4.0, max = 1200.0, step = 1.0, category = "Optics")]
    pub focal_length_mm: f32,
    #[property(min = 0.7, max = 64.0, step = 0.1, category = "Optics")]
    pub f_number: f32,
    /// Transmission stop, carried for exposure (0: the f-number).
    #[property(min = 0.0, max = 64.0, step = 0.1, category = "Optics")]
    pub t_stop: f32,
    #[property(min = 1.0, max = 120.0, step = 0.1, category = "Optics")]
    pub sensor_width_mm: f32,
    /// 0 derives the height at 3:2.
    #[property(min = 0.0, max = 120.0, step = 0.1, category = "Optics")]
    pub sensor_height_mm: f32,
    /// Iris blades (0: circular); shape the flare pupil.
    #[property(min = 0.0, max = 16.0, step = 1.0, category = "Optics")]
    pub aperture_blades: u32,
    #[property(min = -180.0, max = 180.0, step = 1.0, category = "Optics")]
    pub aperture_rotation_deg: f32,
    #[property(min = 0.0, max = 64.0, step = 0.5, category = "Optics")]
    pub max_coc_px: f32,
    #[property(category = "Optics")]
    pub quality: LensGatherQuality,
    #[property(category = "Focus")]
    pub focus_mode: LensFocusMode,
    /// Manual focus distance in metres.
    #[property(min = 0.05, max = 100000.0, step = 0.05, category = "Focus")]
    pub focus_distance_m: f32,
    /// Autofocus point in uv (0..1, y down) and its sample radius.
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Focus")]
    pub af_u: f32,
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Focus")]
    pub af_v: f32,
    #[property(min = 0.0, max = 0.5, step = 0.005, category = "Focus")]
    pub af_radius: f32,
    /// Autofocus ease rate in 1/s (0: instant).
    #[property(min = 0.0, max = 60.0, step = 0.1, category = "Focus")]
    pub af_speed: f32,
    /// Scheimpflug tilt of the focal plane in degrees, about an axis in
    /// the image plane at `tilt_axis_deg` from +x.
    #[property(min = -30.0, max = 30.0, step = 0.1, category = "Focal Surface")]
    pub tilt_deg: f32,
    #[property(min = -180.0, max = 180.0, step = 1.0, category = "Focal Surface")]
    pub tilt_axis_deg: f32,
    /// Petzval field curvature in dioptres at the frame corner.
    #[property(min = -4.0, max = 4.0, step = 0.01, category = "Focal Surface")]
    pub field_curvature_dpt: f32,
    /// The ordered element stack (at most [`LENS_ELEMENT_SLOTS`]); ignored
    /// by a preset other than Custom.
    #[property(category = "Elements")]
    pub elements: Vec<LensElementProps>,
}

impl Default for CameraLensComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            view_id: 0,
            preset: LensPreset::Custom,
            focal_length_mm: 50.0,
            f_number: 2.8,
            t_stop: 0.0,
            sensor_width_mm: 36.0,
            sensor_height_mm: 24.0,
            aperture_blades: 6,
            aperture_rotation_deg: 0.0,
            max_coc_px: 0.0,
            quality: LensGatherQuality::Economical,
            focus_mode: LensFocusMode::Manual,
            focus_distance_m: 3.0,
            af_u: 0.5,
            af_v: 0.5,
            af_radius: 0.0,
            af_speed: 0.0,
            tilt_deg: 0.0,
            tilt_axis_deg: 0.0,
            field_curvature_dpt: 0.0,
            elements: Vec::new(),
        }
    }
}

impl CameraLensComponent {
    /// This lens set to `preset`'s values in its own fields, as `Custom`,
    /// ready to adjust. Returns false for `Custom`.
    pub fn bake_preset(&mut self, preset: LensPreset) -> bool {
        let Some(data) = preset.data() else {
            return false;
        };
        self.focal_length_mm = data.focal_length_mm;
        self.f_number = data.f_number;
        self.aperture_blades = data.aperture_blades;
        self.tilt_deg = data.tilt_deg;
        self.tilt_axis_deg = data.tilt_axis_deg;
        self.field_curvature_dpt = data.field_curvature_dpt;
        if let Some(distance) = data.focus_distance_m {
            self.focus_distance_m = distance;
        }
        self.elements = data.elements.iter().map(LensElementProps::from_preset).collect();
        self.preset = LensPreset::Custom;
        true
    }

    /// The lens with its preset applied: the authored fields, with the
    /// preset's values in place of the optics and the element stack.
    pub fn resolved(&self) -> Self {
        let mut lens = self.clone();
        if lens.bake_preset(self.preset) {
            lens.preset = self.preset;
        }
        lens
    }

    /// The pass rows: the lens, and its element rows from `element_first`
    /// (inert past the stack; at most [`LENS_ELEMENT_SLOTS`] elements).
    pub fn to_rows(
        &self,
        element_first: u32,
    ) -> (LensComponent, [LensElementComponent; LENS_ELEMENT_SLOTS]) {
        let lens = self.resolved();
        let mut elements = [bytemuck::Zeroable::zeroed(); LENS_ELEMENT_SLOTS];
        let mut row = LensComponent::new(lens.view_id);
        row.enabled = u32::from(lens.enabled);
        if !lens.enabled {
            return (row, elements);
        }
        let count = lens.elements.len().min(LENS_ELEMENT_SLOTS);
        for (slot, element) in elements.iter_mut().zip(&lens.elements[..count]) {
            *slot = element.to_row();
        }
        row.focus_mode = match lens.focus_mode {
            LensFocusMode::Manual => FOCUS_MANUAL,
            LensFocusMode::AutoPoint => FOCUS_AUTO_POINT,
        };
        row.quality = match lens.quality {
            LensGatherQuality::Economical => 0,
            LensGatherQuality::High => 1,
        };
        row.focal_length_mm = lens.focal_length_mm.max(1.0);
        row.f_number = lens.f_number.max(0.5);
        row.t_stop = if lens.t_stop > 0.0 { lens.t_stop } else { row.f_number };
        row.sensor_width_mm = lens.sensor_width_mm.max(1.0);
        row.sensor_height_mm = lens.sensor_height_mm.max(0.0);
        row.focus_distance_m = lens.focus_distance_m.max(0.01);
        row.af_u = lens.af_u;
        row.af_v = lens.af_v;
        row.af_radius = lens.af_radius;
        row.af_speed = lens.af_speed;
        row.max_coc_px = lens.max_coc_px;
        row.element_first = element_first;
        row.element_count = count as u32;
        row.tilt_angle = lens.tilt_deg.to_radians();
        row.tilt_axis = lens.tilt_axis_deg.to_radians();
        row.field_curvature = lens.field_curvature_dpt;
        row.aperture_blades = lens.aperture_blades;
        row.aperture_rotation = lens.aperture_rotation_deg.to_radians();
        (row, elements)
    }
}

#[register_world_component]
impl CameraLensComponent {}

/// The lens's script methods.
fn camera_lens_methods() -> Vec<pulsar_reflection::MethodMetadata> {
    use pulsar_reflection::{
        EngineClass, MethodFlags, MethodMetadata, MethodParameter, MethodReturnType,
        RUNTIME_TYPE_REGISTRY,
    };
    let (Some(u32_info), Some(bool_info)) = (
        RUNTIME_TYPE_REGISTRY.get::<u32>(),
        RUNTIME_TYPE_REGISTRY.get::<bool>(),
    ) else {
        return Vec::new();
    };
    vec![MethodMetadata {
        name: "bake_preset",
        display_name: "Bake Preset".into(),
        category: Some("Camera"),
        params: vec![MethodParameter {
            name: "preset_index",
            type_info: u32_info,
        }],
        return_type: Some(MethodReturnType {
            type_info: bool_info,
        }),
        flags: MethodFlags::NONE,
        caller: Box::new(
            |instance: &mut dyn EngineClass, args: Vec<Box<dyn std::any::Any>>| {
                let index = *args.first()?.downcast_ref::<u32>()? as usize;
                let preset = *LensPreset::ALL.get(index)?;
                let lens = instance.as_any_mut().downcast_mut::<CameraLensComponent>()?;
                Some(Box::new(lens.bake_preset(preset)))
            },
        ),
    }]
}

pulsar_reflection::inventory::submit! {
    pulsar_reflection::ComponentMethodRegistration {
        class_name: CAMERA_LENS_CLASS_NAME,
        methods: camera_lens_methods,
    }
}

/// Where each lens's element block lives in `lens_elements`: blocks of
/// [`LENS_ELEMENT_SLOTS`] rows from a free list, per GPU store.
#[derive(Default)]
struct ElementBlocks {
    next: u32,
    free: Vec<u32>,
    by_lens_row: HashMap<u32, u32>,
}

impl ElementBlocks {
    fn block_of(&mut self, lens_row: u32) -> u32 {
        if let Some(&block) = self.by_lens_row.get(&lens_row) {
            return block;
        }
        let block = self.free.pop().unwrap_or_else(|| {
            let block = self.next;
            self.next += 1;
            block
        });
        self.by_lens_row.insert(lens_row, block);
        block
    }

    fn release(&mut self, lens_row: u32) -> Option<u32> {
        let block = self.by_lens_row.remove(&lens_row)?;
        self.free.push(block);
        Some(block)
    }
}

static BLOCKS: Mutex<Option<HashMap<usize, ElementBlocks>>> = Mutex::new(None);

fn with_blocks<T>(mirror: &GpuMirrorHandle, f: impl FnOnce(&mut ElementBlocks) -> T) -> T {
    let key = mirror.store() as *const SceneGpuStore as usize;
    let mut blocks = BLOCKS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    f(blocks.get_or_insert_with(HashMap::new).entry(key).or_default())
}

fn camera_lens_dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const (), is_new_insert: bool) {
    // SAFETY: SceneDB reaches this only through `CameraLensComponent`'s own
    // `ComponentId`, with a pointer to a live value.
    let authored = unsafe { &*(data as *const CameraLensComponent) };
    let block = with_blocks(mirror, |blocks| blocks.block_of(row));
    let first = block * LENS_ELEMENT_SLOTS as u32;
    let (lens, elements) = authored.to_rows(first);
    pulsar_scenedb::gpu::write_derived_row(
        mirror,
        row,
        &CameraLensSourceRow::of(&lens),
        is_new_insert,
    );
    for (slot, element) in elements.iter().enumerate() {
        pulsar_scenedb::gpu::write_derived_row(mirror, first + slot as u32, element, is_new_insert);
    }
}

fn camera_lens_clear(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<CameraLensSourceRow>(mirror, row);
    if let Some(block) = with_blocks(mirror, |blocks| blocks.release(row)) {
        let first = block * LENS_ELEMENT_SLOTS as u32;
        for slot in 0..LENS_ELEMENT_SLOTS as u32 {
            pulsar_scenedb::gpu::clear_derived_row::<LensElementComponent>(mirror, first + slot);
        }
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<CameraLensComponent>,
        dispatch: camera_lens_dispatch,
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<CameraLensComponent>,
        clear: camera_lens_clear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsar_reflection::EngineClass;

    #[test]
    fn the_default_lens_is_the_passes_neutral_lens() {
        let (row, elements) = CameraLensComponent::default().to_rows(0);
        let mut expected = LensComponent::new(0);
        expected.t_stop = expected.f_number;
        assert_eq!(row, expected);
        assert!(elements.iter().all(|e| e.kind == element_kind::NONE));
    }

    #[test]
    fn a_disabled_lens_is_an_inert_row_with_no_elements() {
        let mut lens = CameraLensComponent::default();
        lens.preset = LensPreset::SplitDiopter2Way;
        lens.enabled = false;
        let (row, elements) = lens.to_rows(8);
        assert_eq!(row.enabled, 0);
        assert_eq!(row.element_count, 0);
        assert!(elements.iter().all(|e| e.kind == element_kind::NONE));
    }

    #[test]
    fn fields_reach_the_row_in_the_passes_units() {
        let mut lens = CameraLensComponent::default();
        lens.view_id = 3;
        lens.focal_length_mm = 85.0;
        lens.f_number = 1.4;
        lens.tilt_deg = 90.0;
        lens.tilt_axis_deg = 180.0;
        lens.aperture_rotation_deg = 45.0;
        lens.focus_mode = LensFocusMode::AutoPoint;
        lens.quality = LensGatherQuality::High;
        let (row, _) = lens.to_rows(16);
        assert_eq!(row.view_id, 3);
        assert_eq!((row.focal_length_mm, row.f_number, row.t_stop), (85.0, 1.4, 1.4));
        assert!((row.tilt_angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((row.tilt_axis - std::f32::consts::PI).abs() < 1e-6);
        assert!((row.aperture_rotation - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert_eq!((row.focus_mode, row.quality), (FOCUS_AUTO_POINT, 1));
        assert_eq!(row.element_first, 16);
    }

    #[test]
    fn element_rows_follow_the_stack_and_skip_disabled_ones() {
        let mut lens = CameraLensComponent::default();
        let mut off = LensElementProps::default();
        off.enabled = false;
        lens.elements = vec![
            LensElementProps::default(),
            off,
            LensElementProps {
                kind: LensElementKind::SweetSpot,
                shape: LensRegionShape::Ellipse,
                auto_feather: false,
                feather: 0.1,
                ..LensElementProps::default()
            },
        ];
        let (row, elements) = lens.to_rows(0);
        assert_eq!(row.element_count, 3);
        assert_eq!(elements[0].kind, element_kind::DIOPTER);
        assert_eq!(elements[0].region_kind, region_kind::HALF_PLANE);
        assert!(elements[0].feather < 0.0, "derived from the aperture");
        assert_eq!(elements[1].kind, element_kind::NONE, "disabled: inert");
        assert_eq!(elements[2].kind, element_kind::SWEET_SPOT);
        assert_eq!(elements[2].feather, 0.1);
        assert_eq!(elements[3].kind, element_kind::NONE);
    }

    #[test]
    fn pupil_and_chromatic_elements_carry_their_parameters() {
        use helio_pass_lens::pupil;
        let mut lens = CameraLensComponent::default();
        lens.elements = vec![
            LensElementProps {
                kind: LensElementKind::Pupil,
                shape: LensRegionShape::FullFrame,
                inherit_iris: false,
                blades: 9,
                blade_curvature: 0.5,
                squeeze: 1.8,
                cats_eye: 0.7,
                spherical: 0.4,
                obstruction: 0.3,
                ..LensElementProps::default()
            },
            LensElementProps {
                kind: LensElementKind::LongitudinalCa,
                shape: LensRegionShape::FullFrame,
                red_shift_dpt: -0.2,
                blue_shift_dpt: -0.1,
                ..LensElementProps::default()
            },
            LensElementProps {
                kind: LensElementKind::Pupil,
                shape: LensRegionShape::FullFrame,
                ..LensElementProps::default()
            },
        ];
        let (row, elements) = lens.to_rows(0);
        assert_eq!(row.element_count, 3);
        assert_eq!(elements[0].kind, element_kind::PUPIL);
        let p = &elements[0].params;
        assert_eq!((p[pupil::BLADES], p[pupil::CURVATURE]), (9.0, 0.5));
        assert_eq!((p[pupil::SQUEEZE], p[pupil::CATS_EYE]), (1.8, 0.7));
        assert_eq!((p[pupil::SPHERICAL], p[pupil::OBSTRUCTION]), (0.4, 0.3));
        assert_eq!(elements[1].kind, element_kind::LONGITUDINAL_CA);
        assert_eq!(&elements[1].params[..3], &[-0.2, 0.0, -0.1]);
        assert!(
            elements[2].params[pupil::BLADES] < 0.0,
            "inherits the lens iris"
        );
    }

    #[test]
    fn a_long_stack_is_clamped_to_the_block() {
        let mut lens = CameraLensComponent::default();
        lens.elements = vec![LensElementProps::default(); LENS_ELEMENT_SLOTS + 5];
        let (row, _) = lens.to_rows(0);
        assert_eq!(row.element_count as usize, LENS_ELEMENT_SLOTS);
    }

    #[test]
    fn polygon_regions_carry_their_vertices() {
        let mut lens = CameraLensComponent::default();
        lens.elements = vec![LensElementProps {
            shape: LensRegionShape::Polygon,
            polygon_vertices: 4,
            p3_x: 0.25,
            p3_y: 0.5,
            ..LensElementProps::default()
        }];
        let (_, elements) = lens.to_rows(0);
        assert_eq!(elements[0].region_kind, region_kind::POLYGON);
        assert_eq!(elements[0].region_count, 4);
        assert_eq!(&elements[0].region[6..8], &[0.25, 0.5]);
    }

    #[test]
    fn every_preset_has_data_and_resolves_to_valid_rows() {
        for preset in LensPreset::ALL {
            if *preset == LensPreset::Custom {
                assert!(preset.data().is_none());
                continue;
            }
            let data = preset.data().unwrap_or_else(|| panic!("{preset:?} has no data"));
            assert!(!data.name.is_empty());
            assert!(data.elements.len() <= LENS_ELEMENT_SLOTS, "{}", data.name);
            let mut lens = CameraLensComponent::default();
            lens.preset = *preset;
            let (row, elements) = lens.to_rows(0);
            assert_eq!(row.focal_length_mm, data.focal_length_mm, "{}", data.name);
            assert_eq!(row.f_number, data.f_number, "{}", data.name);
            assert_eq!(row.element_count as usize, data.elements.len(), "{}", data.name);
            for element in &elements[..data.elements.len()] {
                assert_ne!(element.kind, element_kind::NONE, "{}", data.name);
            }
        }
        let names: Vec<_> = LENS_PRESETS.iter().map(|p| p.name).collect();
        assert_eq!(
            names,
            [
                "Standard 50mm f/2.8",
                "Portrait 85mm f/1.4",
                "2-way split diopter",
                "3-way split diopter",
                "Tilt-shift miniature",
                "Lensbaby sweet spot",
            ]
        );
    }

    #[test]
    fn split_diopters_follow_the_thin_lens_focus_sum() {
        // 1/s' = 1/s + D.
        let focus = |s: f32, d: f32| 1.0 / (1.0 / s + d);
        let two = LensPreset::SplitDiopter2Way.data().unwrap();
        assert_eq!(two.elements.len(), 1);
        let near = focus(two.focus_distance_m.unwrap(), two.elements[0].amount);
        assert!((near - 4.0 / (1.0 + 4.0 * 1.5)).abs() < 1e-5);
        assert!(near < two.focus_distance_m.unwrap(), "positive power focuses nearer");

        let three = LensPreset::SplitDiopter3Way.data().unwrap();
        assert_eq!(three.elements.len(), 2);
        let base = three.focus_distance_m.unwrap();
        let (left, right) = (
            focus(base, three.elements[0].amount),
            focus(base, three.elements[1].amount),
        );
        assert!(left < right && right < base, "three distinct focus zones");
        // Left covers x < -0.3, right x > 0.3 (half planes at 180 and 0 deg).
        let angles: Vec<f32> = three
            .elements
            .iter()
            .map(|e| match e.region {
                PresetRegion::HalfPlane { angle_deg, .. } => angle_deg,
                _ => panic!("half planes"),
            })
            .collect();
        assert_eq!(angles, [180.0, 0.0]);
    }

    #[test]
    fn the_tilt_shift_and_sweet_spot_presets_use_their_features() {
        let (tilt, _) = {
            let mut lens = CameraLensComponent::default();
            lens.preset = LensPreset::TiltShiftMiniature;
            lens.to_rows(0)
        };
        assert!(tilt.tilt_angle > 0.0 && tilt.element_count == 0);
        let (spot, elements) = {
            let mut lens = CameraLensComponent::default();
            lens.preset = LensPreset::LensbabySweetSpot;
            lens.to_rows(0)
        };
        assert_eq!(spot.element_count, 1);
        assert_eq!(elements[0].kind, element_kind::SWEET_SPOT);
        assert_eq!(elements[0].region_kind, region_kind::ELLIPSE);
        assert_eq!(elements[0].params[0], 14.0);
        assert_eq!(elements[0].feather, 0.14);
    }

    #[test]
    fn a_preset_keeps_the_authored_focus_unless_it_sets_one() {
        let mut lens = CameraLensComponent::default();
        lens.focus_distance_m = 7.0;
        lens.preset = LensPreset::Portrait85mmF1_4;
        assert_eq!(lens.to_rows(0).0.focus_distance_m, 7.0);
        lens.preset = LensPreset::SplitDiopter2Way;
        assert_eq!(lens.to_rows(0).0.focus_distance_m, 4.0);
    }

    #[test]
    fn baking_a_preset_copies_it_into_the_fields_as_custom() {
        let mut lens = CameraLensComponent::default();
        assert!(!lens.bake_preset(LensPreset::Custom));
        assert!(lens.bake_preset(LensPreset::SplitDiopter3Way));
        assert_eq!(lens.preset, LensPreset::Custom);
        assert_eq!(lens.elements.len(), 2);
        assert_eq!(lens.focus_distance_m, 6.0);
        // The baked lens makes the same rows as the preset did.
        let mut preset = CameraLensComponent::default();
        preset.preset = LensPreset::SplitDiopter3Way;
        assert_eq!(lens.to_rows(0), preset.to_rows(0));
    }

    #[test]
    fn bake_preset_is_a_method_and_the_fields_are_reflected() {
        let methods = <CameraLensComponent as EngineClass>::get_methods();
        let bake = methods
            .iter()
            .find(|m| m.name == "bake_preset")
            .expect("bake_preset is registered");
        let mut lens = CameraLensComponent::default();
        let index = LensPreset::LensbabySweetSpot as u32;
        let done = (bake.caller)(&mut lens, vec![Box::new(index)]).unwrap();
        assert!(*done.downcast_ref::<bool>().unwrap());
        assert_eq!(lens.elements.len(), 1);

        let names: Vec<_> = lens.get_properties().iter().map(|p| p.name).collect();
        for field in ["focal_length_mm", "f_number", "sensor_width_mm", "aperture_blades", "elements", "preset"] {
            assert!(names.contains(&field), "{field} is reflected");
        }
    }

    #[test]
    fn the_authored_value_round_trips_through_json_with_names_or_indices() {
        let mut lens = CameraLensComponent::default();
        lens.preset = LensPreset::SplitDiopter2Way;
        lens.elements = vec![LensElementProps {
            kind: LensElementKind::SweetSpot,
            shape: LensRegionShape::RadialBand,
            ..LensElementProps::default()
        }];
        let json = serde_json::to_value(&lens).unwrap();
        let back: CameraLensComponent = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back.to_rows(0), lens.to_rows(0));
        let by_index: LensElementKind = serde_json::from_value(serde_json::json!(1)).unwrap();
        assert_eq!(by_index, LensElementKind::SweetSpot);
        // A level saved with fewer fields still loads.
        let sparse: CameraLensComponent =
            serde_json::from_value(serde_json::json!({ "f_number": 4.0 })).unwrap();
        assert_eq!(sparse.f_number, 4.0);
        assert_eq!(sparse.focal_length_mm, 50.0);
    }

    #[test]
    fn element_blocks_are_compact_stable_and_reused() {
        let mut blocks = ElementBlocks::default();
        let a = blocks.block_of(5000);
        let b = blocks.block_of(12);
        assert_eq!((a, b), (0, 1), "compact, whatever the entity indices");
        assert_eq!(blocks.block_of(5000), a, "stable");
        assert_eq!(blocks.release(5000), Some(a));
        assert_eq!(blocks.release(5000), None);
        assert_eq!(blocks.block_of(77), a, "a freed block is reused");
    }
}
