//! SceneDB-owned lens records: one [`LensComponent`] per view, and the
//! [`LensElementComponent`] rows it stacks in front of the sensor.
//!
//! A view's lens is an ordered stack of optical elements reduced per pixel
//! to four quantities: a ray map (sensor uv -> scene uv, per wavelength), a
//! focus field `s(uv)`, a pupil function and a transmission. The focus field
//! is resolved by the diopter and sweet-spot elements, the pupil function by
//! [`element_kind::PUPIL`] elements (per-channel focus by
//! [`element_kind::LONGITUDINAL_CA`]), the ray map by the projection,
//! distortion, shift, keystone, breathing, prism, lateral chromatic
//! aberration and underwater port elements applied in stack order, the
//! transmission by the vignetting, filter and T-stop elements. The lens is
//! owned by the camera: post-process volumes never blend lens settings.
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
    /// The pupil function inside its region (aperture shape, anamorphic
    /// squeeze, cat's eye, swirl, spherical aberration, apodization, central
    /// obstruction). `params`: see [`super::pupil`]. Where several cover a
    /// pixel the last wins; with none the lens's own iris
    /// (`aperture_blades`, `aperture_rotation`) is a plain polygon.
    pub const PUPIL: u32 = 3;
    /// Longitudinal chromatic aberration inside its region: `params[0..3]`
    /// are the red, green and blue focus shifts in dioptres (`1/s' = 1/s +
    /// D`), weighted by coverage and summed where several overlap.
    pub const LONGITUDINAL_CA: u32 = 4;
    /// Projection of the rendered rectilinear image: [`super::projection`].
    pub const PROJECTION: u32 = 5;
    /// Brown-Conrady (OpenCV) radial + tangential distortion:
    /// [`super::brown_conrady`].
    pub const BROWN_CONRADY: u32 = 6;
    /// Division-model distortion: [`super::division`].
    pub const DIVISION: u32 = 7;
    /// Anamorphic (mumps) radial distortion with separate x and y
    /// coefficients: [`super::anamorphic_distortion`].
    pub const ANAMORPHIC_DISTORTION: u32 = 8;
    /// Anamorphic squeeze / desqueeze: `params[0]` times the horizontal
    /// field of view.
    pub const ANAMORPHIC_SQUEEZE: u32 = 9;
    /// Lens shift (optical centre offset): `params[0..2]` in image-height
    /// units.
    pub const SHIFT: u32 = 10;
    /// Keystone / perspective correction: [`super::keystone`].
    pub const KEYSTONE: u32 = 11;
    /// Focus breathing: [`super::breathing`].
    pub const FOCUS_BREATHING: u32 = 12;
    /// N weighted displaced / rotated copies of the image: [`super::prism`].
    pub const PRISM: u32 = 13;
    /// Spectral lateral chromatic aberration: [`super::lateral_ca`].
    pub const LATERAL_CA: u32 = 14;
    /// Underwater flat port: [`super::flat_port`].
    pub const FLAT_PORT: u32 = 15;
    /// Underwater dome port: [`super::dome_port`].
    pub const DOME_PORT: u32 = 16;
    /// Natural `cos^4` falloff from the ray's real angle: `params[0]` is the
    /// amount (0..1).
    pub const NATURAL_VIGNETTE: u32 = 17;
    /// Optical (cat's eye) vignetting: [`super::optical_vignette`].
    pub const OPTICAL_VIGNETTE: u32 = 18;
    /// Mechanical vignetting by a hood or matte box:
    /// [`super::mechanical_vignette`].
    pub const MECHANICAL_VIGNETTE: u32 = 19;
    /// Neutral density: `params[0]` stops.
    pub const ND_FILTER: u32 = 20;
    /// Graduated neutral density: [`super::graduated_nd`].
    pub const GRADUATED_ND: u32 = 21;
    /// Colour filter: `params[0..3]` linear transmittance, `params[3]`
    /// amount (0..1).
    pub const COLOR_FILTER: u32 = 22;
    /// Light lost between the f-number and the T-stop: scales by
    /// `(f_number / t_stop)^2`; `params[0]` is the amount (0..1).
    pub const T_STOP_LOSS: u32 = 23;
}

/// `params` slots of an [`element_kind::PROJECTION`] element.
///
/// The ray map works in focal-length units: a sensor point `p` (mm) is
/// `p / focal_length`, y down. The rendered image is rectilinear at the
/// lens's focal length, so a projection says how the *sensor* records the
/// angle `theta` of each ray: a rectilinear lens puts it at `tan(theta)`
/// and the map is the identity; the others at the radii of
/// [`ProjectionModel`].
pub mod projection {
    /// [`super::ProjectionModel`].
    pub const MODEL: usize = 0;
    /// Kannala-Brandt `k1..k4` in `r = theta + k1 theta^3 + k2 theta^5 +
    /// k3 theta^7 + k4 theta^9`.
    pub const K1: usize = 1;
    pub const K2: usize = 2;
    pub const K3: usize = 3;
    pub const K4: usize = 4;
    /// Panini distance `d` (0: rectilinear, 1: classic Panini).
    pub const PANINI_D: usize = 1;
    /// Panini vertical compression `s` (0..1; 1 classic, 0 none).
    pub const PANINI_S: usize = 2;
}

/// Projection models ([`projection::MODEL`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ProjectionModel {
    /// `r = tan(theta)`: the identity.
    Rectilinear = 0,
    /// `r = theta`.
    Equidistant = 1,
    /// `r = 2 sin(theta / 2)`.
    Equisolid = 2,
    /// `r = sin(theta)`.
    Orthographic = 3,
    /// `r = 2 tan(theta / 2)`.
    Stereographic = 4,
    /// Kannala-Brandt polynomial in `theta` ([`projection::K1`]..).
    KannalaBrandt = 5,
    /// Panini ([`projection::PANINI_D`], [`projection::PANINI_S`]).
    Panini = 6,
    /// Cylindrical: horizontal equidistant, vertical rectilinear.
    Cylindrical = 7,
}

/// `params` slots of an [`element_kind::BROWN_CONRADY`] element, the OpenCV
/// model on normalised coordinates `x = X / Z` (y down):
///
/// ```text
/// radial = (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)
/// x_d = x radial + 2 p1 x y + p2 (r^2 + 2 x^2)
/// y_d = y radial + p1 (r^2 + 2 y^2) + 2 p2 x y
/// ```
///
/// `(x, y)` is the ideal (rendered) point and `(x_d, y_d)` where the sensor
/// records it. Negative `k1` is barrel, positive pincushion.
pub mod brown_conrady {
    pub const K1: usize = 0;
    pub const K2: usize = 1;
    pub const K3: usize = 2;
    pub const K4: usize = 3;
    pub const K5: usize = 4;
    pub const K6: usize = 5;
    pub const P1: usize = 6;
    pub const P2: usize = 7;
}

/// `params` slots of an [`element_kind::DIVISION`] element: the sensor point
/// `x_d` records the ideal point `x_d / (1 + l1 r_d^2 + l2 r_d^4)`.
pub mod division {
    pub const L1: usize = 0;
    pub const L2: usize = 1;
}

/// `params` slots of an [`element_kind::ANAMORPHIC_DISTORTION`] element:
/// `x_d = x (1 + kx1 r^2 + kx2 r^4)`, `y_d = y (1 + ky1 r^2 + ky2 r^4)` with
/// the ideal point's `r`. Barrel in one axis and pincushion in the other is
/// the "mumps" of anamorphic lenses.
pub mod anamorphic_distortion {
    pub const KX1: usize = 0;
    pub const KX2: usize = 1;
    pub const KY1: usize = 2;
    pub const KY2: usize = 3;
}

/// `params` slots of an [`element_kind::KEYSTONE`] element: the point `x`
/// maps to `x / (1 + kx x_x + ky x_y)` in focal-length units (a tilted
/// sensor or perspective control: `k` is about `tan(tilt)`).
pub mod keystone {
    pub const KX: usize = 0;
    pub const KY: usize = 1;
}

/// `params` slots of an [`element_kind::FOCUS_BREATHING`] element: the
/// magnification is `1 + k (D - D_ref)` for the focus `D` in dioptres, so
/// focusing closer (larger `D`) with a negative `k` narrows the view.
pub mod breathing {
    /// Magnification change per dioptre of focus.
    pub const PER_DIOPTRE: usize = 0;
    /// Focus (dioptres) where the magnification is 1; 0 is infinity.
    pub const REFERENCE_DIOPTRES: usize = 1;
}

/// `params` slots of an [`element_kind::PRISM`] element. Copy `i` of `N`
/// samples the image displaced by `offset_i` after turning it `i * spin`
/// radians about the frame centre; the copies' weights are `falloff^i`,
/// normalised to sum to 1, so the filter never changes the frame's energy.
/// The first prism element covering a pixel is the one that acts.
pub mod prism {
    /// Copies, 2..=8.
    pub const COUNT: usize = 0;
    /// Displacement in image-height units: spacing of a line of copies, the
    /// radius of a ring.
    pub const OFFSET: usize = 1;
    /// Direction of the line (radians), or the angle of the first copy of a
    /// ring.
    pub const ANGLE: usize = 2;
    /// 0 copies along a line centred on the original, 1 around a ring.
    pub const RING: usize = 3;
    /// Rotation per copy about the frame centre in radians (kaleidoscope).
    pub const SPIN: usize = 4;
    /// Weight ratio between successive copies (<= 0: 1, all equal).
    pub const FALLOFF: usize = 5;
}

/// `params` slots of an [`element_kind::LATERAL_CA`] element. With `t` the
/// wavelength position `(lambda - 550 nm) / 150 nm` (-1 blue .. +1 red) the
/// image is magnified by `1 + d1 t + d2 t^2` about the frame centre, so a
/// positive `d1` puts red outward and blue inward. Colour is rebuilt from 3,
/// 7 or 16 wavelength samples (`quality` 0, 1, 2) with spectral weights that
/// sum to 1 per channel.
pub mod lateral_ca {
    pub const D1: usize = 0;
    pub const D2: usize = 1;
}

/// `params` slots of an [`element_kind::FLAT_PORT`] element.
pub mod flat_port {
    /// Refractive index of water (<= 0: 1.333).
    pub const INDEX: usize = 0;
    /// Index change per unit wavelength position `t` (blue higher); water is
    /// about 0.006.
    pub const DISPERSION: usize = 1;
}

/// `params` slots of an [`element_kind::DOME_PORT`] element. The dome's
/// radius `R` and the entrance pupil's offset `e` from its centre (positive
/// toward the glass) refract each ray; the virtual image of a distant subject
/// is a sphere of radius `R n / (n - 1)` about the dome's centre, which adds
/// field curvature to the focus field.
pub mod dome_port {
    /// Dome radius in metres.
    pub const RADIUS: usize = 0;
    /// Pupil offset from the dome centre in metres.
    pub const PUPIL_OFFSET: usize = 1;
    /// Refractive index of water (<= 0: 1.333).
    pub const INDEX: usize = 2;
    pub const DISPERSION: usize = 3;
    /// Scale of the virtual image's field curvature (0 off, 1 physical).
    pub const CURVATURE: usize = 4;
}

/// `params` slots of an [`element_kind::OPTICAL_VIGNETTE`] element. The
/// transmission is the share of the iris left by the barrel: the unit iris
/// circle intersected with a circle of radius `BARREL_RADIUS` displaced
/// `CATS_EYE * r` toward the axis (`r` the field radius, 1 at the corner):
/// the geometry of the cat's eye of a `PUPIL` element, so give both the same
/// numbers. With a `REFERENCE_F_NUMBER` the barrel is specified at that stop
/// and looks larger against a smaller iris as the lens stops down.
pub mod optical_vignette {
    pub const CATS_EYE: usize = 0;
    pub const BARREL_RADIUS: usize = 1;
    /// F-number at which `BARREL_RADIUS` holds (<= 0: the current one).
    pub const REFERENCE_F_NUMBER: usize = 2;
}

/// `params` slots of an [`element_kind::MECHANICAL_VIGNETTE`] element.
pub mod mechanical_vignette {
    /// 0 circular, 1 rectangular.
    pub const SHAPE: usize = 0;
    /// Radius (circular) or half width (rectangular), image-height units.
    pub const SIZE_X: usize = 1;
    /// Half height (rectangular).
    pub const SIZE_Y: usize = 2;
    /// Width of the soft edge, image-height units.
    pub const FEATHER: usize = 3;
    /// Rotation of the rectangle in radians.
    pub const ROTATION: usize = 4;
}

/// `params` slots of an [`element_kind::GRADUATED_ND`] element: density
/// `STOPS` and a colour `TINT` ramp from clear to full across `SOFTNESS`
/// around the line `dot(q, n) = OFFSET`, `n = (cos ANGLE, sin ANGLE)`
/// pointing to the dark side.
pub mod graduated_nd {
    pub const STOPS: usize = 0;
    pub const ANGLE: usize = 1;
    pub const OFFSET: usize = 2;
    /// Transition width in image-height units (hard ~0.01, soft ~0.5).
    pub const SOFTNESS: usize = 3;
    /// Colour transmittance at full density (red, green, blue), slots 4..7.
    pub const TINT: usize = 4;
}

/// `params` slots of an [`element_kind::PUPIL`] element.
pub mod pupil {
    /// Blade count: 0 circular, 3..=16 polygon, negative inherits the lens's
    /// `aperture_blades` and `aperture_rotation`.
    pub const BLADES: usize = 0;
    /// Blade curvature: 0 straight edges, 1 circle.
    pub const CURVATURE: usize = 1;
    /// Iris rotation in radians (used when `BLADES` >= 0).
    pub const ROTATION: usize = 2;
    /// f-number at and above which the blades are as curved as `CURVATURE`
    /// says; below half of it the iris is a circle, between the two it
    /// blends (wide open rounds, stopped down polygonal). 0: off.
    pub const ROUND_UNTIL_F: usize = 3;
    /// Layer of the aperture texture array.
    pub const TEXTURE_LAYER: usize = 4;
    /// 0..1: how far the texture replaces the polygon (0 ignores it).
    pub const TEXTURE_AMOUNT: usize = 5;
    /// Anamorphic squeeze ratio (>= 1; below that is 1): the bokeh is this
    /// many times taller than wide.
    pub const SQUEEZE: usize = 6;
    /// Cat's eye strength: the clipping circle is displaced `strength * r`
    /// pupil radii toward the optical axis, `r` the field radius (1 at the
    /// corner).
    pub const CATS_EYE: usize = 7;
    /// Radius of the cat's eye clipping circle relative to the iris (the
    /// lens barrel; <= 0: 1).
    pub const BARREL_RADIUS: usize = 8;
    /// Swirl: the bokeh is squashed radially by `1 + swirl * r`, i.e.
    /// stretched tangentially.
    pub const SWIRL: usize = 9;
    /// Spherical aberration, -1..1. Positive: bright rim (soap bubble) behind
    /// the focus; negative: soft, centre-weighted. See `SA_ASYMMETRY`.
    pub const SPHERICAL: usize = 10;
    /// Onion rings: amplitude of the concentric brightness ripple.
    pub const ONION_AMPLITUDE: usize = 11;
    /// Onion rings: ripples across the radius (<= 0: 6).
    pub const ONION_RINGS: usize = 12;
    /// Apodization: standard deviation of the Gaussian radial transmission in
    /// pupil radii (smooth bokeh); 0: off.
    pub const APODIZATION: usize = 13;
    /// Central obstruction (mirror lens donut): hole radius in pupil radii.
    pub const OBSTRUCTION: usize = 14;
    /// 0..1: how much spherical aberration flips with the defocus sign. 0:
    /// the same either side of the focus; 1: the opposite profile in front.
    pub const SA_ASYMMETRY: usize = 15;
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
    /// 0 economical, 1 high (more gather taps), 2 ultra. Also the number of
    /// wavelength samples of lateral chromatic aberration: 3, 7, 16.
    #[gpu]
    pub quality: u32,
    #[gpu]
    pub focal_length_mm: f32,
    #[gpu]
    pub f_number: f32,
    /// Transmission stop; the light lost against the f-number is applied by
    /// a [`element_kind::T_STOP_LOSS`] element.
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

    /// The pupil function inside `region`. Settings start neutral and
    /// inherit the lens's iris; refine with the `with_*` builders.
    pub fn pupil(region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[pupil::BLADES] = -1.0;
        params[pupil::SQUEEZE] = 1.0;
        Self::with_region(element_kind::PUPIL, params, region)
    }

    /// Longitudinal chromatic aberration: focus shifts in dioptres per
    /// channel inside `region`. Negative red and blue against zero green
    /// focus them farther than green: purple foreground fringes, green
    /// background ones.
    pub fn longitudinal_ca(red_dpt: f32, green_dpt: f32, blue_dpt: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = red_dpt;
        params[1] = green_dpt;
        params[2] = blue_dpt;
        Self::with_region(element_kind::LONGITUDINAL_CA, params, region)
    }

    /// A projection of the rendered rectilinear image. Kannala-Brandt and
    /// Panini take their coefficients from [`Self::with_kannala_brandt`] and
    /// [`Self::with_panini`].
    pub fn projection(model: ProjectionModel, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[projection::MODEL] = model as u32 as f32;
        if model == ProjectionModel::Panini {
            params[projection::PANINI_D] = 1.0;
            params[projection::PANINI_S] = 1.0;
        }
        Self::with_region(element_kind::PROJECTION, params, region)
    }

    /// Kannala-Brandt `k1..k4` of a [`ProjectionModel::KannalaBrandt`].
    pub fn with_kannala_brandt(mut self, k: [f32; 4]) -> Self {
        self.params[projection::K1..projection::K1 + 4].copy_from_slice(&k);
        self
    }

    /// Panini distance and vertical compression.
    pub fn with_panini(mut self, d: f32, s: f32) -> Self {
        self.params[projection::PANINI_D] = d;
        self.params[projection::PANINI_S] = s;
        self
    }

    /// OpenCV Brown-Conrady distortion: `k1..k6` and tangential `p1`, `p2`.
    pub fn brown_conrady(k: [f32; 6], p: [f32; 2], region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[..6].copy_from_slice(&k);
        params[brown_conrady::P1] = p[0];
        params[brown_conrady::P2] = p[1];
        Self::with_region(element_kind::BROWN_CONRADY, params, region)
    }

    /// Division-model distortion.
    pub fn division(l1: f32, l2: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[division::L1] = l1;
        params[division::L2] = l2;
        Self::with_region(element_kind::DIVISION, params, region)
    }

    /// Anamorphic distortion with per-axis radial coefficients.
    pub fn anamorphic_distortion(kx: [f32; 2], ky: [f32; 2], region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[anamorphic_distortion::KX1..anamorphic_distortion::KX1 + 2].copy_from_slice(&kx);
        params[anamorphic_distortion::KY1..anamorphic_distortion::KY1 + 2].copy_from_slice(&ky);
        Self::with_region(element_kind::ANAMORPHIC_DISTORTION, params, region)
    }

    /// Anamorphic squeeze (`> 1` sees that many times more horizontally).
    pub fn anamorphic_squeeze(squeeze: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = squeeze;
        Self::with_region(element_kind::ANAMORPHIC_SQUEEZE, params, region)
    }

    /// Lens shift in image-height units.
    pub fn shift(x: f32, y: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = x;
        params[1] = y;
        Self::with_region(element_kind::SHIFT, params, region)
    }

    /// Keystone / perspective correction.
    pub fn keystone(kx: f32, ky: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[keystone::KX] = kx;
        params[keystone::KY] = ky;
        Self::with_region(element_kind::KEYSTONE, params, region)
    }

    /// Focus breathing: magnification change per dioptre of focus.
    pub fn focus_breathing(per_dioptre: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[breathing::PER_DIOPTRE] = per_dioptre;
        Self::with_region(element_kind::FOCUS_BREATHING, params, region)
    }

    /// `count` (2..=8) copies spaced `offset` image heights apart along a
    /// line at `angle`; refine with [`Self::with_prism_ring`],
    /// [`Self::with_prism_spin`] and [`Self::with_prism_falloff`].
    pub fn prism(count: u32, offset: f32, angle: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[prism::COUNT] = count as f32;
        params[prism::OFFSET] = offset;
        params[prism::ANGLE] = angle;
        Self::with_region(element_kind::PRISM, params, region)
    }

    /// Places a prism's copies around a ring instead of along a line.
    pub fn with_prism_ring(mut self) -> Self {
        self.params[prism::RING] = 1.0;
        self
    }

    /// Turns each prism copy `radians` more than the one before.
    pub fn with_prism_spin(mut self, radians: f32) -> Self {
        self.params[prism::SPIN] = radians;
        self
    }

    /// Weight ratio between successive prism copies.
    pub fn with_prism_falloff(mut self, ratio: f32) -> Self {
        self.params[prism::FALLOFF] = ratio;
        self
    }

    /// Spectral lateral chromatic aberration: magnification `1 + d1 t + d2 t^2`.
    pub fn lateral_ca(d1: f32, d2: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[lateral_ca::D1] = d1;
        params[lateral_ca::D2] = d2;
        Self::with_region(element_kind::LATERAL_CA, params, region)
    }

    /// An underwater flat port (`index` <= 0: 1.333; water disperses by
    /// about 0.006).
    pub fn flat_port(index: f32, dispersion: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[flat_port::INDEX] = index;
        params[flat_port::DISPERSION] = dispersion;
        Self::with_region(element_kind::FLAT_PORT, params, region)
    }

    /// An underwater dome port of `radius_m` with the pupil `pupil_offset_m`
    /// from its centre. Water index 1.333 and the physical virtual image
    /// curvature; override with the `params` slots.
    pub fn dome_port(radius_m: f32, pupil_offset_m: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[dome_port::RADIUS] = radius_m;
        params[dome_port::PUPIL_OFFSET] = pupil_offset_m;
        params[dome_port::CURVATURE] = 1.0;
        Self::with_region(element_kind::DOME_PORT, params, region)
    }

    /// Natural `cos^4` falloff, `amount` 0..1.
    pub fn natural_vignette(amount: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = amount;
        Self::with_region(element_kind::NATURAL_VIGNETTE, params, region)
    }

    /// Optical vignetting by the lens barrel; see [`optical_vignette`].
    pub fn optical_vignette(
        cats_eye: f32,
        barrel_radius: f32,
        reference_f_number: f32,
        region: LensRegion<'_>,
    ) -> Self {
        let mut params = [0.0; 16];
        params[optical_vignette::CATS_EYE] = cats_eye;
        params[optical_vignette::BARREL_RADIUS] = barrel_radius;
        params[optical_vignette::REFERENCE_F_NUMBER] = reference_f_number;
        Self::with_region(element_kind::OPTICAL_VIGNETTE, params, region)
    }

    /// A circular hood: clear inside `radius` image heights, `feather` wide.
    pub fn circular_hood(radius: f32, feather: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[mechanical_vignette::SHAPE] = 0.0;
        params[mechanical_vignette::SIZE_X] = radius;
        params[mechanical_vignette::FEATHER] = feather;
        Self::with_region(element_kind::MECHANICAL_VIGNETTE, params, region)
    }

    /// A rectangular matte box with half sizes `half` image heights.
    pub fn matte_box(half: [f32; 2], rotation: f32, feather: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[mechanical_vignette::SHAPE] = 1.0;
        params[mechanical_vignette::SIZE_X] = half[0];
        params[mechanical_vignette::SIZE_Y] = half[1];
        params[mechanical_vignette::FEATHER] = feather;
        params[mechanical_vignette::ROTATION] = rotation;
        Self::with_region(element_kind::MECHANICAL_VIGNETTE, params, region)
    }

    /// Neutral density of `stops` stops.
    pub fn nd_filter(stops: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = stops;
        Self::with_region(element_kind::ND_FILTER, params, region)
    }

    /// Graduated ND: `stops` of density (colour `tint`, linear transmittance
    /// at full density) across a transition of `softness` image heights
    /// about the line at `offset` along the direction `angle`, which points
    /// to the dark side.
    pub fn graduated_nd(
        stops: f32,
        angle: f32,
        offset: f32,
        softness: f32,
        tint: [f32; 3],
        region: LensRegion<'_>,
    ) -> Self {
        let mut params = [0.0; 16];
        params[graduated_nd::STOPS] = stops;
        params[graduated_nd::ANGLE] = angle;
        params[graduated_nd::OFFSET] = offset;
        params[graduated_nd::SOFTNESS] = softness;
        params[graduated_nd::TINT..graduated_nd::TINT + 3].copy_from_slice(&tint);
        Self::with_region(element_kind::GRADUATED_ND, params, region)
    }

    /// A colour filter of linear `transmittance`, `amount` 0..1.
    pub fn color_filter(transmittance: [f32; 3], amount: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[..3].copy_from_slice(&transmittance);
        params[3] = amount;
        Self::with_region(element_kind::COLOR_FILTER, params, region)
    }

    /// The light lost between the f-number and the T-stop.
    pub fn t_stop_loss(amount: f32, region: LensRegion<'_>) -> Self {
        let mut params = [0.0; 16];
        params[0] = amount;
        Self::with_region(element_kind::T_STOP_LOSS, params, region)
    }

    /// Sets an iris: `blades` (0 circular, 3..=16), edge `curvature` (0
    /// straight, 1 circle) and `rotation` in radians. Pupil elements.
    pub fn with_iris(mut self, blades: u32, curvature: f32, rotation: f32) -> Self {
        self.params[pupil::BLADES] = blades as f32;
        self.params[pupil::CURVATURE] = curvature;
        self.params[pupil::ROTATION] = rotation;
        self
    }

    /// Rounds the blades when the lens is wide open; see
    /// [`pupil::ROUND_UNTIL_F`].
    pub fn with_f_number_rounding(mut self, straight_from_f_number: f32) -> Self {
        self.params[pupil::ROUND_UNTIL_F] = straight_from_f_number;
        self
    }

    /// Uses layer `layer` of the aperture texture array as the iris shape,
    /// blended `amount` (0..1) over the polygon.
    pub fn with_aperture_texture(mut self, layer: u32, amount: f32) -> Self {
        self.params[pupil::TEXTURE_LAYER] = layer as f32;
        self.params[pupil::TEXTURE_AMOUNT] = amount;
        self
    }

    /// Anamorphic squeeze ratio (1.33, 1.5, 1.8, 2.0...).
    pub fn with_anamorphic(mut self, squeeze: f32) -> Self {
        self.params[pupil::SQUEEZE] = squeeze;
        self
    }

    /// Cat's eye (optical vignetting); `barrel_radius` <= 0 means 1.
    pub fn with_cats_eye(mut self, strength: f32, barrel_radius: f32) -> Self {
        self.params[pupil::CATS_EYE] = strength;
        self.params[pupil::BARREL_RADIUS] = barrel_radius;
        self
    }

    /// Swirl: tangential stretch growing with the field radius.
    pub fn with_swirl(mut self, swirl: f32) -> Self {
        self.params[pupil::SWIRL] = swirl;
        self
    }

    /// Spherical aberration profile; see [`pupil::SPHERICAL`] and
    /// [`pupil::SA_ASYMMETRY`].
    pub fn with_spherical_aberration(mut self, amount: f32, asymmetry: f32) -> Self {
        self.params[pupil::SPHERICAL] = amount;
        self.params[pupil::SA_ASYMMETRY] = asymmetry;
        self
    }

    /// Onion rings: concentric ripple of `amplitude` with `rings` ripples.
    pub fn with_onion_rings(mut self, amplitude: f32, rings: f32) -> Self {
        self.params[pupil::ONION_AMPLITUDE] = amplitude;
        self.params[pupil::ONION_RINGS] = rings;
        self
    }

    /// Gaussian apodization of standard deviation `sigma` pupil radii.
    pub fn with_apodization(mut self, sigma: f32) -> Self {
        self.params[pupil::APODIZATION] = sigma;
        self
    }

    /// Central obstruction of `ratio` pupil radii (donut bokeh).
    pub fn with_central_obstruction(mut self, ratio: f32) -> Self {
        self.params[pupil::OBSTRUCTION] = ratio;
        self
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
