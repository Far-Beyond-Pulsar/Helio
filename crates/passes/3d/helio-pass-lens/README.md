# Physically based screen-space lens

`LensPass` is the camera's lens: it applies what the optics do to the image the
sensor would record, in scene-linear radiance, before anything is metered,
bloomed or tone mapped. It models the **focus field**, the **thin-lens
defocus** it implies, the **pupil function** (what that defocus looks like)
with longitudinal chromatic aberration, the **ray map** (projection,
distortion, spectral lateral chromatic aberration) and the **transmission**
(vignetting, filters, T-stop). The old depth of field (`helio-pass-dof`) ran on the tone
mapped image and had no optics behind it; it is gone, and so are its
post-process settings.

## The model

A lens is an ordered stack of optical elements. The stack reduces, per pixel,
to four quantities:

| Quantity | Now |
| --- | --- |
| Ray map (sensor uv to scene uv, per wavelength) | composed in stack order from the projection, distortion, shift, keystone, breathing, prism, lateral CA and port elements |
| Focus field `s(uv)` | resolved: base focus, tilt, field curvature, diopters, per-channel focus shift |
| Pupil function | `pupil_shape` and `pupil_profile`: iris, squeeze, cat's eye, swirl, aberration, apodization, obstruction |
| Transmission | natural cos⁴, optical and mechanical vignetting, ND / graduated ND / colour filters, T-stop loss, diopter seam darkening |

Later phases add element kinds (prescription LUTs, more aberrations) to the
same element table; the shader dispatches on `LensElementComponent::kind`.

### Focus field

Focus is carried in dioptres so elements add:

```text
1/s' = 1/s + sum(D)
```

- **Base focus** is `focus_distance_m`, or the autofocus position.
- **Tilt** (Scheimpflug) tilts the focal plane by `tilt_angle` about an image
  plane axis at `tilt_axis`: the ray through a pixel meets it at
  `cos(a) s0 / (cos(a) + sin(a) (axis . d))`.
- **Field curvature** (Petzval) adds `field_curvature * r^2` dioptres, `r` the
  distance from the centre, 1 at the corner.
- **Diopter** elements add `params[0]` dioptres where their region covers the
  pixel. Any number stack: two-way and three-way splits, nested regions, any
  region shape. Where several cover a pixel their powers sum.
- **Sweet spot** elements (Lensbaby) leave focus normal inside their region and
  add `params[0]` pixels of defocus outside it, in quadrature with the thin-lens
  circle.

### Defocus

The signed circle of confusion, a diameter in input pixels, negative in front
of the focal surface:

```text
c = A |s2 - s1| / s2 * f / (s1 - f),   A = f / N
```

`s1` is the pixel's focus distance from the field, `s2` the scene depth along the
optical axis, converted through `sensor_width_mm` to pixels. Sky is at infinity.

### Pupil function

Each source spreads its light over its circle of confusion, and the pupil
function says how: `P(rho, theta; field position, defocus sign, wavelength)`,
evaluated per gather sample in the pupil's own frame (the offset divided by the
source's circle radius). It splits into a **shape** (where light passes, 0..1)
and a **profile** (how bright it is across the disc). Every shape lies inside
the unit circle, so the tile search radii of the circle of confusion stay valid.

Where it lives: lens-wide `aperture_blades` / `aperture_rotation` on
`LensComponent` give the base iris (a straight polygon, or a circle at 0
blades). Everything else is a `PUPIL` **element**, so it can be limited to a
region and different screen areas can have different bokeh. Where several
`PUPIL` elements cover a pixel the last one wins; there is no blending across a
region edge (the region is tested at 50% coverage). `LensComponent` is unchanged.

| `PUPIL` params | Effect |
| --- | --- |
| `BLADES` `CURVATURE` `ROTATION` | Polygon iris, 3..16 blades (0 circle; negative inherits the lens's blades and rotation). Curvature 0 is straight blades, 1 a circle |
| `ROUND_UNTIL_F` | f-number at which the blades are as curved as `CURVATURE` says: a circle wide open (below half of it), blending up to it (wide open round, stopped down polygonal). 0 disables |
| `TEXTURE_LAYER` `TEXTURE_AMOUNT` | A layer of the aperture texture array (coverage in red, the square around the pupil circle, masked by the circle) blended over the polygon |
| `SQUEEZE` | Anamorphic: the pupil is this many times taller than wide (1.33, 1.5, 1.8, 2.0) |
| `CATS_EYE` `BARREL_RADIUS` | Optical vignetting: the iris is clipped by a second circle (radius `BARREL_RADIUS`, default 1) displaced toward the optical axis by `CATS_EYE * r` pupil radii, `r` the field radius (1 at the corner). Round on the axis, lens-shaped toward the corners |
| `SWIRL` | The disc is squashed radially by `1 + SWIRL * r`: tangential stretch growing with the field radius (Petzval, Helios) |
| `SPHERICAL` `SA_ASYMMETRY` | Spherical aberration, a brightness profile `1 + e (3 rho^4 - 1)` (mean 1): `e > 0` a bright rim (soap bubble), `e < 0` centre-weighted (soft). `e = SPHERICAL * mix(1, sign, SA_ASYMMETRY)` with `sign` +1 behind the focus and -1 in front, so an asymmetry of 1 shows a rim on one side of the focus and a soft disc on the other |
| `ONION_AMPLITUDE` `ONION_RINGS` | Concentric ripple `1 + a cos(2 pi n rho)` |
| `APODIZATION` | Gaussian radial transmission of this standard deviation in pupil radii (smooth bokeh) |
| `OBSTRUCTION` | Central obstruction radius ratio: mirror-lens donut bokeh |

**Energy.** The shape is normalised by its area as a fraction of the circle
(integrated per gather pixel with 96 samples) and the profile by its mean over
the shape on each side of the focus. A pupil change reshapes and redistributes
the light of a defocused source but does not change the frame's exposure: the
energy test holds at the phase-1 tolerance for every variant.

The rim coordinate for the profile is the shape's own boundary, so a polygon
has a bright rim along its edges. The profile also modulates the near layer's
coverage, so the aberration shows on foreground bokeh too.

The shape is evaluated against the field position of the gather pixel (the
destination), not of each source, and the region test is also at the
destination; the difference is a disc radius, small against the field.

### Longitudinal chromatic aberration

A `LONGITUDINAL_CA` element shifts the focus of each colour channel in
dioptres (`1/s\x27 = 1/s + D`), `params[0..3]` for red, green, blue, weighted by
the region coverage and summed where several overlap. Green is the reference
the focus field was resolved for, so only the differences from green matter.
Each channel gathers with its own circle of confusion,
`c_ch = c + A f dD / (s1 - f)` (the first-order change, with `s1` the lens's
focus distance), clamped to `max_coc_px`.

With red and blue shifted toward farther focus than green (negative `D`), a
foreground edge has larger red and blue circles, so it fringes purple at the
edge of its bokeh; a background edge has the larger green circle and fringes
green. The size of the effect is `A f / sensor_width * width` pixels per
dioptre. A pixel with no shift runs one gather as before; a shifted one runs
three (one per channel), so the cost triples inside the region. Both quality
tiers do the same; the economical tier only has fewer taps.

### Ray map

The rendered image is rectilinear at the lens's focal length. The ray map says
where on it each sensor position looks, per wavelength, and is applied in
`cs_composite` **after** the defocus, which runs in the undistorted scene.
Elements apply in stack order to a point in focal-length units (`x = p_mm /
f`, y down: the tangent coordinates of the rendered image, as in OpenCV's
normalised camera coordinates). Each element's region is tested at the *sensor*
position and blends the element's map with the identity across its feather.
Order matters: `[SHIFT, BROWN_CONRADY]` distorts about a shifted axis,
`[BROWN_CONRADY, SHIFT]` shifts the distorted image.

| Element | `params` (see `components.rs`) | Map |
| --- | --- | --- |
| `PROJECTION` | model; `k1..k4` (Kannala-Brandt) or `d`, `s` (Panini) | rectilinear (identity), equidistant `r = θ`, equisolid `2 sin(θ/2)`, orthographic `sin θ`, stereographic `2 tan(θ/2)`, Kannala-Brandt `θ + k1 θ³ + …`, Panini, cylindrical |
| `BROWN_CONRADY` | `k1..k6`, `p1`, `p2` | OpenCV rational radial + tangential model, `x_d = x · (1 + k1 r² + k2 r⁴ + k3 r⁶)/(1 + k4 r² + k5 r⁴ + k6 r⁶) +` tangential. `k1 < 0` is barrel |
| `DIVISION` | `l1`, `l2` | `x = x_d / (1 + l1 r_d² + l2 r_d⁴)` |
| `ANAMORPHIC_DISTORTION` | `kx1`, `kx2`, `ky1`, `ky2` | per-axis radial polynomials (mumps) |
| `ANAMORPHIC_SQUEEZE` | factor | the horizontal field of view is that many times wider |
| `SHIFT` | `x`, `y` (image heights) | optical centre offset (tilt-shift) |
| `KEYSTONE` | `kx`, `ky` | `x / (1 + kx x_x + ky x_y)` |
| `FOCUS_BREATHING` | per dioptre, reference dioptres | magnification `1 + k (D − D_ref)` for the lens's focus `D` (after autofocus) |
| `PRISM` | count 2..8, offset, angle, ring, spin, falloff | weighted displaced / rotated copies (below) |
| `LATERAL_CA` | `d1`, `d2` | magnification `1 + d1 t + d2 t²`, `t = (λ − 550 nm)/150 nm` |
| `FLAT_PORT` | index, dispersion | a ray in water at `θ_w` reaches the camera at `sin θ = n sin θ_w`: magnification ≈ n, pincushion, dispersion |
| `DOME_PORT` | radius, pupil offset, index, dispersion, curvature | refraction at the dome for a pupil off its centre, plus field curvature from the virtual image |

Brown-Conrady and the anamorphic distortion are specified as the *forward*
model (ideal point to where the sensor records it), as calibrations are; the
sensor-to-scene map needs the inverse, which the shader finds with eight
Newton steps. A distortion with no inverse over the frame (a barrel stronger
than its polynomial can invert at the edge) falls back to whatever the
iteration reaches; keep the polynomial monotonic over the frame.

**Defocus size.** A circle of confusion is a size on the sensor, but it is
gathered in the undistorted scene, where the map stretches it by the local
magnification. `cs_coc` multiplies the circle by `sqrt(det J)`, `J` the
Jacobian of the sensor-to-scene map at the sensor point that looks at the
pixel (found with Newton steps), so blur comes out the size optics says. The
correction is isotropic: a map that stretches one axis more than the other
blurs by the mean.

**Prism.** Copy `i` of `N` samples the mapped image displaced by an offset (a
line centred on the original, or a ring) after turning it `i · spin` about the
frame centre, weighted `falloff^i` and normalised to sum to 1, so the filter
redistributes light and never changes the frame's energy. The first prism
covering a pixel acts; across its region's feather it blends with the unprismed
image.

**Underwater ports.** The flat port's refraction (n = 1.333, about 0.006
dispersion for water) is exact for a pupil on the port; the dome port's follows
the law of sines in the triangle of pupil, dome centre and glass, `θ_w = θ − i +
asin(sin i / n)`, `sin i = (e/R) sin θ`, which is no deviation for a centred
pupil. The dome's virtual image of a distant subject is a sphere of radius
`R n / (n − 1)` about its centre, so the focus field gains `(1/cos θ − 1) / ρ`
dioptres toward the corners (`CURVATURE` scales it).

**Spectral lateral chromatic aberration.** A `LATERAL_CA` element, or a port
with dispersion, gives each wavelength its own map. The composite samples N
wavelengths over 400–700 nm (3, 7, 16 for `quality` 0, 1, 2), each channel
takes a Gaussian sensor response of them, and the weights are normalised so
each channel's sum is exactly 1: a lens whose wavelengths agree reproduces the
image. A wavelength's map is the stack evaluated at that sample's `t` (−1 blue
… +1 red), so `d1 > 0` magnifies red more than blue and a white line off the
axis splits red outward and blue inward. The cost is N full defocus fetches
per pixel.

**Edges.** Outside the rendered frame the map clamps to the edge pixel (it does
not fade): where a distortion or fisheye looks past the frame it repeats the
last row or column.

### Overscan

The map's reach beyond the frame is published as `ResolvedLens::
required_overscan` (and by `ray_map::required_overscan` on the CPU, from the
lens row and its elements): the factor, at least 1 and exactly 1 for a map that
stays inside the frame, by which the scene extent at the frame's boundary
exceeds the frame's half extent, over the spectrum's ends (`t = −1, 0, 1`) and
every prism copy. It is sampled at 64 boundary points (16 per edge, corners
included) and capped at 16; a map that folds back inside the frame can reach
further than its boundary does.

A camera consumes it by widening its field of view's *tangent extent* by this
factor about the frame centre, rendering that, and handing the lens the wider
image. The lens does not yet crop or rescale a wider image, so until a camera
pads the clamp above applies. A CPU consumer (the camera is authored on the CPU)
builds the lens's `LensComponent` and its `LensElementComponent` stack and calls
`ray_map::required_overscan(&lens, &stack, aspect)`; it reads breathing from
`focus_distance_m`, as only the GPU knows the autofocus position. A GPU consumer
reads `lens_resolved` (a frame late).

### Transmission

The product of the elements covering a sensor position, per colour channel,
applied to the composite:

| Element | Effect |
| --- | --- |
| `NATURAL_VIGNETTE` | `cos⁴ θ` of the ray's real angle after the map (so a fisheye follows its projection), mixed by the amount. The only natural falloff: the flare and post-process passes have none of their own |
| `OPTICAL_VIGNETTE` | iris share left by the barrel: the unit circle against a circle of `BARREL_RADIUS` displaced `CATS_EYE · r` toward the axis, the geometry of the `PUPIL` cat's eye. With a reference f-number the barrel is a fixed size, so it clips less as the iris shrinks (`barrel · N / N_ref`) |
| `MECHANICAL_VIGNETTE` | circular hood or rotated rectangular matte box with a smooth feather |
| `ND_FILTER` | `2^−stops` |
| `GRADUATED_ND` | colour-tinted density ramping across `softness` about a line (hard ≈ 0.01, soft ≈ 0.5) |
| `COLOR_FILTER` | linear transmittance, mixed by the amount |
| `T_STOP_LOSS` | `(f_number / t_stop)²`, mixed by the amount |

`t_stop` is carried on the lens and applied only when a `T_STOP_LOSS` element
asks for it, so existing lenses keep their exposure.

### Seams

A diopter sits in front of the lens, out of focus, so its edge is soft: the
entrance pupil seen from the filter, `A / e` radians for a filter `e` metres
ahead of the lens (`params[2]`, default 0.08 m). The default feather is derived
from that and the aperture, so a seam narrows as the lens stops down; an
authored `feather` (image-height units) replaces it. `params[1]` darkens the
seam.

Across a seam a pixel sees both focus configurations. The two strongest are kept
per pixel (`coc_a`, `coc_b`, weight, at most one half) and the gather runs once
for each, blended by the weight.

### Regions

Regions live in image-height units centred on the frame, `q = (uv - 0.5) *
(aspect, 1)`, y down: full frame, half-plane `{angle, offset}`, ellipse
`{centre, radii, rotation}`, convex polygon of up to 8 vertices (either
winding), radial band `{r0, r1, centre}`. A mask-texture kind is reserved and
covers nothing yet. `LensRegion` builds the rows.

## Settings

Two packed SceneDB components, read straight from GPU memory:

| Row | Buffer | Matched by |
| --- | --- | --- |
| `LensComponent` | `camera_lenses` | `view_id` (u32 bits of the camera's `jitter_frame.w`); the first enabled row wins, none disables the lens |
| `LensElementComponent` | `lens_elements` | rows `element_first .. element_first + element_count` of the owning lens (`element_count` is capped at 32; a `NONE` row is skipped) |

The lens is owned by the camera. Post-process volumes do not blend it.

| `LensComponent` field | Meaning |
| --- | --- |
| `enabled`, `view_id` | Row switch and view match |
| `focal_length_mm`, `f_number`, `t_stop` | Optics; `t_stop` loses light against the f-number through a `T_STOP_LOSS` element |
| `sensor_width_mm`, `sensor_height_mm` | Sensor; height 0 derives 3:2 |
| `focus_mode`, `focus_distance_m` | 0 manual; 1 autofocus on `af_u`, `af_v` |
| `af_radius`, `af_speed` | Autofocus sample radius in uv (0: 0.02) and ease rate in 1/s (0: instant) |
| `quality`, `max_coc_px` | 0 economical (24 taps, 3 wavelengths), 1 high (56, 7) or 2 ultra (56, 16); circle cap in pixels (0: 32) |
| `element_first`, `element_count` | The element rows |
| `tilt_angle`, `tilt_axis`, `field_curvature` | Focus field |
| `aperture_blades`, `aperture_rotation` | Iris: the base pupil polygon of the gather, and the flare's |

Autofocus averages a grid of depth taps in dioptres, eases in dioptres at
`af_speed`, and holds its last focus when the point sees only sky.

## Stages

All dispatches are sized on the GPU by `cs_resolve`, so a view with no lens costs
one workgroup and a copy, with no CPU readback.

1. `cs_resolve`: matches the row, runs autofocus, scans the stack for the
   stages it needs (`map_flags`), measures the ray map's `required_overscan` on
   64 boundary samples, writes the `lens_resolved` buffer and the indirect
   arguments (zero when disabled, except the copy).
2. `cs_coc`: per-pixel circle of confusion from the focus field, scaled by the
   ray map's local magnification; depth is read with a nearest-of-footprint
   fetch when it is rendered below the image's size.
3. `cs_tiles`, `cs_dilate`: 16x16 tile extents of the circles, widened by every
   circle that reaches the tile, bounding each pixel's gather radius and
   skipping the near field where there is none.
4. `cs_prefilter`: half-resolution colour and circles; a block with a near-field
   pixel takes the near class.
5. `cs_gather`: scatter-as-gather. Each source spreads its radiance over a disc of
   its own circle, shaped by the pupil function and weighted by one over the
   disc's area. Far-field samples are
   averaged (a mean, so a uniform blur keeps the image's energy); near-field
   samples form a layer whose alpha is the fraction of the pixel covered, over
   the far layer. Taps are a Vogel spiral rotated per pixel.
6. `cs_composite`: in-focus pixels keep their full-resolution value, blurred ones
   take the far layer, the near layer goes over both; with a ray map this runs
   at the position the map says, once per wavelength sample and prism copy, and
   the spectral weights and the transmission are applied. A lens with no map
   elements copies texel for texel; a disabled lens copies the input.

## Graph contract

```text
FogCompositePass -> Transparent -> TSR/FXAA -> LensPass -> LensFlarePass -> PostProcessPass
                                   (tsr_color | fxaa_color | fogged_hdr)
```

- Reads the configured HDR input (`with_color_input`; default `fogged_hdr`,
  falling back to `pre_aa`) and the scene depth. Depth may be at internal
  resolution while colour is at output resolution after TSR upscaling.
- Writes `lens_color`: Rgba16Float, scene-linear, the size of its input.
  `LensFlarePass` and `PostProcessPass` read it.
- Publishes `lens_resolved`: a storage buffer holding one `ResolvedLens` (80
  bytes: the optics, `natural_vignette`, `required_overscan`, `map_flags`,
  `spectral_samples`). It
  holds the neutral 50 mm f/2.8, 36 mm, six-blade lens when the view has none, so
  consumers always have optics. `LensFlarePass` takes f-number, focal length,
  sensor width and the iris, and the natural falloff for analytic sources,
  from it; post-process settings no longer carry them (nor chromatic
  aberration or a lens vignette: both are the lens's).
- Records on the graphics encoder after its producers. The compute encoder is
  submitted before graphics and would sample the previous frame.
- Hosts register the columns up front, like the other SceneDB rows:
  `LensComponent::register_gpu_columns_growable` and
  `LensElementComponent::register_gpu_columns_growable`. A graph without the
  `camera_lenses` buffer runs with no lens.
- Reads the optional registry entry `lens_aperture_shapes`
  (`APERTURE_SHAPES`): a `D2Array` view of R8Unorm iris shapes, see
  `aperture_shape_array`. Without it one white 1x1 layer is bound, so the
  binding is always valid.

## Limits

- Screen-space: nothing outside the frame or behind opaque geometry contributes,
  so occluded background is reconstructed from its surroundings.
- The ray map resamples the defocused image bilinearly at the pixel's scene
  position, so a strong minification or a heavy distortion aliases a fine
  pattern the blur does not hide. The defocus size correction is isotropic, and
  the focus field, the pupil and the seams are evaluated in the undistorted
  scene.
- The map depends on the wavelength only for `LATERAL_CA` and dispersive ports;
  every other element is one map for all colours. Transmission is evaluated
  from the green wavelength's ray. Analytic flare sources take only the natural
  falloff.
- Outside the rendered frame the map clamps to the edge pixel; the lens does
  not yet take an overscanned image (see Overscan).
- The pupil is evaluated at the gather pixel, not per source, and a `PUPIL`
  region switches at 50% coverage with no blend. Pupil shape detail is limited
  by the half-resolution gather: a disc under about eight pixels across reads
  as a blob whatever its blades.
- The pupil lies inside the circle of confusion, so a swirl or squeeze makes the
  disc smaller along its short axis rather than larger along its long one;
  `max_coc_px` bounds the long axis. A texture aperture is bounded by the
  circle as well.
- Longitudinal chromatic aberration shifts circles by the first-order change in
  the circle (exact for a lens focused far from its focal length) and widens the
  gather radius by it, but the tile reach (`cs_tiles`, `cs_dilate`) and the
  composite's in-focus test follow the green circle: a shifted channel's disc
  that extends beyond the tiles green reaches is cut at the tile edge, and a
  pixel in focus for green shows no chromatic blur. Inside the region the gather
  costs three times as much.
- The spherical-aberration, onion and apodization profiles redistribute light
  within the disc; sources at different circle sizes are mixed by area, so a
  scene mixing sharp and strongly blurred sources shifts slightly (below the
  3% of the energy test) where profiles differ.
- Half-resolution gather: blur of a few pixels is resolved at half size, and a
  half-resolution block that mixes near and far pixels takes the near class.
- Taps are noise-rotated per pixel, trading structure for grain at the cost of a
  fixed budget; very bright, very small highlights under a large circle are
  under-sampled.
- Seam handling keeps the strongest two focus configurations per pixel; a
  junction of three seams approximates.
- Elements address rows by index range; the authoring layer that places element
  rows is not part of this pass.

## Validation

`cargo test -p helio-pass-lens` runs the production WGSL on a GPU (an adapter is
required; tests fail rather than skip). Phase 3 (`src/tests/optics.rs`) is checked
against closed forms where physics gives one: a rectilinear map is an exact
identity; a point under the OpenCV model lands where the formula says; a
fisheye records radius against angle for each projection (equidistant within
2%); spectral weights sum to 1 per channel and a neutral lens at 3, 7 and 16
samples reproduces the image within fp16; a white line off the axis splits red
outward and blue inward by the weighted prediction, and the axis line does not;
cos⁴ follows the ray's real angle (also through a fisheye); a graduated ND is
monotonic, hard against soft and tinted; the T-stop loss is `(N/T)²`; a prism
preserves energy for a line, a ring with falloff, a kaleidoscope and a region;
`required_overscan` is 1 for the identity and a map that stays inside, above 1
for barrel and fisheye, and the squeeze, shift, CA and prism extents exactly,
GPU against CPU; the flat port's magnification and dispersion and the dome's
refraction and focus curvature against their formulas; breathing against the
focus and the circle of confusion halved by a 2x zoom; the optical vignette
against the circle-circle area; every element against the CPU twin on a smooth
image (and the stack order mattering); and clamping outside the frame. The
older tests: exact passthrough with no or disabled
lens, row matching, the thin-lens circle against the formula, energy
preservation of the gather at both quality tiers, near-field coverage, the
split diopter (both halves in focus, the seam width against the aperture),
stacked three-way and nested diopters, every region kind, tilt and field
curvature, autofocus and its easing, depth below image resolution, and the
component ABI.

The pupil tests measure the averaged result, not single taps (a lone point
source in a defocused plane is a few lucky taps, so its disc is noise): the
10-90% or 30-70% width of a blurred edge for the projected shape, and a small
bright blob binned by radius for the profile. They cover polygon blades (a
hexagon turned by a twelfth of a turn and a triangle against a circle,
curvature, wide-open rounding), anamorphic squeeze 1.33 / 1.8 / 2.0 within
10%, cat's eye (round on the axis, a narrower radial than tangential width off
it) and swirl, soap bubble / soft / near-far asymmetric spherical aberration,
apodization, onion rings, the donut hole, purple foreground and green
background fringes from longitudinal chromatic aberration, a texture aperture
and the neutral layer, region limiting, and the frame energy of fifteen pupil
variants at both quality tiers within 3%.
