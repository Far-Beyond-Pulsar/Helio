# Physically based screen-space lens

`LensPass` is the camera's lens: it applies what the optics do to the image the
sensor would record, in scene-linear radiance, before anything is metered,
bloomed or tone mapped. Phase 1 models the **focus field** and the **thin-lens
defocus** it implies. The old depth of field (`helio-pass-dof`) ran on the tone
mapped image and had no optics behind it; it is gone, and so are its
post-process settings.

## The model

A lens is an ordered stack of optical elements. The stack reduces, per pixel,
to four quantities:

| Quantity | Phase 1 |
| --- | --- |
| Ray map (sensor uv to scene uv, per wavelength) | identity |
| Focus field `s(uv)` | resolved: base focus, tilt, field curvature, diopters |
| Pupil function | a circle (`pupil_coverage`) |
| Transmission | 1, except diopter seam darkening |

Later phases add element kinds (pupil shapes, distortion and projection,
chromatic aberration, vignetting, filters, aberrations, prescription LUTs) to
the same element table and replace the identity quantities; the shader
dispatches on `LensElementComponent::kind`.

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
| `focal_length_mm`, `f_number`, `t_stop` | Optics; `t_stop` is carried for exposure |
| `sensor_width_mm`, `sensor_height_mm` | Sensor; height 0 derives 3:2 |
| `focus_mode`, `focus_distance_m` | 0 manual; 1 autofocus on `af_u`, `af_v` |
| `af_radius`, `af_speed` | Autofocus sample radius in uv (0: 0.02) and ease rate in 1/s (0: instant) |
| `quality`, `max_coc_px` | 0 economical (24 taps) or 1 high (56); circle cap in pixels (0: 32) |
| `element_first`, `element_count` | The element rows |
| `tilt_angle`, `tilt_axis`, `field_curvature` | Focus field |
| `aperture_blades`, `aperture_rotation` | Iris, read by the flare |

Autofocus averages a grid of depth taps in dioptres, eases in dioptres at
`af_speed`, and holds its last focus when the point sees only sky.

## Stages

All dispatches are sized on the GPU by `cs_resolve`, so a view with no lens costs
one workgroup and a copy, with no CPU readback.

1. `cs_resolve`: matches the row, runs autofocus, writes the `lens_resolved`
   buffer and the indirect arguments (zero when disabled, except the copy).
2. `cs_coc`: per-pixel circle of confusion from the focus field; depth is read
   with a nearest-of-footprint fetch when it is rendered below the image's size.
3. `cs_tiles`, `cs_dilate`: 16x16 tile extents of the circles, widened by every
   circle that reaches the tile, bounding each pixel's gather radius and
   skipping the near field where there is none.
4. `cs_prefilter`: half-resolution colour and circles; a block with a near-field
   pixel takes the near class.
5. `cs_gather`: scatter-as-gather. Each source spreads its radiance over a disc of
   its own circle, weighted by one over the disc's area. Far-field samples are
   averaged (a mean, so a uniform blur keeps the image's energy); near-field
   samples form a layer whose alpha is the fraction of the pixel covered, over
   the far layer. Taps are a Vogel spiral rotated per pixel.
6. `cs_composite`: in-focus pixels keep their full-resolution value, blurred ones
   take the far layer, the near layer goes over both. A disabled lens copies the
   input.

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
- Publishes `lens_resolved`: a storage buffer holding one `ResolvedLens`. It
  holds the neutral 50 mm f/2.8, 36 mm, six-blade lens when the view has none, so
  consumers always have optics. `LensFlarePass` takes f-number, focal length,
  sensor width and the iris from it; post-process settings no longer carry them.
- Records on the graphics encoder after its producers. The compute encoder is
  submitted before graphics and would sample the previous frame.
- Hosts register the columns up front, like the other SceneDB rows:
  `LensComponent::register_gpu_columns_growable` and
  `LensElementComponent::register_gpu_columns_growable`. A graph without the
  `camera_lenses` buffer runs with no lens.

## Limits

- Screen-space: nothing outside the frame or behind opaque geometry contributes,
  so occluded background is reconstructed from its surroundings.
- The pupil is a circle; iris shapes, optical vignetting and cat's-eye effects
  are later phases. The ray map is the identity: no distortion, projection or
  chromatic aberration.
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
required; tests fail rather than skip): exact passthrough with no or disabled
lens, row matching, the thin-lens circle against the formula, energy
preservation of the gather at both quality tiers, near-field coverage, the
split diopter (both halves in focus, the seam width against the aperture),
stacked three-way and nested diopters, every region kind, tilt and field
curvature, autofocus and its easing, depth below image resolution, and the
component ABI.
