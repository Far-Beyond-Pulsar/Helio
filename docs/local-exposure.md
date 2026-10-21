# Local exposure

Helio's post-process (`helio-pass-postprocess`) applies a local exposure after the global one. It is on
by default. Shade under tree canopies, cloud shadows and other dark regions move toward middle grey,
and bright regions keep most of their exposure. There is no fill light to tune and no halo at
silhouettes.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `local_exposure_shadow_contrast` | 0.6 | Contrast kept by regions darker than middle grey. 1 leaves shade as it is; a region 4 EV under grey is lifted by 4 × (1 − 0.6) = 1.6 EV. |
| `local_exposure_highlight_contrast` | 0.85 | Contrast kept by regions brighter than middle grey. A region 2 EV over grey is darkened by 0.3 EV. |
| `local_exposure_detail_strength` | 1.0 | Contrast within a region. At 1 the texture keeps its contrast; above 1 it is sharpened, below 1 flattened. |

All three at 1 turn the effect off. The GPU kernels then return at once, and `fs_uber` skips the lookup.

The settings live in `PostProcessSettings` and `GpuPostProcessUniforms` (bytes 192..204, which held the
removed post-process chromatic aberration). Post-process volumes blend them as properties 84–86
(`PostProcessProperty::LOCAL_EXPOSURE_*`). In the editor they are the "Local Exposure" category of the
camera and volume components. Their volume override group (`override_local_exposure`) defaults off,
like lens flare's: a volume placed for grading or fog does not replace the camera's local exposure.

## How it works

1. **Pivot.** EVs are measured from the scene luminance that the frame's exposure brings to middle grey
   (0.18) at the tone curve's input. Auto exposure's adapted mean sits at the pivot. Manual exposure and
   `tonemap_exposure` are included.
2. **Grid** (`cs_local_exposure_grid`). This builds a bilateral grid over screen position and log2
   luminance. The screen is split into 32 rows of square cells (57 × 32 at 16:9). Each cell has 24
   luminance bins of 1 EV, covering −13 to +10 EV from the pivot. One workgroup per cell takes a
   16 × 16 lattice of bilinear taps of the HDR input. Each tap is split between its two nearest bins,
   in integer workgroup atomics.
3. **Blur** (`cs_local_exposure_blur`). A binomial blur runs over the grid: 1 4 6 4 1 across cells and
   1 2 1 across bins. The result is written to an `Rgba16Float` 3D texture holding (sum of EV, weight).
4. **Slice** (`fs_uber`). Each pixel samples the grid trilinearly at its screen position and its own
   EV. The result is the mean EV of nearby pixels about as bright as it (range sigma about 1 EV,
   support ±3 EV). Pixels across a strong edge, such as a canopy against the sky or shade against
   sunlit ground, fall into other bins, so they do not mix. The multiplier is
   `exp2(mean × (contrast − 1) + (detail − 1) × (ev − mean))`, where `contrast` is the shadow or
   highlight scale by the sign of `mean`.

Nothing is carried between frames, so a still camera gives a still result. Bloom and the lens
response keep the global exposure.

## Cost

These are GPU timestamps on an RTX 3060 (`indoor_cathedral_hlfs --capture` with
`HLFS_POSTPROCESS_COMPUTE_TIMINGS`, `HLFS_POSTPROCESS_TIMINGS`, `HLFS_RESOLUTION`, and
`HLFS_NO_LOCAL_EXPOSURE` for the off case).

| | 1080p | 4K |
|---|---|---|
| Grid + blur (off: early-out dispatches) | 0.083 ms (0.011) | 0.140 ms (0.012) |
| `fs_uber` increase | +0.020 ms | +0.078 ms |
| Total | ~0.09 ms | ~0.21 ms |

## Tests

- `local_exposure_tests.rs` runs the production kernels on a GPU. It checks that a dark half beside a
  bright half moves each half toward middle grey by its own contrast, all the way to the edge (no
  halo). It also checks that a uniform image metered to its own mean is unchanged, that contrasts of 1
  change nothing, and that a fine texture inside a region keeps its contrast.
- `abi_tests.rs` checks that the resolver blends the three properties like the Rust blender, and onto
  their own bytes.
