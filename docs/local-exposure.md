# Local exposure

Helio's post-process (`helio-pass-postprocess`) applies a local exposure after the global one. It is on
by default. Shade under tree canopies, cloud shadows and other dark regions move toward middle grey,
and bright regions keep most of their exposure. There is no fill light to tune, no halo at
silhouettes, and no dark rim along soft shadow edges.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `local_exposure_shadow_contrast` | 0.5 | Contrast kept by regions darker than middle grey. 1 leaves shade as it is; a region 4 EV under grey is lifted by 4 × (1 − 0.5) = 2 EV. |
| `local_exposure_highlight_contrast` | 0.85 | Contrast kept by regions brighter than middle grey. A region 2 EV over grey is darkened by 0.3 EV. |
| `local_exposure_detail_strength` | 1.0 | Contrast within a region. At 1 the texture keeps its contrast; above 1 it is sharpened, below 1 flattened. |

All three at 1 turn the effect off. The GPU kernels then return at once, and `fs_uber` skips the lookup.

The defaults were tuned on voxel_craft after the flat shade fill was removed. Forest floor there sits
about 4 EV under sunlit grass; AgX punchy's toe shows that as near black, and 0.5 brings it to about
−2 EV, where it reads as shade. Sunlit grass sits near the pivot and keeps its exposure.

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
   1 2 1 across bins. Each bin then holds the mean EV of the nearby pixels about as bright as it (range
   sigma about 1 EV, support ±3 EV). Pixels across a strong edge, such as a canopy against the sky or
   shade against sunlit ground, sit in other bins, so they do not mix.
4. **Curves** (`cs_local_exposure_curve`). Each cell gets a tone curve over its bins. Each bin's mean is
   scaled toward the pivot by the shadow or highlight contrast, and the bin's offset from that mean by
   the detail strength.
   - Taken pixel by pixel, the mean jumps from one region to the other across a penumbra: pixels a
     little darker than sunlight average with the shade. A brighter pixel could then come out darker
     than its neighbour, which shows as a dark rim along soft shadow edges (gradient reversal).
   - Each cell's curve is therefore made non-decreasing, with a minimum slope of 0.2, by weighted
     isotonic regression (pool adjacent violators).
   - Well-populated bins (the real regions) keep their values; the sparse bins between regions are
     pooled.
   - The curves are stored as gains in EV in an `Rgba16Float` 3D texture.
5. **Slice** (`fs_uber`). Each pixel samples the curves trilinearly at its screen position and its own
   EV. Interpolating non-decreasing curves keeps them non-decreasing, so a brighter input never comes
   out darker.

Nothing is carried between frames, so a still camera gives a still result. Bloom and the lens
response keep the global exposure.

Local exposure scales all three colour channels by the same factor, so it never changes hue.

## Cost

These are GPU timestamps on an RTX 3060 (`indoor_cathedral_hlfs --capture` with
`HLFS_POSTPROCESS_COMPUTE_TIMINGS`, `HLFS_POSTPROCESS_TIMINGS`, `HLFS_RESOLUTION`, and
`HLFS_NO_LOCAL_EXPOSURE` for the off case).

| | 1080p | 4K |
|---|---|---|
| Grid, blur and curves (off: early-out dispatches) | 0.096 ms (0.013) | 0.155 ms (0.013) |
| `fs_uber` increase | +0.019 ms | +0.073 ms |
| Total | ~0.10 ms | ~0.23 ms |

In voxel_craft at 1920 × 1080 (`VOXEL_CRAFT_TIMINGS=120`), the PostProcess pass goes from 0.113 ms to
0.204 ms.

## Tests

`local_exposure_tests.rs` runs the production kernels on a GPU and checks that:

- A dark half beside a bright half moves each half toward middle grey by its own contrast, all the way
  to the edge (no halo).
- A uniform image metered to its own mean is unchanged.
- Contrasts of 1 change nothing.
- A fine texture inside a region keeps its contrast.
- Across soft shadow edges (shade 2–8 EV deep, penumbrae of 4–48 pixels), a sweep of input luminances
  at every position never gives a darker output for a brighter input. Without the isotonic pooling
  this test fails.

`abi_tests.rs` checks that the resolver blends the three properties like the Rust blender, and onto
their own bytes.
