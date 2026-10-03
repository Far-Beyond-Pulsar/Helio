use crate::grid::Grid;
use crate::landform::{LandformConstants, WARP_OCTAVES};
use crate::noise::{hash3, mul_fine, noise_fine, FINE_ONE};
use glam::IVec3;

pub(crate) const KNOTS: usize = 33;
pub(crate) const RIDGES: usize = 7;
const SAMPLE_COUNT: usize = 2048;

pub(crate) fn bake_ridge_suffix(
    grid: &Grid,
    k: &LandformConstants,
) -> Result<[[i32; 4]; 66], &'static str> {
    let count = usize::try_from(k.header[0]).map_err(|_| "invalid octave count")?;
    if !(WARP_OCTAVES..=k.octaves.len()).contains(&count) {
        return Err("invalid octave count");
    }
    let ridges: Vec<_> = k.octaves[WARP_OCTAVES..count]
        .iter()
        .filter(|o| o.kind == 2)
        .collect();
    if ridges.len() != RIDGES || ridges.windows(2).any(|o| o[0].shift < o[1].shift) {
        return Err("suffix filtering requires the stock ordered seven-ridge recipe");
    }
    // Current limb multiplication contract. Canonical validation currently
    // accepts values outside it; do not silently saturate the new bake.
    if k.octaves[..count]
        .iter()
        .any(|o| i64::from(o.amplitude).abs() >= (1 << 28))
    {
        return Err("recipe exceeds existing fixed-point amplitude contract");
    }
    let magnitude: i64 = ridges.iter().map(|o| i64::from(o.amplitude).abs()).sum();
    if magnitude >= (1 << 28) {
        return Err("ridge suffix exceeds the fixed-point limb contract");
    }

    let mut samples = Vec::<[i32; RIDGES]>::with_capacity(SAMPLE_COUNT);
    for index in 0..SAMPLE_COUNT {
        let face = grid.faces()[index % grid.faces().len()];
        let i = (hash3(index as i32, 0, 0, 0x51978213) % (grid.cells() as u32)) as i32;
        let j = (hash3(index as i32, 1, 0, 0x51978213) % (grid.cells() as u32)) as i32;
        let p = grid.domain_point(face, i, j, 0);
        let mut warp = [0i32; 3];
        for o in &k.octaves[..WARP_OCTAVES] {
            let axis = (o.kind - 4) as usize;
            if axis >= 3 {
                return Err("invalid warp axis");
            }
            warp[axis] = warp[axis]
                .checked_add(mul_fine(o.amplitude, noise_fine(p, o.shift, o.seed)))
                .ok_or("warp sum overflows")?;
        }
        let q = IVec3::new(
            p.x.checked_add(warp[0]).ok_or("warped domain overflows")?,
            p.y.checked_add(warp[1]).ok_or("warped domain overflows")?,
            p.z.checked_add(warp[2]).ok_or("warped domain overflows")?,
        );
        samples.push(std::array::from_fn(|r| {
            let o = ridges[r];
            let n = noise_fine(q, o.shift, o.seed);
            let ridge = (FINE_ONE - n.abs()).clamp(0, FINE_ONE - 1);
            mul_fine(ridge, ridge)
        }));
    }

    let mut packed = [[0i32; 4]; 66];
    for row in 0..RIDGES {
        for knot in 0..KNOTS {
            let incoming = (FINE_ONE / 4 + (knot as i32) * 393216).min(FINE_ONE - 1);
            let mut sum = 0i64;
            for sample in &samples {
                let mut weight = incoming;
                for r in row..RIDGES {
                    let v = mul_fine(sample[r], weight);
                    sum += i64::from(mul_fine(ridges[r].amplitude, v));
                    weight = (v * 2).clamp(FINE_ONE / 4, FINE_ONE - 1);
                }
            }
            let mean =
                i32::try_from(sum / (SAMPLE_COUNT as i64)).map_err(|_| "suffix mean overflows")?;
            let index = row * KNOTS + knot;
            packed[index / 4][index % 4] = mean;
        }
        for knot in 0..KNOTS - 1 {
            let index = row * KNOTS + knot;
            let next = index + 1;
            let delta =
                i64::from(packed[next / 4][next % 4]) - i64::from(packed[index / 4][index % 4]);
            if delta.abs() >= (1 << 28) {
                return Err("mean interpolation exceeds limb contract");
            }
        }
    }
    // Row7 is the zero terminal suffix. All values remain signed.
    Ok(packed)
}

/// Display and canonical suffixes share the interval from the sum of negative
/// amplitudes to the sum of positive amplitudes. Filtering and mask contraction
/// preserve it, so their difference needs only its width: the omitted absolute
/// amplitude sum already included in the canonical bounds and quantization pad.
pub(crate) fn render_bounds(_grid: &Grid, _k: &LandformConstants, bounds: [i32; 24]) -> [i32; 24] {
    bounds
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Shape;
    use crate::landform::Landform;

    // Exact i64 product is independent of the production limb implementation.
    fn product(a: i32, b: i32) -> i32 {
        (i64::from(a) * i64::from(b) / i64::from(FINE_ONE)) as i32
    }

    fn holdout(grid: &Grid, k: &LandformConstants) -> Vec<[i32; RIDGES]> {
        let ridges: Vec<_> = k.octaves[..k.header[0] as usize]
            .iter()
            .filter(|o| o.kind == 2)
            .collect();
        (0..8192)
            .map(|index| {
                // Distinct salt and sequence from the constructor's 2048 samples.
                let id = index + 19_937;
                let face = grid.faces()[index as usize % grid.faces().len()];
                let i = (hash3(id, 79, -15, 0xA49BA217) % grid.cells() as u32) as i32;
                let j = (hash3(id, -97, 31, 0xA49BA217) % grid.cells() as u32) as i32;
                let p = grid.domain_point(face, i, j, 0);
                let mut q = p;
                for o in &k.octaves[..WARP_OCTAVES] {
                    q[(o.kind - 4) as usize] +=
                        product(o.amplitude, noise_fine(p, o.shift, o.seed));
                }
                std::array::from_fn(|r| {
                    let o = ridges[r];
                    let value =
                        (FINE_ONE - noise_fine(q, o.shift, o.seed).abs()).clamp(0, FINE_ONE - 1);
                    product(value, value)
                })
            })
            .collect()
    }

    #[test]
    fn suffix_means_match_independent_full_ridge_holdout() {
        for grid in [
            Grid::new(6_371_000.0, 0.1).unwrap(),
            Grid::plane(Shape::Plane, 100_000.0, 0.3).unwrap(),
        ] {
            for mountain_km in [20.0, 0.000001, 1_000_000.0] {
                let land = Landform {
                    mountain_km,
                    ..Landform::default()
                };
                let k = LandformConstants::new(&grid, &land, 19);
                let lut = bake_ridge_suffix(&grid, &k).unwrap();
                let samples = holdout(&grid, &k);
                let ridges: Vec<_> = k.octaves[..k.header[0] as usize]
                    .iter()
                    .filter(|o| o.kind == 2)
                    .collect();
                for row in 0..RIDGES {
                    let magnitude: i64 = ridges[row..]
                        .iter()
                        .map(|o| i64::from(o.amplitude).abs())
                        .sum();
                    for half_knot in (0..=64).step_by(2).chain([7, 23, 47]) {
                        let incoming = (FINE_ONE / 4 + half_knot as i32 * 196608).min(FINE_ONE - 1);
                        let mut sum = 0i64;
                        for sample in &samples {
                            let mut weight = incoming;
                            for r in row..RIDGES {
                                let v = product(sample[r], weight);
                                sum += i64::from(product(ridges[r].amplitude, v));
                                weight = (v * 2).clamp(FINE_ONE / 4, FINE_ONE - 1);
                            }
                        }
                        let expected = sum / samples.len() as i64;
                        let delta = incoming - FINE_ONE / 4;
                        let knot = (delta / 393216).min(31) as usize;
                        let fraction = ((delta - knot as i32 * 393216) << 7) / 3;
                        let index = row * KNOTS + knot;
                        let lo = lut[index / 4][index % 4];
                        let hi = lut[(index + 1) / 4][(index + 1) % 4];
                        let actual = i64::from(lo) + i64::from(product(hi - lo, fraction));
                        // A statistical gate, not exact spatial reconstruction.
                        let error = (actual - expected).abs();
                        let tolerance = (magnitude * 15 / 1000).max(16);
                        assert!(error <= tolerance,
                            "wavelength={mountain_km}, row={row}, half-knot={half_knot}: {actual} vs {expected}, error={error}, allowed={tolerance}");
                    }
                }
                let all = i64::from(lut[(32) / 4][(32) % 4]);
                assert!(all > 0, "unresolved positive ridge mass must not vanish");
            }
        }
    }

    #[test]
    fn signed_amplitudes_repeated_shifts_and_unsafe_recipes_are_explicit() {
        let grid = Grid::new(6_371_000.0, 1.0).unwrap();
        let k = LandformConstants::new(&grid, &Landform::default(), 7);
        let positive = bake_ridge_suffix(&grid, &k).unwrap();
        let mut negative = k;
        for o in &mut negative.octaves[..negative.header[0] as usize] {
            if o.kind == 2 {
                o.amplitude = -o.amplitude;
            }
        }
        let neg = bake_ridge_suffix(&grid, &negative).unwrap();
        for index in 0..264 {
            assert_eq!(positive[index / 4][index % 4], -neg[index / 4][index % 4]);
        }
        let mut zero = k;
        for o in &mut zero.octaves[..zero.header[0] as usize] {
            if o.kind == 2 {
                o.amplitude = 0;
                o.shift = 1;
            }
        }
        assert_eq!(bake_ridge_suffix(&grid, &zero).unwrap(), [[0; 4]; 66]);
        let mut invalid = k;
        let first = invalid.octaves.iter().position(|o| o.kind == 2).unwrap();
        invalid.octaves[first].amplitude = 1 << 28;
        assert!(bake_ridge_suffix(&grid, &invalid).is_err());
        invalid = k;
        invalid.octaves[first + 1].shift = invalid.octaves[first].shift + 1;
        assert!(bake_ridge_suffix(&grid, &invalid).is_err());
        // Mixed signs test recipe; no positive-only assumption in the payload.
        let mut mixed = k;
        mixed.octaves[first].amplitude = -mixed.octaves[first].amplitude;
        assert!(bake_ridge_suffix(&grid, &mixed).unwrap()[8][0] < 0);
    }

    #[test]
    fn display_bounds_reuse_canonical_suffix_diameter() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        for signs in [0u32, 0b1111111, 0b0101010] {
            let mut k = LandformConstants::new(&grid, &Landform::default(), 7);
            for (r, o) in k.octaves.iter_mut().filter(|o| o.kind == 2).enumerate() {
                if signs & (1 << r) != 0 {
                    o.amplitude = -o.amplitude;
                }
            }
            let canonical = k.bound_margins(&grid);
            assert_eq!(render_bounds(&grid, &k, canonical), canonical);
            assert_eq!(canonical[17], 7);
            assert_eq!(canonical, k.bound_margins(&grid));
        }
        assert_eq!(std::mem::size_of::<LandformConstants>() + 66 * 16, 1616);
    }

    #[test]
    fn signed_suffix_mask_rounding_preserves_the_diameter() {
        // Mixed signs and non-integral mask products exercise the rounding
        // premise independently of the bake's statistical approximation.
        let negative = -2_105_360;
        let positive = 2_400_000;
        let diameter = positive - negative;
        let masks = [0, 1, FINE_ONE / 7, FINE_ONE / 2, FINE_ONE - 1, FINE_ONE];
        let prefixes = [-4_505_360, -1, 0, 1, 4_505_360];
        let mut minimum = i32::MAX;
        let mut maximum = i32::MIN;
        for prefix in prefixes {
            for region in masks {
                for land in masks {
                    let mask = |x| mul_fine(mul_fine(x, region), land);
                    let baseline = mask(prefix);
                    for tail in [negative, negative + 1, -1, 0, 1, positive - 1, positive] {
                        let actual = mask(prefix + tail) - baseline;
                        assert!((negative..=positive).contains(&actual));
                        minimum = minimum.min(actual);
                        maximum = maximum.max(actual);
                    }
                }
            }
        }
        assert_eq!(maximum - minimum, diameter);
    }

    #[test]
    #[ignore = "constructor microbenchmark; run optimized and without concurrent engine work"]
    fn constructor_bake_cost_and_cached_payload() {
        use crate::landform::LandformField;
        use crate::terrain::TerrainField;
        use std::{hint::black_box, time::Instant};
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let land = Landform::default();
        let k = LandformConstants::new(&grid, &land, 7);
        let mut bake = Vec::new();
        let mut constructor = Vec::new();
        for _ in 0..9 {
            let start = Instant::now();
            black_box(bake_ridge_suffix(&grid, &k).unwrap());
            bake.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            black_box(LandformField::new(&grid, &land, 7));
            constructor.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        bake.sort_by(f64::total_cmp);
        constructor.sort_by(f64::total_cmp);
        let field = LandformField::new(&grid, &land, 7);
        let first = field.program();
        assert_eq!(first.constants.len(), 1616);
        let start = Instant::now();
        for _ in 0..1024 {
            let program = field.program();
            assert_eq!(program.constants, first.constants);
            black_box(program);
        }
        let serialization = start.elapsed().as_secs_f64() * 1000.0 / 1024.0;
        eprintln!("recipe-only ridge bake: median={:.6}ms max={:.6}ms; full constructor: median={:.6}ms max={:.6}ms; cached program serialization={serialization:.6}ms/call",bake[4],bake[8],constructor[4],constructor[8]);
    }
}
