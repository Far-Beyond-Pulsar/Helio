use super::*;

fn next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

#[test]
fn noise_intervals_enclose_exact_integer_samples() {
    let mut state = 0x48654c49;
    let mut single_cell = 0;
    let mut crossing = 0;
    let mut checked = 0;
    for shift in [7, 10, 14] {
        for seed in [0, 419, 73, 191, 311, u32::MAX] {
            for case in 0..384 {
                let low =
                    std::array::from_fn(|_| (next(&mut state) % 199_900_000) as i32 - 99_950_000);
                let width = if case < 192 {
                    1 + case % 8
                } else {
                    1 << (case % (shift + 1))
                };
                let high = std::array::from_fn(|a| low[a] + width as i32 - 1);
                let bounds = noise_range(low, high, shift, seed);
                if single_cell_noise_range(low, high, shift, seed).is_some() {
                    single_cell += 1;
                } else {
                    crossing += 1;
                }
                let mut check = |c| {
                    let actual = noise(c, shift, seed);
                    assert!((bounds[0]..=bounds[1]).contains(&actual),
                        "{low:?}..{high:?} shift={shift} seed={seed} cell={c:?}: {actual} outside {bounds:?}");
                    checked += 1;
                    if width == 1 {
                        assert_eq!(bounds, [actual, actual]);
                    }
                };
                if width <= 8 {
                    for z in low[2]..=high[2] {
                        for y in low[1]..=high[1] {
                            for x in low[0]..=high[0] {
                                check([x, y, z]);
                            }
                        }
                    }
                } else {
                    for i in 0..72 {
                        check(std::array::from_fn(|a| {
                            if i < 8 {
                                if i & (1 << a) == 0 {
                                    low[a]
                                } else {
                                    high[a]
                                }
                            } else {
                                low[a] + (next(&mut state) % width) as i32
                            }
                        }));
                    }
                }
            }
        }
    }
    assert!(single_cell > 4000 && crossing > 300 && checked > 500_000);
    eprintln!("NOISE_INTERVALS samples={checked} single_cell={single_cell} crossing={crossing}");
}

#[test]
fn lattice_boundaries_and_extreme_cells_keep_conservative_ranges() {
    for shift in [7, 10, 14] {
        for seed in [0u32, 419, u32::MAX] {
            let size = 1i32 << shift;
            let offsets = [
                seed.wrapping_mul(0x9e3779b9) ^ 0xa341316c,
                seed.wrapping_mul(0x85ebca6b) ^ 0xc8013ea4,
                seed.wrapping_mul(0xc2b2ae35) ^ 0xad90777d,
            ]
            .map(|x| (x & (size as u32 - 1)) as i32);
            for anchor in [-99_900_000i32, -size, 0, size, 99_900_000] {
                let boundary =
                    std::array::from_fn::<_, 3, _>(|a| anchor.div_euclid(size) * size - offsets[a]);
                for axis in 0..3 {
                    let low = std::array::from_fn(|a| boundary[a] + if a == axis { -3 } else { 1 });
                    let high = low.map(|v| v + 6);
                    assert!(single_cell_noise_range(low, high, shift, seed).is_none());
                    let bounds = noise_range(low, high, shift, seed);
                    for z in low[2]..=high[2] {
                        for y in low[1]..=high[1] {
                            for x in low[0]..=high[0] {
                                assert!((bounds[0]..=bounds[1]).contains(&noise(
                                    [x, y, z],
                                    shift,
                                    seed
                                )));
                            }
                        }
                    }
                }
            }
            for c in [[-LIMIT; 3], [LIMIT; 3], [-LIMIT, 0, LIMIT]] {
                assert_eq!(noise_range(c, c, shift, seed), [noise(c, shift, seed); 2]);
            }
        }
    }
}
