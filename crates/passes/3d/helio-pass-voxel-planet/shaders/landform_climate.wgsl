// Landform's bound uses absolute octave amplitudes and gradient bounds,
// so it bounds both signs of the change from a coarse centre to any finer
// column inside it. The extra two cells also cover top quantization.
// Height only affects basin classification and the clamped alpine weight.
// Outside those transitions every possible height in the interval produces
// the same material, for any moisture, depth, slope or strata layer.
fn climate_height_reusable(top: i32, level: u32) -> bool {
    if terrain.levels.z <= 0 { return false; }
    let cell_mm = f32(world.grid.y) * f32(1u << level);
    let margin = f32(bound_margin(level));
    let lo = (f32(top) - margin) * cell_mm;
    let hi = (f32(top) + margin) * cell_mm;
    let basin = f32(terrain.levels.w);
    if hi < basin - 1.0 { return true; }
    let rock_min = f32(terrain.levels.z - terrain.levels.z / 8 - terrain.levels.z / 3) - 1.0;
    let snow_max = f32(terrain.levels.z + terrain.levels.z / 8) + 1.0;
    return lo >= basin + 1.0 && (hi < rock_min || lo > snow_max);
}
