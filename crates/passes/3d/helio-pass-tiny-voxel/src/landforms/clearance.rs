/// Constants for a continuous bound of the quantized radial/detail field.
/// Shared by build-time WGSL generation and the CPU query adapter.
/// See tools/voxel-planet/FIELD-CLEARANCE.md for the derivation.
#[derive(Clone, Copy, Debug)]
pub struct FieldClearance {
    pub lipschitz: f64,
    pub quantization_guard: f64,
    pub maximum_radius: f64,
}
impl FieldClearance {
    pub fn from_atlas(n: u32, heights: &[i32]) -> Self {
        assert!(n.is_power_of_two() && n <= 256);
        let side = n as usize + 1;
        assert_eq!(heights.len(), 6 * side * side);
        let mut edge = 0i64;
        for face in 0..6 {
            for y in 0..side {
                for x in 0..side {
                    let i = face * side * side + y * side + x;
                    if x + 1 < side {
                        edge = edge.max((i64::from(heights[i]) - i64::from(heights[i + 1])).abs());
                    }
                    if y + 1 < side {
                        edge =
                            edge.max((i64::from(heights[i]) - i64::from(heights[i + side])).abs());
                    }
                }
            }
        }
        let minimum_radius =
            (127_420_000.0 + f64::from(*heights.iter().min().unwrap()) - 8280.0) * 0.05;
        assert!(minimum_radius > 0.0);
        let base_lipschitz = 5.0 * f64::from(n) * edge as f64 * 0.05 / minimum_radius;
        let axis_detail: f64 = [(14, 8000.0), (10, 480.0), (7, 40.0)]
            .into_iter()
            .map(|(shift, amplitude)| {
                0.05 * amplitude / 32768.0 * 1.5 * 65535.0 / (0.1 * f64::from(1u32 << shift))
            })
            .sum();
        let base_error = 0.05 * (2.0 * edge as f64 / 65536.0 + 2.0);
        let detail_error = 0.05 * (6.0 * 8520.0 / 32768.0 + 3.0);
        // Upward binary-grid rounding; these constants are exactly representable
        // in f32 over the validated snapshot ranges. Radius rounds down to metres.
        Self {
            lipschitz: ((1.0 + base_lipschitz + 3.0f64.sqrt() * axis_detail) * 1024.0).ceil()
                / 1024.0,
            quantization_guard: ((2.0 * (base_error + detail_error) + 1.0) * 1024.0).ceil()
                / 1024.0,
            maximum_radius: (minimum_radius * 0.5).floor(),
        }
    }
    /// Caller has already established procedural air at this canonical centre.
    /// Voxel extents, query offset and later additions must be guarded separately.
    pub fn empty_radius(self, air_depth: f64) -> f64 {
        if !air_depth.is_finite() {
            return 0.0;
        }
        ((air_depth * 0.99999 - self.quantization_guard) / self.lipschitz * 0.99999)
            .clamp(0.0, self.maximum_radius)
    }
}
