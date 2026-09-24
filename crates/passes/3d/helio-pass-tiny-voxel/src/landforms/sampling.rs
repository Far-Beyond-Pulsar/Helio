use super::LandformSnapshot;

pub const SAMPLING_SHADER: &str = include_str!("sampling.wgsl");
pub const MAX_CELL: i32 = 100_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceAddress {
    pub face: u32,
    pub u: u32,
    pub v: u32,
    pub fraction: [u32; 2],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellSample {
    pub address: FaceAddress,
    pub height_units: i32,
    pub solid: bool,
}

impl FaceAddress {
    pub fn from_cell(c: [i32; 3], resolution: u32) -> Option<Self> {
        if c.iter().any(|v| !(-MAX_CELL..=MAX_CELL).contains(v))
            || !resolution.is_power_of_two()
            || resolution > 256
        {
            return None;
        }
        let center = c.map(|v| i64::from(v) * 2 + 1);
        let mut axis = 0;
        for i in 1..3 {
            if center[i].abs() > center[axis].abs() {
                axis = i;
            }
        }
        let face = axis as u32 * 2 + u32::from(center[axis] < 0);
        let other = match axis {
            0 => [1, 2],
            1 => [0, 2],
            2 => [0, 1],
            _ => unreachable!(),
        };
        let radius = center[axis].unsigned_abs();
        let denominator = radius * 2;
        let project = |a: usize| {
            let numerator = (center[a] + radius as i64) as u64 * u64::from(resolution);
            let whole = (numerator / denominator).min(u64::from(resolution - 1));
            let remainder = numerator - whole * denominator;
            (whole as u32, ((remainder * 65536) / denominator) as u32)
        };
        let (u, fx) = project(other[0]);
        let (v, fy) = project(other[1]);
        Some(Self {
            face,
            u,
            v,
            fraction: [fx, fy],
        })
    }
}

impl LandformSnapshot {
    pub fn sample_cell(&self, c: [i32; 3]) -> Option<CellSample> {
        let address = FaceAddress::from_cell(c, self.resolution)?;
        let side = self.resolution as usize + 1;
        let index =
            address.face as usize * side * side + address.v as usize * side + address.u as usize;
        let lerp = |a: i32, b: i32, t: u32| {
            a + (((i64::from(b) - i64::from(a)) * i64::from(t)) >> 16) as i32
        };
        let a = lerp(
            self.heights[index],
            self.heights[index + 1],
            address.fraction[0],
        );
        let b = lerp(
            self.heights[index + side],
            self.heights[index + side + 1],
            address.fraction[0],
        );
        let height_units = lerp(a, b, address.fraction[1]);
        let radius = 127_420_000_i64 + i64::from(height_units);
        let squared = c
            .iter()
            .map(|v| {
                let value = i64::from(*v) * 2 + 1;
                (value * value) as u64
            })
            .sum::<u64>();
        Some(CellSample {
            address,
            height_units,
            solid: squared <= (radius * radius) as u64,
        })
    }
}
