//! Scans the canonical height field around a spawn for discontinuities.
use helio_pass_voxel_planet::{field, Planet, PlanetRecipe};
fn main() {
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let g = *planet.grid();
    let n = f64::from(g.cells());
    // Same search as the flight harness ground spawn.
    let (mut fi, mut fj) = (0.0, 0.0);
    'find: for step in 0..4000 {
        let a = 0.47 + 0.002 * f64::from(step % 60);
        let b = 0.53 + 0.002 * f64::from(step / 60);
        let (i, j) = ((a * n) as i32, (b * n) as i32);
        if planet.column_top(2, i, j, 0) > 200 {
            fi = f64::from(i);
            fj = f64::from(j);
            break 'find;
        }
    }
    let (ci, cj) = (fi as i32, fj as i32);
    let mut jumps = 0;
    for dj in -600..600 {
        for di in -600..600 {
            let (i, j) = (ci + di, cj + dj);
            let a = planet.column_height(2, i, j, 0);
            let b = planet.column_height(2, i + 1, j, 0);
            if (a - b).abs() > 400 {
                jumps += 1;
                if jumps <= 6 {
                    let p = g.domain_point(2, i, j, 0);
                    let q = g.domain_point(2, i + 1, j, 0);
                    eprintln!("jump at di {di} dj {dj}: {a} -> {b} mm; domain {p} -> {q}");
                    let k = planet.field();
                    let warp = |p: glam::IVec3| {
                        let mut w = [0i32; 3];
                        for o in &k.octaves[..6] {
                            w[(o.kind - 4) as usize] += field::scale(field::noise(p, o.shift, o.seed), o.amplitude);
                        }
                        p + glam::IVec3::from_array(w)
                    };
                    let (qa, qb) = (warp(p), warp(q));
                    eprintln!("  warped {qa} -> {qb}");
                    for o in &k.octaves[6..k.header[1] as usize] {
                        let (sa, sb) = if o.kind == 7 { (p, q) } else { (qa, qb) };
                        let na = field::scale(field::noise(sa, o.shift, o.seed), o.amplitude);
                        let nb = field::scale(field::noise(sb, o.shift, o.seed), o.amplitude);
                        if (na - nb).abs() > 20 {
                            eprintln!("  octave shift {} kind {} amp {}: raw {} -> {} = {na} -> {nb}", o.shift, o.kind, o.amplitude, field::noise(sa, o.shift, o.seed), field::noise(sb, o.shift, o.seed));
                        }
                    }
                }
            }
        }
    }
    eprintln!("jumps > 0.4 m between x-neighbours: {jumps} in 1200x1200 cells");
}
