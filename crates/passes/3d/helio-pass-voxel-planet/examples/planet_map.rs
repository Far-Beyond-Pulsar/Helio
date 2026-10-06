//! Writes an equirectangular height/material preview (PPM) of the default planet.
use helio_pass_voxel_planet::{grid::face_of, terrain, Planet, PlanetRecipe};
use glam::DVec3;
fn main() {
    let out = std::env::args().nth(1).unwrap_or("planet_map.ppm".into());
    let (w, h) = (1024usize, 512usize);
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let g = *planet.grid();
    let mut img = vec![0u8; w * h * 3];
    let (mut land, mut lo, mut hi) = (0usize, f64::MAX, f64::MIN);
    for y in 0..h {
        for x in 0..w {
            let lon = (x as f64 + 0.5) / w as f64 * std::f64::consts::TAU;
            let lat = (0.5 - (y as f64 + 0.5) / h as f64) * std::f64::consts::PI;
            let d = DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
            let face = face_of(d);
            let c = g.face_coords(face, d * g.radius()).unwrap();
            let (i, j) = (c[0] as i32, c[1] as i32);
            let level = 8;
            let hh = planet.column_height(face, i >> level, j >> level, level);
            let m = f64::from(hh) / 1000.0;
            lo = lo.min(m); hi = hi.max(m);
            let rgb = if m < 0.0 { land += 0; let t = (1.0 + m / 3000.0).clamp(0.0, 1.0); [(20.0 + 40.0 * t) as u8, (50.0 + 80.0 * t) as u8, (110.0 + 90.0 * t) as u8] } else {
                land += 1;
                let top = terrain::top_cells(&g, hh, level);
                let p = g.domain_point(face, i >> level, j >> level, level);
                let surface = planet.field().surface(p, level + g.level_offset(), hh) & 0xffff;
                let mat = planet.field().ground_material(p, surface, hh, 0, 0, top - 1) & terrain::material::ID;
                let base = match mat { 1 => [80, 130, 50], 4 => [210, 190, 130], 5 => [240, 240, 245], 8 | 12 => [190, 110, 70], 3 | 9 => [120, 120, 120], _ => [140, 100, 70] };
                let shade = (0.6 + m / 6000.0).clamp(0.4, 1.3);
                base.map(|v: i32| (f64::from(v) * shade).min(255.0) as u8)
            };
            img[(y * w + x) * 3..][..3].copy_from_slice(&rgb);
        }
    }
    let mut data = format!("P6 {w} {h} 255\n").into_bytes();
    data.extend_from_slice(&img);
    std::fs::write(&out, data).unwrap();
    println!("land fraction {:.2}  height {lo:.0}..{hi:.0} m", land as f64 / (w * h) as f64);
}
