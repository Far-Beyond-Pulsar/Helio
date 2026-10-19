//! Jungle giant (kapok): a tall bending trunk on buttress roots under a
//! wide, layered, flattened canopy.
//!
//! A Rust port of `kapok_worker.py` of NGNT/treegen-pinegen (MIT, see
//! assets/trees/LICENSE-treegen-pinegen): the tapering trunk with its spine,
//! the root wedges with their ridges and twist, the per-layer watertight fill,
//! and the canopy: tiered branching with flattening and outward drift, and
//! the gravity-biased leaf walks. The vertical axis is +Y.

use super::rng::Rng;
use super::shapes::{clamp, normalize, smooth5, Volume, CENTRE};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct JungleParams {
    /// Trunk height (blocks).
    pub height: f64,
    pub trunk_iter: u32,
    pub trunk_size: f64,
    pub bend: f64,
    pub root_twist: f64,
    pub root_profile: f64,
    pub root_spread: f64,
    pub root_count: u32,
    pub root_variance: f64,
    pub canopy_iter: u32,
    pub size: f64,
    pub wide: f64,
    pub spread: f64,
    pub canopy_twist: f64,
    /// Leaf walk density (the original's `leaves`).
    pub leaves: f64,
    pub gravity: f64,
    pub canopy_thick: f64,
    pub canopy_profile: f64,
    pub canopy_flat: f64,
    pub canopy_start: f64,
}

impl JungleParams {
    pub fn jungle() -> Self {
        Self {
            height: 15.0,
            trunk_iter: 24,
            trunk_size: 1.25,
            bend: 0.5,
            root_twist: 0.3,
            root_profile: 0.5,
            root_spread: 0.45,
            root_count: 5,
            root_variance: 1.0,
            canopy_iter: 8,
            size: 0.22,
            wide: 0.6,
            spread: 0.5,
            canopy_twist: 0.5,
            leaves: 0.5,
            gravity: 0.0,
            canopy_thick: 0.25,
            canopy_profile: 0.9,
            canopy_flat: 0.75,
            canopy_start: 0.7,
        }
    }
}

struct Root {
    height: f64,
    fan_cap: f64,
    peak_scale: f64,
    wedge_half: f64,
    steps: i32,
    ridge: f64,
    sink_start: f64,
    phase_jitter: f64,
}

pub fn generate(p: &JungleParams, seed: u64) -> Volume {
    let mut rng = Rng::new(seed);
    let mut v = Volume::default();

    let iter = p.trunk_iter.clamp(8, 40) as usize;
    let h = p.height;
    let seg = h / iter as f64;
    let base_r = 6.0 * p.trunk_size * p.size;
    let bend_var = 0.06 * p.bend;
    const NAT_TAPER: f64 = 0.18;
    const BLEND_ZONE: f64 = 0.40;
    const JOIN_BLEND: f64 = 2.0;
    const JOIN_HOLD: u32 = 5;

    let root_height_base = 0.62 + 0.36 * p.root_profile;
    let spread_s = p.root_spread.powf(2.2);
    let fan_base = (0.25 + 2.20 * p.root_profile) * base_r * 0.60;
    let fan_cap_base = fan_base * spread_s;
    let wedge_half_base = 0.12 + 0.18 * p.root_profile;
    let ang_steps_base = (22.0 + 18.0 * p.root_profile).floor().max(14.0);
    let ridge_amp_base = 0.35 + 0.25 * p.root_profile;
    const RIDGE_SHARP: f64 = 1.6;
    let sink_start_base = 0.35 + 0.35 * p.root_profile;
    const SEAM_OVERLAP: f64 = 3.2;
    const RADIAL_PAD: f64 = 1.0;

    let mut cx = vec![0.0; iter + 3];
    let mut cy = vec![0.0; iter + 3];
    let mut cz = vec![0.0; iter + 3];
    let mut cr = vec![0.0; iter + 3];

    let radius_at = |t: f64| {
        let mut r = base_r * (1.0 - NAT_TAPER * t.powf(1.2));
        let t0 = 1.0 - BLEND_ZONE;
        if t > t0 {
            let u = smooth5((t - t0) / BLEND_ZONE);
            r = r * (1.0 - u) + base_r * (1.0 - NAT_TAPER) * u;
        }
        r
    };

    // The trunk: a bending chain of capsules, remembering its spine.
    let (mut x, mut y, mut z) = (CENTRE, 0.0, CENTRE);
    let (mut dx, mut dz) = (0.0f64, 0.0f64);
    for i in 1..=iter {
        dx += rng.uniform(-bend_var, bend_var);
        dz += rng.uniform(-bend_var, bend_var);
        let d = normalize([dx, 1.0, dz], [0.0, 1.0, 0.0]);
        let end = [x + d[0] * seg, y + d[1] * seg, z + d[2] * seg];
        let (t0, t1) = ((i - 1) as f64 / iter as f64, i as f64 / iter as f64);
        v.capsule([x, y, z], end, radius_at(t0), radius_at(t1), true);
        cx[i] = end[0];
        cy[i] = end[1];
        cz[i] = end[2];
        cr[i] = radius_at(t1);
        (x, y, z) = (end[0], end[1], end[2]);
    }
    for k in [1usize, 2] {
        cx[iter + k] = cx[iter];
        cy[iter + k] = cy[iter];
        cz[iter + k] = cz[iter];
        cr[iter + k] = cr[iter];
    }
    // The core's stitch keeps a thin solid spine through the whole trunk.
    for i in 2..=iter {
        v.capsule([cx[i - 1], cy[i - 1], cz[i - 1]], [cx[i], cy[i], cz[i]], cr[i - 1] * 0.14, cr[i] * 0.14, true);
    }
    let sample = |yf: f64| -> (f64, f64, f64) {
        let f = yf / h * iter as f64;
        let i = (f.floor() as i64).clamp(1, iter as i64 - 1) as usize;
        let a = f - i as f64;
        (cx[i] + (cx[i + 1] - cx[i]) * a, cz[i] + (cz[i + 1] - cz[i]) * a, cr[i] + (cr[i + 1] - cr[i]) * a)
    };

    // Canopy anchor: the trunk's top and its direction.
    let anchor = [cx[iter], cy[iter], cz[iter]];
    let anchor_dir = normalize([cx[iter] - cx[iter - 1], cy[iter] - cy[iter - 1], cz[iter] - cz[iter - 1]], [0.0, 1.0, 0.0]);
    let join_radius = (cr[iter] - 0.8).max(0.0);
    let trunk_size_base = 6.0 * p.trunk_size * p.size;
    let base_scale = clamp(join_radius / (trunk_size_base + 1e-6), 0.15, 4.0);

    // Buttress roots.
    let make_root = |rng: &mut Rng| {
        let mut j = |amount: f64| rng.uniform(-1.0, 1.0) * amount * p.root_variance;
        let height_f = clamp(1.0 + j(0.40), 0.65, 1.25);
        let spread_f = clamp(1.0 + j(0.50), 0.0, 1.50);
        let thick_f = clamp(1.0 + j(0.45), 0.25, 1.75);
        let wedge_f = clamp(1.0 + j(0.35), 0.60, 1.50);
        let ridge_f = clamp(1.0 + j(0.50), 0.40, 2.00);
        let sink = clamp(sink_start_base + j(0.18), 0.05, 0.90);
        let phase = j(0.25);
        Root {
            height: clamp(root_height_base * height_f, 0.55, 0.985),
            fan_cap: (fan_cap_base * spread_f).max(0.0),
            peak_scale: thick_f,
            wedge_half: clamp(wedge_half_base * wedge_f, 0.07, 0.35),
            steps: ((ang_steps_base * (0.9 + 0.4 * p.root_variance)).floor() as i32).max(12),
            ridge: ridge_amp_base * ridge_f,
            sink_start: sink,
            phase_jitter: phase,
        }
    };
    let mut draw_root = |v: &mut Volume, theta0: f64, r: &Root| {
        let y_max = ((h * clamp(r.height, 0.0, 0.985) + 0.5).floor()).min(h.floor() - 1.0).max(0.0) as i32;
        let twist = p.root_twist * 1.3 * std::f64::consts::PI;
        let thin = 0.12 + 0.88 * p.root_spread.powf(1.2);
        for y in 0..=y_max {
            let t_h = if y_max == 0 { 0.0 } else { f64::from(y) / f64::from(y_max) };
            let m_raw = if (1.0 - r.sink_start) != 0.0 { (t_h - r.sink_start) / (1.0 - r.sink_start) } else { 0.0 };
            let u = smooth5(m_raw);
            let hold = if u < 0.8 { 1.0 } else { 1.0 - smooth5((u - 0.8) / 0.2) };
            let theta_c = theta0 + t_h * twist;
            let (sx, sz, r_trunk) = sample(f64::from(y) + 0.35);
            let off = (r.fan_cap * (1.0 - t_h.powi(2))) * (1.0 - u);
            let peak_h = (1.0 - t_h.powf(0.40)) * r_trunk;
            let peak = (0.45 + 0.45 * p.root_profile) * peak_h * (1.0 - u) * thin * r.peak_scale;
            let seam = SEAM_OVERLAP * (1.0 - 0.35 * t_h);
            let sink = u * r_trunk;
            let r_in = (r_trunk - seam - sink).max(0.0);
            let r_surf = r_trunk + off;
            let r_hold = r_trunk - 1.0;
            let r_skin = r_hold * hold + (r_surf * (1.0 - u)) * (1.0 - hold);
            for s in 0..=r.steps {
                let w = f64::from(s) / f64::from(r.steps) * 2.0 - 1.0;
                let ang = theta_c + w * r.wedge_half;
                let tri = (1.0 - w.abs()).max(0.0);
                let ridge = (r.ridge * thin) * tri.powf(RIDGE_SHARP) * (1.0 - t_h) * (1.0 - u) * r_trunk;
                let thick = peak * tri + ridge;
                let mut r_out = (r_skin + thick) * (1.0 - u);
                if r_out < r_in {
                    r_out = r_in;
                }
                let (ux, uz) = (ang.cos(), ang.sin());
                let from = (r_in - RADIAL_PAD * (1.0 - u)).max(0.0).floor() as i32;
                for rr in from..=r_out.floor() as i32 {
                    let cell = [(sx + ux * f64::from(rr) + 0.5).floor() as i32, y, (sz + uz * f64::from(rr) + 0.5).floor() as i32];
                    if v.get(cell).is_none() {
                        v.set_wood(cell);
                    }
                }
            }
        }
        for yy in (y_max - 3).max(0)..=(y_max.min(h.floor() as i32 - 1)) {
            let (sx, sz, _) = sample(f64::from(yy) + 0.5);
            v.set_wood([(sx + 0.5).floor() as i32, yy, (sz + 0.5).floor() as i32]);
        }
    };
    if p.root_count > 0 {
        let base_phase = rng.uniform(0.0, std::f64::consts::TAU);
        for k in 1..=p.root_count {
            let r = make_root(&mut rng);
            let theta = base_phase
                + f64::from(k - 1) * (std::f64::consts::TAU / f64::from(p.root_count.max(1)))
                + r.phase_jitter * (std::f64::consts::TAU / f64::from(p.root_count.max(3)));
            draw_root(&mut v, theta, &r);
        }
    }
    fill_watertight(&mut v, base_r + fan_base + 6.0, h);

    // The canopy has its own stream so roots never change it.
    let mut rng = Rng::new(seed.wrapping_add(1337));
    let canopy_iter = p.canopy_iter.clamp(5, 15);
    let fiter = f64::from(canopy_iter);
    let c_size = 150.0 * p.size / fiter;
    let g_trunk = 6.0 * p.trunk_size * p.size;
    let len0 = c_size * (1.0 - p.wide);
    let len1 = c_size * p.wide;
    let t_of = |i: u32| ((f64::from(i) - 1.0) / fiter).sqrt();
    let flatness = |i: u32| {
        let t = t_of(i);
        if t <= p.canopy_start || p.canopy_flat <= 0.0 {
            return 0.0;
        }
        let denom = 1.0 - p.canopy_start;
        if denom <= 1e-4 {
            return 0.0;
        }
        clamp(clamp((t - p.canopy_start) / denom, 0.0, 1.0) * p.canopy_flat, 0.0, 1.0)
    };
    let branch_size = |i: u32| {
        let alpha = 1.0 + 2.5 * p.canopy_thick;
        (1.0 - t_of(i).powf(alpha)) * g_trunk
    };
    let branch_length = |i: u32| len0 + t_of(i) * (len1 - len0);
    let branch_angle = |i: u32| 2.0 * p.spread * t_of(i);
    let branch_probability = |i: u32| {
        let t = t_of(i);
        let strength = 0.15 + 0.55 * p.canopy_profile;
        let gauss = |x: f64, m: f64, s: f64| (-((x - m) * (x - m)) / (2.0 * s * s)).exp();
        let q = t * (1.0 - 0.5 * strength)
            + strength
                * (1.00 * gauss(t, 0.60, 0.10 - 0.04 * p.canopy_profile)
                    + 0.75 * gauss(t, 0.78, 0.08 - 0.03 * p.canopy_profile)
                    + 0.55 * gauss(t, 0.92, 0.06 - 0.02 * p.canopy_profile));
        clamp(q, 0.0, 1.0)
    };

    let mut tips: Vec<[f64; 3]> = Vec::new();
    struct Canopy<'a> {
        sample: &'a dyn Fn(f64) -> (f64, f64, f64),
        flatness: &'a dyn Fn(u32) -> f64,
        size: &'a dyn Fn(u32) -> f64,
        length: &'a dyn Fn(u32) -> f64,
        angle: &'a dyn Fn(u32) -> f64,
        probability: &'a dyn Fn(u32) -> f64,
        anchor_dir: [f64; 3],
        base_scale: f64,
        join_radius: f64,
        iters: u32,
    }
    #[allow(clippy::too_many_arguments)]
    fn grow(
        c: &Canopy,
        p: &JungleParams,
        rng: &mut Rng,
        v: &mut Volume,
        tips: &mut Vec<[f64; 3]>,
        pos: [f64; 3],
        dir: [f64; 3],
        i: u32,
    ) {
        let l = (c.length)(i);
        let mut s0 = (c.size)(i) * c.base_scale;
        let mut s1 = (c.size)(i + 1) * c.base_scale;
        if i <= JOIN_HOLD {
            let hold = f64::from(i - 1) / f64::from(JOIN_HOLD.max(1));
            s0 = c.join_radius * (1.0 - hold) + s0 * hold;
            s1 = c.join_radius * (1.0 - hold) + s1 * hold;
        }
        if p.canopy_thick > 0.0 {
            let boost = 1.0 + 0.25 * p.canopy_thick * (-dir[1]).max(0.0);
            s0 *= boost;
            s1 *= boost;
        }
        let end = [pos[0] + dir[0] * l, pos[1] + dir[1] * l, pos[2] + dir[2] * l];
        v.capsule(pos, end, s0, s1, true);
        if i < c.iters {
            let mut forks = 1;
            let mut var = f64::from(i) * 0.2 * p.canopy_twist;
            if rng.f64() < (c.probability)(i) {
                forks = 2;
                var = (c.angle)(i);
            }
            for _ in 0..forks {
                let mut d = [dir[0] + rng.uniform(-var, var), dir[1] + rng.uniform(-var, var), dir[2] + rng.uniform(-var, var)];
                let (sx, sz, _) = (c.sample)(end[1]);
                let (rx, rz) = (end[0] - sx, end[2] - sz);
                let orad = rx.hypot(rz);
                let t = f64::from(i) / f64::from(c.iters);
                if orad > 1e-4 && t > p.canopy_start {
                    let ow = ((t - p.canopy_start) / (1.0 - p.canopy_start)) * (0.35 * p.canopy_profile);
                    d[0] += rx / orad * ow;
                    d[2] += rz / orad * ow;
                }
                d[1] *= 1.0 - (c.flatness)(i + 1);
                let blend = clamp((JOIN_BLEND - f64::from(i - 1)) / JOIN_BLEND, 0.0, 1.0);
                for k in 0..3 {
                    d[k] = d[k] * (1.0 - blend) + c.anchor_dir[k] * blend;
                }
                grow(c, p, rng, v, tips, end, normalize(d, [0.0, 1.0, 0.0]), i + 1);
            }
        } else {
            tips.push(end);
            tips.push([(pos[0] + end[0]) / 2.0, (pos[1] + end[1]) / 2.0, (pos[2] + end[2]) / 2.0]);
        }
    }
    let canopy = Canopy {
        sample: &sample,
        flatness: &flatness,
        size: &branch_size,
        length: &branch_length,
        angle: &branch_angle,
        probability: &branch_probability,
        anchor_dir,
        base_scale,
        join_radius,
        iters: canopy_iter,
    };
    grow(&canopy, p, &mut rng, &mut v, &mut tips, anchor, anchor_dir, 1);

    // Leaves: gravity-biased random walks from every tip.
    let chunks = (5.0 * p.leaves).ceil() as i32;
    let per_chunk = (50.0 * p.leaves).ceil() as i32;
    let vert = (2.0 / 6.0) * (1.0 - 0.85 * p.canopy_profile);
    let step_prob = (1.0 - vert) / 4.0;
    let gravity = clamp(p.gravity - 0.35 * p.canopy_profile, -1.0, 1.0);
    let mut seen: HashSet<[i32; 3]> = HashSet::new();
    let mut cells: Vec<[i32; 3]> = Vec::new();
    for t in &tips {
        for _ in 0..chunks {
            let mut c = [t[0] as i32, t[1] as i32, t[2] as i32];
            for _ in 0..per_chunk {
                if seen.insert(c) {
                    cells.push(c);
                }
                let r = rng.f64();
                if r < step_prob {
                    c[0] -= 1;
                } else if r < step_prob * 2.0 {
                    c[0] += 1;
                } else if r < step_prob * 3.0 {
                    c[2] -= 1;
                } else if r < step_prob * 4.0 {
                    c[2] += 1;
                } else if rng.uniform(-1.0, 1.0) > gravity {
                    c[1] -= 1;
                } else {
                    c[1] += 1;
                }
            }
        }
    }
    for c in cells {
        if c[1] >= 3 {
            v.set_leaf(c);
        }
    }
    v
}

/// Fill every enclosed hole of each horizontal layer of the trunk and roots:
/// air not reachable from outside the layer's window becomes wood.
fn fill_watertight(v: &mut Volume, reach: f64, height: f64) {
    let r = reach.ceil() as i32;
    let (c0, c1) = (CENTRE as i32 - r, CENTRE as i32 + r);
    let mut fills = Vec::new();
    for y in 0..=(height.ceil() as i32) {
        let mut outside: HashSet<[i32; 2]> = HashSet::new();
        let mut queue: Vec<[i32; 2]> = Vec::new();
        let mut push = |x: i32, z: i32, v: &Volume, outside: &mut HashSet<[i32; 2]>, queue: &mut Vec<[i32; 2]>| {
            if v.get([x, y, z]).is_none() && outside.insert([x, z]) {
                queue.push([x, z]);
            }
        };
        for a in c0..=c1 {
            push(a, c0, v, &mut outside, &mut queue);
            push(a, c1, v, &mut outside, &mut queue);
            push(c0, a, v, &mut outside, &mut queue);
            push(c1, a, v, &mut outside, &mut queue);
        }
        let mut head = 0;
        while head < queue.len() {
            let [x, z] = queue[head];
            head += 1;
            for [nx, nz] in [[x - 1, z], [x + 1, z], [x, z - 1], [x, z + 1]] {
                if (c0..=c1).contains(&nx) && (c0..=c1).contains(&nz) {
                    push(nx, nz, v, &mut outside, &mut queue);
                }
            }
        }
        for x in c0..=c1 {
            for z in c0..=c1 {
                if v.get([x, y, z]).is_none() && !outside.contains(&[x, z]) {
                    fills.push([x, y, z]);
                }
            }
        }
    }
    for c in fills {
        v.set_wood(c);
    }
}
